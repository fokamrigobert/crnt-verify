"""
crnt_gym.py — the crnt-verify RL environment (Phase 2 of the project)
=======================================================================

Turns crnt-verify from a static grader into something an RL training loop
can actually run against: reset() hands the policy a freshly generated
reaction-network problem, step(action) grades the policy's completion and
returns a reward, exactly like any Gymnasium-style environment.

Design principles (stated up front because they drove every choice below):

1. Framework-agnostic core. Verifier libraries in this space move fast --
   see the Prime Intellect `verifiers` note at the bottom of this file for
   a concrete, dated example of exactly that happening mid-project. So the
   core (task generation, prompt rendering, answer parsing, reward) has NO
   dependency on any specific RL/eval framework. Adapters to a specific
   framework are thin, separate, and clearly marked as such.

2. Reuse, don't reimplement. Task ground truth comes from crnt_solver.py's
   solve_network(); grading reuses crnt_checker.py's check_prediction() and
   its subspace/randomized-substitution logic directly. If the checker is
   later improved, the environment improves for free.

3. Deterministic task generation. Every task is generated from an explicit
   integer seed. Same seed -> same network, same ground truth, every time
   -- necessary for reproducible curricula and for regression-testing the
   environment itself the same way crnt_checker.py's own self-tests do.

4. Reward has two modes, and the difference is deliberate:
     - "binary": exactly check_prediction()'s verdict. This is the honest
       verifiable-reward signal -- no partial credit for a wrong answer
       that merely looks close.
     - "shaped": average of the three per-invariant sub-checks. Useful as
       a denser training signal early on (all-or-nothing binary reward is
       a known problem for cold-start RL), but it is explicitly a
       *training convenience*, not a claim that partial CRNT answers are
       partially correct science. Said so directly in reset()'s info dict
       and again here so it can't be missed.
"""

from __future__ import annotations
import json
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import signal
import sympy as sp

import os
# Import Phase 1 from THIS file's own directory, not the caller's working
# directory -- so `python3 /anywhere/crnt-verify/crnt_gym.py` works, and so
# does importing it from another folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crnt_solver import ReactionNetwork, Reaction, solve_network       # noqa: E402
from crnt_checker import (                                             # noqa: E402
    check_prediction, _normalize, _check_subspace, _check_steady_states,
    _SPECIES_PATTERN, _FLUX_PATTERN,
)


# --------------------------------------------------------------------------- #
# 1. Procedural task generator
# --------------------------------------------------------------------------- #
# Every generated network is a single directed cycle of complexes, each
# complex either a lone species or a pair -- the same family as the worked
# examples in the research note, chosen because mass-action steady states on
# a simple cycle solve reliably (equal-flux condition) rather than because
# it's the only interesting family. Extending the generator to branched /
# multi-linkage-class networks is a natural Phase 3, flagged at the bottom.

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


ANSWER_KEYS = ("conservation laws", "steady states", "flux balance")


@dataclass
class CRNTTask:
    seed: int
    difficulty: int                 # number of complexes/reactions in the cycle
    species: list
    reactions: list                 # list of (reactant_name, product_name, coeffs) for display
    net: ReactionNetwork
    ground_truth: dict              # ONLY the three ANSWER_KEYS -- guaranteed JSON-serializable
    diagnostics: dict = field(default_factory=dict)   # rank, deficiency, etc., stringified


def _split_solution(full: dict):
    """solve_network() returns the three answer keys plus a `_diagnostics` entry
    holding raw sympy objects (Matrix, Integer, ...). Those are useful to read
    but are NOT JSON-serializable, and leaking them into `ground_truth` broke
    both the HuggingFace Dataset builder and the eval-results writer during
    development. Keep ground_truth strictly clean; stringify diagnostics."""
    clean = {k: full[k] for k in ANSWER_KEYS}
    diags = {}
    for k, v in (full.get("_diagnostics") or {}).items():
        diags[k] = str(v)
    return clean, diags


class _SolveTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _SolveTimeout()


# SIGALRM exists on Unix/macOS but NOT on Windows. Detected once at import
# rather than assumed -- an earlier version assumed Unix and crashed
# immediately on Windows with "module 'signal' has no attribute 'SIGALRM'".
_HAS_SIGALRM = hasattr(signal, "SIGALRM")


def _solve_with_budget(net: ReactionNetwork, seconds: float = 4.0) -> Optional[dict]:
    """Wall-clock-bounded call to solve_network(). Returns None on timeout or on
    any solver failure, rather than ever letting reset() hang -- a training loop
    cannot tolerate an environment step that might not return.

    Two implementations, chosen automatically:
      * Unix/macOS: SIGALRM interrupts the running solve directly (clean).
      * Windows:    the solve runs in a daemon thread which is *abandoned* on
                    timeout. Python cannot forcibly kill a thread, so an
                    abandoned solve keeps running in the background until it
                    finishes or the process exits. This is acceptable here
                    because timeouts are rare and the fallback path generates a
                    simpler network immediately; but it does mean a timed-out
                    task costs some background CPU on Windows. A
                    multiprocessing-based version would be genuinely killable,
                    at the cost of pickling overhead on every task.
    """
    if _HAS_SIGALRM:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return solve_network(net)
        except _SolveTimeout:
            return None
        except Exception:
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    # --- Windows path: thread with abandonment ---
    import threading

    box = {}

    def _work():
        try:
            box["result"] = solve_network(net)
        except Exception:
            box["result"] = None

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        return None                      # timed out; thread abandoned (daemon)
    return box.get("result")


def generate_task(seed: int, difficulty: int = 3, dimerization_prob: float = 0.35,
                   _retry_depth: int = 0) -> CRNTTask:
    """Deterministically generate a solvable single-linkage-class CRN of a given size.

    difficulty = number of complexes = number of reactions in the cycle (>=3).
    dimerization_prob = chance each complex is a 2-species sum (e.g. "A+B")
                         rather than a single species -- raises deficiency
                         and stoichiometric-coefficient variety.

    Reliability note: symbolic mass-action solving (sympy.solve) is occasionally
    very slow on dimerized topologies -- observed up to ~12s, and worse for
    larger difficulty, in testing. Every attempt is wall-clock-bounded; on
    timeout this falls back to a pure single-species cycle (dimerization_prob=0)
    at the same difficulty and seed, which was verified fast and reliable
    across every seed tested. This bug and its fix are the reason
    reset()/generate_task() should never be assumed instantaneous in a training
    loop without this guard.
    """
    if difficulty < 3:
        raise ValueError("difficulty (cycle length) must be >= 3")

    rng = random.Random(seed)
    n_species_needed = difficulty + 1          # generous pool, not all necessarily used identically
    species = [_LETTERS[i] for i in range(min(n_species_needed, len(_LETTERS)))]

    def _sample_complex():
        if rng.random() < dimerization_prob and len(species) >= 2:
            a, b = rng.sample(species, 2)
            if a == b:
                return f"2{a}", {a: 2}
            name = f"{a}+{b}" if a < b else f"{b}+{a}"
            return name, {a: 1, b: 1}
        a = rng.choice(species)
        return a, {a: 1}

    complexes = {}
    complex_order = []
    used_species = set()
    for i in range(difficulty):
        # Resample until this complex's SPECIES COMPOSITION differs from its
        # predecessor (and, on the final step, from the first complex too,
        # since the cycle wraps). An earlier version appended a "'" tag to
        # de-duplicate the *name* while leaving the composition identical --
        # that silently produced no-op reactions like "A -> A'" where A' == A
        # chemically, which in turn produced degenerate rank-deficient tasks
        # with meaningless square-root steady states. Comparing compositions,
        # not names, is the fix.
        for _attempt in range(50):
            name, comp = _sample_complex()
            clashes_prev = bool(complex_order) and comp == complexes[complex_order[-1]]
            clashes_first = (i == difficulty - 1) and bool(complex_order) and comp == complexes[complex_order[0]]
            if not clashes_prev and not clashes_first:
                break
        else:
            # Could not find a distinct complex (tiny species pool): fall back
            # to a guaranteed-distinct single species not equal to the neighbours.
            forbidden = set()
            if complex_order:
                forbidden |= set(complexes[complex_order[-1]].keys())
                if i == difficulty - 1:
                    forbidden |= set(complexes[complex_order[0]].keys())
            choices = [s for s in species if s not in forbidden] or species
            a = choices[0]
            name, comp = a, {a: 1}
        # distinct compositions may still collide by name if re-sampled identically
        while name in complexes and complexes[name] != comp:
            name = name + "_"
        complexes[name] = comp
        complex_order.append(name)
        used_species |= set(comp.keys())

    species = sorted(used_species)
    reactions = []
    rate_syms = []
    for i in range(difficulty):
        r_name, p_name = complex_order[i], complex_order[(i + 1) % difficulty]
        k = sp.Symbol(f"k{i+1}", positive=True)
        rate_syms.append(k)
        reactions.append(Reaction(str(i + 1), reactants=complexes[r_name],
                                   products=complexes[p_name], k=k))

    net = ReactionNetwork(species=species, reactions=reactions)
    ground_truth = _solve_with_budget(net, seconds=4.0)
    if ground_truth is None:
        if _retry_depth >= 2 or dimerization_prob == 0.0:
            # Last resort: pure cycle at this difficulty is verified fast/reliable.
            return generate_task(seed=seed, difficulty=difficulty, dimerization_prob=0.0,
                                  _retry_depth=_retry_depth + 1)
        return generate_task(seed=seed, difficulty=difficulty, dimerization_prob=0.0,
                              _retry_depth=_retry_depth + 1)

    display_reactions = [(complex_order[i], complex_order[(i + 1) % difficulty])
                          for i in range(difficulty)]

    clean_gt, diagnostics = _split_solution(ground_truth)

    return CRNTTask(seed=seed, difficulty=difficulty, species=species,
                     reactions=display_reactions, net=net,
                     ground_truth=clean_gt, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# 2. Prompt rendering (what the policy actually sees)
# --------------------------------------------------------------------------- #

_ANSWER_FORMAT_INSTRUCTIONS = """\
Answer with a single JSON object with exactly these three keys:
{"conservation laws": [...], "steady states": [...], "flux balance": [...]}
Each value is a list of strings. Species concentrations may be written as
plain names or in brackets, e.g. "A" or "[A]". Wrap your final answer in
<answer>...</answer> tags and put nothing else inside them."""


def render_prompt(task: CRNTTask) -> str:
    rxn_lines = []
    for r, p in task.reactions:
        rxn_lines.append(f"  {r} -> {p}")
    return (
        "You are given a chemical reaction network.\n\n"
        f"Species: {', '.join(task.species)}\n"
        f"Reactions:\n" + "\n".join(rxn_lines) + "\n\n"
        "Determine, exactly:\n"
        "1. The conservation law(s) -- linear combinations of species concentrations "
        "that are invariant under the dynamics.\n"
        "2. The steady-state concentrations under mass-action kinetics (rate constants "
        f"k1..k{task.difficulty}, one per reaction in the order listed), in terms of a "
        "conserved total T0 and the rate constants.\n"
        "3. The flux-balance relation among the reaction rates v1..v{} at steady state.\n\n"
        .format(task.difficulty)
        + _ANSWER_FORMAT_INSTRUCTIONS
    )


# --------------------------------------------------------------------------- #
# 3. Completion parsing (raw model text -> pred dict)
# --------------------------------------------------------------------------- #

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_completion(text: str) -> Optional[dict]:
    """Best-effort, crash-proof extraction of a pred dict from raw model output.
    Returns None on failure -- callers must treat that as reward 0, never raise."""
    if not text:
        return None
    tagged = _ANSWER_TAG.search(text)
    candidate = tagged.group(1) if tagged else text
    m = _JSON_OBJECT.search(candidate)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return _normalize(obj)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 4. Reward
# --------------------------------------------------------------------------- #

def compute_reward(pred: Optional[dict], expected: dict, mode: str = "binary") -> float:
    if pred is None:
        return 0.0
    if mode == "binary":
        try:
            return 1.0 if check_prediction(pred, expected) else 0.0
        except Exception:
            return 0.0
    elif mode == "shaped":
        keys = ("conservation laws", "flux balance", "steady states")
        if any(k not in pred or k not in expected for k in keys):
            return 0.0
        scores = []
        try:
            scores.append(1.0 if _check_subspace(pred["conservation laws"], expected["conservation laws"],
                                                   _SPECIES_PATTERN, True) else 0.0)
        except Exception:
            scores.append(0.0)
        try:
            scores.append(1.0 if _check_subspace(pred["flux balance"], expected["flux balance"],
                                                   _FLUX_PATTERN, False) else 0.0)
        except Exception:
            scores.append(0.0)
        try:
            scores.append(1.0 if _check_steady_states(pred["steady states"], expected["steady states"]) else 0.0)
        except Exception:
            scores.append(0.0)
        return sum(scores) / len(scores)
    else:
        raise ValueError(f"unknown reward mode: {mode}")


# --------------------------------------------------------------------------- #
# 5. The environment: Gymnasium-style reset()/step()
# --------------------------------------------------------------------------- #

class CRNTVerifyEnv:
    """
    Single-turn (one-shot) RL environment, Gymnasium signature convention:

        obs, info   = env.reset(seed=...)
        obs, reward, terminated, truncated, info = env.step(action_text)

    `action_text` is the policy's raw completion (a string) -- exactly what
    an LLM would actually produce. Parsing failures are graded 0.0, not
    raised, because a training loop must never crash on a malformed rollout.

    difficulty can be a fixed int or a (low, high) range for curriculum use
    -- a new difficulty is sampled every reset() when a range is given.
    """

    def __init__(self, difficulty=3, reward_mode: str = "binary", dimerization_prob: float = 0.35):
        self.difficulty_spec = difficulty
        self.reward_mode = reward_mode
        self.dimerization_prob = dimerization_prob
        self._task: Optional[CRNTTask] = None
        self._episode_seed: Optional[int] = None

    def _sample_difficulty(self, rng: random.Random) -> int:
        if isinstance(self.difficulty_spec, tuple):
            lo, hi = self.difficulty_spec
            return rng.randint(lo, hi)
        return self.difficulty_spec

    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randrange(2**31)
        self._episode_seed = seed
        rng = random.Random(seed)
        difficulty = self._sample_difficulty(rng)
        self._task = generate_task(seed=seed, difficulty=difficulty,
                                    dimerization_prob=self.dimerization_prob)
        observation = render_prompt(self._task)
        info = {
            "seed": seed,
            "difficulty": difficulty,
            "species": self._task.species,
            "ground_truth": self._task.ground_truth,
            "diagnostics": self._task.diagnostics,
            "reward_mode": self.reward_mode,
            "note": ("shaped reward is a training-signal convenience -- it is NOT a claim "
                     "that a partially-correct CRNT answer is partially scientifically valid")
                    if self.reward_mode == "shaped" else None,
        }
        return observation, info

    def step(self, action_text: str):
        if self._task is None:
            raise RuntimeError("call reset() before step()")
        pred = parse_completion(action_text)
        reward = compute_reward(pred, self._task.ground_truth, mode=self.reward_mode)
        terminated = True          # single-turn: episode always ends after one step
        truncated = False
        info = {
            "seed": self._episode_seed,
            "parsed_ok": pred is not None,
            "ground_truth": self._task.ground_truth,
            "prediction": pred,
        }
        observation = None         # no further observation after episode end
        self._task = None
        return observation, reward, terminated, truncated, info


# --------------------------------------------------------------------------- #
# 6. Self-test: prove the environment works end-to-end before trusting it
# --------------------------------------------------------------------------- #

def _oracle_policy(task: CRNTTask) -> str:
    """A 'perfect' policy: emits the exact ground truth, wrapped as instructed."""
    gt = task.ground_truth
    payload = {k: gt[k] for k in ("conservation laws", "steady states", "flux balance")}
    return f"<answer>{json.dumps(payload)}</answer>"


def _rewritten_correct_policy(task: CRNTTask) -> str:
    """A policy that gives a genuinely CORRECT but differently-written answer
    (scaled conservation law, reordered flux chain) -- must still score 1.0."""
    gt = task.ground_truth
    cons = gt["conservation laws"]
    cons_rewritten = []
    for law in cons:
        if "=" in law:
            lhs, rhs = law.split("=", 1)
            cons_rewritten.append(f"2*({lhs.strip()}) = 2*{rhs.strip()}")
        else:
            cons_rewritten.append(law)
    payload = {
        "conservation laws": cons_rewritten,
        "steady states": gt["steady states"],
        "flux balance": gt["flux balance"],
    }
    return f"<answer>{json.dumps(payload)}</answer>"


def _broken_policy(task: CRNTTask) -> str:
    """A policy that confidently gives a wrong conservation law."""
    payload = {
        "conservation laws": ["[A] + 99*[B] = const"],
        "steady states": ["nonsense"],
        "flux balance": ["v1 = 0"],
    }
    return f"<answer>{json.dumps(payload)}</answer>"


def _garbage_policy(task: CRNTTask) -> str:
    """A policy that fails to even produce parseable output."""
    return "I think the answer involves some kind of balance but I'm not sure."


def _run_policy(env: CRNTVerifyEnv, policy_fn, n_episodes: int, base_seed: int = 0) -> float:
    total = 0.0
    for i in range(n_episodes):
        seed = base_seed + i
        obs, info = env.reset(seed=seed)
        # policy needs the task, not just the prompt, to construct its answer here
        # (self-test only -- a real LLM policy would read `obs` instead)
        task = generate_task(seed=seed, difficulty=env._sample_difficulty(random.Random(seed)),
                              dimerization_prob=env.dimerization_prob)
        action = policy_fn(task)
        _, reward, terminated, truncated, step_info = env.step(action)
        assert terminated and not truncated
        total += reward
    return total / n_episodes


if __name__ == "__main__":
    print("===== crnt_gym.py self-test =====\n")

    env = CRNTVerifyEnv(difficulty=(3, 4), reward_mode="binary")
    n = 15

    oracle_avg = _run_policy(env, _oracle_policy, n)
    rewritten_avg = _run_policy(env, _rewritten_correct_policy, n)
    broken_avg = _run_policy(env, _broken_policy, n)
    garbage_avg = _run_policy(env, _garbage_policy, n)

    print(f"Oracle policy (exact ground truth):          mean reward = {oracle_avg:.3f}  (expect 1.000)")
    print(f"Rewritten-but-correct policy (scaled law):    mean reward = {rewritten_avg:.3f}  (expect 1.000)")
    print(f"Broken policy (wrong coefficients):           mean reward = {broken_avg:.3f}  (expect 0.000)")
    print(f"Garbage / unparseable policy:                 mean reward = {garbage_avg:.3f}  (expect 0.000)")

    assert oracle_avg == 1.0, "oracle policy must score perfectly"
    assert rewritten_avg == 1.0, "differently-written-but-correct policy must score perfectly"
    assert broken_avg == 0.0, "wrong policy must score zero"
    assert garbage_avg == 0.0, "unparseable policy must score zero"

    # single fixed episode demo, printed in full for inspection
    print("\n--- one full episode, difficulty=3, seed=42 ---")
    env2 = CRNTVerifyEnv(difficulty=3, reward_mode="shaped")
    obs, info = env2.reset(seed=42)
    print("PROMPT:\n", obs)
    task42 = generate_task(seed=42, difficulty=3, dimerization_prob=0.35)
    action = _oracle_policy(task42)
    _, reward, terminated, truncated, step_info = env2.step(action)
    print("REWARD (shaped):", reward, " terminated:", terminated)

    print("\nAll self-tests passed.")
