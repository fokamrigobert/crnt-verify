# crnt-verify — Phase 2: the RL Environment

Status: **implemented and passing its own tests**, built and smoke-tested
against real, installed packages (not just documentation) at every step.

## Why this exists

Phase 1 (`crnt_solver.py` + `crnt_checker.py`) is a static grader:
`check_prediction(pred, expected) -> bool`. That's necessary but not
sufficient for RL training or automated evals, which need something that
can *generate* problems on demand and *score* a policy's response in a
standard interaction loop. Phase 2 turns the grader into an environment
without changing or duplicating any of its logic — everything below
imports and reuses Phase 1 directly.

## Interface contract

```python
from crnt_gym import CRNTVerifyEnv

env = CRNTVerifyEnv(difficulty=(3, 5), reward_mode="binary")
observation, info   = env.reset(seed=42)
# observation: str, the full problem prompt the policy should see
# info: {"seed", "difficulty", "species", "ground_truth", "reward_mode", "note"}

observation, reward, terminated, truncated, info = env.step(action_text)
# action_text: str, the policy's raw completion (exactly what an LLM emits)
# reward: float in [0, 1]
# terminated: True (every episode is currently single-turn)
# info: {"seed", "parsed_ok", "ground_truth", "prediction"}
```

This is the Gymnasium five-tuple convention on purpose — it's the most
widely recognized interaction contract in the space, so anything built to
consume it (or adapted from it) doesn't require re-deriving the shape.

## What each piece does, and why it's built that way

| Concern | Design choice | Why |
|---|---|---|
| Task generation | Procedural generator (`generate_task`), single directed cycle of complexes, deterministic from an integer seed | Infinite unique training instances at controllable difficulty, not a fixed example set. Same seed always reproduces the same task — required for reproducible curricula and for regression-testing the environment the same way `crnt_checker.py` tests itself |
| Ground truth | Computed once per task via Phase 1's `solve_network()` | No duplicated math. If the solver improves, tasks improve for free |
| Grading | `compute_reward()` calls `check_prediction()` (binary) or the three sub-checks directly (shaped) | Binary is the honest verifiable-reward signal — no credit for a wrong answer that merely looks close. Shaped is offered as a *training convenience* for cold-start density, explicitly labeled as such in `info["note"]`, not as a claim that partial CRNT answers are partially valid science |
| Answer parsing | Regex-extract `<answer>...</answer>`, parse JSON, run through Phase 1's own `_normalize()` | A malformed or unparseable completion must score 0, never crash the rollout. Verified by a dedicated garbage-input test case |
| Reliability | Wall-clock-bounded symbolic solve with automatic fallback | Real bug found while building this: some dimerized topologies made `sympy.solve` take up to ~12s, worse at higher difficulty. An environment's `reset()` cannot be allowed to hang. Every `generate_task()` call is bounded (4s) and falls back to a simpler cycle at the same seed/difficulty. Cross-platform: `SIGALRM` on Unix/macOS, an abandoned daemon thread on Windows (which has no `SIGALRM`) — both paths tested |

## Self-tests (what's actually been checked, not just claimed)

`python3 crnt_gym.py` runs four policies across 15 episodes each and asserts:

| Policy | Behavior | Required mean reward |
|---|---|---|
| Oracle | Emits the exact ground truth | 1.0 |
| Rewritten-but-correct | Scales a conservation law by 2, still mathematically correct | 1.0 |
| Broken | Confidently wrong coefficients | 0.0 |
| Garbage | Not even parseable as an answer | 0.0 |

The second row is the one that matters most — it's a direct regression test
that the environment inherits Phase 1's actual point (equivalence, not
string matching), not just its name.

## Publishing to Prime Intellect's `verifiers` library

`crnt_verifiers_env.py` wraps the environment as a `verifiers` package:
pre-generates a fixed task pool as a HuggingFace `Dataset`, defines a
reward function against it, and constructs `vf.SingleTurnEnv` +
`vf.Rubric`. Also implemented and passing, against the real installed
package (`pip install verifiers` → v0.3.0, August 2026), not assumed from
docs.

**Worth knowing before you publish, because it happened live while
building this**: the public docs describe a `SingleTurnEnv` + `Rubric` +
`Parser` pattern. The installed package still supports it — confirmed by
constructing one, not by trusting the docs — but it now sits alongside a
substantially larger architecture (`verifiers.v1`: `Task`, `Taskset`,
`Env`, `Judge`, `Harness`) that none of the documentation surfaced by
search even mentions. The library's own `LegacyEnvConfig` docstring calls
the pattern used here *"a classic (v0) environment... loaded through the
legacy bridge"* — current and working today, explicitly framed as the old
path. Before actually pushing to the Environments Hub: reinstall fresh,
rerun `crnt_verifiers_env.py`'s smoke test, and skim `verifiers.v1` to see
whether the newer system has become the expected submission format by
then. Six months was enough for this to have already changed once.

## Known limitations, stated directly

- Task family is currently restricted to single-linkage-class directed
  cycles (with single- or two-species complexes). Branched networks and
  multiple linkage classes are supported by Phase 1's solver but not yet
  by the generator — natural next step, not attempted here.
- The symbolic-solve timeout means a small fraction of generated tasks
  silently fall back to a simpler (non-dimerized) network at the same
  seed. Fine for training signal; worth knowing if you're auditing exact
  task-distribution statistics.
- Single-turn only. A multi-turn variant — submit conservation laws, get
  turn-level feedback, then flux balance, then steady states — is a
  natural Phase 2b for agentic/long-horizon training, which is where a
  meaningful share of current RL-environment demand actually sits. Not
  built yet; flagging it as the next real decision point rather than
  half-building it now.

## Files

- `crnt_gym.py` — the environment core. No dependency beyond `sympy` (via
  Phase 1) and the standard library.
- `crnt_verifiers_env.py` — the `verifiers`/Prime Intellect adapter.
  Requires `verifiers` and `datasets`, installed separately; not a
  dependency of the core.
