"""
crnt_verifiers_env.py — publishing crnt-verify's environment to Prime
Intellect's `verifiers` library / Environments Hub
=======================================================================

A note on how this file came to exist, because it's a useful data point in
its own right: the public docs for `verifiers` (README, docs.primeintellect.ai)
describe a `SingleTurnEnv` + `Rubric` + `Parser` pattern. Installing the
library fresh (pip install verifiers -> v0.3.0, August 2026) shows that
pattern is still supported -- confirmed by actually constructing a
SingleTurnEnv below, not by trusting the docs -- but it now sits alongside a
much larger, newer architecture (Task / Taskset / Env / Judge / Harness,
under `verifiers.v1`) that the docs found by search do not mention at all.
`LegacyEnvConfig`'s own docstring calls the SingleTurnEnv pattern "a classic
(v0) environment, loaded ... through the legacy bridge" -- meaning it works
today, but is explicitly framed as the old path, not the current one.

Practical takeaway, stated plainly: this adapter is built and smoke-tested
against the real installed package as of this writing. Library APIs in this
space move fast enough that "the docs said so" was already not good enough
proof six months in. Before actually publishing to the Environments Hub,
re-run this file's __main__ block against whatever version is current then,
and skim `verifiers.v1` to see whether the newer Task/Taskset/Judge system
has become the expected path for new submissions.

Everything domain-specific (task generation, prompt, parsing, grading)
lives in crnt_gym.py and is reused here unchanged -- this file is only the
translation into verifiers' Dataset + Rubric shape.
"""

from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset

from crnt_gym import generate_task, render_prompt, parse_completion, compute_reward


def build_dataset(n_tasks: int = 200, difficulty=(3, 5), dimerization_prob: float = 0.35,
                   base_seed: int = 0) -> Dataset:
    """Pre-generate a fixed pool of tasks as a HF Dataset (verifiers' expected
    input shape). Pre-generating rather than generating inside the reward
    function keeps rollouts fast and keeps the (slow, sympy-based) solving
    off the training hot path entirely -- solved once here, reused for every
    rollout against that row."""
    import random
    rows = []
    for i in range(n_tasks):
        seed = base_seed + i
        d = random.Random(seed).randint(*difficulty) if isinstance(difficulty, tuple) else difficulty
        task = generate_task(seed=seed, difficulty=d, dimerization_prob=dimerization_prob)
        # task.ground_truth is guaranteed JSON/Arrow-serializable (see
        # _split_solution in crnt_gym.py -- raw sympy objects are kept out of it).
        rows.append({
            "question": render_prompt(task),
            "answer": "",                         # unused; ground truth carried in `info`
            "info": {"ground_truth": task.ground_truth, "seed": seed, "difficulty": d},
        })
    return Dataset.from_list(rows)


def crnt_reward(completion, info, **kwargs) -> float:
    """verifiers reward-function signature: (prompt, completion, answer, info, ...) -> float.
    Only `completion` and `info` are needed here."""
    text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
    pred = parse_completion(text)
    return compute_reward(pred, info["ground_truth"], mode="binary")


def load_environment(n_tasks: int = 200, difficulty=(3, 5), dimerization_prob: float = 0.35):
    """Entry point matching verifiers' `load_environment()` convention
    (see PyPI package examples: `def load_environment(...) -> vf.Environment`)."""
    import verifiers as vf

    dataset = build_dataset(n_tasks=n_tasks, difficulty=difficulty,
                             dimerization_prob=dimerization_prob)
    rubric = vf.Rubric(funcs=[crnt_reward], weights=[1.0])
    system_prompt = (
        "You are a careful chemical reaction network theorist. Work through "
        "the linear algebra exactly; do not guess."
    )
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric, system_prompt=system_prompt)
    return env


if __name__ == "__main__":
    print("===== crnt_verifiers_env.py: construction + reward smoke test =====\n")

    env = load_environment(n_tasks=5, difficulty=(3, 4))
    print("Environment constructed:", type(env))
    print("Dataset size:", len(env.dataset) if hasattr(env, "dataset") else "n/a")

    # Exercise the reward function directly (no live model call) against the
    # first row, using an oracle-correct completion, mirroring the
    # correctness checks already run in crnt_gym.py's own self-test.
    row = env.dataset[0]
    gt = row["info"]["ground_truth"]
    payload = {k: gt[k] for k in ("conservation laws", "steady states", "flux balance")}
    oracle_completion = [{"role": "assistant", "content": f"<answer>{json.dumps(payload)}</answer>"}]
    r = crnt_reward(oracle_completion, row["info"])
    print(f"Oracle completion reward on dataset row 0: {r}  (expect 1.0)")
    assert r == 1.0

    broken_completion = [{"role": "assistant", "content": "<answer>{\"conservation laws\": [\"nonsense\"], "
                                                            "\"steady states\": [], \"flux balance\": []}</answer>"}]
    r2 = crnt_reward(broken_completion, row["info"])
    print(f"Broken completion reward on dataset row 0:  {r2}  (expect 0.0)")
    assert r2 == 0.0

    print("\nSmoke test passed against the installed verifiers version.")
