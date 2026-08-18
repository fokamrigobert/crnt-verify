"""
evaluate.py — score a REAL model against the crnt-verify environment
======================================================================

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 evaluate.py --n 20 --difficulty 3

This is what "does the environment perform well" actually means in
practice. Two separate questions, easy to conflate:

  1. Is the ENVIRONMENT correct?      -> python3 crnt_gym.py  (self-tests)
  2. How well does a MODEL score?     -> this file

The environment doesn't have an accuracy; it has correctness. The number
this file prints is the model's accuracy ON the environment, which is also
the number that tells you whether your benchmark is well-calibrated:

    ~100%  -> too easy, no training signal, raise difficulty
    ~0%    -> too hard (or your prompt/parsing is broken), investigate
    20-80% -> useful. This is the band where RL training gets traction.

Needs `pip install anthropic`. To use a different provider, replace
call_model() -- nothing else in this file is provider-specific.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crnt_gym import CRNTVerifyEnv


def call_model(prompt: str, model: str = "claude-sonnet-4-6", max_tokens: int = 2000) -> str:
    """Single provider-specific function. Swap this to use another API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit(
            "The `anthropic` package is not installed.\n"
            "  pip install anthropic\n"
            "Or edit call_model() to use whichever provider you prefer."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "No ANTHROPIC_API_KEY found in your environment.\n\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...      (macOS/Linux)\n"
            "  setx ANTHROPIC_API_KEY sk-ant-...        (Windows)\n\n"
            "Get one at console.anthropic.com. Note this costs money per call —\n"
            "20 problems is a few cents, but be aware before running with --n 500.\n\n"
            "To see the environment work WITHOUT an API key, run:  python3 demo.py"
        )
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on crnt-verify.")
    ap.add_argument("--n", type=int, default=20, help="number of problems (default 20)")
    ap.add_argument("--difficulty", type=int, default=3, help="cycle length, >=3 (default 3)")
    ap.add_argument("--model", type=str, default="claude-sonnet-4-6")
    ap.add_argument("--seed-start", type=int, default=1000, help="first task seed")
    ap.add_argument("--save", type=str, default="eval_results.json")
    args = ap.parse_args()

    env = CRNTVerifyEnv(difficulty=args.difficulty, reward_mode="binary")

    print(f"Evaluating {args.model} on {args.n} problems at difficulty {args.difficulty}\n")

    records, rewards = [], []
    for i in range(args.n):
        seed = args.seed_start + i
        observation, info = env.reset(seed=seed)

        t0 = time.time()
        try:
            completion = call_model(observation, model=args.model)
            error = None
        except SystemExit:
            raise
        except Exception as exc:                      # network hiccup, rate limit, etc.
            completion, error = "", f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0

        _, reward, _, _, step_info = env.step(completion)
        rewards.append(reward)

        status = "PASS" if reward == 1.0 else "fail"
        parse_note = "" if step_info["parsed_ok"] else "  [unparseable]"
        err_note = f"  [{error}]" if error else ""
        print(f"  [{i+1:>3}/{args.n}] seed={seed}  {status}  {elapsed:5.1f}s{parse_note}{err_note}")

        records.append({
            "seed": seed,
            "difficulty": args.difficulty,
            "reward": reward,
            "parsed_ok": step_info["parsed_ok"],
            "error": error,
            "ground_truth": step_info["ground_truth"],
            "model_output": completion,
        })

    n = len(rewards)
    accuracy = sum(rewards) / n if n else 0.0
    unparseable = sum(1 for r in records if not r["parsed_ok"])

    print("\n" + "=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Difficulty:   {args.difficulty}")
    print(f"  Problems:     {n}")
    print(f"  ACCURACY:     {accuracy:.1%}")
    print(f"  Unparseable:  {unparseable}/{n}")
    print("=" * 60)

    if accuracy > 0.95:
        print("\n  -> Near-ceiling. Raise --difficulty for a useful training signal.")
    elif accuracy < 0.05:
        print("\n  -> Near-floor. Before concluding the task is hard, check a few")
        print("     saved model_output entries: a formatting/parsing mismatch looks")
        print("     identical to genuine failure from the accuracy number alone.")
    else:
        print("\n  -> Useful range for benchmarking and RL training.")

    with open(args.save, "w") as f:
        json.dump({"model": args.model, "difficulty": args.difficulty,
                   "accuracy": accuracy, "records": records}, f, indent=2)
    print(f"\n  Full transcripts saved to {args.save}")
    print("  (Inspect these before trusting any headline number.)")


if __name__ == "__main__":
    main()
