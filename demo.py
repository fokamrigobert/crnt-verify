"""
demo.py — see the crnt-verify environment work, step by step
=============================================================

Run this first:   python3 demo.py

No API key, no model, no training framework needed. It walks through one
full episode slowly and prints what happens at every stage, so you can see
the machinery instead of reading it.

Mental model in one line:
    Phase 1 (crnt_solver + crnt_checker) = the EXAMINER
    Phase 2 (crnt_gym)                   = the EXAMINATION HALL
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crnt_gym import CRNTVerifyEnv


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


banner("STEP 1 — Create the environment")
print("""
    env = CRNTVerifyEnv(difficulty=3, reward_mode="binary")

difficulty=3   -> reaction networks with 3 complexes / 3 reactions
reward_mode    -> "binary": 1.0 only if the WHOLE answer is right
                  "shaped": partial credit per sub-part (training aid only)
""")
env = CRNTVerifyEnv(difficulty=3, reward_mode="binary")
print("Environment created.")


banner("STEP 2 — reset(): the environment invents a fresh problem")
print("""
    observation, info = env.reset(seed=7)

`seed` makes it reproducible: seed=7 always produces this exact network.
Leave it out and you get a random new one every time.
""")
observation, info = env.reset(seed=7)

print("--- `observation` is the exact text you would send to a model: ---\n")
print(observation)

print("\n--- `info` is bookkeeping YOU see but the model must not: ---\n")
print("  seed:       ", info["seed"])
print("  difficulty: ", info["difficulty"])
print("  species:    ", info["species"])
print("  ground_truth (the answer key, computed by Phase 1's solver):")
for key, value in info["ground_truth"].items():
    if key.startswith("_"):
        continue
    print(f"     {key}: {value}")


banner("STEP 3a — step(): submit a CORRECT answer")
gt = info["ground_truth"]
correct_payload = {k: gt[k] for k in ("conservation laws", "steady states", "flux balance")}
correct_answer = f"<answer>{json.dumps(correct_payload)}</answer>"

print("What a model would send back:\n")
print(correct_answer[:400] + ("..." if len(correct_answer) > 400 else ""))

_, reward, terminated, truncated, step_info = env.step(correct_answer)
print(f"\n  -> reward     = {reward}   (1.0 = fully correct)")
print(f"  -> terminated = {terminated}  (episode over; this task is single-turn)")
print(f"  -> parsed_ok  = {step_info['parsed_ok']}")


banner("STEP 3b — the important one: a CORRECT answer, WRITTEN DIFFERENTLY")
observation, info = env.reset(seed=7)          # same task again
gt = info["ground_truth"]

# Scale the conservation law by 2. Mathematically identical. Textually different.
rescaled = []
for law in gt["conservation laws"]:
    lhs, rhs = law.split("=", 1)
    rescaled.append(f"2*({lhs.strip()}) = 2*{rhs.strip()}")

variant_payload = {
    "conservation laws": rescaled,
    "steady states": gt["steady states"],
    "flux balance": gt["flux balance"],
}
variant_answer = f"<answer>{json.dumps(variant_payload)}</answer>"

print("Original conservation law: ", gt["conservation laws"])
print("Model's version (scaled):  ", rescaled)
print("\nA string-matching grader would mark this WRONG. It is not wrong.\n")

_, reward, _, _, _ = env.step(variant_answer)
print(f"  -> reward = {reward}   (still 1.0 — this is the whole point of the project)")


banner("STEP 3c — a WRONG answer")
observation, info = env.reset(seed=7)
wrong_answer = '<answer>{"conservation laws": ["[A] + 99*[B] = const"], ' \
               '"steady states": ["A = 1"], "flux balance": ["v1 = 0"]}</answer>'
print("Model's answer:\n")
print(wrong_answer)
_, reward, _, _, _ = env.step(wrong_answer)
print(f"\n  -> reward = {reward}   (0.0 — wrong coefficients are caught)")


banner("STEP 3d — unparseable rambling (models do this)")
observation, info = env.reset(seed=7)
_, reward, _, _, step_info = env.step("Hmm, I think there's some kind of balance here?")
print(f"  -> reward    = {reward}")
print(f"  -> parsed_ok = {step_info['parsed_ok']}  (graded 0, did NOT crash)")


banner("STEP 4 — the loop, which is all training/evaluation really is")
print("""
    for i in range(N):
        observation, info = env.reset(seed=i)     # get a problem
        answer = my_model(observation)            # ask the model
        _, reward, _, _, _ = env.step(answer)     # score it
        scores.append(reward)

Swap `my_model` for a real LLM  -> that's an EVALUATION (see evaluate.py)
Feed `reward` back into training -> that's REINFORCEMENT LEARNING

Running it now with a deliberately mediocre fake model (correct half the
time) across 6 different problems:
""")

def fake_model(observation, info, be_correct):
    gt = info["ground_truth"]
    if be_correct:
        payload = {k: gt[k] for k in ("conservation laws", "steady states", "flux balance")}
    else:
        payload = {"conservation laws": ["[A] = const"],
                   "steady states": ["A = 0"], "flux balance": ["v1 = 0"]}
    return f"<answer>{json.dumps(payload)}</answer>"

scores = []
for i in range(6):
    observation, info = env.reset(seed=100 + i)
    answer = fake_model(observation, info, be_correct=(i % 2 == 0))
    _, reward, _, _, _ = env.step(answer)
    scores.append(reward)
    print(f"   problem {i}: reward = {reward}")

print(f"\n   MEAN SCORE = {sum(scores)/len(scores):.2f}   <- this is the number you report")

banner("Done")
print("""
Next:
  python3 crnt_gym.py    -> the environment's own correctness tests
  python3 evaluate.py    -> score a REAL model (needs an API key)
""")
