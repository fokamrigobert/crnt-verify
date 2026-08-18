# crnt-verify

**An RL environment and deterministic verifier for chemical reaction network invariants.**

Generates reaction-network problems on demand, and grades answers by testing
genuine **mathematical equivalence** rather than string matching — because a
correct answer to these problems can be rescaled, reordered, or algebraically
rearranged and still be correct.

```python
from crnt_gym import CRNTVerifyEnv

env = CRNTVerifyEnv(difficulty=3, reward_mode="binary")
observation, info = env.reset(seed=42)          # a fresh reaction network
observation, reward, terminated, truncated, info = env.step(model_output)
```

Gymnasium-style `reset` / `step` / `reward`, so it plugs into an RL training
loop or an eval harness directly. A Prime Intellect `verifiers` adapter is
included (`crnt_verifiers_env.py`).

---

## Why equivalence testing, not string matching

Conservation laws and flux-balance modes are **bases of a subspace** — any
nonzero scalar multiple or basis recombination is equally correct. Steady
states are algebraic expressions, correct up to rearrangement. Comparing text
therefore fails on correct answers, which is exactly the failure mode that
poisons a reward signal.

| Invariant | Underlying object | Equivalence test |
|---|---|---|
| Conservation laws | basis of ker(S^T) | **subspace equality by rank**: rank(P) = rank(E) = rank(P‖E) |
| Flux balance | basis of ker(S) | same rank test |
| Steady states | algebraic expressions | **deterministic randomized substitution** at fixed-seed rational points |

Binary verdict, fixed seed, no LLM-as-judge. This is the verifier pattern that
reinforcement learning with verifiable rewards (RLVR) depends on, applied to a
real scientific domain.

**It works on real model output.** In live evaluation a model answered flux
balance as three redundant pairwise equations (`v1=v2`, `v2=v3`, `v1=v3`)
instead of the key's single chain — correctly accepted as the same subspace.
Another gave conservation laws in a different basis whose recombination
reproduced the key — also accepted.

---

## Measured result

**gemini-3.5-flash-lite, difficulty 3, n=20: 15% (3/20)**, 95% CI 5–36%.
Zero API errors, zero unparseable responses.

Broken down by sub-problem:

| Sub-check | Pass rate |
|---|---|
| Conservation laws (linear) | 60% |
| Flux balance (linear) | 65% |
| **Steady states (nonlinear)** | **15%** |

The two linear sub-problems cluster together; the nonlinear one collapses.
That gradient matches the mathematics and was not designed for. Failures were
verified independently — the model's claimed conservation laws give
`c^T S ≠ 0` — so this is genuine model error, not a grader artifact.

Full detail, including the defects found and fixed along the way, in
[RESULTS.md](RESULTS.md).

---

## Quick start

```bash
pip install -r requirements.txt        # sympy only
py demo.py                             # narrated walkthrough, no API key needed
py crnt_gym.py                         # environment self-tests
py evaluate.py --n 20 --provider google --model <model> --delay 4 --save run.json
py inspect_results.py --file run.json --only-failures
```

Providers supported: `anthropic`, `google`, `groq`, `openrouter`, and
`manual` (paste answers by hand — no key, no cost). See
[QUICKSTART.md](QUICKSTART.md) for step-by-step setup.

## Files

| File | Purpose |
|---|---|
| `crnt_solver.py` | computes ground truth (exact rational linear algebra) |
| `crnt_checker.py` | `check_prediction(pred, expected) -> bool` — the grader |
| `crnt_gym.py` | the environment: task generation, `reset`/`step`, reward |
| `crnt_verifiers_env.py` | Prime Intellect `verifiers` adapter |
| `demo.py` | narrated walkthrough |
| `evaluate.py` | score a model, with retry/quota/error handling |
| `inspect_results.py` | diagnose *why* answers failed, per sub-check |
| `compare_models.py` | compare models on identical seeds |
| `ENVIRONMENT_SPEC.md` | design rationale |
| `RESULTS.md` | measured results, defect log, limitations |

## Limitations

Networks with more than one conservation law are excluded, because the steady
state then depends on a non-unique choice of basis — the question is genuinely
ill-posed, not merely underspecified. The conservation basis is also not
saturated (Smith normal form would fix it; measured frequency of the problem
on the current task distribution: 0 in 112 networks). Full list in
[RESULTS.md](RESULTS.md).

## About

Built by [Rigobert Fokam Souop](https://github.com/fokamrigobert), PhD in
Applied Mathematics (isometric graph embeddings into Cayley graphs of abelian
groups, University of Ngaoundéré). The subspace-equality logic here —
deciding whether two differently-written objects are secretly the same one via
a canonical representation — is the same idea as the Cocycle/Quotient Labeling
Theorem in the author's dissertation, applied to a different domain.

## License

MIT — see [LICENSE](LICENSE).
