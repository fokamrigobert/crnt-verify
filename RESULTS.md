# RESULTS

Measured findings, verified defects, and known limitations for crnt-verify.
Everything here was checked computationally; nothing is asserted from memory.

---

## Schema version

**Schema v1** (current). A task is a single-linkage-class directed cycle of
complexes with **at most one conservation law**, generated deterministically
from an integer seed.

Results measured before schema v1 are **not comparable** and are listed under
[Superseded](#superseded-measurements) rather than deleted, since the reason
they were superseded is itself part of the record.

---

## Measured result

**gemini-3.5-flash-lite, difficulty 3, n=20, schema v1**

| Metric | Value | 95% CI (Wilson) |
|---|---|---|
| **Overall accuracy** | **15%** (3/20) | 5% – 36% |
| Conservation laws sub-check | 60% (12/20) | 39% – 78% |
| Flux balance sub-check | 65% (13/20) | 43% – 82% |
| **Steady states sub-check** | **15%** (3/20) | 5% – 36% |

API errors: 0/20. Unparseable: 0/20. Mean latency ≈ 2.7 s.

### What this shows

The two **linear** sub-problems (conservation laws, flux balance — both null
space computations) cluster at 60–65%. The **nonlinear** sub-problem
(mass-action steady states) collapses to 15%. That ordering matches the
mathematical difficulty gradient and was not designed for; its emerging
unprompted is evidence the benchmark measures something coherent.

### Verified, not assumed

The failures were checked independently before being called genuine. At seed
1000 the model claimed `[A]+[B]` is conserved; direct computation gives
`cᵀS = (-1,1,0) ≠ 0`, so the claim is false, while the key's `[A]+[D]` gives
exactly zero. Same at seed 1002. At seed 1003 the model asserted `D = 0` and
`v₂ = 0`, which would require a rate constant to vanish.

Conversely the grader accepted answers that *were* correct but differently
written — flux balance given as three redundant pairwise equations
(`v1=v2`, `v2=v3`, `v1=v3`) was correctly recognised as the same subspace.

### Superseded measurements

| Model | Result | Why superseded |
|---|---|---|
| gemini-3.5-flash | 85% (17/20) | Pre-dates the well-posedness fix; 3 failures were ill-posed multi-law tasks |
| gemini-2.5-flash | 80% (16/20) | Pre-dates the empty-conservation-lattice checker fix; ≥2 failures were false negatives |

Both need re-measuring on schema v1 before any cross-model claim is made.

---

## Defects found and fixed

Five real defects surfaced during one day of live evaluation. Four were in
this codebase; the fifth was in the companion research note. Recorded because
the debugging history is itself evidence the tooling has been stressed.

| # | Defect | How it was caught | Fix |
|---|---|---|---|
| 1 | **Flux naming false negative** — `v1` and `v_1` treated as different variables, so a correct `v1 = v2 = v3` was marked wrong | Live eval; `inspect_results.py` localised it to notation, not chemistry | Fold both spellings before comparison; regression test both directions |
| 2 | **Empty-lattice false negative** — a model correctly answering "no conservation laws" as `["none"]` instead of `[]` was marked wrong | Perfect correlation: both zero-law tasks failed, 2/2 | Treat semantic emptiness as empty; assert that inventing or denying a law still fails |
| 3 | **Ill-posed multi-law tasks** — with dim ker(Sᵀ) ≥ 2 the steady state depends on a *non-unique* basis choice, so the question cannot be answered without guessing the solver's basis | Perfect correlation: 3/3 multi-law tasks failed, 17/17 single-law passed | Restrict schema v1 to ≤1 conservation law; document the deeper fix as open |
| 4 | **Same-seed non-determinism** — the wall-clock solve timeout could fall back differently between calls, so one seed could yield two different networks | Oracle policy dropped to 0.933 when it must be 1.000 | Budget raised 4s → 15s; harness reads the environment's own ground truth; caveat documented |
| 5 | **Invalid counterexample in the research note** — the matrix in the Theorem 3 proof had none of its quoted vectors in the kernel | Direct re-verification when asked whether HNF was implemented | Replaced with two fully verified witnesses (saturation indices 2 and 8) |

Infrastructure fixes on top of these: Windows `SIGALRM` incompatibility,
non-JSON-serialisable sympy objects in ground truth, API errors being counted
as model failures, and permanent errors (quota, retired model, bad key) being
retried instead of aborting.

---

## Known limitations

Stated plainly, because a benchmark whose limits are undocumented is not
usable by anyone else.

1. **Multi-conservation-law networks are excluded.** These are chemically the
   *more* realistic case — real metabolic networks have many conservation
   laws. Supporting them needs either a canonical basis (Hermite normal form
   would do it) or a basis-invariant steady-state check. Deferred
   deliberately, not overlooked.
2. **The conservation-law basis is not saturated.** `conservation_laws()` uses
   rational elimination plus per-vector gcd clearing, which can miss genuine
   integer conservation laws. Measured frequency on the current task
   distribution: **0 occurrences in 112 networks** (21 of rank ≥ 2), so it is a
   latent hazard for larger networks rather than a live bug. Fix = Smith
   normal form.
3. **Seeds are reproducible in practice, not guaranteed across machines** — see
   defect 4. For hard reproducibility, persist a generated task set rather
   than regenerating from seeds.
4. **Single-turn only.** No multi-turn variant where a model submits one
   invariant, receives feedback, then continues.
5. **Task family is narrow** — single-linkage-class cycles with one- or
   two-species complexes. Branched networks and multiple linkage classes are
   supported by the solver but not the generator.
6. **Small n.** All results are n=20. The confidence intervals above are wide
   and are reported for that reason.

---

## Reproducing these numbers

```bash
py crnt_gym.py                     # environment self-tests (no API key)
py crnt_checker.py                 # grader self-tests, incl. regressions for defects 1 and 2
py evaluate.py --n 20 --provider google --model gemini-3.5-flash-lite \
               --delay 4 --save eval_35flashlite.json
py inspect_results.py --file eval_35flashlite.json --only-failures
```

Free-tier quotas are typically 20 requests per model per day, which is exactly
one run. `evaluate.py` aborts immediately on quota exhaustion rather than
retrying.
