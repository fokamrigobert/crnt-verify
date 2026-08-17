# crnt-verify

**A deterministic solver *and* a deterministic grader for Chemical Reaction Network invariants.**

Given a set of species and reactions, `crnt-verify` computes the three canonical reaction invariants — **conservation laws**, **flux balance**, and **mass-action steady states** — using exact rational linear algebra (no floats, no sampling, no approximation). It also ships a `check_prediction()` function that grades *any* candidate answer to these problems — from a student, a script, or an LLM — by testing genuine mathematical equivalence rather than string matching.

```python
["conservation laws", "steady states", "flux balance"]
    = ["[A]+[B]+[C]=const",
       "[A]*=T·k2·k3/(k1k2+k1k3+k2k3), ...",
       "v1=v2=v3"]
```

## Why a *grader*, not just a solver

Conservation laws and flux-balance modes are only defined **up to a choice of basis and a scalar multiple** — `2[A]+2[B]+2[C]=const` is exactly as correct as `[A]+[B]+[C]=const`. Steady-state expressions are only defined **up to algebraic rearrangement**. That means the obvious way to check an answer — compare it to a reference string — fails on correct answers that are simply written differently, which is precisely the failure mode you hit the moment you try to auto-grade an LLM's output on problems like this.

`check_prediction(pred, expected) -> bool` handles it with two different equivalence tests, chosen by what kind of object is being compared:

| Invariant | What it really is | Equivalence test |
|---|---|---|
| Conservation laws | A basis of $\ker(S^T)$ | **Subspace equality** via rank: $\operatorname{rank}(M_{pred}) = \operatorname{rank}(M_{exp}) = \operatorname{rank}(M_{pred} \Vert M_{exp})$ |
| Flux balance | A basis of $\ker(S)$ | Same subspace-equality test, applied to the null space |
| Steady states | Algebraic expressions in rate constants | **Deterministic randomized-substitution equivalence** — evaluate the symbolic difference at fixed-seed rational test points |

Binary output, fixed seed, no LLM-as-judge, no partial credit for "close enough." This same pattern — turn "is this mathematically the same answer, however it's written" into a cheap, deterministic, binary check — is exactly the kind of verifier that reward pipelines for reasoning-focused LLM training (RLVR) run millions of times per training run. This repo is a small, self-contained example of that pattern applied to a real scientific domain.

## What's inside

```
crnt_solver.py    # builds S, computes ker(S^T), ker(S), deficiency, mass-action steady states
crnt_checker.py   # check_prediction(pred, expected) -> bool
```

### `crnt_solver.py`

```python
from crnt_solver import ReactionNetwork, Reaction, solve_network
import sympy as sp

species = ["A", "B", "C"]
reactions = [
    Reaction("1", reactants={"A": 1}, products={"B": 1}, k=sp.Symbol("k1", positive=True)),
    Reaction("2", reactants={"B": 1}, products={"C": 1}, k=sp.Symbol("k2", positive=True)),
    Reaction("3", reactants={"C": 1}, products={"A": 1}, k=sp.Symbol("k3", positive=True)),
]
net = ReactionNetwork(species=species, reactions=reactions)
result = solve_network(net)
```

For this catalytic cycle $A \to B \to C \to A$: rank$(S)=2$, deficiency $\delta = 3-1-2=0$ (Deficiency Zero Theorem applies — a unique, locally stable steady state is guaranteed for *any* choice of positive rate constants), and the solver returns:

- **Conservation law:** $[A]+[B]+[C]=\text{const}$
- **Flux balance:** $v_1=v_2=v_3$
- **Steady state:** $[A]^* = \dfrac{T\,k_2k_3}{k_1k_2+k_1k_3+k_2k_3}$ (and cyclic permutations)

### `crnt_checker.py`

```python
from crnt_checker import check_prediction

expected = {...}   # reference answer, same shape as solve_network()'s output
pred     = {...}   # candidate answer — could be scaled, reordered, differently written

check_prediction(pred, expected)   # -> True / False
```

Run `python crnt_checker.py` to see the self-test suite: exact match, a rewritten-but-equivalent answer (scaled laws, re-chained flux equalities, reshuffled fractions), a wrong coefficient, a wrong steady state, and a missing category — the grader passes the first two and fails the rest, as it should.

## Install

```bash
pip install -r requirements.txt
```

No other dependencies — pure Python + sympy.

## Limitations (honestly stated)

- `crnt_solver.py` uses `sympy.solve` for the polynomial steady-state system, which is fine for small networks but won't scale to large or multistationary ones — a real production tool would need Gröbner bases or numerical continuation (see the [mantis-delta](https://www.biorxiv.org/content/10.64898/2026.05.14.725189v1) project for a much more complete solver).
- `check_prediction` assumes rate-constant/parameter names match by literal string between `pred` and `expected` — it can't infer that two differently-named symbols mean the same physical quantity.
- This is a portfolio-scale demonstration of a verification *pattern*, not a validated benchmark — no claim is made here about LLM accuracy on CRNT problems at scale.

## About

Built by [Rigobert Fokam Souop](https://github.com/fokamrigobert), PhD candidate in Applied Mathematics (Cayley-graph embeddings of graphs, University of Ngaoundéré). The subspace-equality logic here — checking whether two differently-written objects are secretly the same one, via a canonical/quotient representation — is the same underlying idea as the Cocycle/Quotient Labeling Theorem in the author's dissertation, applied to a different domain.

## License

MIT — see [LICENSE](LICENSE).
