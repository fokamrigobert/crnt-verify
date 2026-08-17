"""
crnt_checker.py
================
check_prediction(pred: dict, expected: dict) -> bool

Grades a model's answer to the CRNT reaction-invariant task:
    { "conservation laws": [...], "steady states": [...], "flux balance": [...] }
(or the wrapped canonical form {"result": '["conservation laws", ...]=[...]'} ).

Why not exact string match?
----------------------------
Conservation-law and flux-balance bases are only defined up to choice of basis
and scalar multiples (any nonzero scalar multiple / linear recombination of a
correct basis is still correct). Steady-state expressions are only defined up
to algebraic rearrangement. A checker that string-matches would fail
mathematically-correct answers written differently, so this is a **rule-based**
checker:

  * conservation laws  -> compared as a SUBSPACE (row-space of ker(S^T)),
                          via rank equality: rank(pred) == rank(expected)
                          == rank(pred stacked with expected).
  * flux balance       -> same subspace test, applied to ker(S).
  * steady states       -> compared per-species via symbolic difference,
                          confirmed by deterministic randomized-numeric
                          substitution (fixed seed -> reproducible).

The function is deterministic (fixed RNG seed), binary (returns only
True/False), and context-aware (the two "shapes" of subspace-vs-expression
comparison are chosen based on which invariant is being checked).
"""

from __future__ import annotations
import json
import re
import random
import sympy as sp


# --------------------------------------------------------------------------- #
# 0. Top-level entry point
# --------------------------------------------------------------------------- #

def check_prediction(pred: dict, expected: dict) -> bool:
    try:
        p = _normalize(pred)
        e = _normalize(expected)
    except Exception:
        return False  # malformed input is always incorrect, never an exception

    required_keys = ("conservation laws", "steady states", "flux balance")
    if any(k not in p or k not in e for k in required_keys):
        return False

    try:
        ok_cons = _check_subspace(p["conservation laws"], e["conservation laws"],
                                   var_pattern=_SPECIES_PATTERN, bracket_style=True)
        ok_flux = _check_subspace(p["flux balance"], e["flux balance"],
                                   var_pattern=_FLUX_PATTERN, bracket_style=False)
        ok_ss = _check_steady_states(p["steady states"], e["steady states"])
    except Exception:
        return False

    return bool(ok_cons and ok_flux and ok_ss)


# --------------------------------------------------------------------------- #
# 1. Input normalization: accept either the raw dict or the wrapped
#    {"result": '["conservation laws", ...]=[...]'} canonical string form.
# --------------------------------------------------------------------------- #

_CANON_KEYS = ["conservation laws", "steady states", "flux balance"]


def _normalize(d: dict) -> dict:
    if not isinstance(d, dict):
        raise ValueError("prediction/expected must be a dict")

    if "result" in d and isinstance(d["result"], str):
        s = d["result"]
        idx = s.find("]=[")
        if idx == -1:
            raise ValueError("wrapped result string missing '][' array boundary")
        keys_part = s[: idx + 1]
        values_part = s[idx + 2:]
        keys = json.loads(keys_part)
        values = json.loads(values_part)
        out = dict(zip(keys, values))
    else:
        out = dict(d)

    # allow single strings instead of length-1 lists
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = [v]
    return out


# --------------------------------------------------------------------------- #
# 2. Subspace-equivalence check (conservation laws / flux balance)
# --------------------------------------------------------------------------- #

_SPECIES_PATTERN = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")           # matches [A], [B2] ...
_FLUX_PATTERN = re.compile(r"\bv_?[A-Za-z0-9]+\b")                        # matches v1, v_1, v_R2 ...


def _variable_universe(equations: list, pattern: re.Pattern, bracket_style: bool) -> list:
    names = set()
    for eq in equations:
        for m in pattern.finditer(eq):
            names.add(m.group(1) if bracket_style else m.group(0))
    return sorted(names)


def _equation_to_rows(eq: str, variables: list, symmap: dict) -> list:
    """
    Split a (possibly chained) equality 'lhs = mid = rhs' into pairwise
    homogeneous constraints [lhs-mid, mid-rhs, ...], then return each
    constraint's coefficient row over `variables` (all other symbols,
    e.g. 'const', 'T0', are treated as irrelevant nuisance terms and ignored).
    """
    # bracket species notation [A] -> plain symbol A so sympify can parse it
    clean = re.sub(r"\[([A-Za-z_][A-Za-z0-9_]*)\]", r"\1", eq)
    sides = [s.strip() for s in clean.split("=")]
    if len(sides) < 2:
        sides = [sides[0], "0"]

    rows = []
    for a, b in zip(sides, sides[1:]):
        expr = sp.sympify(a, locals=symmap) - sp.sympify(b, locals=symmap)
        expr = sp.expand(expr)
        row = [sp.nsimplify(expr.coeff(symmap[v], 1)) for v in variables]
        if any(c != 0 for c in row):
            rows.append(row)
    return rows


def _check_subspace(pred_eqs: list, exp_eqs: list, var_pattern: re.Pattern, bracket_style: bool) -> bool:
    if len(pred_eqs) == 0 or len(exp_eqs) == 0:
        return len(pred_eqs) == len(exp_eqs) == 0

    variables = _variable_universe(exp_eqs, var_pattern, bracket_style)
    # if pred references variables never seen in the reference answer, it's wrong
    pred_vars = _variable_universe(pred_eqs, var_pattern, bracket_style)
    if not set(pred_vars).issubset(set(variables)):
        return False
    if not variables:
        return False

    symmap = {v: sp.Symbol(v) for v in variables}

    pred_rows, exp_rows = [], []
    for eq in pred_eqs:
        pred_rows.extend(_equation_to_rows(eq, variables, symmap))
    for eq in exp_eqs:
        exp_rows.extend(_equation_to_rows(eq, variables, symmap))

    if not pred_rows or not exp_rows:
        return False

    Mp = sp.Matrix(pred_rows)
    Me = sp.Matrix(exp_rows)
    Mstack = Mp.col_join(Me)

    return Mp.rank() == Me.rank() == Mstack.rank()


# --------------------------------------------------------------------------- #
# 3. Steady-state equivalence: per-species symbolic + randomized-numeric check
# --------------------------------------------------------------------------- #

_SS_LABEL = re.compile(r"^\s*\[?([A-Za-z_][A-Za-z0-9_]*)\]?\s*\*?\s*=\s*(.+)$")


def _parse_steady_states(entries: list) -> dict:
    out = {}
    for e in entries:
        m = _SS_LABEL.match(e)
        if not m:
            # fallback text like "no closed-form solution found" -> keyed by whole text
            out[e.strip().lower()] = None
            continue
        species, expr = m.group(1), m.group(2)
        out[species] = expr
    return out


def _check_steady_states(pred_entries: list, exp_entries: list, seed: int = 1234, trials: int = 5,
                          tol: float = 1e-6) -> bool:
    pred_map = _parse_steady_states(pred_entries)
    exp_map = _parse_steady_states(exp_entries)

    if set(pred_map.keys()) != set(exp_map.keys()):
        return False

    # non-formula fallback entries (e.g. both say "system is nonlinear ...")
    formula_keys = [k for k, v in exp_map.items() if v is not None]
    for k in exp_map:
        if exp_map[k] is None or pred_map[k] is None:
            if exp_map[k] != pred_map[k]:  # both must be the identical fallback text
                return False

    if not formula_keys:
        return True

    try:
        pred_exprs = {k: sp.sympify(pred_map[k]) for k in formula_keys}
        exp_exprs = {k: sp.sympify(exp_map[k]) for k in formula_keys}
    except Exception:
        return False

    free_syms = set()
    for k in formula_keys:
        free_syms |= pred_exprs[k].free_symbols | exp_exprs[k].free_symbols
    free_syms = sorted(free_syms, key=str)

    rng = random.Random(seed)
    for _ in range(trials):
        subs = {s: sp.Rational(rng.randint(2, 37), rng.randint(1, 5)) for s in free_syms}
        for k in formula_keys:
            try:
                diff = (pred_exprs[k] - exp_exprs[k]).evalf(subs=subs)
                if abs(complex(diff)) > tol:
                    return False
            except Exception:
                return False
    return True


# --------------------------------------------------------------------------- #
# 4. Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    expected = {
        "conservation laws": ["[A] + [B] + [C] = const"],
        "steady states": [
            "[A]* = T0*k2*k3/(k1*k2 + k1*k3 + k2*k3)",
            "[B]* = T0*k1*k3/(k1*k2 + k1*k3 + k2*k3)",
            "[C]* = T0*k1*k2/(k1*k2 + k1*k3 + k2*k3)",
        ],
        "flux balance": ["v_1 = v_2 = v_3"],
    }

    # 1. exact match
    assert check_prediction(expected, expected) is True

    # 2. mathematically equivalent but differently-written answer -> must PASS
    equivalent = {
        "conservation laws": ["2*[B] + 2*[A] + 2*[C] = const2"],           # scaled + reordered
        "steady states": [
            "[C]* = T0*k1*k2/(k2*k3 + k1*k3 + k1*k2)",                    # reordered denom
            "[A]* = T0*k2*k3/(k1*k2 + k1*k3 + k2*k3)",
            "[B]* = T0*k1*k3/(k1*k2 + k1*k3 + k2*k3)",
        ],
        "flux balance": ["v_1 = v_3", "v_3 = v_2"],                        # same subspace, re-chained
    }
    assert check_prediction(equivalent, expected) is True

    # 3. wrong conservation law -> must FAIL
    wrong_cons = json.loads(json.dumps(expected))
    wrong_cons["conservation laws"] = ["[A] + 2*[B] + [C] = const"]
    assert check_prediction(wrong_cons, expected) is False

    # 4. wrong steady state -> must FAIL
    wrong_ss = json.loads(json.dumps(expected))
    wrong_ss["steady states"][0] = "[A]* = T0*k2*k3/(k1*k2 + k1*k3)"
    assert check_prediction(wrong_ss, expected) is False

    # 5. missing category -> must FAIL
    incomplete = {"conservation laws": expected["conservation laws"],
                  "steady states": expected["steady states"]}
    assert check_prediction(incomplete, expected) is False

    # 6. wrapped canonical-string form, round-tripped -> must PASS
    wrapped = {
        "result": json.dumps(_CANON_KEYS) + "=" + json.dumps(
            [expected["conservation laws"], expected["steady states"], expected["flux balance"]]
        )
    }
    assert check_prediction(wrapped, expected) is True

    print("All self-tests passed.")
