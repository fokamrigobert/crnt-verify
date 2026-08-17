"""
crnt_solver.py
===============
Deterministic Chemical Reaction Network Theory (CRNT) engine.

Implements exactly the algorithm specified in reaction_network_framework.tex:

    Conservation laws  -> basis of ker(S^T)   (left null space), Gaussian elimination over Q
    Flux balance       -> basis of ker(S)     (null space),      Gaussian elimination over Q
    Steady states      -> solve S . v(x) = 0, linear if v free, polynomial if mass-action,
                           constrained to the stoichiometric compatibility class
                           x0 + Im(S)  (equivalently, the level sets of the conservation laws)
    Deficiency         -> delta = n - l - rank(S)   (Feinberg-Horn-Jackson)

All linear algebra is done with sympy.Rational entries -> exact, no stochasticity,
no sampling, no approximation, as required by the framework document.

Input
-----
species   : list[str]                      e.g. ["A", "B", "C"]
reactions : list[Reaction]                 each reaction has .reactants, .products
                                            (dict species -> stoichiometric coefficient)
                                            and belongs to a linkage class (auto-detected
                                            from the complex graph's connected components).

Output
------
A dict with the three canonical reaction invariants, plus deficiency and the
Deficiency-Zero-Theorem verdict, printed as clean JSON.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import itertools
import json
import sympy as sp


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #

@dataclass
class Reaction:
    name: str
    reactants: dict          # species -> stoichiometric coeff (reactant complex)
    products: dict            # species -> stoichiometric coeff (product complex)
    k: sp.Symbol | float = None   # rate constant (symbol by default)

    def complex_key(self, side: str):
        d = self.reactants if side == "reactant" else self.products
        return tuple(sorted(d.items()))


@dataclass
class ReactionNetwork:
    species: list
    reactions: list = field(default_factory=list)

    # ---------- core matrices ----------
    def stoichiometric_matrix(self) -> sp.Matrix:
        """S in Z^{m x n}: S_ij = product_coeff - reactant_coeff of species i in reaction j."""
        m, n = len(self.species), len(self.reactions)
        S = sp.zeros(m, n)
        for j, r in enumerate(self.reactions):
            for i, s in enumerate(self.species):
                S[i, j] = sp.Integer(r.products.get(s, 0) - r.reactants.get(s, 0))
        return S

    def rate_vector(self, x: list) -> sp.Matrix:
        """Mass-action rate vector v_j(x) = k_j * prod_i x_i^{a_ij}."""
        xmap = dict(zip(self.species, x))
        v = []
        for r in self.reactions:
            kj = r.k if r.k is not None else sp.Symbol(f"k_{r.name}", positive=True)
            term = kj
            for s, a in r.reactants.items():
                term *= xmap[s] ** a
            v.append(term)
        return sp.Matrix(v)

    # ---------- linkage classes (for deficiency) ----------
    def linkage_classes(self) -> int:
        complexes = set()
        edges = []
        for r in self.reactions:
            rc, pc = r.complex_key("reactant"), r.complex_key("product")
            complexes.add(rc)
            complexes.add(pc)
            edges.append((rc, pc))
        parent = {c: c for c in complexes}

        def find(c):
            while parent[c] != c:
                c = parent[c]
            return c

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in edges:
            union(a, b)
        return len({find(c) for c in complexes})


# --------------------------------------------------------------------------- #
# 2. Deterministic linear-algebraic core (Gaussian elimination over Q)
# --------------------------------------------------------------------------- #

def _clear_and_normalize(vecs: list) -> list:
    """Scale each nullspace basis vector to smallest integer entries, canonical sign."""
    out = []
    for v in vecs:
        denoms = [term.q if hasattr(term, "q") else sp.nsimplify(term).q for term in v]
        lcm = sp.ilcm(*denoms) if denoms else 1
        v_int = [sp.nsimplify(term) * lcm for term in v]
        g = sp.igcd(*[sp.Integer(t) for t in v_int]) if any(v_int) else 1
        g = g if g != 0 else 1
        v_int = [sp.Integer(t) / g for t in v_int]
        # canonical sign: first nonzero entry positive
        for t in v_int:
            if t != 0:
                if t < 0:
                    v_int = [-t for t in v_int]
                break
        out.append(v_int)
    return out


def conservation_laws(S: sp.Matrix, species: list) -> dict:
    """Basis of ker(S^T) = LeftNull(S). Returns dim, basis vectors, and human-readable laws."""
    basis = S.T.nullspace()
    basis_int = _clear_and_normalize(basis) if basis else []
    laws = []
    for vec in basis_int:
        terms = [f"{int(c)}*[{sp_name}]" if c != 1 else f"[{sp_name}]"
                  for c, sp_name in zip(vec, species) if c != 0]
        laws.append(" + ".join(terms) + " = const")
    return {
        "dimension": len(basis_int),
        "rank_S": S.rank(),
        "num_species": len(species),
        "basis_vectors": [[int(c) for c in v] for v in basis_int],
        "laws": laws,
    }


def flux_balance(S: sp.Matrix, reaction_names: list) -> dict:
    """Basis of ker(S). These are the extreme-pathway / EFM generators."""
    basis = S.nullspace()
    basis_int = _clear_and_normalize(basis) if basis else []
    modes = []
    for vec in basis_int:
        terms = [f"{int(c)}*v_{name}" if c != 1 else f"v_{name}"
                  for c, name in zip(vec, reaction_names) if c != 0]
        modes.append(" = ".join(terms) if len(terms) > 1 else (terms[0] if terms else ""))
    return {
        "dimension": len(basis_int),
        "rank_S": S.rank(),
        "num_reactions": len(reaction_names),
        "basis_vectors": [[int(c) for c in v] for v in basis_int],
        "elementary_flux_modes": modes,
    }


def deficiency(S: sp.Matrix, linkage_classes: int) -> dict:
    n = S.cols
    rank_S = S.rank()
    delta = n - linkage_classes - rank_S
    return {"n_reactions": n, "linkage_classes": linkage_classes,
            "rank_S": rank_S, "deficiency": delta,
            "deficiency_zero": delta == 0}


def steady_states_mass_action(net: ReactionNetwork, total_symbol="T") -> dict:
    """
    Solve S.v(x) = 0 under mass-action kinetics, restricted to the stoichiometric
    compatibility class defined by the conservation law(s) found above
    (x lies on c^T x = T for each conservation vector c).
    Returns the symbolic steady-state solution.
    """
    S = net.stoichiometric_matrix()
    x = sp.symbols(f"x0:{len(net.species)}", positive=True)
    v = net.rate_vector(list(x))
    ode_rhs = S * v  # dx/dt

    cons = conservation_laws(S, net.species)
    T_syms = sp.symbols(f"{total_symbol}0:{cons['dimension']}", positive=True)

    cons_eqs = []
    for idx, cvec in enumerate(cons["basis_vectors"]):
        expr = sum(sp.Integer(c) * xi for c, xi in zip(cvec, x))
        cons_eqs.append(sp.Eq(expr, T_syms[idx]))

    ode_eqs = [sp.Eq(expr, 0) for expr in ode_rhs]

    # Steady state = ODE equations (rank-deficient by construction) + conservation laws
    # Use only rank(S) independent ODE rows + all conservation equations -> square system.
    eqs = ode_eqs + cons_eqs
    sol = sp.solve(eqs, list(x), dict=True)

    sol_str = []
    for s in sol:
        sol_str.append({str(k): sp.simplify(v_) for k, v_ in s.items()})

    return {
        "variables": [str(xi) for xi in x],
        "conservation_constants": [str(t) for t in T_syms],
        "ode_system": [f"d[{sp_name}]/dt = {sp.simplify(expr)}" for sp_name, expr in zip(net.species, ode_rhs)],
        "solutions": [{k: str(v_) for k, v_ in s.items()} for s in sol_str],
    }


def solve_network(net: ReactionNetwork) -> dict:
    S = net.stoichiometric_matrix()
    cons = conservation_laws(S, net.species)
    flux = flux_balance(S, [r.name for r in net.reactions])
    defc = deficiency(S, net.linkage_classes())
    ss = steady_states_mass_action(net)

    result = {
        "conservation laws": cons["laws"],
        "steady states": [
            f"[{sp_name}]* = {ss['solutions'][0][xi]}" if ss["solutions"] else "no closed-form solution found"
            for sp_name, xi in zip(net.species, ss["variables"])
        ] if ss["solutions"] else ["system is nonlinear / requires numerical or Groebner-basis solve"],
        "flux balance": flux["elementary_flux_modes"],
        "_diagnostics": {
            "stoichiometric_matrix": S.tolist(),
            "rank_S": S.rank(),
            "deficiency": defc,
            "conservation_law_dimension": cons["dimension"],
            "flux_balance_dimension": flux["dimension"],
        },
    }
    return result


# --------------------------------------------------------------------------- #
# 3. Demonstration network (deficiency-zero cycle: A -> B -> C -> A)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    species = ["A", "B", "C"]
    reactions = [
        Reaction("1", reactants={"A": 1}, products={"B": 1}, k=sp.Symbol("k1", positive=True)),
        Reaction("2", reactants={"B": 1}, products={"C": 1}, k=sp.Symbol("k2", positive=True)),
        Reaction("3", reactants={"C": 1}, products={"A": 1}, k=sp.Symbol("k3", positive=True)),
    ]
    net = ReactionNetwork(species=species, reactions=reactions)

    full = solve_network(net)

    canonical = {
        "result": json.dumps(["conservation laws", "steady states", "flux balance"]) +
                   "=" + json.dumps([full["conservation laws"], full["steady states"], full["flux balance"]])
    }

    print("===== FULL DIAGNOSTIC OUTPUT =====")
    print(json.dumps(full, indent=2, default=str))
    print("\n===== CANONICAL JSON RESULT =====")
    print(json.dumps(canonical, indent=2))
