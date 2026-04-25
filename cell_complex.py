"""
Demo: incremental cell-complex enumeration in R^3.

Cells are stored as combinatorial sign vectors during construction; no cell
witness is ever stored on a Cell while the complex is being built. Slice
tests run LP feasibility for both possible signs of the new plane (with the
chirotope GP forbidden-pattern rule as a pre-filter that skips the LP when
it would provably be infeasible).

Vertex witnesses ARE stored on Vertex: they are 3D realizations of the
chirotope, and the LP needs concrete coordinates to run. The combinatorial
shape of the cell complex is determined by the chirotope alone (= the
relative positions of the named vertices); the specific realization affects
only the witnesses, not the cell complex.

A note on what the GP rule does and doesn't do here: the 5-tuple
forbidden-pattern rule (proved in csaszar.pdf as the affine identity
star_abcd) is *necessary* for an extension of a cell sign vector to be
realizable, and it suffices to recognize the 'missing 16th cell' of every
named tetrahedron. It is *not* sufficient on its own to characterize all
realizable cells once n>=5: there are additional constraints (the
'wedge-ordering' constraints around lines through pairs of named points,
and higher Plucker relations) that GP3 misses. So we use it only as a
pre-filter and rely on the LP for the final yes/no decision.

At the END, cell witnesses are computed via LP and three numerical
validations are run:
  1. Each cell's witness gives the expected bracket signs.
  2. Random affine combinations of the named vertices land in some
     enumerated cell.
  3. The bracket-sign segment-triangle piercing test agrees with a
     numerical (Moeller-Trumbore-style) test on random samples.
"""

import time
import numpy as np
from itertools import combinations
from collections import Counter
from scipy.optimize import linprog


# === Bracket primitives ===

def bracket(p_i, p_j, p_k, p_l):
    """4-bracket [i,j,k,l] = det of [[1, P_x] for P_x in (P_i,P_j,P_k,P_l)]."""
    M = np.array([[1.0, *p_i], [1.0, *p_j], [1.0, *p_k], [1.0, *p_l]])
    return np.linalg.det(M)


def bracket_linear_part(p_i, p_j, p_k):
    """Affine decomposition: bracket(P_i, P_j, P_k, p) = a @ p + b for any p."""
    b = bracket(p_i, p_j, p_k, np.zeros(3))
    a = np.array([
        bracket(p_i, p_j, p_k, e) - b for e in np.eye(3)
    ])
    return a, b


# === Vertex / Cell ===

class Vertex:
    """A named point with a stored 3D realization (the witness).

    The witness is a specific point in R^3 chosen to realize the chirotope
    so far, computed via an LP at the moment the vertex is created. The
    cell complex shape is determined by the chirotope; the witness is just
    one such realization that the LP-based slicer needs to run.
    """
    def __init__(self, index, witness, combinatorial=None):
        self.index = index
        self.witness = np.asarray(witness, dtype=float)
        self.combinatorial = dict(combinatorial) if combinatorial else {}

    def __repr__(self):
        x, y, z = self.witness
        return f"V{self.index}({x:+.3f},{y:+.3f},{z:+.3f})"


class Cell:
    """Combinatorial cell.

    During construction, only `signs` is set. `witness` stays None until the
    validation phase, where it is computed from `signs` by an LP (and
    nothing in the construction logic consults it).
    """
    def __init__(self, signs):
        self.signs = dict(signs)
        self.witness = None

    def __repr__(self):
        return f"Cell(<{len(self.signs)} signs>)"

    def facets(self, chirotope, n_total):
        return {p for p in self.signs if is_facet(p, self.signs, chirotope, n_total)}


# === Chirotope-only forbidden-pattern test ===

def is_facet(plane, signs, chirotope, n_total):
    """Is `plane`'s sign in `signs` essential, or implied by other signs?

    Implied (= redundant) iff flipping it triggers a forbidden pattern in
    some 4-tuple containing `plane`'s three indices. Pure chirotope test;
    no coordinates.
    """
    flipped = -signs[plane]
    a, b, c = plane
    for d in range(n_total):
        if d in (a, b, c):
            continue
        tup = tuple(sorted((a, b, c, d)))
        chir = chirotope.get(tup)
        if chir is None:
            continue
        a_s, b_s, c_s, d_s = tup
        planes_4 = (
            (a_s, b_s, c_s),
            (a_s, b_s, d_s),
            (a_s, c_s, d_s),
            (b_s, c_s, d_s),
        )
        actual = tuple(flipped if p == plane else signs[p] for p in planes_4)
        forbidden = (-chir, +chir, -chir, +chir)
        if actual == forbidden:
            return False
    return True


def precompute_forbidden_patterns(plane, chirotope, n_total):
    """Per-new-plane precomputation: list of (planes_4, forbidden_signs).
    Reused across all cells in a single slice-by-this-plane pass."""
    a, b, c = plane
    out = []
    for d in range(n_total):
        if d in (a, b, c):
            continue
        tup = tuple(sorted((a, b, c, d)))
        chir = chirotope.get(tup)
        if chir is None:
            continue
        a_s, b_s, c_s, d_s = tup
        planes_4 = (
            (a_s, b_s, c_s),
            (a_s, b_s, d_s),
            (a_s, c_s, d_s),
            (b_s, c_s, d_s),
        )
        forbidden = (-chir, +chir, -chir, +chir)
        out.append((planes_4, forbidden))
    return out


def check_forbidden_patterns(signs, plane, sign_value, patterns):
    """Pre-filter: returns False the moment any forbidden pattern is hit by
    the proposed extension (signs U {plane: sign_value}); returns True if
    no forbidden pattern fires (LP is then needed to decide realizability)."""
    for planes_4, forbidden in patterns:
        actual = []
        all_known = True
        for p in planes_4:
            if p == plane:
                actual.append(sign_value)
            elif p in signs:
                actual.append(signs[p])
            else:
                all_known = False
                break
        if all_known and tuple(actual) == forbidden:
            return False
    return True


# === LP-based witness finder (the slice oracle and the validation oracle) ===

def find_witness(signs, vertices, bracket_cache=None, slack_max=1.0, eps=1e-9):
    """Find p in R^3 with sgn(bracket(P_i,P_j,P_k,p)) == signs[(i,j,k)] for
    every triple. Returns p or None.

    Used in two places:
      - inside the slicer, to decide whether each side of a new plane is
        realizable. The witness is *not* stored on the resulting Cell.
      - inside the validation phase, to attach a witness to each cell so
        bracket signs and piercing can be checked numerically.
    """
    if not signs:
        return np.zeros(3)
    A_ub = []
    b_ub = []
    for triple, s in signs.items():
        if bracket_cache is not None and triple in bracket_cache:
            a, b = bracket_cache[triple]
        else:
            i, j, k = triple
            a, b = bracket_linear_part(
                vertices[i].witness, vertices[j].witness, vertices[k].witness
            )
            if bracket_cache is not None:
                bracket_cache[triple] = (a, b)
        A_ub.append([-s * a[0], -s * a[1], -s * a[2], 1.0])
        b_ub.append(s * b)
    c = [0.0, 0.0, 0.0, -1.0]
    bounds = [(None, None)] * 3 + [(None, slack_max)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and -res.fun > eps:
        return res.x[:3]
    return None


# === Cell complex (LP-based slicing, no cell witnesses stored) ===

class CellComplex:
    def __init__(self, initial_witnesses):
        assert len(initial_witnesses) == 3
        self.vertices = [Vertex(i, w) for i, w in enumerate(initial_witnesses)]
        self._bracket_cache = {}
        plane = (0, 1, 2)
        self.cells = [Cell({plane: +1}), Cell({plane: -1})]
        # Sanity: initial cells must each be non-empty. We only check this;
        # we don't store the resulting witnesses on the cells.
        for cell in self.cells:
            if find_witness(cell.signs, self.vertices, self._bracket_cache) is None:
                raise ValueError(f"Initial cell {cell.signs} is empty.")
        self.chirotope = {}
        self.lp_calls = 2  # the two sanity LPs above
        self.filter_hits = 0

    def add_point_in_cell(self, target_cell):
        """Add a new vertex inside target_cell, then slice every cell with
        each new plane via two LPs (one per sign), with the GP forbidden-
        pattern rule as a pre-filter to skip provably-empty extensions.

        target_cell.witness is and stays None: we do not consult it. The
        new vertex's witness is found by an LP from target_cell.signs.
        """
        n = len(self.vertices)
        new_witness = find_witness(target_cell.signs, self.vertices, self._bracket_cache)
        if new_witness is None:
            raise ValueError("Target cell is empty (cannot add a point inside it).")
        self.lp_calls += 1
        new_vertex = Vertex(n, new_witness, combinatorial=dict(target_cell.signs))
        self.vertices.append(new_vertex)

        # The new chirotope entries are pinned by the chosen cell's signs.
        for triple, sign in target_cell.signs.items():
            a, b, c = triple
            self.chirotope[(a, b, c, n)] = sign

        n_total = len(self.vertices)
        cells = list(self.cells)
        new_planes = [(i, j, n) for i, j in combinations(range(n), 2)]
        for new_plane in new_planes:
            patterns = precompute_forbidden_patterns(new_plane, self.chirotope, n_total)
            next_cells = []
            for cell in cells:
                next_cells.extend(self._slice_cell(cell, new_plane, patterns))
            cells = next_cells
        self.cells = cells

    def _slice_cell(self, cell, new_plane, patterns):
        """Test both signs of new_plane for `cell`. For each side, the GP
        forbidden-pattern pre-filter is consulted first; only when it
        cannot definitively rule the side out do we run the LP."""
        results = []
        for sign in (+1, -1):
            if not check_forbidden_patterns(cell.signs, new_plane, sign, patterns):
                self.filter_hits += 1
                continue
            new_signs = {**cell.signs, new_plane: sign}
            self.lp_calls += 1
            if find_witness(new_signs, self.vertices, self._bracket_cache) is not None:
                results.append(Cell(new_signs))
        return results


# === Validation phase ===

def attach_cell_witnesses(complex):
    """Compute witnesses for all cells via LP (validation only)."""
    n_with = 0
    for cell in complex.cells:
        cell.witness = find_witness(cell.signs, complex.vertices, complex._bracket_cache)
        if cell.witness is not None:
            n_with += 1
    return n_with


def verify_cell_witnesses(complex):
    n_cells = len(complex.cells)
    n_correct = 0
    failures = []
    for i, cell in enumerate(complex.cells):
        if cell.witness is None:
            failures.append((i, '(no witness)', None, None, None))
            continue
        ok = True
        for triple, expected in cell.signs.items():
            a, b, k = triple
            br = bracket(
                complex.vertices[a].witness,
                complex.vertices[b].witness,
                complex.vertices[k].witness,
                cell.witness,
            )
            actual = int(np.sign(br))
            if actual != expected:
                failures.append((i, triple, expected, actual, br))
                ok = False
        if ok:
            n_correct += 1
    print(f"  {n_correct}/{n_cells} cells: every bracket sign matches the witness.")
    for f in failures[:5]:
        print(f"    cell {f[0]} triple {f[1]}: expected {f[2]}, got {f[3]} (bracket={f[4]})")
    return n_correct == n_cells


def random_combination_test(complex, n_samples=1000, alpha_lo=-1.0, alpha_hi=2.0, seed=12345):
    rng = np.random.default_rng(seed)
    P = np.array([v.witness for v in complex.vertices])
    n = P.shape[0]
    triples = list(combinations(range(n), 3))
    cell_signs_set = frozenset(
        tuple(sorted(c.signs.items())) for c in complex.cells
    )
    n_matched = n_skipped = n_unmatched = 0
    for _ in range(n_samples):
        alpha = rng.uniform(alpha_lo, alpha_hi, size=n)
        s = alpha.sum()
        if abs(s) < 1e-3:
            n_skipped += 1
            continue
        p = (alpha @ P) / s
        signs = {}
        any_zero = False
        for t in triples:
            i, j, k = t
            br = bracket(P[i], P[j], P[k], p)
            if abs(br) < 1e-10:
                any_zero = True
                break
            signs[t] = int(np.sign(br))
        if any_zero:
            n_skipped += 1
            continue
        if tuple(sorted(signs.items())) in cell_signs_set:
            n_matched += 1
        else:
            n_unmatched += 1
    print(f"  matched: {n_matched}/{n_samples}    unmatched: {n_unmatched}    "
          f"skipped: {n_skipped} (degenerate)")
    return n_unmatched == 0


def pierces(i, j, k, l, m, vertices):
    """Combinatorial pierce test using bracket signs."""
    P = [v.witness for v in vertices]
    s1 = int(np.sign(bracket(P[i], P[j], P[k], P[l])))
    s2 = int(np.sign(bracket(P[i], P[j], P[l], P[m])))
    s3 = int(np.sign(bracket(P[i], P[j], P[m], P[k])))
    s4 = int(np.sign(bracket(P[k], P[l], P[m], P[i])))
    s5 = int(np.sign(bracket(P[k], P[l], P[m], P[j])))
    if 0 in (s1, s2, s3, s4, s5):
        return None
    return (s1 == s2 == s3) and (s4 != s5)


def numerical_pierces(P_i, P_j, P_k, P_l, P_m, eps=1e-9):
    edge1 = P_l - P_k
    edge2 = P_m - P_k
    direction = P_j - P_i
    pvec = np.cross(direction, edge2)
    det_val = float(np.dot(edge1, pvec))
    if abs(det_val) < eps:
        return False
    inv_det = 1.0 / det_val
    tvec = P_i - P_k
    u = float(np.dot(tvec, pvec)) * inv_det
    if u <= eps or u >= 1 - eps:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inv_det
    if v <= eps or u + v >= 1 - eps:
        return False
    t = float(np.dot(edge2, qvec)) * inv_det
    if t <= eps or t >= 1 - eps:
        return False
    return True


def pierces_validation_test(complex, n_samples=1000, seed=54321):
    rng = np.random.default_rng(seed)
    n = len(complex.vertices)
    if n < 5:
        print(f"  Need at least 5 vertices (have {n}); skipping.")
        return
    n_agree = n_disagree = n_sym_none = 0
    n_pierce_true = 0
    n_parallel = 0
    for _ in range(n_samples):
        idx = rng.permutation(n)[:5]
        i, j, k, l, m = (int(x) for x in idx)
        sym = pierces(i, j, k, l, m, complex.vertices)
        num = numerical_pierces(
            complex.vertices[i].witness, complex.vertices[j].witness,
            complex.vertices[k].witness, complex.vertices[l].witness,
            complex.vertices[m].witness,
        )
        edge1 = complex.vertices[l].witness - complex.vertices[k].witness
        edge2 = complex.vertices[m].witness - complex.vertices[k].witness
        direction = complex.vertices[j].witness - complex.vertices[i].witness
        det_val = float(np.dot(edge1, np.cross(direction, edge2)))
        if abs(det_val) < 1e-9:
            n_parallel += 1
        if sym is None:
            n_sym_none += 1
            continue
        if sym == num:
            n_agree += 1
            if sym:
                n_pierce_true += 1
        else:
            n_disagree += 1
            if n_disagree <= 5:
                print(f"    DISAGREE: ({i},{j}) vs ({k},{l},{m}): sym={sym}, num={num}")
    print(f"  agree: {n_agree}/{n_samples}    disagree: {n_disagree}    "
          f"sym None: {n_sym_none}    (parallel-line cases: {n_parallel})    "
          f"(agreeing pierces: {n_pierce_true})")


def print_facet_stats(complex, indent="    "):
    counts = []
    for cell in complex.cells:
        counts.append(len(cell.facets(complex.chirotope, len(complex.vertices))))
    distribution = Counter(counts)
    histogram = ", ".join(f"{k}:{v}" for k, v in sorted(distribution.items()))
    mean = sum(counts) / len(counts) if counts else 0.0
    n_planes = len(list(combinations(range(len(complex.vertices)), 3)))
    print(f"{indent}facet count distribution (out of {n_planes} planes per cell): "
          f"{{{histogram}}}    mean={mean:.2f}")


# === Demo ===

def main():
    rng = np.random.default_rng(200)
    initial_witnesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    print("=== Building cell complex ===")
    print("    No cell witnesses are stored during construction. The slicer")
    print("    uses LP feasibility on each (sign +/- 1) extension, with the")
    print("    GP forbidden-pattern rule as a chirotope pre-filter.")
    print()
    cc = CellComplex(initial_witnesses)
    print(f"  Initial: {len(cc.vertices)} vertices, {len(cc.cells)} cells")

    t0 = time.perf_counter()
    for step in range(4):
        idx = int(rng.integers(len(cc.cells)))
        before = len(cc.cells)
        before_lp = cc.lp_calls
        before_filt = cc.filter_hits
        cc.add_point_in_cell(cc.cells[idx])
        added = cc.vertices[-1]
        lp_done = cc.lp_calls - before_lp
        filt = cc.filter_hits - before_filt
        print(f"  step {step+1}: added {added} via cell {idx}  ->  "
              f"{len(cc.cells)} cells (was {before});  "
              f"LP={lp_done}, filtered={filt}")
        print_facet_stats(cc)
    elapsed_build = time.perf_counter() - t0
    print(f"\n  Construction: {len(cc.vertices)} vertices, {len(cc.cells)} cells, "
          f"{elapsed_build:.2f}s, total LP={cc.lp_calls}, filtered={cc.filter_hits}")

    print("\n=== Validation phase: computing cell witnesses ===")
    t0 = time.perf_counter()
    n_with_witness = attach_cell_witnesses(cc)
    elapsed_witness = time.perf_counter() - t0
    print(f"  {n_with_witness}/{len(cc.cells)} cells with witnesses, {elapsed_witness:.2f}s")

    print("\n=== Verifying cell witnesses against their sign vectors ===")
    verify_cell_witnesses(cc)

    print("\n=== Random affine-combination test (1000 samples, coefficients in [-1,2]) ===")
    random_combination_test(cc)

    print("\n=== Random segment-triangle piercing test (1000 samples) ===")
    pierces_validation_test(cc)


if __name__ == "__main__":
    main()
