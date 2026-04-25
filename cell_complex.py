"""
Demo: incremental cell-complex enumeration in R^3.

Each Vertex carries:
  - a combinatorial part: dict (i,j,k) -> sign in {-1,+1} for every i<j<k<self.index,
    recording the sign of the 4-bracket [i,j,k,self.index]. This pins down which
    cell of the prior arrangement this vertex was added in.
  - a realized 3D witness, used pedagogically and as input to the LP that
    decides cell-plane slices.

The demo:
  1. Start with 3 points P_0, P_1, P_2 at fixed coordinates and 2 cells (the
     two sides of plane 012).
  2. Iterate 4 times: pick a random cell, add a new point inside it
     (witness = the cell's LP-found witness), then slice every existing cell
     with all new planes ij(n).  A chirotope-based pre-filter (the
     forbidden-pattern rule from the doc) skips the LP when the slice is
     forced by the chirotope.
  3. Verify each cell's witness gives the expected bracket signs.
  4. Take 1000 random affine combinations of the 7 named points and check
     that each falls in one of the enumerated cells.
  5. Take 1000 random (segment, triangle) splits of the 7 vertices and
     verify the bracket-sign-based `pierces` test agrees with a numerical
     (Moeller-Trumbore-style) segment-triangle intersection.
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


# === Vertex and Cell classes ===

class Vertex:
    def __init__(self, index, witness, combinatorial=None):
        self.index = index
        self.witness = np.asarray(witness, dtype=float)
        self.combinatorial = dict(combinatorial) if combinatorial else {}

    def __repr__(self):
        x, y, z = self.witness
        return f"V{self.index}({x:+.3f},{y:+.3f},{z:+.3f})"


class Cell:
    def __init__(self, signs, witness=None):
        self.signs = dict(signs)
        self.witness = np.asarray(witness, dtype=float) if witness is not None else None

    def __repr__(self):
        return f"Cell(<{len(self.signs)} signs>)"

    def facets(self, chirotope, n_total):
        """Return the set of triples whose constraint is essential (non-redundant)."""
        return {p for p in self.signs if is_facet(p, self.signs, chirotope, n_total)}


# === LP-based witness finder ===

def find_witness(signs, vertices, bracket_cache=None, slack_max=1.0, eps=1e-9):
    """Find p in R^3 with sgn(bracket(P_i,P_j,P_k,p)) == signs[(i,j,k)] for each triple.

    `bracket_cache`, if provided, is a dict triple -> (a, b) for the affine
    decomposition of [i,j,k,p] = a @ p + b; misses are filled in. Caching
    matters: a single LP iterates over all triples in `signs` (~35 of them at
    n=7), and we run thousands of LPs.
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


# === Chirotope-based forbidden-pattern test (the GP rule from the doc) ===

def precompute_forbidden_patterns(plane, chirotope, n_total):
    """For each 4-tuple {plane's 3 indices, d}, return the list of records
       (planes_4, forbidden_signs) used by check_forbidden_patterns.

    Computed once per new plane (NOT per cell), so the cell loop never
    re-walks `d in range(n_total)` or re-builds the sorted 4-plane list.
    """
    a, b, c = plane
    out = []
    for d in range(n_total):
        if d in (a, b, c):
            continue
        tuple_sorted = tuple(sorted((a, b, c, d)))
        chir_sign = chirotope.get(tuple_sorted)
        if chir_sign is None:
            continue
        a_s, b_s, c_s, d_s = tuple_sorted
        planes_4 = (
            (a_s, b_s, c_s),
            (a_s, b_s, d_s),
            (a_s, c_s, d_s),
            (b_s, c_s, d_s),
        )
        forbidden_signs = (-chir_sign, +chir_sign, -chir_sign, +chir_sign)
        out.append((planes_4, forbidden_signs))
    return out


def check_forbidden_patterns(signs, plane, sign_value, patterns):
    """Does extending `signs` with (plane: sign_value) trigger any forbidden
    pattern in the precomputed list? Returns False on first hit, True if
    none fires (and uncheckable 4-tuples are silently skipped)."""
    for planes_4, forbidden_signs in patterns:
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
        if all_known and tuple(actual) == forbidden_signs:
            return False
    return True


def is_extension_consistent(signs, plane, sign_value, chirotope, n_total):
    """Convenience wrapper: precomputes patterns then checks. Use the two-step
    form (precompute_forbidden_patterns + check_forbidden_patterns) inside the
    cell loop to amortize across cells."""
    patterns = precompute_forbidden_patterns(plane, chirotope, n_total)
    return check_forbidden_patterns(signs, plane, sign_value, patterns)


def is_facet(plane, signs, chirotope, n_total):
    """Is `plane`'s sign in `signs` essential (a facet) or redundant?
    Redundant <=> flipping it would create an empty cell <=> a forbidden
    pattern fires after the flip.
    """
    flipped = -signs[plane]
    signs_minus = {p: s for p, s in signs.items() if p != plane}
    return is_extension_consistent(signs_minus, plane, flipped, chirotope, n_total)


# === Cell complex with incremental updates ===

class CellComplex:
    def __init__(self, initial_witnesses):
        assert len(initial_witnesses) == 3
        self.vertices = [Vertex(i, w) for i, w in enumerate(initial_witnesses)]
        self._bracket_cache = {}  # triple -> (a_grad, b_const)
        plane = (0, 1, 2)
        self.cells = [Cell({plane: +1}), Cell({plane: -1})]
        for cell in self.cells:
            cell.witness = find_witness(cell.signs, self.vertices, self._bracket_cache)
            assert cell.witness is not None
        self.chirotope = {}
        self.lp_calls = 0
        self.filtered_calls = 0

    def get_bracket_part(self, triple):
        cached = self._bracket_cache.get(triple)
        if cached is not None:
            return cached
        i, j, k = triple
        result = bracket_linear_part(
            self.vertices[i].witness, self.vertices[j].witness, self.vertices[k].witness
        )
        self._bracket_cache[triple] = result
        return result

    def add_point_in_cell(self, target_cell, plane_order='lex_asc'):
        """plane_order in {'lex_asc', 'lex_desc', 'i_desc'} controls the order
        in which the new planes ij(n) are processed in the slice loop.
        The cell complex result is identical; only the timing of the
        chirotope pre-filter (and so the LP / filter ratio) is affected."""
        n = len(self.vertices)  # index of the new vertex (will be appended)
        if target_cell.witness is None:
            target_cell.witness = find_witness(target_cell.signs, self.vertices, self._bracket_cache)
        new_witness = target_cell.witness.copy()
        new_vertex = Vertex(n, new_witness, combinatorial=dict(target_cell.signs))
        self.vertices.append(new_vertex)

        # Update chirotope with new 4-tuples (a,b,c,n) for each old plane (a,b,c).
        for triple, sign in target_cell.signs.items():
            a, b, c = triple
            self.chirotope[(a, b, c, n)] = sign

        # Order the new planes
        pairs = list(combinations(range(n), 2))
        if plane_order == 'lex_asc':
            pass
        elif plane_order == 'lex_desc':
            pairs.sort(key=lambda p: (-p[0], -p[1]))
        elif plane_order == 'i_desc':
            pairs.sort(key=lambda p: (-p[0], p[1]))
        else:
            raise ValueError(f"unknown plane_order: {plane_order}")
        new_planes = [(i, j, n) for i, j in pairs]

        # Process plane by plane across all (sub-)cells. Precompute the
        # forbidden patterns for the current plane once, share across cells.
        n_total = len(self.vertices)
        cells = list(self.cells)
        for new_plane in new_planes:
            patterns = precompute_forbidden_patterns(new_plane, self.chirotope, n_total)
            a_lp, b_lp = self.get_bracket_part(new_plane)
            next_cells = []
            for cell in cells:
                next_cells.extend(self._slice_cell(cell, new_plane, a_lp, b_lp, patterns))
            cells = next_cells
        self.cells = cells

    def _slice_cell(self, c, new_plane, a_lp, b_lp, patterns):
        s_at_witness = a_lp @ c.witness + b_lp

        if abs(s_at_witness) < 1e-9:
            # Witness lies on the new plane (it IS the just-added point).
            # Try both sides; pre-filter rejects forbidden extensions.
            results = []
            for s in (+1, -1):
                if not check_forbidden_patterns(c.signs, new_plane, s, patterns):
                    self.filtered_calls += 1
                    continue
                self.lp_calls += 1
                ns = {**c.signs, new_plane: s}
                w = find_witness(ns, self.vertices, self._bracket_cache)
                if w is not None:
                    results.append(Cell(ns, witness=w))
            return results

        s_int = int(np.sign(s_at_witness))
        same_signs = {**c.signs, new_plane: s_int}
        cell_same = Cell(same_signs, witness=c.witness.copy())

        # Pre-filter: check the OTHER side via the chirotope. If a forbidden
        # pattern fires, the LP is guaranteed infeasible and we skip it.
        if not check_forbidden_patterns(c.signs, new_plane, -s_int, patterns):
            self.filtered_calls += 1
            return [cell_same]

        self.lp_calls += 1
        other_signs = {**c.signs, new_plane: -s_int}
        p_other = find_witness(other_signs, self.vertices, self._bracket_cache)
        if p_other is not None:
            return [cell_same, Cell(other_signs, witness=p_other)]
        return [cell_same]


# === Segment-triangle piercing (bracket-sign test from the doc) ===

def pierces(i, j, k, l, m, vertices):
    """Combinatorial test: does segment P_i P_j pierce the open triangle P_k P_l P_m?

    Conditions:
      (1) sgn[i,j,k,l] == sgn[i,j,l,m] == sgn[i,j,m,k] in {+,-}
          (the line through P_i,P_j enters the triangle's interior)
      (2) sgn[k,l,m,i] != sgn[k,l,m,j]
          (the segment endpoints are on opposite sides of the triangle's plane)

    Returns True / False, or None if any of the five brackets vanishes (the
    five points are not in general position).
    """
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
    """Numerical Moeller-Trumbore-style segment-triangle test (open interior).

    The determinant det_val below works out to [k,l,m,P_j] - [k,l,m,P_i].
    When the named points are in general position (no 4 coplanar) this
    difference can still vanish: it is exactly the case where line
    P_i-P_j is parallel to plane klm. In that case the line is OFF the
    plane (by general position), so the segment cannot pierce; we
    return False rather than None.
    """
    edge1 = P_l - P_k
    edge2 = P_m - P_k
    direction = P_j - P_i

    pvec = np.cross(direction, edge2)
    det_val = float(np.dot(edge1, pvec))
    if abs(det_val) < eps:
        return False  # line parallel to plane -> no piercing

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


# === Verification routines ===

def verify_cell_witnesses(complex):
    n_cells = len(complex.cells)
    n_correct = 0
    failures = []
    for i, cell in enumerate(complex.cells):
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
        print(f"    cell {f[0]} triple {f[1]}: expected {f[2]:+d}, got {f[3]:+d} (bracket={f[4]:.3e})")
    return n_correct == n_cells


def random_combination_test(complex, n_samples=1000, alpha_lo=-1.0, alpha_hi=2.0, seed=12345):
    rng = np.random.default_rng(seed)
    P = np.array([v.witness for v in complex.vertices])
    n = P.shape[0]
    triples = list(combinations(range(n), 3))
    cell_signs_list = [c.signs for c in complex.cells]

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
        if signs in cell_signs_list:
            n_matched += 1
        else:
            n_unmatched += 1
    print(f"  matched: {n_matched}/{n_samples}    unmatched: {n_unmatched}    skipped: {n_skipped} (degenerate)")
    return n_unmatched == 0


def pierces_validation_test(complex, n_samples=1000, seed=54321):
    """Pick 1000 random 5-element subsets of vertex indices, partition into a
    segment (2) and a triangle (3) per the random permutation, and check that
    `pierces` (bracket signs) and `numerical_pierces` (Moeller-Trumbore) agree.
    """
    rng = np.random.default_rng(seed)
    n = len(complex.vertices)
    if n < 5:
        print(f"  Need at least 5 vertices (have {n}); skipping.")
        return

    n_agree = n_disagree = 0
    n_sym_none = n_num_none = n_parallel = 0
    n_pierce_true = 0
    for _ in range(n_samples):
        idx = rng.permutation(n)[:5]
        i, j, k, l, m = (int(x) for x in idx)
        sym = pierces(i, j, k, l, m, complex.vertices)
        num = numerical_pierces(
            complex.vertices[i].witness, complex.vertices[j].witness,
            complex.vertices[k].witness, complex.vertices[l].witness,
            complex.vertices[m].witness,
        )
        # Track parallels separately (where Moeller-Trumbore det vanishes).
        # In general position these resolve to "no piercing", so num returns False.
        edge1 = complex.vertices[l].witness - complex.vertices[k].witness
        edge2 = complex.vertices[m].witness - complex.vertices[k].witness
        direction = complex.vertices[j].witness - complex.vertices[i].witness
        det_val = float(np.dot(edge1, np.cross(direction, edge2)))
        if abs(det_val) < 1e-9:
            n_parallel += 1
        if sym is None:
            n_sym_none += 1
            continue
        if num is None:
            n_num_none += 1
            continue
        if sym == num:
            n_agree += 1
            if sym:
                n_pierce_true += 1
        else:
            n_disagree += 1
            if n_disagree <= 5:
                print(f"    DISAGREE: segment ({i},{j}) vs triangle ({k},{l},{m}): "
                      f"sym={sym}, num={num}")
    print(f"  agree: {n_agree}/{n_samples}    disagree: {n_disagree}    "
          f"sym None: {n_sym_none}    num None: {n_num_none}    "
          f"(parallel-line cases: {n_parallel})    "
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

def run_iteration(plane_order, seed=200, label=None, verbose=False):
    rng = np.random.default_rng(seed)
    initial_witnesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    cc = CellComplex(initial_witnesses)
    if verbose:
        print(f"  Initial: 3 points, {len(cc.cells)} cells")
    t0 = time.perf_counter()
    for step in range(4):
        idx = int(rng.integers(len(cc.cells)))
        chosen = cc.cells[idx]
        before = len(cc.cells)
        before_lp = cc.lp_calls
        before_filtered = cc.filtered_calls
        cc.add_point_in_cell(chosen, plane_order=plane_order)
        added = cc.vertices[-1]
        lp_done = cc.lp_calls - before_lp
        filt = cc.filtered_calls - before_filtered
        total = lp_done + filt
        skipped_pct = 100 * filt / total if total > 0 else 0.0
        if verbose:
            print(f"  step {step+1} [{plane_order}]: added {added} in cell {idx}  -> "
                  f" {len(cc.cells)} cells (was {before}); LP={lp_done}, "
                  f"filtered={filt} ({skipped_pct:.1f}%)")
            print_facet_stats(cc)
    elapsed = time.perf_counter() - t0
    return cc, elapsed


def main():
    initial_witnesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    print("=== Reference run (lex_asc order, with full verifications) ===")
    cc, elapsed = run_iteration('lex_asc', verbose=True)
    print(f"\n  Final: {len(cc.vertices)} points, {len(cc.cells)} cells, "
          f"{elapsed:.2f}s, LP={cc.lp_calls}, filtered={cc.filtered_calls} "
          f"({100*cc.filtered_calls/(cc.lp_calls+cc.filtered_calls):.1f}% skipped)")

    print("\n=== Verifying cell witnesses against their sign vectors ===")
    verify_cell_witnesses(cc)

    print("\n=== Random affine-combination test (1000 samples, coefficients in [-1,2]) ===")
    random_combination_test(cc, n_samples=1000)

    print("\n=== Random segment-triangle piercing test (1000 samples) ===")
    pierces_validation_test(cc, n_samples=1000)

    print("\n=== Plane-ordering comparison ===")
    print("  (same seed, same final cells; only the slice-loop ordering differs)")
    print(f"  {'order':<10}  {'cells':>6}  {'LP calls':>9}  {'filtered':>9}  {'skip%':>7}  {'time (s)':>9}")
    for order in ('lex_asc', 'i_desc', 'lex_desc'):
        cc_o, elapsed_o = run_iteration(order, verbose=False)
        total = cc_o.lp_calls + cc_o.filtered_calls
        pct = 100.0 * cc_o.filtered_calls / total if total else 0.0
        print(f"  {order:<10}  {len(cc_o.cells):>6}  {cc_o.lp_calls:>9}  "
              f"{cc_o.filtered_calls:>9}  {pct:>6.1f}%  {elapsed_o:>9.2f}")


if __name__ == "__main__":
    main()
