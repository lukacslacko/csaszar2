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
     with all new planes ij(n).
  3. After 4 steps we have 7 points and the cell complex of the arrangement
     of binom(7,3)=35 planes.
  4. Verify each cell's witness yields the expected bracket signs.
  5. Take 1000 random affine combinations of the 7 named points and check
     that each falls in one of the enumerated cells.
"""

import numpy as np
from itertools import combinations
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


# === LP-based witness finder ===

def find_witness(signs, vertices, slack_max=1.0, eps=1e-9):
    """Find p in R^3 with sgn(bracket(P_i,P_j,P_k,p)) == signs[(i,j,k)] for each triple.

    Solves:  max t  s.t.  s_ijk * (a_ijk @ p + b_ijk) >= t for all triples, t <= slack_max.
    If max t > eps, returns p; otherwise returns None (cell is empty).
    """
    if not signs:
        return np.zeros(3)

    A_ub = []
    b_ub = []
    for (i, j, k), s in signs.items():
        a, b = bracket_linear_part(
            vertices[i].witness, vertices[j].witness, vertices[k].witness
        )
        # s * (a @ p + b) >= t   <=>   -s*(a @ p) + t <= s*b
        A_ub.append([-s * a[0], -s * a[1], -s * a[2], 1.0])
        b_ub.append(s * b)

    c = [0.0, 0.0, 0.0, -1.0]                     # minimize -t  ==  maximize t
    bounds = [(None, None)] * 3 + [(None, slack_max)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and -res.fun > eps:
        return res.x[:3]
    return None


# === Cell complex with incremental updates ===

class CellComplex:
    def __init__(self, initial_witnesses):
        assert len(initial_witnesses) == 3, "Start with exactly 3 points"
        self.vertices = [Vertex(i, w) for i, w in enumerate(initial_witnesses)]
        plane = (0, 1, 2)
        self.cells = [Cell({plane: +1}), Cell({plane: -1})]
        for cell in self.cells:
            cell.witness = find_witness(cell.signs, self.vertices)
            assert cell.witness is not None, f"Initial cell {cell.signs} is empty?"

    def add_point_in_cell(self, target_cell):
        """Add a new point inside target_cell, then slice every cell with new planes."""
        n = len(self.vertices)
        if target_cell.witness is None:
            target_cell.witness = find_witness(target_cell.signs, self.vertices)
        new_witness = target_cell.witness.copy()
        new_vertex = Vertex(n, new_witness, combinatorial=dict(target_cell.signs))
        self.vertices.append(new_vertex)

        new_planes = [(i, j, n) for i, j in combinations(range(n), 2)]

        new_cells = []
        for c in self.cells:
            current = [c]
            for new_plane in new_planes:
                next_current = []
                for cc in current:
                    next_current.extend(self._slice_cell(cc, new_plane))
                current = next_current
            new_cells.extend(current)
        self.cells = new_cells

    def _slice_cell(self, c, new_plane):
        """Slice cell c with new_plane. Returns list of resulting cells (1 or 2)."""
        a, b = bracket_linear_part(*[self.vertices[idx].witness for idx in new_plane])
        s_at_witness = a @ c.witness + b

        # If the witness happens to lie on the new plane (e.g., it IS the new
        # point), test both sides explicitly.
        if abs(s_at_witness) < 1e-9:
            results = []
            for s in (+1, -1):
                ns = {**c.signs, new_plane: s}
                w = find_witness(ns, self.vertices)
                if w is not None:
                    results.append(Cell(ns, witness=w))
            return results

        s_int = int(np.sign(s_at_witness))
        same_signs = {**c.signs, new_plane: s_int}
        cell_same = Cell(same_signs, witness=c.witness.copy())

        other_signs = {**c.signs, new_plane: -s_int}
        p_other = find_witness(other_signs, self.vertices)
        if p_other is not None:
            return [cell_same, Cell(other_signs, witness=p_other)]
        return [cell_same]


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


# === Demo ===

def main():
    rng = np.random.default_rng(200)

    # Concrete witnesses for the first three points (any non-collinear triple works).
    initial_witnesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    print("=== Initial: 3 points (plane 012), 2 cells ===")
    cc = CellComplex(initial_witnesses)
    for v in cc.vertices:
        print(f"  {v}")
    print(f"  cells: {len(cc.cells)}")

    print("\n=== Adding 4 more points ===")
    for step in range(4):
        idx = int(rng.integers(len(cc.cells)))
        chosen = cc.cells[idx]
        before = len(cc.cells)
        cc.add_point_in_cell(chosen)
        added = cc.vertices[-1]
        print(f"  step {step+1}: added {added} in cell {idx} -> {len(cc.cells)} cells (was {before})")

    print(f"\n=== Final: {len(cc.vertices)} points, {len(cc.cells)} cells ===")

    print("\n=== Verifying cell witnesses against their sign vectors ===")
    verify_cell_witnesses(cc)

    print("\n=== Random affine-combination test (1000 samples, coefficients in [-1,2]) ===")
    random_combination_test(cc, n_samples=1000)


if __name__ == "__main__":
    main()
