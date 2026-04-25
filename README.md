# csaszar2

A coordinate-free, chirotope-based exploration of cell complexes formed by the
arrangement of all planes through triples of named points in $\mathbb{R}^3$,
with the long-term goal of rediscovering the Császár polyhedron by
enumeration.

## What's here

- **`csaszar.pdf`** / **`csaszar.tex`** — the worked-out math:
  - Notation: $4$-bracket $[a,b,c,d]$, signed volume of tetrahedron $abcd$.
  - The condition `3 ∈ 012⁺` written in three equivalent forms (determinant,
    scalar triple product, fully unparenthesized polynomial) — the same
    treatment given to `012⁺ ∩ 013⁺ ≠ ∅`.
  - Why `012⁺ ∩ 013⁺` is non-empty (an explicit witness $p^* = 2P_3 - P_2$)
    and why `012⁻ ∩ 013⁺ ∩ 023⁻ ∩ 123⁺` is empty (the missing 16th cell of
    tetrahedron $0123$), both proved from the affine identity
    $$[0,1,2,p] - [0,1,3,p] + [0,2,3,p] - [1,2,3,p] = [0,1,2,3].$$
  - The chirotope-only **slice algorithm**: deciding whether a new plane
    $ij(n+1)$ slices an existing cell of the prior arrangement, expressed as
    a small CSP with $\binom{n+1}{2}$ Boolean variables and $\binom{n+1}{3}$
    forbidden-pattern clauses (the "missing $16$th cell" pattern applied to
    every $4$-tuple of named indices).
  - A Farkas-style proof that the forbidden-pattern rule is *sufficient*, not
    only necessary: every infeasibility certificate of the slice LP
    corresponds to a violated GP relation, because the only signed
    $4$-term linear dependency among plane-normals is the one inherited from
    the affine identity.

- **`cell_complex.py`** — a Python demo of the incremental construction.

## The Python demo

`Vertex` carries
- a *combinatorial* part (dict mapping triples $(i,j,k)$ with $i<j<k<$ self-index
  to $\pm 1$, recording the sign of $[i,j,k,$ self-index$]$ — i.e. which cell of
  the prior arrangement this vertex was added in);
- a *realized* 3D witness (used pedagogically and as input to the LP that
  decides cell-plane slices).

`Cell` carries a sign vector over the current planes plus a witness in the
cell's interior. The slice test is implemented as an LP feasibility check via
`scipy.optimize.linprog` — by Farkas duality this is the chirotope-based
forbidden-pattern test, just expressed in its dual (LP) form.

Run it:

```bash
pip install numpy scipy
python3 cell_complex.py
```

Sample output:

```
=== Initial: 3 points (plane 012), 2 cells ===
  V0(+0.000,+0.000,+0.000)
  V1(+1.000,+0.000,+0.000)
  V2(+0.000,+1.000,+0.000)
  cells: 2

=== Adding 4 more points ===
  step 1: added V3 in cell 0 ->   15 cells (was   2)
  step 2: added V4 in cell ?? ->   90 cells (was  15)
  step 3: added V5 in cell ?? ->  559 cells (was  90)
  step 4: added V6 in cell ?? -> 3112 cells (was 559)

=== Verifying cell witnesses against their sign vectors ===
  3112/3112 cells: every bracket sign matches the witness.

=== Random affine-combination test (1000 samples, coefficients in [-1,2]) ===
  matched: 1000/1000    unmatched: 0    skipped: 0 (degenerate)
```

The exact cell counts depend on the random cell choices at each step — the
$3 \to 4$ transition always gives $15$ cells (the four-plane tetrahedron
arrangement) but the later counts vary because different cells produce
different chirotopes.

The two verifications:

1. **Witness consistency.** Every cell's stored witness is checked against
   every sign in the cell's sign vector. All 3112 cells match.
2. **Coverage.** 1000 random affine combinations of the 7 named points are
   computed, their bracket-sign vectors are read off, and each is matched
   against the enumerated cells. All 1000 land in some cell, which gives
   numerical confirmation that the cells tile $\mathbb{R}^3$ minus a
   measure-zero set of plane points.

## Status / next

The cell-complex construction is solid. Next milestones:

- For each cell of the 7-point arrangement (or earlier, if structurally
  sufficient), check whether the underlying chirotope produces the Császár
  combinatorics — 14 triangular faces forming a torus, every pair of
  vertices joined by an edge, no diagonals.
- This requires reading off the chirotope (the $\binom{7}{4}=35$ bracket
  signs) from each cell and testing the Császár face-set conditions on it.
