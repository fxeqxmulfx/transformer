/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Transformer.Sturm.Variations

/-!
# Generalized Sturm chain conditions

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

/-- Sign variations of the chain at `+∞`: the sign of each element there is the
sign of its leading coefficient, so this is the zero-skipping variation count
of the leading coefficients. The zero polynomial contributes leading
coefficient `0`, which the zero-skipping convention drops.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def sturmVarPosInf (chain : List (Polynomial ℝ)) : ℕ :=
  signVariations (chain.map Polynomial.leadingCoeff)

/-- Sign variations of the chain at `−∞`: the sign of an element there is the
sign of its leading coefficient times `(-1) ^ degree`, so this is the
zero-skipping variation count of `leadingCoeff · (-1) ^ natDegree`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def sturmVarNegInf (chain : List (Polynomial ℝ)) : ℕ :=
  signVariations (chain.map (fun q => q.leadingCoeff * (-1) ^ q.natDegree))

/-- A generalized Sturm chain for a real polynomial.

At a root of the first polynomial, the product of the first two entries changes
from negative to positive. At a root of an interior entry, its neighbors have
opposite signs. The last entry has no real roots, and every entry is nonzero.

These conditions allow the one-element chain of a nonzero constant polynomial.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
structure IsSturmChain (p : Polynomial ℝ) (chain : List (Polynomial ℝ)) : Prop where
  /-- The head of the chain is `p`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
  head : chain.head? = some p
  /-- At every real root `r` of `p`, the chain has a second element `q`,
  nonzero at `r`, with `p * q` negative just left of `r` and positive just
  right of `r`.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
  root_flank : ∀ r : ℝ, p.IsRoot r → ∃ q : Polynomial ℝ, chain[1]? = some q ∧
    q.eval r ≠ 0 ∧
    (∀ᶠ x in 𝓝[<] r, (p * q).eval x < 0) ∧
    (∀ᶠ x in 𝓝[>] r, 0 < (p * q).eval x)
  /-- No chain element is the zero polynomial.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
  nonzero_mem : ∀ q ∈ chain, q ≠ 0
  /-- Whenever the interior element `b = chain[i+1]` vanishes at `x`, its two
  neighbours `a = chain[i]` and `c = chain[i+2]` are nonzero there and have
  opposite signs.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
  interior_alternates : ∀ (i : ℕ) (x : ℝ) (a b c : Polynomial ℝ),
    chain[i]? = some a → chain[i + 1]? = some b → chain[i + 2]? = some c →
    b.eval x = 0 → a.eval x ≠ 0 ∧ c.eval x ≠ 0 ∧ a.eval x * c.eval x < 0
  /-- The last element of the chain has no real zero.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
  last_no_root : ∀ q : Polynomial ℝ, chain.getLast? = some q → ∀ x : ℝ, q.eval x ≠ 0

namespace IsSturmChain

variable {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}

/-- A Sturm chain is nonempty.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem nonempty (h : IsSturmChain p chain) : chain ≠ [] := by
  rintro rfl
  simpa using h.head

/-- The polynomial counted by a Sturm chain is its first entry.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem head_mem (h : IsSturmChain p chain) : p ∈ chain :=
  List.mem_of_head? h.head

/-- A polynomial admitting a Sturm chain is nonzero.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem ne_zero (h : IsSturmChain p chain) : p ≠ 0 :=
  h.nonzero_mem p h.head_mem

end IsSturmChain

end Transformer.Sturm
