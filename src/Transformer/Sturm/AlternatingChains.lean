/-
# Alternating chains for signed real-root queries

These conditions omit the positive head crossing needed for unsigned
Sturm counts. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.EuclideanSigns

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- A nonzero polynomial chain with opposite neighbors at every
interior zero and a terminal entry without real roots. The head may
cross in either direction in a signed query. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
structure IsRootAlternatingChain (p : Polynomial ℝ) (chain : List (Polynomial ℝ)) : Prop where
  head : chain.head? = some p
  nonzero_mem : ∀ f ∈ chain, f ≠ 0
  interior_alternates : ∀ (i : ℕ) (x : ℝ) (a b c : Polynomial ℝ),
    chain[i]? = some a → chain[i + 1]? = some b → chain[i + 2]? = some c →
    b.eval x = 0 → a.eval x ≠ 0 ∧ c.eval x ≠ 0 ∧ a.eval x * c.eval x < 0
  last_no_root : ∀ f : Polynomial ℝ, chain.getLast? = some f → ∀ x : ℝ, f.eval x ≠ 0

/-- A Sturm chain has the alternating-chain properties.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem IsSturmChain.toRootAlternatingChain {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}
    (h : IsSturmChain p chain) : IsRootAlternatingChain p chain :=
  ⟨h.head, h.nonzero_mem, h.interior_alternates, h.last_no_root⟩

/-- Signed Euclidean division constructs an alternating chain for any
nonzero first polynomial and pair without common real zeros.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signedRemainderChain_isRootAlternating {p q : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → q.eval x ≠ 0) :
    IsRootAlternatingChain p (signedRemainderChain p q) :=
  ⟨signedRemainderChain_head p q, signedRemainderChain_nonzero hp,
    signedRemainderChain_interior h, signedRemainderChain_last_no_root h⟩

/-- The head belongs to an alternating chain. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem IsRootAlternatingChain.head_mem {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}
    (h : IsRootAlternatingChain p chain) : p ∈ chain := List.mem_of_head? h.head

/-- The head of an alternating chain is nonzero. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem IsRootAlternatingChain.ne_zero {p : Polynomial ℝ} {chain : List (Polynomial ℝ)}
    (h : IsRootAlternatingChain p chain) : p ≠ 0 := h.nonzero_mem p h.head_mem

/-- The pair `X, 1` witnesses the nonzero and absence-of-common-zero
hypotheses used to construct an alternating chain. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X : Polynomial ℝ) ≠ 0 ∧
    ∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (1 : Polynomial ℝ).eval x ≠ 0 := by
  exact ⟨X_ne_zero, fun _ _ => by simp⟩

end Transformer.Sturm
