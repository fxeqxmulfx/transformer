/-
# Bit-level feature permutations as butterfly weights

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`prop: butterfly-hyena`. A feature-axis factor pairs coordinates differing
in one binary digit. Its common partner permutation is itself a butterfly
matrix, and therefore an explicitly represented width-one K matrix.
-/

import Transformer.Zoology.Appendix_ButterflyIdentity

namespace Transformer.Zoology

/-- Factor exchanging its two halves.
Source: Appendix `eq: butterfly-split`, off-diagonal partner read. -/
def swapButterflyFactor (m : ℕ) : ButterflyFactor m := {
  upperLeft := fun _ => 0
  upperRight := fun _ => 1
  lowerLeft := fun _ => 1
  lowerRight := fun _ => 0
}

/-- Toggle binary digit t of an index of width 2^k. For t ≥ k this is
the identity, so the auxiliary definition needs no partial index.
Source: Appendix `def: butterfly`, the pairs in each block factor. -/
def butterflyToggleIndex : (k t : ℕ) → Fin (butterflyWidth k) →
    Fin (butterflyWidth k)
  | 0, _, i => i
  | k + 1, t, i =>
      let p := finProdFinEquiv.symm i
      if t = k then finProdFinEquiv (if p.1 = 0 then 1 else 0, p.2)
      else finProdFinEquiv (p.1, butterflyToggleIndex k t p.2)

/-- A recursive butterfly implementing exactly the binary partner read.
Source: Appendix `def: butterfly`, recursive representation. -/
def butterflyToggleTree : (k t : ℕ) → ButterflyTree k
  | 0, _ => .leaf
  | k + 1, t =>
      if t = k then .node (swapButterflyFactor (butterflyWidth k))
        (identityButterflyTree k) (identityButterflyTree k)
      else .node (identityButterflyFactor (butterflyWidth k))
        (butterflyToggleTree k t) (butterflyToggleTree k t)

/-- The tree's linear action is the common feature partner permutation.
Source: Appendix `def: butterfly` and `eq: butterfly-split`. -/
theorem butterflyToggleTree_apply (k t : ℕ)
    (u : Fin (butterflyWidth k) → ℝ) :
    (butterflyToggleTree k t).apply u = fun i => u (butterflyToggleIndex k t i) := by
  induction k generalizing t with
  | zero => rfl
  | succ k ih =>
      funext j
      have hj := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply j
      generalize hp : finProdFinEquiv.symm j = p at hj
      rcases p with ⟨half, i⟩
      rw [← hj]
      fin_cases half <;> by_cases ht : t = k <;>
        simp [butterflyToggleTree, butterflyToggleIndex, ht,
          ButterflyTree.apply, swapButterflyFactor, identityButterflyFactor,
          ButterflyFactor.apply, identityButterflyTree_apply, ih]

/-- Pairing a feature twice restores its original coordinate.
Source: Appendix `def: butterfly`, two-half pairing. -/
theorem butterflyToggleIndex_involutive (k t : ℕ) :
    Function.Involutive (butterflyToggleIndex k t) := by
  induction k with
  | zero => intro i; rfl
  | succ k ih =>
      intro j
      have hj := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply j
      generalize hp : finProdFinEquiv.symm j = p at hj
      rcases p with ⟨half, i⟩
      rw [← hj]
      fin_cases half <;> by_cases ht : t = k <;>
        (simp [butterflyToggleIndex, ht, ih i]; rfl)

/-- Store the partner permutation as a width-one K projection.
Source: Appendix `def: W-kmat`, allowed butterfly weight. -/
def butterflyToggleK (k t : ℕ) : ExpandedKaleidoscope k 1 :=
  butterflyTreeK (butterflyToggleTree k t)

/-- K decoding acts by the exact partner permutation.
Source: Appendix `def: W-kmat` and `eq: butterfly-split`. -/
theorem butterflyToggleK_apply (k t : ℕ)
    (u : Fin (butterflyWidth k) → ℝ) :
    (butterflyToggleK k t).apply u = fun i => u (butterflyToggleIndex k t i) := by
  rw [butterflyToggleK, butterflyTreeK_apply, butterflyToggleTree_apply]

/-- The shared feature projection reads the partner at each token.
Source: Appendix `prop: butterfly-hyena`, feature-axis off-diagonal term. -/
theorem butterflyToggleK_linearProjection {n : ℕ} (k t : ℕ)
    (u : RealSequence n (butterflyWidth k)) (i : Fin n)
    (q : Fin (butterflyWidth k)) :
    linearProjection u (kaleidoscopeWeight (butterflyToggleK k t)) i q =
      u i (butterflyToggleIndex k t q) := by
  have h := ExpandedKaleidoscope.apply_eq_matrix (butterflyToggleK k t) (u i) q
  simpa [linearProjection, kaleidoscopeWeight, mul_comm,
    butterflyToggleK_apply] using h.symm

/-- The partner projection always uses one BB* factor.
Source: Appendix `def: kaleidoscope`, width-one representation. -/
theorem butterflyToggleK_width (k t : ℕ) :
    (butterflyToggleK k t).inner.width = 1 := rfl

end Transformer.Zoology
