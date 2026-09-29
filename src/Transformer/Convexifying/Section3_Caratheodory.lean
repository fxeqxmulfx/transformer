/-
# Sparse convex combinations of token response vectors

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, Appendix A.4.  The paper
uses Carathéodory's theorem to bound the number of active token directions
by the sample count plus one.  This file proves the finite-dimensional
weight-reduction step used by the sparse-optimum theorem.
-/

import Transformer.Convexifying.Section3_Factorization

open scoped BigOperators

namespace Transformer.Convexifying

/-- Pull an affine-independent convex combination back to token weights;
the affine-independent cardinal bound gives at most `N + 1` active tokens.
Source: arXiv:2211.11052v1, Appendix A.4. -/
private theorem sparse_weights_from_affine {N n : ℕ} {ι : Type} [Fintype ι]
    (a : Fin n → Vec N) (w : Fin n → ℝ) (z : ι → Vec N) (ω : ι → ℝ)
    (hz : Set.range z ⊆ Set.range a) (hAI : AffineIndependent ℝ z)
    (hωpos : ∀ i, 0 < ω i) (hωsum : ∑ i, ω i = 1)
    (hcombo : (∑ i, ω i • z i) = ∑ k, w k • a k) :
    ∃ w' : Fin n → ℝ,
      (∀ k, 0 ≤ w' k) ∧ (∑ k, w' k = 1) ∧
      (∀ r, ∑ k, w' k * a k r = ∑ k, w k * a k r) ∧
      ((Finset.univ.filter fun k => w' k ≠ 0).card ≤ N + 1) := by
  classical
  let token (i : ι) : Fin n := Classical.choose (hz ⟨i, rfl⟩)
  have htoken (i : ι) : a (token i) = z i :=
    Classical.choose_spec (hz ⟨i, rfl⟩)
  let w' : Fin n → ℝ := fun k => ∑ i, if token i = k then ω i else 0
  have hw' (k : Fin n) : 0 ≤ w' k := by
    apply Finset.sum_nonneg
    intro i _
    split_ifs with h
    · exact (hωpos i).le
    · norm_num
  have hsum' : ∑ k, w' k = 1 := by
    change (∑ k, ∑ i, if token i = k then ω i else 0) = 1
    rw [Finset.sum_comm]
    simpa using hωsum
  have hpred (r : Fin N) : ∑ k, w' k * a k r = ∑ k, w k * a k r := by
    change (∑ k, (∑ i, if token i = k then ω i else 0) * a k r) = _
    simp_rw [Finset.sum_mul]
    rw [Finset.sum_comm]
    have hleft : (∑ i, ∑ k, (if token i = k then ω i else 0) * a k r) =
        ∑ i, ω i * z i r := by
      apply Finset.sum_congr rfl
      intro i _
      simp [← htoken i]
    rw [hleft]
    have hright := congrArg (fun v : Vec N => v r) hcombo
    simpa [Finset.sum_apply, Pi.smul_apply, smul_eq_mul] using hright
  have hsupport : (Finset.univ.filter fun k => w' k ≠ 0) ⊆
      Finset.univ.image token := by
    intro k hk
    by_contra hnot
    have hforall : ∀ i, token i ≠ k := by
      intro i hi
      apply hnot
      exact Finset.mem_image.mpr ⟨i, Finset.mem_univ _, hi⟩
    have hzero : w' k = 0 := by
      simp [w', hforall]
    exact (Finset.mem_filter.mp hk).2 hzero
  have hcard : Fintype.card ι ≤ N + 1 := by
    have h₁ := AffineIndependent.card_le_finrank_succ hAI
    have h₂ := Submodule.finrank_le (vectorSpan ℝ (Set.range z))
    have h₃ : Module.finrank ℝ (Vec N) = N := by
      simp [Vec]
    calc
      Fintype.card ι ≤ Module.finrank ℝ ↥(vectorSpan ℝ (Set.range z)) + 1 := h₁
      _ ≤ N + 1 := Nat.add_le_add_right (by simpa [Vec, h₃] using h₂) 1
  refine ⟨w', hw', hsum', hpred, ?_⟩
  calc
    (Finset.univ.filter fun k => w' k ≠ 0).card ≤ (Finset.univ.image token).card :=
      Finset.card_le_card hsupport
    _ ≤ Fintype.card ι := by
      simpa using (Finset.card_image_le (s := Finset.univ) (f := token))
    _ ≤ N + 1 := hcard

/-- Carathéodory reduction for a convex combination of `n` response vectors
in `ℝ^N`: the same vector has nonnegative weights on at most `N + 1`
tokens.  The bound concerns the actual support of the new weights.
Source: arXiv:2211.11052v1, Appendix A.4, paragraph after
`eq:dual_cons_multihead_bidual`. -/
theorem sparse_convex_weights {N n : ℕ} (a : Fin n → Vec N)
    (w : Fin n → ℝ) (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) :
    ∃ w' : Fin n → ℝ,
      (∀ k, 0 ≤ w' k) ∧ (∑ k, w' k = 1) ∧
      (∀ r, ∑ k, w' k * a k r = ∑ k, w k * a k r) ∧
      ((Finset.univ.filter fun k => w' k ≠ 0).card ≤ N + 1) := by
  classical
  let x : Vec N := ∑ k, w k • a k
  have hx : x ∈ convexHull ℝ (Set.range a) :=
    mem_convexHull_of_exists_fintype w a hw hsum
      (fun k => ⟨k, rfl⟩) rfl
  obtain ⟨ι, hι, z, ω, hz, hAI, hωpos, hωsum, hcombo⟩ :=
    eq_pos_convex_span_of_mem_convexHull hx
  exact @sparse_weights_from_affine N n ι hι a w z ω hz hAI hωpos hωsum hcombo

/-- The nonnegative unit-sum weight hypotheses hold for one token. -/
example : (∀ _k : Fin 1, 0 ≤ (1 : ℝ)) ∧
    (∑ _k : Fin 1, (1 : ℝ)) = 1 := by
  constructor
  · intro _
    norm_num
  · simp

end Transformer.Convexifying
