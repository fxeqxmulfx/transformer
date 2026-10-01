/-
# Genuine second derivative bounds from compact parameter ranges

arXiv:2506.12543v1, Section 4.3, Theorem 1's corrected regularity conditions.
The signed drift is smooth on positive noise parameters. Compactness bounds
its outer derivatives before the actual state-dependent parameters are composed.
-/

import Transformer.BatchSize.Section4_SmoothCoefficients
import Mathlib.Analysis.Calculus.ContDiff.Bounds
import Mathlib.Analysis.Normed.Group.Bounded

open scoped BigOperators NNReal

noncomputable section

namespace Transformer.BatchSize

/-- An outer C2 function on an open parameter domain has a genuine
global second derivative bound after composition with a bounded smooth
parameter map whose range is compact inside that domain,
Section 4.3, Theorem 1. Only true derivatives of the specified maps occur. -/
theorem compact_range_second_derivative_bound {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (g : F → G) (U K : Set F) (hU : IsOpen U) (hK : IsCompact K) (hKU : K ⊆ U)
    (hg : ContDiffOn ℝ 2 g U) (f : E → F) (hf : ContDiff ℝ 2 f) (hrange : Set.range f ⊆ K)
    (D : NNReal) (hD : ∀ i, 1 ≤ i → i ≤ 2 → ∀ x, ‖iteratedFDeriv ℝ i f x‖ ≤ (D : ℝ) ^ i) :
    ∃ H : NNReal, ∀ x, ‖iteratedFDeriv ℝ 2 (g ∘ f) x‖ ≤ H := by
  have hbound (j : Fin 3) : ∃ C : ℝ, ∀ y ∈ K,
      ‖iteratedFDerivWithin ℝ j.val g U y‖ ≤ C :=
    hK.exists_bound_of_continuousOn
      ((hg.continuousOn_iteratedFDerivWithin
        (by exact_mod_cast (show j.val ≤ 2 from by omega)) hU.uniqueDiffOn).mono hKU)
  choose C hC using hbound
  let R : ℝ := (∑ j : Fin 3, |C j|) + 1
  have hR : 0 ≤ R := by dsimp only [R]; positivity
  refine ⟨⟨2 * R * (D : ℝ) ^ 2, by positivity⟩, fun x => ?_⟩
  have hb (i : ℕ) (hi : i ≤ 2) : ‖iteratedFDerivWithin ℝ i g U (f x)‖ ≤ R := by
    let j : Fin 3 := ⟨i, by omega⟩
    calc
      _ ≤ C j := hC j (f x) (hrange ⟨x, rfl⟩)
      _ ≤ |C j| := le_abs_self _
      _ ≤ ∑ k : Fin 3, |C k| := Finset.single_le_sum (fun k _ => abs_nonneg (C k)) (Finset.mem_univ j)
      _ ≤ R := by dsimp only [R]; linarith
  have h := norm_iteratedFDeriv_comp_le' (hrange.trans hKU) hU.uniqueDiffOn hg hf
    (by norm_num : (2 : WithTop ℕ∞) ≤ 2) x hb (fun i hi hj => hD i hi hj x)
  norm_num at h
  exact h

/-- The two real parameter derivatives can be bounded together
without changing the true product norm, Section 4.3, Theorem 1. -/
theorem iteratedFDeriv_prod_norm_le {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E → F) (g : E → G) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (i : ℕ) (hi : i ≤ 2) (x : E) (L : NNReal)
    (hfL : ‖iteratedFDeriv ℝ i f x‖ ≤ L) (hgL : ‖iteratedFDeriv ℝ i g x‖ ≤ L) :
    ‖iteratedFDeriv ℝ i (fun x => (f x, g x)) x‖ ≤ L := by
  rw [iteratedFDeriv_prodMk hf.contDiffAt hg.contDiffAt (by exact_mod_cast hi)]
  apply ContinuousMultilinearMap.opNorm_le_bound L.coe_nonneg
  intro v
  rw [ContinuousMultilinearMap.prod_apply, Prod.norm_def]
  apply max_le
  · exact ((iteratedFDeriv ℝ i f x).le_opNorm v).trans
      (mul_le_mul_of_nonneg_right hfL (Finset.prod_nonneg fun _ _ => norm_nonneg _))
  · exact ((iteratedFDeriv ℝ i g x).le_opNorm v).trans
      (mul_le_mul_of_nonneg_right hgL (Finset.prod_nonneg fun _ _ => norm_nonneg _))

/-- Joint nonvacuity of compact composition hypotheses,
Section 4.3: a smooth nonconstant outer sine function and a nonzero
constant parameter in a compact interval inside the full open domain. -/
example : IsOpen (Set.univ : Set ℝ) ∧ IsCompact (Set.Icc (-1 : ℝ) 1) ∧
    Set.Icc (-1 : ℝ) 1 ⊆ Set.univ ∧ ContDiffOn ℝ 2 Real.sin Set.univ ∧
    ContDiff ℝ 2 (fun _ : ℝ => (1 / 2 : ℝ)) ∧
    Set.range (fun _ : ℝ => (1 / 2 : ℝ)) ⊆ Set.Icc (-1) 1 ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ 2 → ∀ x : ℝ,
      ‖iteratedFDeriv ℝ i (fun _ : ℝ => (1 / 2 : ℝ)) x‖ ≤ (1 : ℝ) ^ i) := by
  refine ⟨isOpen_univ, isCompact_Icc, Set.subset_univ _, Real.contDiff_sin.contDiffOn,
    contDiff_const, ?_, ?_⟩
  · rintro _ ⟨x, rfl⟩
    norm_num
  · intro i hi hj x
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp

/-- Joint nonvacuity of the product derivative hypotheses,
Section 4.3: two actual linear scalar parameter maps. -/
example : ContDiff ℝ 2 (id : ℝ → ℝ) ∧ ContDiff ℝ 2 (id : ℝ → ℝ) ∧
    (1 : ℕ) ≤ 2 ∧ ‖iteratedFDeriv ℝ 1 (id : ℝ → ℝ) 0‖ ≤ (1 : NNReal) := by
  refine ⟨contDiff_id, contDiff_id, by omega, ?_⟩
  rw [norm_iteratedFDeriv_one]
  simp

end Transformer.BatchSize
