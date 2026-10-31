module

public import Transformer.SequenceModel.Attention.Finite.Hinges
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# A finite invertible genuine-gate evaluation matrix

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F2.
Distinct projective coordinates give an invertible matrix at some positive
finite gain. The determinant limit supplies existence, not an infinite-width
model or an activation substituted into the actual forward.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open Filter
open scoped Topology BigOperators

variable {A : Type*} [Fintype A] [DecidableEq A]

/-- F2: finitely many distinct real coordinates have a common positive
separation. Singletons are allowed, using one as the empty-pair scale. -/
theorem exists_separation [Nonempty A] (α : A → ℝ) (hα : Function.Injective α) :
    ∃ δ > 0, ∀ a b, a ≠ b → δ ≤ |α a - α b| := by
  let gaps : A × A → ℝ := fun ab => if ab.1 = ab.2 then 1 else |α ab.1 - α ab.2|
  let δ := Finset.univ.inf' (Finset.univ_nonempty : (Finset.univ : Finset (A × A)).Nonempty) gaps
  have hδ : 0 < δ := by
    apply (Finset.lt_inf'_iff _).mpr
    intro ab _
    dsimp only [gaps]
    split_ifs with he
    · norm_num
    · exact abs_pos.mpr (sub_ne_zero.mpr (fun h => he (hα h)))
  refine ⟨δ, hδ, ?_⟩
  intro a b hab
  have h := Finset.inf'_le gaps (Finset.mem_univ (a, b))
  simpa only [gaps, hab, ↓reduceIte] using h

/-- Evaluation of three SiLU features at each separated center, including
the strictly positive value-channel and projective normalization factors. -/
def evaluation (α u : A → ℝ) (δ gain : ℝ) : Matrix A A ℝ :=
  fun a b => (u a) ^ 2 * softTriangle (gain * u a) δ (α a - α b)

omit [Fintype A] in
/-- F2: the evaluation matrices converge to an invertible positive diagonal. -/
theorem tendsto_evaluation (α u : A → ℝ) (hu : ∀ a, 0 < u a) {δ : ℝ}
    (hδ : 0 < δ) (hsep : ∀ a b, a ≠ b → δ ≤ |α a - α b|) :
    Tendsto (evaluation α u δ) atTop (𝓝 (Matrix.diagonal (fun a => (u a) ^ 2 * δ))) := by
  apply tendsto_pi_nhds.mpr
  intro a
  apply tendsto_pi_nhds.mpr
  intro b
  have hg : Tendsto (fun gain : ℝ => gain * u a) atTop atTop := by
    simpa only [id_eq, mul_comm] using (tendsto_const_mul_atTop_of_pos (hu a)).mpr tendsto_id
  have ht := ((tendsto_softTriangle δ (α a - α b)).comp hg).const_mul ((u a) ^ 2)
  by_cases hab : a = b
  · subst b
    simpa only [evaluation, Function.comp_def, sub_self, triangle_zero hδ,
      Matrix.diagonal_apply_eq] using ht
  · simpa only [evaluation, Function.comp_def, triangle_outside hδ (hsep a b hab),
      mul_zero, Matrix.diagonal_apply_ne _ hab] using ht

/-- F2: there is one positive, finite gain whose actual smooth features
have full row rank. The same gain works simultaneously for every row. -/
theorem exists_invertible_evaluation [Nonempty A] (α u : A → ℝ)
    (hα : Function.Injective α) (hu : ∀ a, 0 < u a) :
    ∃ δ > 0, ∃ gain > 0, (evaluation α u δ gain).det ≠ 0 := by
  obtain ⟨δ, hδ, hsep⟩ := exists_separation α hα
  let D := Matrix.diagonal (fun a => (u a) ^ 2 * δ)
  have hD : D.det ≠ 0 := by
    rw [Matrix.det_diagonal]
    exact Finset.prod_ne_zero_iff.mpr (fun a _ => mul_ne_zero (pow_ne_zero 2 (hu a).ne') hδ.ne')
  have hc : Continuous (fun M : Matrix A A ℝ => M.det) := continuous_id.matrix_det
  have ht := (hc.tendsto D).comp (tendsto_evaluation α u hu hδ hsep)
  obtain ⟨gain, hg, hdet⟩ := ((eventually_gt_atTop (0 : ℝ)).and
    (ht.eventually (eventually_ne_nhds hD))).exists
  exact ⟨δ, hδ, gain, hg, hdet⟩

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- Fixed vector-valued interpolation coefficients of an invertible matrix. -/
def interpolationCoefficients (M : Matrix A A ℝ) (target : A → E) (b : A) : E :=
  ∑ a, (M⁻¹ b a) • target a

/-- F2: the coefficients interpolate all vector targets simultaneously,
without changing the input-dependent evaluation matrix. -/
theorem interpolate_matrix (M : Matrix A A ℝ) (hM : M.det ≠ 0)
    (target : A → E) (a : A) :
    ∑ b, M a b • interpolationCoefficients M target b = target a := by
  have hi (b : A) : (∑ c, M a c * M⁻¹ c b) = if a = b then 1 else 0 := by
    have h := congrArg (fun B : Matrix A A ℝ => B a b)
      (Matrix.mul_nonsing_inv (A := M) (isUnit_iff_ne_zero.mpr hM))
    simpa only [Matrix.mul_apply, Matrix.one_apply] using h
  simp only [interpolationCoefficients, Finset.smul_sum, smul_smul]
  rw [Finset.sum_comm]
  simp_rw [← Finset.sum_smul, hi]
  simp [ite_smul]

private def coords (a : Fin 3) : ℝ := a.val
private def values (a : Fin 3) : ℝ := a.val + 1
private theorem coords_injective : Function.Injective coords := by
  intro a b h
  change (a.val : ℝ) = b.val at h
  apply Fin.ext
  exact_mod_cast h

example := exists_separation coords coords_injective
example := tendsto_evaluation coords values (by intro a; dsimp [values]; positivity) (δ := 1) (by norm_num)
  (by intro a b h; fin_cases a <;> fin_cases b <;> norm_num [coords] at *)
example := exists_invertible_evaluation coords values coords_injective (by intro a; dsimp [values]; positivity)
example := interpolate_matrix (!![(2 : ℝ), 0; 0, 3])
  (by norm_num [Matrix.det_fin_two]) (fun a : Fin 2 => (a.val : ℝ) + 1) 1

end Transformer.SequenceModel.Attention.Finite
