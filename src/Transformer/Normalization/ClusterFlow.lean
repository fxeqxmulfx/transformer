/-
# Positive normalized attention and sphere residence

The sign and sphere estimates used with Table 2 and Appendix C, proof of
Theorem 4.3 of arXiv:2510.22026v2. Squared norm error avoids regularity
assumptions on the speed coefficients.
-/

import Transformer.Normalization.ClusterWeights
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- In a cluster with positive pairwise inner products, the attention
vector cannot vanish. Source: arXiv:2510.22026v2, Appendix C, the lower bound
on the attention magnitude used in the speed estimates. -/
theorem attentionVec_id_ne_zero_of_positive_cluster (β m : ℝ) (hm : 0 < m)
    (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d)
    (hΘ : ∀ j k, m ≤ inner (𝕜 := ℝ) (Θ j) (Θ k)) (j : Idx n) :
    attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j ≠ 0 := by
  intro hzero
  have h := inner_attentionVec_id_ge β m Q K Θ hΘ j
  rw [hzero, inner_zero_right] at h
  exact (not_le_of_gt hm) h

/-- A unit singleton is a positive cluster. -/
example : (0 : ℝ) < 1 ∧ ∀ j k : Idx 1, (1 : ℝ) ≤
    inner (𝕜 := ℝ) ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) j)
      ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) :=
  ⟨one_pos, fun _ _ => by simp⟩

/-- Every scheme has a positive speed factor in a positive cluster with
positive radii and positive nGPT scale. Source: arXiv:2510.22026v2, Table 2
and Appendix C, proof of Theorem 4.3. -/
theorem speedFactor_pos_of_positive_cluster (β m τ t : ℝ) (hm : 0 < m) (ht : 0 ≤ t)
    (Q K : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ) (scheme : Scheme)
    (hα : 0 < α t) (hr : ∀ j, 0 < r t j)
    (hθ : ∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) (j : Idx n) :
    0 < speedFactor d n β Q K (idParams d) α θ r τ scheme t j := by
  have hA : 0 < ‖attentionVec d n β (Q t) (K t) (idParams d t) (θ t) j‖ :=
    norm_pos_iff.mpr (attentionVec_id_ne_zero_of_positive_cluster β m hm (Q t) (K t)
      (θ t) hθ j)
  cases scheme <;> simp only [speedFactor]
  · exact one_pos
  · exact hr j
  · split_ifs
    · exact one_pos
    · exact hr j
  · exact mul_pos (hr j) hA
  · exact mul_pos (inv_pos.mpr hα) hA
  · exact Real.sqrt_pos.mpr (by linarith)

/-- Constant unit directions and radii, at `t = 0`, meet all sign conditions. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
    (∀ j : Idx 1, 0 < (fun _ : ℝ => fun _ : Idx 1 => (1 : ℝ)) 0 j) ∧
    ∀ j k : Idx 1, (1 : ℝ) ≤
      inner (𝕜 := ℝ) ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) j)
        ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) :=
  ⟨one_pos, le_rfl, one_pos, fun _ => one_pos, fun _ _ => by simp⟩

/-- The radius never decreases while all pairwise inner products are
positive. Source: arXiv:2510.22026v2, Table 2 and Appendix C, radial growth
in the proof of Theorem 4.3. -/
theorem radialDerivative_nonneg_of_positive_cluster (β m τ t : ℝ) (hm : 0 ≤ m)
    (Q K : ℝ → ParamMatrix d) (θ : ℝ → Idx n → EucSpace d) (scheme : Scheme)
    (hθ : ∀ j k, m ≤ inner (𝕜 := ℝ) (θ t j) (θ t k)) (j : Idx n) :
    0 ≤ radialDerivative d n β Q K (idParams d) θ τ scheme t j := by
  have h := hm.trans (inner_attentionVec_id_ge β m (Q t) (K t) (θ t) hθ j)
  cases scheme <;> simp only [radialDerivative]
  · exact le_rfl
  · exact h
  · split_ifs
    · exact h
    · exact le_rfl
  · exact div_nonneg h (norm_nonneg _)
  · exact le_rfl
  · exact le_rfl

/-- Unit directions meet the nonnegative-cluster hypothesis. -/
example : (0 : ℝ) ≤ 1 ∧ ∀ j k : Idx 1, (1 : ℝ) ≤
    inner (𝕜 := ℝ) ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) j)
      ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) :=
  ⟨zero_le_one, fun _ _ => by simp⟩

/-- The squared norm error satisfies a scalar decay equation. Source:
arXiv:2510.22026v2, equation (NA), the sphere constraint used in Appendix C. -/
theorem hasDerivWithinAt_norm_error_sq (x : ℝ → EucSpace d) (a : ℝ)
    (y : EucSpace d) (s : Set ℝ) (t : ℝ)
    (hx : HasDerivWithinAt x (a • proj d (x t) y) s t) :
    HasDerivWithinAt (fun u => (‖x u‖ ^ 2 - 1) ^ 2)
      (-4 * a * inner (𝕜 := ℝ) (x t) y * (‖x t‖ ^ 2 - 1) ^ 2) s t := by
  convert (hx.norm_sq.sub_const 1).pow 2 using 1
  simp only [real_inner_smul_right, proj, inner_sub_right, real_inner_self_eq_norm_sq]
  ring

/-- A zero vector field realizes the derivative hypothesis. -/
example : HasDerivWithinAt (fun _ : ℝ => (0 : EucSpace 1))
    ((1 : ℝ) • proj 1 (0 : EucSpace 1) 0) (Ici 0) 0 := by
  simpa [proj] using (hasDerivAt_const 0 (0 : EucSpace 1)).hasDerivWithinAt

/-- Nonnegative speed and radial component preserve the initial unit norm
on a finite forward interval. Source: arXiv:2510.22026v2, equation (NA) and
Appendix C, proof of Theorem 4.3. -/
theorem norm_eq_one_of_nonnegative_radial_flow (T : ℝ) (hT : 0 ≤ T)
    (x : ℝ → EucSpace d) (a : ℝ → ℝ) (y : ℝ → EucSpace d)
    (hx : ∀ t ∈ Icc 0 T, HasDerivWithinAt x (a t • proj d (x t) (y t)) (Ici 0) t)
    (ha : ∀ t ∈ Icc 0 T, 0 ≤ a t)
    (hy : ∀ t ∈ Icc 0 T, 0 ≤ inner (𝕜 := ℝ) (x t) (y t))
    (h0 : ‖x 0‖ = 1) : ‖x T‖ = 1 := by
  let f : ℝ → ℝ := fun t => (‖x t‖ ^ 2 - 1) ^ 2
  let v : ℝ → ℝ := fun t =>
    -4 * a t * inner (𝕜 := ℝ) (x t) (y t) * (‖x t‖ ^ 2 - 1) ^ 2
  have hd : ∀ t ∈ Icc 0 T, HasDerivWithinAt f (v t) (Ici 0) t := fun t ht =>
    hasDerivWithinAt_norm_error_sq x (a t) (y t) (Ici 0) t (hx t ht)
  have hc : ContinuousOn f (Icc 0 T) := fun t ht =>
    ((hd t ht).continuousWithinAt).mono (Icc_subset_Ici_self)
  have hdAt : ∀ t ∈ Ioo 0 T, HasDerivAt f (v t) t := fun t ht =>
    (hd t (Ioo_subset_Icc_self ht)).hasDerivAt (Ici_mem_nhds ht.1)
  have hv : ∀ t ∈ Icc 0 T, v t ≤ 0 := by
    intro t ht
    exact mul_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonpos_of_nonneg (by nlinarith [ha t ht]) (hy t ht)) (sq_nonneg _)
  have hmono : AntitoneOn f (Icc 0 T) := antitoneOn_of_deriv_nonpos (convex_Icc 0 T) hc
    (by
      rw [interior_Icc]
      intro t ht
      exact (hdAt t ht).differentiableAt.differentiableWithinAt)
    (by
      simpa only [interior_Icc] using
        (fun t ht => (hdAt t ht).deriv ▸ hv t (Ioo_subset_Icc_self ht)))
  have hle := hmono (left_mem_Icc.mpr hT) (right_mem_Icc.mpr hT) hT
  have hzero : f 0 = 0 := by simp [f, h0]
  rw [hzero] at hle
  have hsq : (‖x T‖ ^ 2 - 1) ^ 2 = 0 := le_antisymm hle (sq_nonneg _)
  have he := sq_eq_zero_iff.mp hsq
  nlinarith [norm_nonneg (x T)]

/-- A constant unit path with zero forcing satisfies every hypothesis. -/
example :
    (0 : ℝ) ≤ 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      HasDerivWithinAt (fun _ : ℝ => EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
        ((1 : ℝ) • proj 1 (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 0) (Ici 0) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, 0 ≤ (fun _ : ℝ => (1 : ℝ)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      0 ≤ inner (𝕜 := ℝ) ((fun _ : ℝ => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) t)
        ((fun _ : ℝ => (0 : EucSpace 1)) t)) ∧
    ‖((fun _ : ℝ => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 0)‖ = 1 := by
  refine ⟨zero_le_one, fun t _ => ?_, fun _ _ => zero_le_one, fun _ _ => by simp,
    by simp⟩
  simpa [proj] using (hasDerivAt_const t (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))).hasDerivWithinAt

end Transformer.Normalization
