/-
# A smooth ambient extension of the original causal attention field

ArXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`.
The ambient projected field uses the actual causal weights and value
average. The original spherical equation is exactly its restriction.
The common scale `(n+1)/(1+sum_j norm(y_j)^2)` controls the ambient
projection's growth, while it is exactly one on unit token tuples.

That scale is already proved for the full attention flow; here it
multiplies the actual masked field, whose row average has its own
proved positivity, normalization and operator-norm bound. The resulting
smooth field has linear growth and agrees with the manuscript on the
sphere. Sphere invariance is a separate consequence of its ODE.
No invariant-set or flow existence conclusion is assumed below.
-/

import Transformer.Causal.Section4_ParticleWeights
import Transformer.Perspective.Section2_ExtendedField

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual original causal field, extended by its same polynomial projection to ambient tuples.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, the field in the proof of `thm1`. -/
noncomputable def csaAmbientField (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun i => proj d (Y i) (csaAverage β Q K V Y i)

/-- A smooth ambient growth extension whose restriction is the original causal sphere field.
Source: arXiv:2411.04990v2, §4, the autonomous flow used in `thm1`.
The auxiliary scale is the proved ambient extension factor for the spherical projection. -/
noncomputable def csaGlobalField (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  Perspective.saExtensionScale Y • csaAmbientField β Q K V Y

/-- The original causal equation is precisely the coordinate ODE of the actual ambient field on sphere tuples.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, with every original masked term and normalization. -/
theorem CSA_iff_csaAmbientField (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d n) :
    CSA d n β Q K V X ↔ ∀ t : ℝ, ∀ i : Idx n,
      HasDerivAt (fun s => (X s i : EucSpace d))
        (csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) i) t := by
  simp only [csaAmbientField, csaAverage_eq_original, csaPartition, CSA]

/-- The actual ambient causal field is smooth for arbitrary original query, key and value matrices.
Source: arXiv:2411.04990v2, §4, the smooth autonomous system of `thm1`. -/
theorem contDiff_csaAmbientField (β : ℝ) (Q K V : ParamMatrix d) :
    ContDiff ℝ (⊤ : ℕ∞) (csaAmbientField (n := n) β Q K V) := by
  apply contDiff_pi.mpr
  intro i
  have hcoord : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Y i) := by fun_prop
  exact (contDiff_csaAverage β Q K V i).sub
    ((hcoord.inner ℝ (contDiff_csaAverage β Q K V i)).smul hcoord)

/-- The causal growth extension is smooth on the whole ambient token space.
Source: arXiv:2411.04990v2, §4, the flow construction behind `thm1`. -/
theorem contDiff_csaGlobalField (β : ℝ) (Q K V : ParamMatrix d) :
    ContDiff ℝ (⊤ : ℕ∞) (csaGlobalField (n := n) β Q K V) :=
  Perspective.contDiff_saExtensionScale.smul (contDiff_csaAmbientField β Q K V)

/-- The original ambient causal field is locally Lipschitz, as required for uniqueness of its ODE.
Source: arXiv:2411.04990v2, §4, the smooth autonomous flow in the proof of `thm1`. -/
theorem locallyLipschitz_csaAmbientField (β : ℝ) (Q K V : ParamMatrix d) :
    LocallyLipschitz (csaAmbientField (n := n) β Q K V) :=
  ((contDiff_csaAmbientField β Q K V).of_le
    (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz

/-- The actual growth extension is locally Lipschitz on the entire ambient token space.
Source: arXiv:2411.04990v2, §4, global-flow construction for the original `thm1`. -/
theorem locallyLipschitz_csaGlobalField (β : ℝ) (Q K V : ParamMatrix d) :
    LocallyLipschitz (csaGlobalField (n := n) β Q K V) :=
  ((contDiff_csaGlobalField β Q K V).of_le
    (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz

/-- On every original unit tuple, the growth extension is exactly the source's causal field.
Source: arXiv:2411.04990v2, §2, `eqn:csa`; the scale is a proof device outside the sphere. -/
theorem csaGlobalField_eq_of_unit (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (hY : ∀ j, ‖Y j‖ = 1) :
    csaGlobalField β Q K V Y = csaAmbientField β Q K V Y := by
  rw [csaGlobalField, Perspective.saExtensionScale_eq_one Y hY, one_smul]

/-- The original polynomial tangent projection has the expected cubic ambient growth bound.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, retaining the original value matrix. -/
theorem norm_csaAmbientField_coord_le (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) :
    ‖csaAmbientField β Q K V Y i‖ ≤ (1 + ‖Y i‖ ^ 2) * (‖V‖ * ‖Y‖) := by
  let v := csaAverage β Q K V Y i
  have hv : ‖v‖ ≤ ‖V‖ * ‖Y‖ := norm_csaAverage_le β Q K V Y i
  have hinner : |inner ℝ (Y i) v| ≤ ‖Y i‖ * (‖V‖ * ‖Y‖) :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left hv (norm_nonneg _))
  calc
    ‖csaAmbientField β Q K V Y i‖ ≤ ‖v‖ + ‖inner ℝ (Y i) v • Y i‖ := norm_sub_le _ _
    _ = ‖v‖ + |inner ℝ (Y i) v| * ‖Y i‖ := by rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖V‖ * ‖Y‖ + (‖Y i‖ * (‖V‖ * ‖Y‖)) * ‖Y i‖ :=
      add_le_add hv (mul_le_mul_of_nonneg_right hinner (norm_nonneg _))
    _ = (1 + ‖Y i‖ ^ 2) * (‖V‖ * ‖Y‖) := by ring

/-- The actual smooth causal extension has linear growth, uniformly over all ambient initial tuples.
Source: arXiv:2411.04990v2, §4, global-flow justification for the original `thm1`.
The scale absorbs precisely the original polynomial projection's quadratic norm factor. -/
theorem norm_csaGlobalField_le (β : ℝ) (Q K V : ParamMatrix d) (Y : Idx n → EucSpace d) :
    ‖csaGlobalField β Q K V Y‖ ≤ (((n : ℝ) + 1) * ‖V‖) * ‖Y‖ := by
  apply (pi_norm_le_iff_of_nonneg (by positivity)).mpr
  intro i
  have hsum : ‖Y i‖ ^ 2 ≤ ∑ j : Idx n, inner ℝ (Y j) (Y j) := by
    rw [← real_inner_self_eq_norm_sq]
    apply Finset.single_le_sum _ (Finset.mem_univ i)
    intro j hj
    rw [real_inner_self_eq_norm_sq]
    positivity
  have hscale : Perspective.saExtensionScale Y * (1 + ‖Y i‖ ^ 2) ≤ (n : ℝ) + 1 := by
    unfold Perspective.saExtensionScale
    rw [div_mul_eq_mul_div]
    apply (div_le_iff₀ (Perspective.saExtensionDenom_pos Y)).mpr
    exact mul_le_mul_of_nonneg_left (add_le_add le_rfl hsum) (by positivity)
  change ‖Perspective.saExtensionScale Y • csaAmbientField β Q K V Y i‖ ≤ _
  rw [norm_smul, Real.norm_of_nonneg (Perspective.saExtensionScale_pos Y).le]
  calc
    Perspective.saExtensionScale Y * ‖csaAmbientField β Q K V Y i‖
        ≤ Perspective.saExtensionScale Y * ((1 + ‖Y i‖ ^ 2) * (‖V‖ * ‖Y‖)) :=
      mul_le_mul_of_nonneg_left (norm_csaAmbientField_coord_le β Q K V Y i)
        (Perspective.saExtensionScale_pos Y).le
    _ = (Perspective.saExtensionScale Y * (1 + ‖Y i‖ ^ 2)) * (‖V‖ * ‖Y‖) := by ring
    _ ≤ ((n : ℝ) + 1) * (‖V‖ * ‖Y‖) :=
      mul_le_mul_of_nonneg_right hscale (by positivity)
    _ = (((n : ℝ) + 1) * ‖V‖) * ‖Y‖ := by ring

/-- A common original token makes the original causal average equal its actual value image.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, including the original row normalization. -/
theorem csaAverage_consensus (β : ℝ) (Q K V : ParamMatrix d) (x : EucSpace d) (i : Idx n) :
    csaAverage β Q K V (fun _ => x) i = V x := by
  rw [csaAverage, ← Finset.sum_smul, sum_csaTokenWeight, one_smul]

/-- At identity values every original unit consensus tuple is stationary for the original causal field.
Source: arXiv:2411.04990v2, §4, the first class of equilibria in the proof of `thm1`. -/
theorem csaAmbientField_consensus_id (β : ℝ) (Q K : ParamMatrix d) (x : SSphere d) :
    csaAmbientField (n := n) β Q K (ContinuousLinearMap.id ℝ _) (fun _ => (x : EucSpace d)) = 0 := by
  funext i
  rw [csaAmbientField, csaAverage_consensus, ContinuousLinearMap.id_apply]
  simpa only [one_smul, Pi.zero_apply] using proj_smul_self (mem_sphere_zero_iff_norm.mp x.2) 1

/-- A nonzero actual consensus tuple satisfies the extension's unit-input hypothesis. -/
example : ∀ i : Idx 2, ‖(fun _ => (basePoint 1 : EucSpace 2)) i‖ = 1 := by
  intro i
  exact mem_sphere_zero_iff_norm.mp (basePoint 1).2

end Transformer.Causal
