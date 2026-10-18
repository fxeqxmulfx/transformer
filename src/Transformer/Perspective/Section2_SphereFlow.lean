/-
# Global existence and uniqueness of the actual spherical attention flow

ArXiv:2312.10794v5, §2, `SA` and `eq: SA`, and arXiv:2411.04551v3,
§5, the deterministic flow in `prop:generic-discrete`.
The smooth ambient extension has linear growth, so it has a global
solution. Its squared-norm defect satisfies a scalar homogeneous ODE.
Uniqueness of that scalar ODE forces the defect to stay zero.

Consequently the extension factor is one along the constructed solution,
and the solution follows the original spherical attention equation.
No sphere constraint is assumed along the ambient path. Uniqueness is
then obtained for the original equation by the same ambient extension.

Existence and uniqueness are proved for every real temperature; the
source's positive-temperature case is included without changing its field.
-/

import Transformer.Perspective.Section2_ExtendedField
import Transformer.Landscape.GlobalFlow.Existence

open scoped BigOperators
open Real Set Filter Topology

namespace Transformer.Perspective

variable {d n : ℕ}

/-- A scalar homogeneous ODE with continuous coefficient and zero initial
datum has the zero solution. This proves the norm-defect invariant for
arXiv:2312.10794v5, §2, `SA`, using local ODE uniqueness on finite intervals. -/
theorem scalar_eq_zero_of_linearODE (a u : ℝ → ℝ) (ha : Continuous a)
    (hu : ∀ t : ℝ, HasDerivAt u (a t * u t) t) (h0 : u 0 = 0) (t : ℝ) : u t = 0 := by
  let T := |t| + 1
  obtain ⟨B, hB⟩ := (isCompact_Icc (a := -T) (b := T)).exists_bound_of_continuousOn
    ha.continuousOn
  have hbound (s : ℝ) (hs : s ∈ Ioo (-T) T) : ‖a s‖ ≤ |B| :=
    (hB s (Ioo_subset_Icc_self hs)).trans (le_abs_self B)
  have hlip (s : ℝ) (hs : s ∈ Ioo (-T) T) :
      LipschitzOnWith (Real.toNNReal |B|) (fun v : ℝ => a s * v) univ := by
    apply LipschitzWith.lipschitzOnWith
    rw [lipschitzWith_iff_dist_le_mul]
    intro x y
    simp only [dist_eq_norm, ← mul_sub, norm_mul, Real.coe_toNNReal _ (abs_nonneg B)]
    exact mul_le_mul_of_nonneg_right (hbound s hs) (norm_nonneg _)
  have hzero (s : ℝ) : HasDerivAt (fun _ : ℝ => (0 : ℝ)) (a s * 0) s := by
    simpa only [mul_zero] using hasDerivAt_const s (0 : ℝ)
  have heq := ODE_solution_unique_of_mem_Ioo (v := fun s v => a s * v)
    (s := fun _ => (univ : Set ℝ)) hlip
    (show (0 : ℝ) ∈ Ioo (-T) T by dsimp [T]; constructor <;> linarith [abs_nonneg t])
    (fun s _ => ⟨hu s, mem_univ s⟩) (fun s _ => ⟨hzero s, mem_univ 0⟩) h0
  exact heq ⟨by dsimp [T]; linarith [neg_abs_le t], by dsimp [T]; linarith [le_abs_self t]⟩

example : Continuous (fun _ : ℝ => (1 : ℝ)) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (1 * 0) t) ∧
    (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨continuous_const, fun t => by simpa using hasDerivAt_const t (0 : ℝ), rfl⟩

/-- Unit initial tokens stay on the sphere along the actual ambient
extension: the norm defect solves a homogeneous scalar ODE.
Source: arXiv:2312.10794v5, §2, the projection in `SA`. -/
theorem norm_eq_one_of_saGlobalField (β : ℝ) (Y : ℝ → Idx n → EucSpace d)
    (hY : ∀ t : ℝ, HasDerivAt Y (saGlobalField β (Y t)) t)
    (h0 : ∀ i : Idx n, ‖Y 0 i‖ = 1) (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let v (s : ℝ) := ∑ j : Idx n, softmaxWeight (saScore d n β (Y s) i) j • Y s j
  let a (s : ℝ) := -2 * saExtensionScale (Y s) * inner (𝕜 := ℝ) (Y s i) (v s)
  have hv : Continuous v := (contDiff_saAvg β i).continuous.comp hc
  have ha : Continuous a :=
    (continuous_const.mul (contDiff_saExtensionScale.continuous.comp hc)).mul
      (((continuous_apply i).comp hc).inner hv)
  have hdefect (s : ℝ) : HasDerivAt (fun r => ‖Y r i‖ ^ 2 - 1)
      (a s * (‖Y s i‖ ^ 2 - 1)) s := by
    convert ((hasDerivAt_pi.mp (hY s)) i).norm_sq.sub_const 1 using 1
    simp only [saGlobalField, Pi.smul_apply, saField, proj, real_inner_smul_right,
      inner_sub_right, real_inner_self_eq_norm_sq]
    dsimp [a, v]
    ring
  have hzero := scalar_eq_zero_of_linearODE a (fun s => ‖Y s i‖ ^ 2 - 1) ha hdefect
    (by rw [h0 i]; norm_num) t
  nlinarith [norm_nonneg (Y t i)]

/-- A nonzero stationary unit token satisfies every ambient ODE and initial
norm hypothesis. Source: arXiv:2312.10794v5, §2, the one-token case of `SA`. -/
example (β : ℝ) :
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => fun _ : Idx 1 => (basePoint 0 : EucSpace 1))
      (saGlobalField β (fun _ : Idx 1 => (basePoint 0 : EucSpace 1))) t) ∧
    (∀ i : Idx 1, ‖(fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) i‖ = 1) := by
  have hunit : ∀ i : Idx 1, ‖(fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) i‖ = 1 :=
    fun _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hfield : saGlobalField β (fun _ : Idx 1 => (basePoint 0 : EucSpace 1)) = 0 := by
    funext i
    change saExtensionScale _ • saField 1 1 β _ i = 0
    rw [saField_of_one 1 β hunit i, smul_zero]
  exact ⟨fun t => by rw [hfield]; exact hasDerivAt_const t _, hunit⟩

/-- **Every spherical initial tuple has an actual global attention
trajectory.** Source: arXiv:2312.10794v5, §2, `SA`, and
arXiv:2411.04551v3, §5, the flow used in `prop:generic-discrete`.
The original vector field is recovered by the proved sphere invariant. -/
theorem exists_SA (β : ℝ) (X₀ : SphereTuple d n) :
    ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ SA d n β X := by
  have hF : LocallyLipschitz (saGlobalField (d := d) (n := n) β) :=
    ((contDiff_saGlobalField β).of_le (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz
  obtain ⟨Y, hY0, hY⟩ := Landscape.GlobalFlow.exists_global hF (C := (n : ℝ) + 1)
    (by positivity) (norm_saGlobalField_le β) (fun i => (X₀ i : EucSpace d))
  have hunit (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 :=
    norm_eq_one_of_saGlobalField β Y hY
      (fun j => by rw [hY0]; exact mem_sphere_zero_iff_norm.mp (X₀ j).2) t i
  let X (t : ℝ) (i : Idx n) : SSphere d := ⟨Y t i, mem_sphere_zero_iff_norm.mpr (hunit t i)⟩
  refine ⟨X, ?_, (SA_iff d n β X).mpr ?_⟩
  · funext i
    exact Subtype.ext (congrFun hY0 i)
  · intro t i
    simpa only [saGlobalField, saExtensionScale_eq_one (Y t) (hunit t), one_smul] using
      (hasDerivAt_pi.mp (hY t)) i

/-- **The actual spherical attention trajectory is unique.**
Source: arXiv:2312.10794v5, §2, the deterministic flow of `SA`, and
arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem SA_unique (β : ℝ) (X Y : ℝ → SphereTuple d n)
    (hX : SA d n β X) (hY : SA d n β Y) (h0 : X 0 = Y 0) : X = Y := by
  have hF : LocallyLipschitz (saGlobalField (d := d) (n := n) β) :=
    ((contDiff_saGlobalField β).of_le (show (1 : WithTop ℕ∞) ≤ (⊤ : ℕ∞) by simp)).locallyLipschitz
  have hpath (W : ℝ → SphereTuple d n) (hW : SA d n β W) (t : ℝ) :
      HasDerivAt (fun s i => (W s i : EucSpace d))
        (saGlobalField β (fun i => (W t i : EucSpace d))) t := by
    apply hasDerivAt_pi.mpr
    intro i
    have hunit : ∀ j, ‖(W t j : EucSpace d)‖ = 1 :=
      fun j => mem_sphere_zero_iff_norm.mp (W t j).2
    simpa only [saGlobalField, saExtensionScale_eq_one _ hunit, one_smul] using
      (SA_iff d n β W).mp hW t i
  have heq := Landscape.GlobalFlow.eq_of_hasDerivAt hF (hpath X hX) (hpath Y hY)
    (by rw [h0])
  funext t i
  exact Subtype.ext (congrFun (congrFun heq t) i)

example (β : ℝ) (x : SSphere 1) :
    SA 1 1 β (fun _ _ => x) ∧ SA 1 1 β (fun _ _ => x) ∧
    (fun _ : ℝ => fun _ : Idx 1 => x) 0 = (fun _ : ℝ => fun _ : Idx 1 => x) 0 :=
  ⟨SA_const_consensus 1 1 one_pos β x, SA_const_consensus 1 1 one_pos β x, rfl⟩

/-- The deterministic flow used in arXiv:2411.04551v3, §5,
`prop:generic-discrete`, exists for every spherical initial tuple and
is uniquely determined by that tuple. The equation is the original
`SA` of arXiv:2312.10794v5, §2. -/
theorem exists_unique_SA (β : ℝ) (X₀ : SphereTuple d n) :
    ∃! X : ℝ → SphereTuple d n, X 0 = X₀ ∧ SA d n β X := by
  obtain ⟨X, h0, hSA⟩ := exists_SA β X₀
  refine ⟨X, ⟨h0, hSA⟩, ?_⟩
  intro Y hY
  exact SA_unique β Y X hY.2 hSA (hY.1.trans h0.symm)

end Transformer.Perspective
