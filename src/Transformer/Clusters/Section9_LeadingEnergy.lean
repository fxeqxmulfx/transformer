/-
# An energy estimate needs positivity only on the leading space

ArXiv:2305.05465v6, §9, `eq:phistarvar` and `e:majorizeweights`,
in the proof of `l:3hyperplanes11`. The attention form on the leading
line is represented by an injective scalar multiple of the identity.
The form off that line may remain nonsymmetric or indefinite.

The actual normalized bilinear scores have the proved exponential
error from their leading form. The perturbed-softmax estimate therefore
bounds the derivative of the leading squared norm exactly as in §10,
without a global positive attention root. The factor `λ` is retained.

Compatibility of an attention metric with a subspace is a bilinear
identity on that subspace. It assumes no dynamics, convergence,
gain estimate, or energy property. Such a metric is constructed below
from the original positive coefficient along the simple leading line.
-/

import Transformer.Clusters.Section9_LeadingScores
import Transformer.Clusters.Section10_ProjectedEnergy
import Mathlib.Analysis.Real.Sqrt

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A metric represents the attention form on the specified subspace.
Source: arXiv:2305.05465v6, §9, the coefficient `c₁₁` in `e:majorizeweights`. -/
def IsAttentionFormOn (Q K A : ParamMatrix d) (F : Submodule ℝ (EucSpace d)) : Prop :=
  ∀ x ∈ F, ∀ y ∈ F, inner (𝕜 := ℝ) (Q x) (K y) = inner (𝕜 := ℝ) (A x) (A y)

/-- A positive attention root represents the form on every subspace.
Source: arXiv:2305.05465v6, §8, `eq: A`, and §10, its leading restriction. -/
theorem isAttentionFormOn_of_isAttentionRoot (Q K A : ParamMatrix d)
    (F : Submodule ℝ (EucSpace d)) (hA : IsAttentionRoot Q K A) : IsAttentionFormOn Q K A F :=
  fun x _ y _ => hA.2.2 x y

example : IsAttentionRoot (1 : ParamMatrix d) 1 1 := isAttentionRoot_id d

/-- Projected leading scores agree with the representing metric.
Source: arXiv:2305.05465v6, §9, `e:majorizeweights`, the leading term. -/
theorem IsAttentionFormOn.inner_projected_eq (Q K A proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (hform : IsAttentionFormOn Q K A F)
    (hproj : IsProjOnto proj F G) (x y : EucSpace d) :
    inner (𝕜 := ℝ) (Q (proj x)) (K (proj y)) = inner (𝕜 := ℝ) (A (proj x)) (A (proj y)) :=
  hform (proj x) (hproj.1 x) (proj y) (hproj.1 y)

example : IsAttentionFormOn (1 : ParamMatrix d) 1 1 ⊤ ∧
    IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ :=
  ⟨isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id d), isProjOnto_one d⟩

/-- An injective representing metric makes the original form strictly
positive at every nonzero leading vector. Source: arXiv:2305.05465v6,
§4, `d:good`, the required positivity on the leading line. -/
theorem IsAttentionFormOn.inner_self_pos (Q K A : ParamMatrix d)
    (F : Submodule ℝ (EucSpace d)) (hform : IsAttentionFormOn Q K A F)
    (hA : Function.Injective A) (x : EucSpace d) (hx : x ∈ F) (hne : x ≠ 0) :
    0 < inner (𝕜 := ℝ) (Q x) (K x) := by
  rw [hform x hx x hx]
  exact real_inner_self_pos.mpr fun hz => hne (hA (by simpa only [map_zero] using hz))

example : IsAttentionFormOn (1 : ParamMatrix 1) 1 1 ⊤ ∧
    Function.Injective (1 : ParamMatrix 1) ∧
    (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) ∈ (⊤ : Submodule ℝ (EucSpace 1)) ∧
    (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) ≠ 0 :=
  ⟨isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id 1),
    Function.injective_id, Submodule.mem_top, by simp⟩

/-- Positivity along one nonzero eigenvector constructs an injective
metric for that whole line. Source: arXiv:2305.05465v6, §4, `d:good`,
and §9, `c₁₁ > 0` in `e:majorizeweights`. The metric is a positive
scalar identity, with no assertion about the attention form off the line. -/
theorem exists_injective_attentionFormOn_span_singleton (Q K : ParamMatrix d)
    (φ : EucSpace d) (hφ : φ ≠ 0) (hpositive : 0 < inner (𝕜 := ℝ) (Q φ) (K φ)) :
    ∃ A : ParamMatrix d, Function.Injective A ∧ IsAttentionFormOn Q K A (Submodule.span ℝ {φ}) := by
  let c := inner (𝕜 := ℝ) (Q φ) (K φ)
  let a := Real.sqrt c / ‖φ‖
  have hnorm := norm_pos_iff.mpr hφ
  have ha : 0 < a := div_pos (Real.sqrt_pos.mpr hpositive) hnorm
  have hcoeff : a ^ 2 * ‖φ‖ ^ 2 = c := by
    dsimp only [a]
    rw [div_pow, div_mul_cancel₀ _ (pow_ne_zero 2 hnorm.ne'), Real.sq_sqrt hpositive.le]
  refine ⟨a • (1 : ParamMatrix d), ?_, ?_⟩
  · intro x y hxy
    change a • x = a • y at hxy
    exact smul_right_injective (EucSpace d) ha.ne' hxy
  · intro x hx y hy
    obtain ⟨u, rfl⟩ := Submodule.mem_span_singleton.mp hx
    obtain ⟨v, rfl⟩ := Submodule.mem_span_singleton.mp hy
    calc
      _ = u * v * c := by
        simp only [map_smul, real_inner_smul_left, real_inner_smul_right, c]
        ring
      _ = u * v * (a ^ 2 * ‖φ‖ ^ 2) := by rw [hcoeff]
      _ = _ := by
        simp only [smul_apply, one_apply_eq_self, real_inner_smul_left,
          real_inner_smul_right, real_inner_self_eq_norm_sq]
        ring

example : (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) ≠ 0 ∧
    0 < inner (𝕜 := ℝ) ((1 : ParamMatrix 1) (EuclideanSpace.single 0 (1 : ℝ)))
      ((1 : ParamMatrix 1) (EuclideanSpace.single 0 (1 : ℝ))) :=
  ⟨by simp, by simp⟩

/-- **The actual leading energy derivative dominates every leading
attention gap up to the proved exponentially decaying errors.**
Source: arXiv:2305.05465v6, §9, `eq:phistarvar`, `e:majorizeweights`,
and §10, `e:derivativenorm123`. Only representation of the form on
the leading subspace is needed; no global attention root is assumed. -/
theorem RescaledDynamics.exists_projected_energy_bound_of_leading_form
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hform : IsAttentionFormOn Q K A F) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ∃ v : ℝ, HasDerivAt (fun s => ‖A (proj (Z s i))‖ ^ 2) v t ∧ ∀ j : Idx n,
        2 * lam * (inner (𝕜 := ℝ) (A (proj (Z t i)))
          (A (proj (Z t j) - proj (Z t i))) - (n : ℝ) / Real.exp (2 * lam * t) -
          2 * D * Real.exp (-δ * t)) ≤ v := by
  obtain ⟨δ, D, hδ, hD, herror⟩ :=
    hZ.exists_bound_normalized_bilinear_error Q K V proj F G lam hV hproj Z
  refine ⟨δ, D, hδ, hD, fun t ht i => ?_⟩
  let X : Idx n → EucSpace d := fun k => expTime V t (Z t k)
  let p : Idx n → ℝ := attentionMatrix Q K X i
  let u : Idx n → ℝ := fun k => inner (𝕜 := ℝ)
    (Q (Real.exp (-lam * t) • X i)) (K (Real.exp (-lam * t) • X k))
  let w : Idx n → ℝ := fun k => inner (𝕜 := ℝ) (A (proj (Z t i))) (A (proj (Z t k)))
  have hp : ∀ k, Perspective.softmaxWeight (fun l => Real.exp (2 * lam * t) * u l) k = p k :=
    fun k => (attentionMatrix_eq_normalized_bilinear_softmax Q K X lam t i k).symm
  have herr : ∀ k, |u k - w k| ≤ D * Real.exp (-δ * t) := by
    intro k
    simpa only [hform.inner_projected_eq Q K A proj F G hproj] using herror t ht i k
  have hsum : ∑ k : Idx n, p k = 1 := sum_attentionMatrix (Fin.pos_iff_nonempty.mpr ⟨i⟩) _ _ _ _
  refine ⟨2 * lam * ∑ k : Idx n, p k * (w k - w i), ?_, fun j => ?_⟩
  · simpa only [map_sub, inner_sub_right, w, p, X] using
      hZ.hasDerivAt_norm_sq_projected Q K V proj A F G lam hV hproj Z t i
  · have haverage := score_sub_card_div_sub_error_le_softmax_average u w
      (Real.exp (2 * lam * t)) (Real.exp_pos _) (D * Real.exp (-δ * t)) herr j
    simp_rw [hp] at haverage
    have hid : ∑ k : Idx n, p k * (w k - w i) = (∑ k : Idx n, p k * w k) - w i := by
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
    rw [map_sub, inner_sub_right, hid]
    apply mul_le_mul_of_nonneg_left _ (mul_nonneg (show (0 : ℝ) ≤ 2 by norm_num) hV.2.2.2.1.le)
    change w j - w i - (n : ℝ) / Real.exp (2 * lam * t) -
      2 * D * Real.exp (-δ * t) ≤ (∑ k : Idx n, p k * w k) - w i
    linarith

example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionFormOn (1 : ParamMatrix 1) 1 1 ⊤ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1,
    isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id 1),
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
