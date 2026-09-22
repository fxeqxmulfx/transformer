/-
# Cone collapse — weighted flows and the chart of a hemisphere

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
§6.1, `lem: hemisphere.clustering`.  The survey remarks that "we only make use
of the positivity of the coefficients `a_{i,j}(t)` throughout the proof", which
is how it covers `SA`, `USA` and their `Q, K` variants at once.  Its step 2
does use more: the uniform floor `a_ij(t) ≥ n⁻¹ e^{-2β}`, and so does the rate
here, with weights in some `[m, M]`, `m > 0`.  This file sets up the common
form,

  `ẋ_i = Proj_{x_i}( Σ_j a_ij(t) x_j )`   (`IsWeightedFlow`),

and the two facts the proof is built on:

* the floor `min_i ⟨x_i(t), w⟩` of a hemisphere does not decrease
  (`floor_of_weightedFlow`), so the particles stay in the hemisphere;
* in the chart `y = x / ⟨x, w⟩` of the hemisphere onto the hyperplane
  `⟨·, w⟩ = 1`, the flow is a linear consensus dynamics
  `ẏ_i = Σ_j a_ij (s_j / s_i) (y_j - y_i)`, `s_i = ⟨x_i, w⟩`
  (`hasDerivAt_hemiChart`).

The survey's own proof tracks `α(t) = min_i ⟨x_i(t), x⋆⟩` instead; the chart
replaces its qualitative first step (convergence to some `x⋆`) and its
differential inequality by one exponential contraction, proved in
`Perspective.ConeWidth`.
-/

import Transformer.Basic
import Transformer.Perspective.MinCurve
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **A weighted flow on the sphere:** `ẋ_i = Proj_{x_i}(Σ_j a_ij(t) x_j)`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering` (the
coefficients `a_ij(t)`). -/
def IsWeightedFlow (X : ℝ → SphereTuple d n) (a : ℝ → Idx n → Idx n → ℝ) : Prop :=
  ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => (X s i : EucSpace d))
    (proj d (X t i : EucSpace d) (∑ j : Idx n, a t i j • (X t j : EucSpace d))) t

/-- The chart `x ↦ x / ⟨x, w⟩` of the open hemisphere around `w` onto the
hyperplane `⟨·, w⟩ = 1`. -/
noncomputable def hemiChart (w x : EucSpace d) : EucSpace d :=
  (inner (𝕜 := ℝ) x w)⁻¹ • x

variable {X : ℝ → SphereTuple d n} {a : ℝ → Idx n → Idx n → ℝ}

theorem norm_coe_tuple (X : SphereTuple d n) (i : Idx n) : ‖(X i : EucSpace d)‖ = 1 :=
  mem_sphere_zero_iff_norm.mp (X i).2

/-- The height `s_i = ⟨x_i, w⟩` of a particle above the equator of `w` moves by
`ṡ_i = Σ_j a_ij (s_j - ⟨x_i, x_j⟩ s_i)`. -/
theorem hasDerivAt_height (hX : IsWeightedFlow X a) (w : EucSpace d) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => inner (𝕜 := ℝ) (X s i : EucSpace d) w)
      (∑ j : Idx n, a t i j * (inner (𝕜 := ℝ) (X t j : EucSpace d) w
        - inner (𝕜 := ℝ) (X t i : EucSpace d) (X t j : EucSpace d)
          * inner (𝕜 := ℝ) (X t i : EucSpace d) w)) t := by
  refine ((hX t i).inner ℝ (hasDerivAt_const t w)).congr_deriv ?_
  simp only [inner_zero_right, zero_add, proj, inner_sub_left, real_inner_smul_left,
    real_inner_smul_right, sum_inner, inner_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- **The particles stay in the hemisphere.**  With non-negative weights, a
floor `r₀ > 0` of the heights at time `0` is a floor at every later time: at a
particle attaining the minimum `r ≥ 0`, every term `a_ij (s_j - ⟨x_i, x_j⟩ s_i)`
is non-negative.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, step 1. -/
theorem floor_of_weightedFlow (hX : IsWeightedFlow X a) (ha : ∀ t i j, 0 ≤ a t i j)
    (hn : 0 < n) (w : EucSpace d) {r₀ : ℝ} (hr₀ : 0 < r₀)
    (h0 : ∀ i, r₀ ≤ inner (𝕜 := ℝ) (X 0 i : EucSpace d) w) {t : ℝ} (ht : 0 ≤ t) (i : Idx n) :
    r₀ ≤ inner (𝕜 := ℝ) (X t i : EucSpace d) w := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  have hne : (Finset.univ : Finset (Idx n)).Nonempty := Finset.univ_nonempty
  set f : Idx n → ℝ → ℝ := fun i s => inner (𝕜 := ℝ) (X s i : EucSpace d) w
  set r : ℝ → ℝ := fun s => Finset.univ.inf' hne fun i => f i s
  have hmin : ∀ s i, r s ≤ f i s := fun s i => Finset.inf'_le _ (Finset.mem_univ i)
  have hatt : ∀ s, ∃ i, r s = f i s := fun s => by
    obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_inf' hne fun i => f i s
    exact ⟨i, hi⟩
  have hr0 : r₀ ≤ r 0 := Finset.le_inf' _ _ fun i _ => h0 i
  have hmono := le_min_curve_of_deriv_nonneg (f := f) (r := r)
    (fun i s => hasDerivAt_height hX w s i) hmin hatt ?_ (hr₀.trans_le hr0) ht
  · exact hr0.trans (hmono.trans (hmin t i))
  intro s k hr hk
  refine Finset.sum_nonneg fun j _ => mul_nonneg (ha s k j) ?_
  have hij : inner (𝕜 := ℝ) (X s k : EucSpace d) (X s j : EucSpace d) ≤ 1 := by
    have := real_inner_le_norm (X s k : EucSpace d) (X s j : EucSpace d)
    rwa [norm_coe_tuple, norm_coe_tuple, mul_one] at this
  have hkj : f k s ≤ f j s := hk ▸ hmin s j
  have hk0 : 0 ≤ f k s := hk ▸ hr
  change 0 ≤ f j s - _ * f k s
  nlinarith

/-- **The flow in the chart is a linear consensus dynamics:**
`ẏ_i = Σ_j a_ij (s_j / s_i) (y_j - y_i)` while all heights are positive. -/
theorem hasDerivAt_hemiChart (hX : IsWeightedFlow X a) (w : EucSpace d) {t : ℝ}
    (hs : ∀ j, 0 < inner (𝕜 := ℝ) (X t j : EucSpace d) w) (i : Idx n) :
    HasDerivAt (fun s => hemiChart w (X s i : EucSpace d))
      (∑ j : Idx n, (a t i j * (inner (𝕜 := ℝ) (X t j : EucSpace d) w
          / inner (𝕜 := ℝ) (X t i : EucSpace d) w))
        • (hemiChart w (X t j : EucSpace d) - hemiChart w (X t i : EucSpace d))) t := by
  set s : Idx n → ℝ := fun j => inner (𝕜 := ℝ) (X t j : EucSpace d) w
  set x : Idx n → EucSpace d := fun j => (X t j : EucSpace d)
  set v : EucSpace d := ∑ j : Idx n, a t i j • x j
  have hsi : s i ≠ 0 := (hs i).ne'
  refine ((HasDerivAt.fun_inv (hasDerivAt_height hX w t i) hsi).smul (hX t i)).congr_deriv ?_
  have hvw : inner (𝕜 := ℝ) v w = ∑ j : Idx n, a t i j * s j := by
    simp only [v, sum_inner, real_inner_smul_left]; rfl
  have hvx : inner (𝕜 := ℝ) (x i) v = ∑ j : Idx n, a t i j * inner (𝕜 := ℝ) (x i) (x j) := by
    simp only [v, inner_sum, real_inner_smul_right]
  have hterm : ∀ j : Idx n, (a t i j * (s j / s i)) • (hemiChart w (x j) - hemiChart w (x i))
      = (s i)⁻¹ • (a t i j • x j) - (a t i j * s j / s i ^ 2) • x i := by
    intro j
    have hsj : s j ≠ 0 := (hs j).ne'
    rw [hemiChart, hemiChart, smul_sub, smul_smul, smul_smul, smul_smul]
    change (a t i j * (s j / s i) * (s j)⁻¹) • x j - (a t i j * (s j / s i) * (s i)⁻¹) • x i = _
    congr 2
    · field_simp
    · field_simp
  rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_sub_distrib, ← Finset.smul_sum,
    ← Finset.sum_smul, ← Finset.sum_div, ← hvw]
  have hsum : ∑ j : Idx n, a t i j * (s j - inner (𝕜 := ℝ) (x i) (x j) * s i)
      = inner (𝕜 := ℝ) v w - inner (𝕜 := ℝ) (x i) v * s i := by
    rw [hvw, hvx, Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  change (s i)⁻¹ • proj d (x i) v + (-(∑ j : Idx n, a t i j * (s j
      - inner (𝕜 := ℝ) (x i) (x j) * s i)) / s i ^ 2) • x i = _
  rw [hsum, proj, smul_sub, smul_smul, sub_add_eq_add_sub, sub_eq_sub_iff_add_eq_add, add_assoc,
    ← add_smul]
  congr 2
  field_simp
  ring

/-- The hypotheses of `hasDerivAt_height`, `floor_of_weightedFlow` and
`hasDerivAt_hemiChart` are satisfiable: one resting particle with zero weights,
in the hemisphere around itself. -/
example : IsWeightedFlow (fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) (fun _ _ _ => 0) ∧
    0 < inner (𝕜 := ℝ) ((basePoint 0 : SSphere 1) : EucSpace 1) (basePoint 0 : SSphere 1) := by
  refine ⟨fun t _ => ?_, ?_⟩
  · simpa [proj] using hasDerivAt_const t ((basePoint 0 : SSphere 1) : EucSpace 1)
  · rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (basePoint 0).2]
    norm_num

/-- In the chart, a unit vector of height `s > 0` has norm `1 / s`, and it is
recovered as the direction of its image. -/
theorem hemiChart_normalize {w x : EucSpace d} (hx : ‖x‖ = 1) (hs : 0 < inner (𝕜 := ℝ) x w) :
    ‖hemiChart w x‖ = (inner (𝕜 := ℝ) x w)⁻¹ ∧ ‖hemiChart w x‖⁻¹ • hemiChart w x = x := by
  have hn : ‖hemiChart w x‖ = (inner (𝕜 := ℝ) x w)⁻¹ := by
    rw [hemiChart, norm_smul, hx, mul_one, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
  refine ⟨hn, ?_⟩
  rw [hn, inv_inv, hemiChart, smul_smul, mul_inv_cancel₀ hs.ne', one_smul]

/-- The hypotheses of `hemiChart_normalize` are satisfiable: `x = w = e₀` in
`ℝ¹`. -/
example : ‖(EuclideanSpace.single 0 1 : EucSpace 1)‖ = 1 ∧
    0 < inner (𝕜 := ℝ) (EuclideanSpace.single 0 1 : EucSpace 1) (EuclideanSpace.single 0 1) := by
  simp

end Perspective
end Transformer
