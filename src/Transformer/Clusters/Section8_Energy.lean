/-
# The emergence of clusters in self-attention dynamics — the energy at
  `V = -I_d`

`e:finiteinegral` of arXiv:2305.05465v6, §8, `s:c<0`.

**What the source says and what is carried here.**

* `∫_0^{+∞} ‖ẋ_i(t)‖² dt < +∞` is `IntegrableOn` over `[0,+∞)`; the
  integrand is continuous, so the two say the same.

Source: arXiv:2305.05465v6, `e:finiteinegral`.
-/

import Transformer.Clusters.Section8_Origin
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real Filter MeasureTheory

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **The unnormalized drift.**  `Σ_j e^{⟨x_i,x_j⟩} x_j = -Z_i ẋ_i` along
`e:-Iddyn`, where `Z_i = Σ_j e^{⟨x_i,x_j⟩}`.

Source: arXiv:2305.05465v6, the computation in the proof of
`e:finiteinegral`. -/
theorem sum_exp_smul_eq_neg_smul_negIdDrift (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (Y : Idx n → EucSpace d) (i : Idx n) :
    ∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) (Y i) (Y j)) • Y j
      = -(∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) (Y i) (Y j))) • negIdDrift Q K Y i := by
  have hZ : 0 < ∑ k : Idx n, Real.exp (inner (𝕜 := ℝ) (Y i) (Y k)) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ _⟩
  rw [negIdDrift, neg_smul, smul_neg, neg_neg, Finset.smul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [smul_smul]
  congr 1
  simp only [attentionMatrix, Perspective.softmaxWeight,
    show ∀ u v, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) u v from hQK]
  field_simp

/-- **Lemma (e:finiteinegral).**  The trajectories of `e:-Iddyn` satisfy
`∫_0^{+∞} ‖ẋ_i(t)‖² dt < +∞` for every `i ∈ [n]`.

As in the source, `𝓛(t) = Σ_i Σ_j e^{⟨x_i(t),x_j(t)⟩}` is non-negative and
`d𝓛/dt = -2 Σ_i Z_i ‖ẋ_i‖²`, so `-𝓛` is non-decreasing, bounded, and its
derivative is integrable on `[0,+∞)`.  The source then bounds
`Z_i ≥ n e^{-R²}` by `l:cas1circle`; here `Z_i ≥ e^{‖x_i‖²} ≥ 1` already
suffices, and `l:cas1circle` is not used.

Source: arXiv:2305.05465v6, `e:finiteinegral`. -/
theorem integrableOn_sq_norm_negIdDrift (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) (i : Idx n) :
    MeasureTheory.IntegrableOn (fun t => ‖negIdDrift Q K (X t) i‖ ^ 2) (Set.Ici 0) := by
  set a := fun t k j => inner (𝕜 := ℝ) (X t k) (X t j)
  set Z := fun t k => ∑ j : Idx n, Real.exp (a t k j)
  set v := fun t k => negIdDrift Q K (X t) k
  set g := fun t => -∑ k : Idx n, Z t k
  set g' := fun t => 2 * ∑ k : Idx n, Z t k * ‖v t k‖ ^ 2
  have hZ1 : ∀ t k, 1 ≤ Z t k := fun t k => by
    have h := Finset.single_le_sum (f := fun j => Real.exp (a t k j))
      (fun j _ => (Real.exp_pos _).le) (Finset.mem_univ k)
    have : 1 ≤ Real.exp (a t k k) :=
      Real.one_le_exp (by simp only [a]; rw [real_inner_self_eq_norm_sq]; positivity)
    exact this.trans h
  have hg : ∀ t, HasDerivAt g (g' t) t := by
    intro t
    have hd : HasDerivAt (fun s => ∑ k : Idx n, Z s k)
        (∑ k : Idx n, ∑ j : Idx n, Real.exp (a t k j) *
          (inner (𝕜 := ℝ) (X t k) (v t j) + inner (𝕜 := ℝ) (v t k) (X t j))) t :=
      HasDerivAt.fun_sum fun k _ => HasDerivAt.fun_sum fun j _ =>
        ((hX t k).inner ℝ (hX t j)).exp.congr_deriv (by simp only [a, v])
    refine hd.neg.congr_deriv ?_
    have hsym : ∑ k : Idx n, ∑ j : Idx n, Real.exp (a t k j) * inner (𝕜 := ℝ) (X t k) (v t j)
        = ∑ k : Idx n, ∑ j : Idx n, Real.exp (a t k j) * inner (𝕜 := ℝ) (v t k) (X t j) := by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun j _ => ?_
      simp only [a]
      rw [real_inner_comm (X t k) (X t j), real_inner_comm (X t j) (v t k)]
    have hrow : ∀ k, ∑ j : Idx n, Real.exp (a t k j) * inner (𝕜 := ℝ) (v t k) (X t j)
        = -(Z t k * ‖v t k‖ ^ 2) := fun k => by
      have := congrArg (fun w => inner (𝕜 := ℝ) (v t k) w)
        (sum_exp_smul_eq_neg_smul_negIdDrift Q K hQK (X t) k)
      simp only [inner_sum, inner_smul_right] at this
      simpa [a, Z, v, neg_mul] using this
    simp only [mul_add, Finset.sum_add_distrib, hsym, hrow, g', Finset.sum_neg_distrib]
    ring
  have hg'0 : ∀ t, 0 ≤ g' t := fun t =>
    mul_nonneg zero_le_two (Finset.sum_nonneg fun k _ =>
      mul_nonneg (zero_le_one.trans (hZ1 t k)) (sq_nonneg _))
  have hmono : Monotone g := monotone_of_hasDerivAt_nonneg hg hg'0
  have hbdd : BddAbove (Set.range g) := ⟨0, by
    rintro _ ⟨t, rfl⟩
    exact neg_nonpos.2 (Finset.sum_nonneg fun k _ => zero_le_one.trans (hZ1 t k))⟩
  have hint := integrableOn_Ioi_deriv_of_nonneg (a := 0) (hg 0).continuousAt.continuousWithinAt
    (fun t _ => hg t) (fun t _ => hg'0 t) (tendsto_atTop_ciSup hmono hbdd)
  have hXc : ∀ k, Continuous fun t => X t k := fun k =>
    continuous_iff_continuousAt.2 fun t => (hX t k).continuousAt
  have hZc : Continuous fun t => Z t i := continuous_finsetSum _ fun j _ =>
    Real.continuous_exp.comp ((hXc i).inner (hXc j))
  have hv : (fun t => v t i)
      = fun t => (-(Z t i)⁻¹) • ∑ j : Idx n, Real.exp (a t i j) • X t j := funext fun t => by
    simp only [a, Z, v]
    rw [sum_exp_smul_eq_neg_smul_negIdDrift Q K hQK (X t) i, smul_smul, neg_mul_neg,
      inv_mul_cancel₀ (zero_lt_one.trans_le (hZ1 t i)).ne', one_smul]
  have hvc : Continuous fun t => v t i := by
    rw [hv]
    exact ((hZc.inv₀ fun t => (zero_lt_one.trans_le (hZ1 t i)).ne').neg).smul
      (continuous_finsetSum _ fun j _ =>
        (Real.continuous_exp.comp ((hXc i).inner (hXc j))).smul (hXc j))
  rw [integrableOn_Ici_iff_integrableOn_Ioi]
  refine hint.mono' (hvc.norm.pow 2).aestronglyMeasurable (ae_of_all _ fun t => ?_)
  have hk := Finset.single_le_sum (f := fun k => Z t k * ‖v t k‖ ^ 2)
    (fun k _ => mul_nonneg (zero_le_one.trans (hZ1 t k)) (sq_nonneg _)) (Finset.mem_univ i)
  have h1 : ‖v t i‖ ^ 2 ≤ Z t i * ‖v t i‖ ^ 2 :=
    le_mul_of_one_le_left (sq_nonneg _) (hZ1 t i)
  have h2 : 0 ≤ ∑ k : Idx n, Z t k * ‖v t k‖ ^ 2 := hk.trans' (h1.trans' (sq_nonneg _))
  rw [Real.norm_of_nonneg (sq_nonneg _)]
  simp only [g']
  linarith

/-- The hypotheses of `integrableOn_sq_norm_negIdDrift` are satisfiable:
`Q = K = I_d` and the configuration sitting at the origin. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    NegIdDynamics (n := n) (1 : ParamMatrix d) 1 (fun _ _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, (transformerDynamics_neg_one_iff _ _ _).mp (transformerDynamics_zero 1 1 _)⟩

end Clusters
end Transformer
