module

public import Transformer.Kinetic.Section6_HardyKernel
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Kinetic theory for Transformers — uniqueness on a short time interval

Differences of actual classical solutions in arXiv:2605.09213v1, §6,
`prop:Bessel`, solve the homogeneous Volterra equation. On a finite
time rectangle their original kernel is bounded. We prove that a
bounded homogeneous solution with zero initial value is zero whenever
the time length times the actual operator bound is less than one.
The bound is taken over the actual solution values; the mean value
theorem and the original spatial integral give the contraction.
-/

@[expose] public section

open Real Set MeasureTheory

namespace Transformer
namespace Kinetic

/-- The actual homogeneous Volterra field of a difference of source solutions.
Source: arXiv:2605.09213v1, §6, uniqueness for `prop:Bessel`, subtracting `eq:Hardy-hom`. -/
noncomputable def homogeneousHardyField (lam a s₀ : ℝ) (f : ℝ → ℝ) (s : ℝ) : ℝ :=
  a * ∫ r in s₀..s, graphon lam s r * f r

/-- The original homogeneous field vanishes on the zero function.
Source: arXiv:2605.09213v1, §6, the homogeneous `eq:Hardy-hom`. -/
theorem homogeneousHardyField_zero (lam a s₀ s : ℝ) :
    homogeneousHardyField lam a s₀ (fun _ => 0) s = 0 := by
  simp [homogeneousHardyField]

/-- The norm of the actual Volterra field is bounded by the actual
kernel bound times the original function bound; its spatial interval
has length at most one. Source: arXiv:2605.09213v1, §6, `eq:Hardy-hom`. -/
theorem homogeneousHardyField_norm_le (lam a : ℝ) {s₀ s K C : ℝ} (hs₀ : 0 < s₀)
    (hs : s ∈ Ioc s₀ 1) (hK : 0 ≤ K) (hC : 0 ≤ C)
    (hkernel : ∀ s ∈ Icc s₀ 1, ∀ r ∈ Icc s₀ 1, |graphon lam s r| ≤ K)
    (f : ℝ → ℝ) (hf : ∀ r ∈ Ioc s₀ 1, |f r| ≤ C) :
    ‖homogeneousHardyField lam a s₀ f s‖ ≤ |a| * K * C := by
  have hi : ‖∫ r in s₀..s, graphon lam s r * f r‖ ≤ (K * C) * |s - s₀| := by
    apply intervalIntegral.norm_integral_le_of_norm_le_const
    rw [uIoc_of_le hs.1.le]
    intro r hr
    rw [Real.norm_eq_abs, abs_mul]
    exact mul_le_mul (hkernel s ⟨hs.1.le, hs.2⟩ r ⟨hr.1.le, hr.2.trans hs.2⟩)
      (hf r ⟨hr.1, hr.2.trans hs.2⟩) (abs_nonneg _) hK
  have hd : |s - s₀| ≤ 1 := by rw [abs_of_nonneg (sub_nonneg.mpr hs.1.le)]; linarith [hs.2]
  rw [homogeneousHardyField, norm_mul, Real.norm_eq_abs]
  calc |a| * ‖∫ r in s₀..s, graphon lam s r * f r‖ ≤ |a| * ((K * C) * |s - s₀|) :=
      mul_le_mul_of_nonneg_left hi (abs_nonneg _)
    _ ≤ |a| * K * C := by nlinarith [mul_nonneg (abs_nonneg a) (mul_nonneg hK hC)]

/-- A bounded actual homogeneous Volterra solution with zero initial
value is zero on a sufficiently short actual time interval.
Source: arXiv:2605.09213v1, §6, uniqueness for the original `prop:Bessel`. -/
theorem homogeneousHardy_eq_zero_on_short_interval (lam a : ℝ) {s₀ K T₀ T₁ : ℝ}
    (hs₀ : 0 < s₀) (hs₁ : s₀ < 1) (hK : 0 ≤ K) (hT₀ : 0 ≤ T₀) (hT : T₀ ≤ T₁)
    (hsmall : |a| * K * (T₁ - T₀) < 1)
    (hkernel : ∀ s ∈ Icc s₀ 1, ∀ r ∈ Icc s₀ 1, |graphon lam s r| ≤ K)
    (u : ℝ → ℝ → ℝ) (hzero : ∀ s ∈ Ioc s₀ 1, u T₀ s = 0)
    (hbound : ∃ B : ℝ, ∀ t ∈ Icc T₀ T₁, ∀ s ∈ Ioc s₀ 1, |u t s| ≤ B)
    (hd : ∀ t ∈ Icc T₀ T₁, ∀ s ∈ Ioc s₀ 1,
      HasDerivWithinAt (fun t => u t s) (homogeneousHardyField lam a s₀ (u t) s) (Ici 0) t) :
    ∀ t ∈ Icc T₀ T₁, ∀ s ∈ Ioc s₀ 1, u t s = 0 := by
  let S : Set ℝ := {z | ∃ t ∈ Icc T₀ T₁, ∃ s ∈ Ioc s₀ 1, z = |u t s|}
  obtain ⟨B, hB⟩ := hbound
  have hbdd : BddAbove S := ⟨B, by rintro z ⟨t, ht, s, hs, rfl⟩; exact hB t ht s hs⟩
  have hS0 : 0 ∈ S := by
    refine ⟨T₀, ⟨le_rfl, hT⟩, 1, ⟨hs₁, le_rfl⟩, ?_⟩
    rw [hzero 1 ⟨hs₁, le_rfl⟩, abs_zero]
  let C := sSup S
  have hC : 0 ≤ C := le_csSup hbdd hS0
  have hle : ∀ t ∈ Icc T₀ T₁, ∀ s ∈ Ioc s₀ 1, |u t s| ≤ C :=
    fun t ht s hs => le_csSup hbdd ⟨t, ht, s, hs, rfl⟩
  have hcontract : ∀ t ∈ Icc T₀ T₁, ∀ s ∈ Ioc s₀ 1,
      |u t s| ≤ (|a| * K * (T₁ - T₀)) * C := by
    intro t ht s hs
    have hder : ∀ q ∈ Icc T₀ T₁,
        HasDerivWithinAt (fun t => u t s) (homogeneousHardyField lam a s₀ (u q) s) (Icc T₀ T₁) q :=
      fun q hq => (hd q hq s hs).mono (by intro r hr; exact hT₀.trans hr.1)
    have hm := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le hder
      (fun q hq => homogeneousHardyField_norm_le lam a hs₀ hs hK hC hkernel (u q) (hle q hq))
      (convex_Icc T₀ T₁) ⟨le_rfl, hT⟩ ht
    rw [hzero s hs, sub_zero, Real.norm_eq_abs,
      Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr ht.1)] at hm
    have hnonneg : 0 ≤ |a| * K * C := by positivity
    nlinarith [mul_le_mul_of_nonneg_left ht.2 hnonneg]
  have hsup : C ≤ (|a| * K * (T₁ - T₀)) * C := by
    apply csSup_le ⟨0, hS0⟩
    rintro z ⟨t, ht, s, hs, rfl⟩
    exact hcontract t ht s hs
  have hC0 : C = 0 := by nlinarith
  intro t ht s hs
  apply abs_nonpos_iff.mp
  simpa only [hC0] using hle t ht s hs

/-- A locally bounded actual homogeneous Volterra solution with zero
initial value is zero for all nonnegative time. The proved short-interval
argument is iterated over actual finite time intervals.
Source: arXiv:2605.09213v1, §6, uniqueness in `prop:Bessel`. -/
theorem homogeneousHardy_eq_zero (lam a : ℝ) {s₀ : ℝ} (hs₀ : 0 < s₀) (hs₁ : s₀ < 1)
    (u : ℝ → ℝ → ℝ) (hzero : ∀ s ∈ Ioc s₀ 1, u 0 s = 0)
    (hbound : ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |u t s| ≤ B)
    (hd : ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc s₀ 1,
      HasDerivWithinAt (fun t => u t s) (homogeneousHardyField lam a s₀ (u t) s) (Ici 0) t) :
    ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc s₀ 1, u t s = 0 := by
  obtain ⟨K, hK, hkernel⟩ := exists_bound_graphon lam hs₀
  let δ : ℝ := 1 / (2 * (|a| * K + 1))
  have hδ : 0 < δ := by dsimp [δ]; positivity
  have hsmall : |a| * K * δ < 1 := by
    dsimp [δ]
    rw [mul_one_div, div_lt_one (by positivity : (0 : ℝ) < 2 * (|a| * K + 1))]
    nlinarith [mul_nonneg (abs_nonneg a) hK]
  have hn : ∀ n : ℕ, ∀ t ∈ Icc (0 : ℝ) (n * δ), ∀ s ∈ Ioc s₀ 1, u t s = 0 := by
    intro n
    induction n with
    | zero =>
      intro t ht s hs
      simp only [Nat.cast_zero, zero_mul] at ht
      have ht0 : t = 0 := le_antisymm ht.2 ht.1
      rw [ht0]
      exact hzero s hs
    | succ n ih =>
      have hT₀ : 0 ≤ (n : ℝ) * δ := by positivity
      have hT : (n : ℝ) * δ ≤ ((n + 1 : ℕ) : ℝ) * δ := by push_cast; nlinarith
      obtain ⟨B, hB⟩ := hbound (((n + 1 : ℕ) : ℝ) * δ) (by positivity)
      have hs : |a| * K * (((n + 1 : ℕ) : ℝ) * δ - n * δ) < 1 := by
        convert hsmall using 1
        push_cast
        ring
      have hstep := homogeneousHardy_eq_zero_on_short_interval lam a hs₀ hs₁ hK hT₀ hT hs hkernel u
        (fun s hs => ih (n * δ) ⟨hT₀, le_rfl⟩ s hs)
        ⟨B, fun t ht s hs => hB t ⟨hT₀.trans ht.1, ht.2⟩ s hs⟩
        (fun t ht s hs => hd t (hT₀.trans ht.1) s hs)
      intro t ht s hs
      by_cases hp : t ≤ (n : ℝ) * δ
      · exact ih t ⟨ht.1, hp⟩ s hs
      · exact hstep t ⟨le_of_not_ge hp, ht.2⟩ s hs
  intro t ht s hs
  obtain ⟨n, hnT⟩ := exists_nat_ge (t / δ)
  exact hn n t ⟨ht, (div_le_iff₀ hδ).mp hnT⟩ s hs

/-- A zero original homogeneous solution witnesses all time and value
hypotheses of the contraction theorem on a genuine short interval.
Its original kernel bound is supplied by the proved compact bound.
Source: arXiv:2605.09213v1, §6, the zero difference in `prop:Bessel`. -/
example : ∃ K : ℝ, 0 ≤ K ∧
    (∀ s ∈ Icc (1 / 2 : ℝ) 1, ∀ r ∈ Icc (1 / 2 : ℝ) 1, |graphon 0 s r| ≤ K) ∧
    ∃ δ : ℝ, 0 < δ ∧ K * δ < 1 ∧
      (∀ s ∈ Ioc (1 / 2 : ℝ) 1, (fun _ _ : ℝ => (0 : ℝ)) 0 s = 0) ∧
      (∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) δ, ∀ s ∈ Ioc (1 / 2 : ℝ) 1,
        |(fun _ _ : ℝ => (0 : ℝ)) t s| ≤ B) ∧
      (∀ t ∈ Icc (0 : ℝ) δ, ∀ s ∈ Ioc (1 / 2 : ℝ) 1,
        HasDerivWithinAt (fun _ : ℝ => (0 : ℝ))
          (homogeneousHardyField 0 1 (1 / 2) (fun _ => 0) s) (Ici 0) t) := by
  obtain ⟨K, hK, hb⟩ := exists_bound_graphon 0 one_half_pos
  have hp : 0 < 2 * (K + 1) := by positivity
  refine ⟨K, hK, hb, 1 / (2 * (K + 1)), by positivity, ?_, fun _ _ => rfl,
    ⟨0, fun _ _ _ _ => by simp⟩, ?_⟩
  · rw [mul_one_div, div_lt_one hp]
    linarith
  · intro t ht s hs
    rw [homogeneousHardyField_zero]
    exact (hasDerivAt_const t (0 : ℝ)).hasDerivWithinAt

/-- The same genuine zero solution witnesses all global hypotheses,
with an actual finite bound on every finite time rectangle.
Source: arXiv:2605.09213v1, §6, the homogeneous zero difference in `prop:Bessel`. -/
example : ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc (1 / 2 : ℝ) 1, (fun _ _ : ℝ => (0 : ℝ)) t s = 0 := by
  apply homogeneousHardy_eq_zero 0 1 one_half_pos one_half_lt_one
  · intro s hs
    rfl
  · intro T hT
    exact ⟨0, fun _ _ _ _ => by simp⟩
  · intro t ht s hs
    rw [homogeneousHardyField_zero]
    exact (hasDerivAt_const t (0 : ℝ)).hasDerivWithinAt

end Kinetic
end Transformer
