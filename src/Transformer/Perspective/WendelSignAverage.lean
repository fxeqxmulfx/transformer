/-
# Averaging the sign patterns in Wendel's theorem

The probability of the common-hemisphere event is the expected fraction of
sign patterns that put a sample in a common open hemisphere. This isolates
the probabilistic part of Wendel's argument from the remaining deterministic
count of sign patterns in general position.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelSignSymmetry
import Mathlib.MeasureTheory.Integral.Lebesgue.Add

open MeasureTheory

namespace Transformer.Perspective

/-- The number of coordinatewise sign choices that put `X` in an open
hemisphere. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def hemisphereSignCount (d n : ℕ) (X : SphereTuple d n) : ℕ :=
  by
    classical
    exact (Finset.univ.filter fun mask : Idx n → Bool =>
      ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((flipSigns d n mask X i : EucSpace d))
          ((w : EucSpace d))).card

/-- If all sample vectors are independent, every sign choice can be put in a
common open hemisphere. This checks the `n ≤ d` endpoint of the count.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem hemisphereSignCount_eq_pow_of_linearIndependent (d n : ℕ) (hn : 1 ≤ n)
    (X : SphereTuple d n)
    (hX : LinearIndependent ℝ fun i : Idx n => (X i : EucSpace d)) :
    hemisphereSignCount d n X = 2 ^ n := by
  classical
  have hgood (mask : Idx n → Bool) :
      ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((flipSigns d n mask X i : EucSpace d))
          ((w : EucSpace d)) := by
    let s : Idx n → ℝˣ := fun i => if mask i then -1 else 1
    have heq : (fun i : Idx n => (flipSigns d n mask X i : EucSpace d)) =
        s • (fun i : Idx n => (X i : EucSpace d)) := by
      funext i
      cases h : mask i with
      | false => simp [flipSigns, s, h]
      | true => simp [flipSigns, s, h, sphereMap]
    have hflip : LinearIndependent ℝ fun i : Idx n =>
        (flipSigns d n mask X i : EucSpace d) := by
      rw [heq]
      exact (LinearIndependent.units_smul_iff _ s).2 hX
    exact exists_common_hemisphere_of_linearIndependent d _
      hflip hn
  simp [hemisphereSignCount, hgood, Fintype.card_bool]

/-- Every sign choice has the same probability of putting the sample in an
open hemisphere. Summing over the `2^n` masks gives this finite average.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem sum_measure_hemisphere_flipSigns (d n : ℕ)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P) :
    (∑ mask : Idx n → Bool,
      P ((flipSigns d n mask) ⁻¹' {X : SphereTuple d n |
        ∃ w : SSphere d, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))})) =
      (2 : ENNReal) ^ n * P {X : SphereTuple d n |
        ∃ w : SSphere d, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
  classical
  simp_rw [measure_hemisphereEvent_flipSigns d n P hP]
  simp [Fintype.card_bool]

/-- The expected number of successful sign patterns is `2^n` times the
hemisphere probability. The only remaining ingredient for Wendel's formula is
to show that this count is `2 ∑_{k<d} C(n-1,k)` in general position.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem lintegral_hemisphereSignCount (d n : ℕ)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P) :
    (∫⁻ X, (hemisphereSignCount d n X : ENNReal) ∂P) =
      (2 : ENNReal) ^ n * P {X : SphereTuple d n |
        ∃ w : SSphere d, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} := by
  classical
  let E : Set (SphereTuple d n) := {X | ∃ w : SSphere d, ∀ i : Idx n,
    0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))}
  have hE : MeasurableSet E := (isOpen_hemisphereEvent d n).measurableSet
  have hflip (mask : Idx n → Bool) : Measurable (flipSigns d n mask) := by
    obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
    let _ : IsProbabilityMeasure σ := hσprob
    exact (measurePreserving_flipSigns d n σ hσinv mask).measurable
  have hset (mask : Idx n → Bool) : MeasurableSet ((flipSigns d n mask) ⁻¹' E) :=
    hE.preimage (hflip mask)
  have hcount (X : SphereTuple d n) :
      (hemisphereSignCount d n X : ENNReal) =
        ∑ mask : Idx n → Bool,
          (((flipSigns d n mask) ⁻¹' E).indicator (fun _ => (1 : ENNReal))) X := by
    simp [hemisphereSignCount, E, Set.indicator]
  calc
    (∫⁻ X, (hemisphereSignCount d n X : ENNReal) ∂P) =
        ∫⁻ X, ∑ mask : Idx n → Bool,
          (((flipSigns d n mask) ⁻¹' E).indicator (fun _ => (1 : ENNReal))) X ∂P := by
          congr 1
          funext X
          exact hcount X
    _ = ∑ mask : Idx n → Bool, ∫⁻ X,
          (((flipSigns d n mask) ⁻¹' E).indicator (fun _ => (1 : ENNReal))) X ∂P := by
          exact lintegral_finsetSum _ (fun mask _ => measurable_const.indicator (hset mask))
    _ = ∑ mask : Idx n → Bool, P ((flipSigns d n mask) ⁻¹' E) := by
          apply Finset.sum_congr rfl
          intro mask _
          exact lintegral_indicator_one (hset mask)
    _ = (2 : ENNReal) ^ n * P E := by
          exact sum_measure_hemisphere_flipSigns d n P hP

/-- The deterministic sign-pattern count is the sole remaining input to
Wendel's probability formula. This implication uses only the independent
sign symmetry, not a further probabilistic theorem.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem wendel_of_ae_sign_count (d n : ℕ) (hn : 1 ≤ n)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P)
    (hcount : ∀ᵐ X ∂P, hemisphereSignCount d n X =
      2 * ∑ k ∈ Finset.range d, (n - 1).choose k) :
    (P {X : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
      0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))}).toReal =
        (∑ k ∈ Finset.range d, ((n - 1).choose k : ℝ)) / 2 ^ (n - 1) := by
  let _ : IsProbabilityMeasure P := by
    obtain ⟨σ, hσ, -, rfl⟩ := hP
    let _ : IsProbabilityMeasure σ := hσ
    infer_instance
  let S : ℕ := ∑ k ∈ Finset.range d, (n - 1).choose k
  have hL : (∫⁻ X, (hemisphereSignCount d n X : ENNReal) ∂P) =
      ((2 * S : ℕ) : ENNReal) := by
    calc
      _ = ∫⁻ _ : SphereTuple d n, ((2 * S : ℕ) : ENNReal) ∂P := by
        apply lintegral_congr_ae
        filter_upwards [hcount] with X hX
        exact congrArg (fun k : ℕ => (k : ENNReal)) hX
      _ = _ := by simp
  have heq : ((2 * S : ℕ) : ENNReal) = (2 : ENNReal) ^ n *
      P {X : SphereTuple d n | ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w : EucSpace d))} :=
    hL.symm.trans (lintegral_hemisphereSignCount d n P hP)
  have heqreal := congrArg ENNReal.toReal heq
  simp only [ENNReal.toReal_mul, ENNReal.toReal_pow,
    ENNReal.toReal_ofNat, ENNReal.toReal_natCast, Nat.cast_mul] at heqreal
  have hS : (S : ℝ) = ∑ k ∈ Finset.range d, ((n - 1).choose k : ℝ) := by
    simp [S]
  have hpow : (2 : ℝ) ^ n = 2 * 2 ^ (n - 1) := by
    conv_lhs => rw [← Nat.sub_add_cancel hn, pow_succ]
    ring
  rw [hpow, hS] at heqreal
  apply (eq_div_iff (by positivity : (2 : ℝ) ^ (n - 1) ≠ 0)).2
  linear_combination -heqreal / 2

/-- The count hypothesis above holds in the smallest nonempty case. -/
example (P : Measure (SphereTuple 1 1)) (hP : UniformTuple 1 1 P) :
    ∀ᵐ X ∂P, hemisphereSignCount 1 1 X =
      2 * ∑ k ∈ Finset.range 1, (1 - 1).choose k := by
  filter_upwards [ae_linearIndependent_of_uniformTuple 1 le_rfl P hP] with X hX
  rw [hemisphereSignCount_eq_pow_of_linearIndependent 1 1 le_rfl X hX]
  norm_num

end Transformer.Perspective
