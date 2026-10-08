import Transformer.Modes.Section5_PhaseSublevel
import Mathlib.Analysis.Calculus.Deriv.Pow

/-!
# A quantitative estimate near a stationary point

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, invokes the
method of stationary phase on neighborhoods where the second derivative
is bounded away from zero. This module proves the needed local estimate
by cutting at two levels of the actual first derivative instead.

Suppose `φ' = p`, `p' = q`, `q ≥ 0`, and `q ≥ κ > 0` wherever
`|p| ≤ δ`. The integral is split into at most three intervals. The two
outer intervals each contribute at most `4/(|ρ| δ)` by the first derivative
test. The middle interval has length at most `2δ/κ`, and the oscillatory
kernel has norm one. Their sum gives `8/(|ρ| δ) + 2δ/κ`.

Here `δ` is a threshold for the phase derivative; the source's `δ` is the
radius of a stationary neighborhood. Both cuts may reach the endpoints,
and the argument includes the case when the middle interval is absent.
The proof retains boundary terms and uses no asymptotic expansion.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- Split the phase into two regions with large first derivative and
an interval whose length is controlled by the second derivative.
Source: arXiv:2412.09080v3, §5.5, the stationary-phase estimate. -/
theorem oscillatory_integral_le_of_sublevel_derivative {a b ρ δ κ : ℝ} {φ p q : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ) (hκ : 0 < κ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t)
    (hmin : ∀ t ∈ Set.uIcc a b, |p t| ≤ δ → κ ≤ q t) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t‖ ≤ 8 / (|ρ| * δ) + 2 * δ / κ := by
  have hC : 0 ≤ 4 / (|ρ| * δ) := by positivity
  have hmono := phase_derivative_monotone hab hp hq
  have hcp : ContinuousOn p (Set.Icc a b) :=
    continuousOn_of_forall_continuousAt fun t ht =>
      (hp t (by rwa [Set.uIcc_of_le hab])).continuousAt
  have hsub (x y : ℝ) (hax : a ≤ x) (hxy : x ≤ y) (hyb : y ≤ b) :
      Set.uIcc x y ⊆ Set.uIcc a b := by
    intro t ht
    rw [Set.uIcc_of_le hxy] at ht
    rw [Set.uIcc_of_le hab]
    exact ⟨hax.trans ht.1, ht.2.trans hyb⟩
  have hfirst (x y : ℝ) (hax : a ≤ x) (hxy : x ≤ y) (hyb : y ≤ b)
      (hminxy : ∀ t ∈ Set.uIcc x y, δ ≤ |p t|) :
      ‖∫ t in x..y, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
    have hs := hsub x y hax hxy hyb
    exact first_derivative_test_nonneg hxy hρ hδ (fun t ht => hφ t (hs ht))
      (fun t ht => hp t (hs ht)) (hcq.mono hs) (fun t ht => hq t (hs ht)) hminxy
  by_cases hpa : δ ≤ p a
  · have hf := hfirst a b le_rfl hab le_rfl (fun t ht => by
      rw [Set.uIcc_of_le hab] at ht
      exact (hpa.trans (hmono ⟨le_rfl, hab⟩ ht ht.1)).trans (le_abs_self _))
    calc _ ≤ _ := hf
      _ ≤ _ := by
        have hL : 0 ≤ 2 * δ / κ := by positivity
        have h8 : 8 / (|ρ| * δ) = 2 * (4 / (|ρ| * δ)) := by ring
        rw [h8]
        linarith
  by_cases hpb : p b ≤ -δ
  · have hf := hfirst a b le_rfl hab le_rfl (fun t ht => by
      rw [Set.uIcc_of_le hab] at ht
      have h := (hmono ht ⟨hab, le_rfl⟩ ht.2).trans hpb
      exact (by linarith : δ ≤ -p t).trans (neg_le_abs _))
    calc _ ≤ _ := hf
      _ ≤ _ := by
        have hL : 0 ≤ 2 * δ / κ := by positivity
        have h8 : 8 / (|ρ| * δ) = 2 * (4 / (|ρ| * δ)) := by ring
        rw [h8]
        linarith
  obtain ⟨u, v, hu, hv, huv, hpu, hpv, heu, hev⟩ :=
    exists_phase_sublevel_interval hab hδ hcp hmono (not_le.mp hpa).le (not_le.mp hpb).le
  have hleft : ‖∫ t in a..u, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
    rcases heu with heu | heu
    · rw [heu, intervalIntegral.integral_same, norm_zero]
      exact hC
    apply hfirst a u le_rfl hu.1 hu.2
    intro t ht
    rw [Set.uIcc_of_le hu.1] at ht
    have h := hmono ⟨ht.1, ht.2.trans hu.2⟩ hu ht.2
    rw [heu] at h
    exact (by linarith : δ ≤ -p t).trans (neg_le_abs _)
  have hright : ‖∫ t in v..b, oscillatoryKernel ρ φ t‖ ≤ 4 / (|ρ| * δ) := by
    rcases hev with hev | hev
    · rw [hev, intervalIntegral.integral_same, norm_zero]
      exact hC
    apply hfirst v b hv.1 hv.2 le_rfl
    intro t ht
    rw [Set.uIcc_of_le hv.2] at ht
    have h := hmono hv ⟨hv.1.trans ht.1, ht.2⟩ ht.1
    rw [hev] at h
    exact h.trans (le_abs_self _)
  have hs := hsub u v hu.1 huv hv.2
  have hmonouv : MonotoneOn p (Set.Icc u v) := by
    intro x hx y hy hxy
    exact hmono ⟨hu.1.trans hx.1, hx.2.trans hv.2⟩
      ⟨hu.1.trans hy.1, hy.2.trans hv.2⟩ hxy
  have hlen := phase_sublevel_length_le huv hκ (fun t ht => hp t (hs ht))
    hmonouv (fun t ht => hmin t (hs ht)) hpu hpv
  have hmid : ‖∫ t in u..v, oscillatoryKernel ρ φ t‖ ≤ 2 * δ / κ := by
    have h := intervalIntegral.norm_integral_le_of_norm_le_const (a := u) (b := v)
      (C := 1) (fun t _ => (norm_oscillatoryKernel ρ φ t).le)
    rw [one_mul, abs_of_nonneg (sub_nonneg.mpr huv)] at h
    exact h.trans hlen
  have hcφ : ContinuousOn φ (Set.uIcc a b) :=
    continuousOn_of_forall_continuousAt fun t ht => (hφ t ht).continuousAt
  have hK : ContinuousOn (oscillatoryKernel ρ φ) (Set.uIcc a b) := by
    unfold oscillatoryKernel
    fun_prop
  rw [← intervalIntegral.integral_add_adjacent_intervals
    ((hK.mono (hsub a v le_rfl hv.1 hv.2)).intervalIntegrable)
    ((hK.mono (hsub v b hv.1 hv.2 le_rfl)).intervalIntegrable),
    ← intervalIntegral.integral_add_adjacent_intervals
    ((hK.mono (hsub a u le_rfl hu.1 hu.2)).intervalIntegrable)
    ((hK.mono hs).intervalIntegrable)]
  calc _ ≤ (4 / (|ρ| * δ) + 2 * δ / κ) + 4 / (|ρ| * δ) :=
        norm_add_le_of_le (norm_add_le_of_le hleft hmid) hright
    _ = _ := by ring


example : (-1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => 2 * s) 2 t) ∧
    ContinuousOn (fun _ : ℝ => (2 : ℝ)) (Set.uIcc (-1 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, (0 : ℝ) ≤ 2) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, |2 * t| ≤ 1 → (1 : ℝ) ≤ 2) := by
  refine ⟨by norm_num, one_ne_zero, one_pos, one_pos, ?_, ?_, continuousOn_const,
    fun _ _ => by norm_num, fun _ _ _ => by norm_num⟩
  · intro t _
    simpa using (hasDerivAt_id t).fun_pow 2
  · intro t _
    simpa using (hasDerivAt_id t).const_mul 2

/-- Balancing the small-derivative interval and the integration-by-parts
terms gives the square-root scale of stationary phase.
Source: arXiv:2412.09080v3, §5.5, the `ρ⁻¹ᐟ²` stationary-phase bound. -/
theorem oscillatory_threshold_balance {ρ δ κ : ℝ}
    (hρ : ρ ≠ 0) (hδ : 0 < δ) (hκ : 0 < κ) (hmul : |ρ| * δ ^ 2 = κ) :
    8 / (|ρ| * δ) + 2 * δ / κ = 10 / Real.sqrt (|ρ| * κ) := by
  have hρpos : 0 < |ρ| := abs_pos.mpr hρ
  have hscale : Real.sqrt (|ρ| * κ) = |ρ| * δ := by
    apply (Real.sqrt_eq_iff_eq_sq (mul_nonneg hρpos.le hκ.le)
      (mul_nonneg hρpos.le hδ.le)).mpr
    rw [← hmul]
    ring
  rw [hscale, ← hmul]
  field_simp
  ring

example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    |(1 : ℝ)| * 1 ^ 2 = 1 := by
  exact ⟨one_ne_zero, one_pos, one_pos, by norm_num⟩

end Transformer.Modes
