import Transformer.Modes.Section5_WeightedPhase
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# The interval where a monotone phase derivative is small

In the proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, stationary
points are separated from the part where the phase derivative is bounded
away from zero. For a monotone derivative `p`, its small values lie between
two cuts at `-δ` and `δ`, or at the endpoints of the original interval.
The cuts are constructed by the intermediate value theorem.

If the actual derivative `q` of `p` satisfies `q ≥ κ > 0` whenever
`|p| ≤ δ`, the mean value theorem bounds the distance between the cuts
by `2δ/κ`. Outside that interval, monotonicity supplies the lower bound
`|p| ≥ δ` needed by the first derivative test. This is a quantitative
substitute for choosing unspecified neighborhoods of stationary points.

All derivatives in this module are derivatives of the given phase
functions. No stationary-phase estimate or Fourier decay is assumed.
The interval construction allows either cut to coincide with an endpoint,
so that it also covers phases without a stationary point on the interval.
The length estimate needs no separate continuity assumption on `p`:
its `HasDerivAt` hypotheses already imply continuity at every point of
the closed interval, including both endpoints.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- A nonnegative second derivative makes the actual first derivative
monotone. Source: arXiv:2412.09080v3, §5.5, the stationary intervals. -/
theorem phase_derivative_monotone {a b : ℝ} {p q : ℝ → ℝ}
    (hab : a ≤ b) (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hq : ∀ t ∈ Set.uIcc a b, 0 ≤ q t) : MonotoneOn p (Set.Icc a b) := by
  intro x hx y hy hxy
  rcases eq_or_lt_of_le hxy with hxy | hxy
  · rw [hxy]
  have hsub : Set.Icc x y ⊆ Set.uIcc a b := by
    intro t ht
    rw [Set.uIcc_of_le hab]
    exact ⟨hx.1.trans ht.1, ht.2.trans hy.2⟩
  have hc : ContinuousOn p (Set.Icc x y) :=
    continuousOn_of_forall_continuousAt fun t ht => (hp t (hsub ht)).continuousAt
  obtain ⟨c, hcxy, hce⟩ := exists_hasDerivAt_eq_slope p q hxy hc
    (fun t ht => hp t (hsub ⟨ht.1.le, ht.2.le⟩))
  have hqc := hq c (hsub ⟨hcxy.1.le, hcxy.2.le⟩)
  have he : p y - p x = q c * (y - x) := by
    rw [hce, div_mul_cancel₀ _ (sub_ne_zero.mpr hxy.ne')]
  rw [← sub_nonneg, he]
  exact mul_nonneg hqc (sub_nonneg.mpr hxy.le)

example : (-1 : ℝ) ≤ 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, (0 : ℝ) ≤ 1) := by
  exact ⟨by norm_num, fun t _ => hasDerivAt_id t, fun _ _ => zero_le_one⟩

/-- Cut a monotone phase derivative at the two threshold values,
using an endpoint if that threshold is outside the image.
Source: arXiv:2412.09080v3, §5.5, the stationary neighborhoods `J_j`. -/
theorem exists_phase_sublevel_interval {a b δ : ℝ} {p : ℝ → ℝ}
    (hab : a ≤ b) (hδ : 0 < δ) (hp : ContinuousOn p (Set.Icc a b))
    (hmono : MonotoneOn p (Set.Icc a b)) (ha : p a ≤ δ) (hb : -δ ≤ p b) :
    ∃ u v : ℝ, u ∈ Set.Icc a b ∧ v ∈ Set.Icc a b ∧ u ≤ v ∧
      -δ ≤ p u ∧ p v ≤ δ ∧ (u = a ∨ p u = -δ) ∧ (v = b ∨ p v = δ) := by
  have hu : ∃ u ∈ Set.Icc a b, -δ ≤ p u ∧ (u = a ∨ p u = -δ) := by
    by_cases hau : -δ ≤ p a
    · exact ⟨a, ⟨le_rfl, hab⟩, hau, Or.inl rfl⟩
    · obtain ⟨u, hu, heu⟩ := intermediate_value_Icc hab hp ⟨(not_le.mp hau).le, hb⟩
      exact ⟨u, hu, heu ▸ le_rfl, Or.inr heu⟩
  have hv : ∃ v ∈ Set.Icc a b, p v ≤ δ ∧ (v = b ∨ p v = δ) := by
    by_cases hbv : p b ≤ δ
    · exact ⟨b, ⟨hab, le_rfl⟩, hbv, Or.inl rfl⟩
    · obtain ⟨v, hv, hev⟩ := intermediate_value_Icc hab hp ⟨ha, (not_le.mp hbv).le⟩
      exact ⟨v, hv, hev ▸ le_rfl, Or.inr hev⟩
  obtain ⟨u, hu, hpu, heu⟩ := hu
  obtain ⟨v, hv, hpv, hev⟩ := hv
  refine ⟨u, v, hu, hv, ?_, hpu, hpv, heu, hev⟩
  rcases heu with heu | heu
  · rw [heu]
    exact hv.1
  rcases hev with hev | hev
  · rw [hev]
    exact hu.2
  by_contra huv
  have h := hmono hv hu (not_le.mp huv).le
  rw [heu, hev] at h
  linarith

example : (-2 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    ContinuousOn (fun t : ℝ => t) (Set.Icc (-2 : ℝ) 2) ∧
    MonotoneOn (fun t : ℝ => t) (Set.Icc (-2 : ℝ) 2) ∧
    (-2 : ℝ) ≤ 1 ∧ (-1 : ℝ) ≤ 2 := by
  exact ⟨by norm_num, one_pos, continuousOn_id, fun _ _ _ _ h => h,
    by norm_num, by norm_num⟩

/-- Monotonicity keeps every intermediate derivative between the
thresholds at the two cuts.
Source: arXiv:2412.09080v3, §5.5, the stationary neighborhoods `J_j`. -/
theorem phase_sublevel_abs_le {a b δ : ℝ} {p : ℝ → ℝ}
    (hmono : MonotoneOn p (Set.Icc a b)) (ha : -δ ≤ p a) (hb : p b ≤ δ)
    {t : ℝ} (ht : t ∈ Set.Icc a b) : |p t| ≤ δ := by
  have hab := ht.1.trans ht.2
  apply abs_le.mpr
  constructor
  · exact ha.trans (hmono ⟨le_rfl, hab⟩ ht ht.1)
  · exact (hmono ht ⟨hab, le_rfl⟩ ht.2).trans hb

example : MonotoneOn (fun t : ℝ => t) (Set.Icc (-1 : ℝ) 1) ∧
    (-1 : ℝ) ≤ -1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 := by
  exact ⟨fun _ _ _ _ h => h, le_rfl, le_rfl, by norm_num⟩

/-- The length of a small-derivative interval is at most `2δ/κ`,
provided the actual second derivative is at least `κ` there.
Source: arXiv:2412.09080v3, §5.5, the nondegenerate stationary intervals. -/
theorem phase_sublevel_length_le {a b δ κ : ℝ} {p q : ℝ → ℝ}
    (hab : a ≤ b) (hκ : 0 < κ)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hmono : MonotoneOn p (Set.Icc a b))
    (hmin : ∀ t ∈ Set.uIcc a b, |p t| ≤ δ → κ ≤ q t)
    (ha : -δ ≤ p a) (hb : p b ≤ δ) : b - a ≤ 2 * δ / κ := by
  have hpab := hmono ⟨le_rfl, hab⟩ ⟨hab, le_rfl⟩ hab
  have hδ : 0 ≤ δ := by linarith
  rcases eq_or_lt_of_le hab with hab | hab
  · rw [hab, sub_self]
    positivity
  have hc : ContinuousOn p (Set.Icc a b) :=
    continuousOn_of_forall_continuousAt fun t ht =>
      (hp t (by rwa [Set.uIcc_of_le hab.le])).continuousAt
  obtain ⟨c, hc, hce⟩ := exists_hasDerivAt_eq_slope p q hab hc
    (fun t ht => hp t (by rw [Set.uIcc_of_le hab.le]; exact ⟨ht.1.le, ht.2.le⟩))
  have hpc := phase_sublevel_abs_le hmono ha hb ⟨hc.1.le, hc.2.le⟩
  have hqc := hmin c (by rw [Set.uIcc_of_le hab.le]; exact ⟨hc.1.le, hc.2.le⟩) hpc
  rw [hce] at hqc
  have hmul := (le_div_iff₀ (sub_pos.mpr hab)).mp hqc
  apply (le_div_iff₀ hκ).mpr
  nlinarith

example : (-1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    MonotoneOn (fun t : ℝ => t) (Set.Icc (-1 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (-1 : ℝ) 1, |t| ≤ 1 → (1 : ℝ) ≤ 1) ∧
    (-1 : ℝ) ≤ -1 ∧ (1 : ℝ) ≤ 1 := by
  exact ⟨by norm_num, one_pos, fun t _ => hasDerivAt_id t,
    fun _ _ _ _ h => h, fun _ _ _ => le_rfl, le_rfl, le_rfl⟩

end Transformer.Modes
