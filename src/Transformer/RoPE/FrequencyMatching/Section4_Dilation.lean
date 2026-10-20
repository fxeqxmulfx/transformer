module

public import Transformer.RoPE.FrequencyMatching.Section3_Optimality

/-!
# Position interpolation under dependency dilation

Wu, Liu and Jadbabaie, arXiv:2607.07678v1, §4, Theorem 2 and Corollary 1.
The longer-context distribution is an actual pushforward measure. Merely
increasing a sequence's length is not assumed to dilate its dependencies.
-/

@[expose] public section

open MeasureTheory Set

namespace Transformer.RoPE.FrequencyMatching

/-- Theorem 2: PI preserves positional utility when the dependency distribution
is dilated by the matching scale. No density or atomlessness is required. -/
theorem utility_dilation (mu : Measure ℝ) {scale : ℝ} (hscale : scale ≠ 0)
    (omega : ℝ) :
    utility (Measure.map (fun r => scale * r) mu) (omega / scale) = utility mu omega := by
  unfold utility
  rw [integral_map (by fun_prop : AEMeasurable (fun r : ℝ => scale * r) mu)
    (show AEStronglyMeasurable (contrast (omega / scale)) _ from
      (show Continuous (contrast (omega / scale)) by unfold contrast; fun_prop
        ).aestronglyMeasurable)]
  apply integral_congr_ae
  exact Filter.Eventually.of_forall fun r => by
    change 1 - Real.cos ((omega / scale) * (scale * r)) = 1 - Real.cos (omega * r)
    rw [interpolated_phase hscale]

example : utility (Measure.map (fun r : ℝ => 2 * r) (Measure.dirac 1))
    (Real.pi / 2) = utility (Measure.dirac 1) Real.pi :=
  utility_dilation (Measure.dirac 1) (by norm_num) Real.pi

/-- §4.1: admissibility scales with dependency width under positive dilation. -/
theorem admissible_dilation {scale width omega : ℝ} (hscale : 0 < scale) :
    Admissible (scale * width) (omega / scale) ↔ Admissible width omega := by
  simp only [Admissible, lt_div_iff₀ hscale, zero_mul, interpolated_phase hscale.ne']

example : Admissible (2 * 1) (Real.pi / 2) ↔ Admissible 1 Real.pi :=
  admissible_dilation (by norm_num)

/-- Corollary 1: every maximizer of the source objective transports to a
maximizer under matched dilation. The maximizing property is the input that
Theorem 1 supplies, rather than an unproved model-performance assumption. -/
theorem maximizer_dilation (mu : Measure ℝ) {scale width best : ℝ}
    (hscale : 0 < scale) (hbest : Admissible width best)
    (hmax : ∀ omega, Admissible width omega → utility mu omega ≤ utility mu best) :
    Admissible (scale * width) (best / scale) ∧
      ∀ omega, Admissible (scale * width) omega →
        utility (Measure.map (fun r => scale * r) mu) omega ≤
          utility (Measure.map (fun r => scale * r) mu) (best / scale) := by
  refine ⟨(admissible_dilation hscale).2 hbest, fun omega homega => ?_⟩
  have homega' : Admissible width (omega * scale) := by
    apply (admissible_dilation hscale).1
    simpa [mul_div_cancel_right₀ _ hscale.ne'] using homega
  have hleft := utility_dilation mu hscale.ne' (omega * scale)
  rw [mul_div_cancel_right₀ _ hscale.ne'] at hleft
  rw [hleft, utility_dilation mu hscale.ne']
  exact hmax _ homega'

example : Admissible (2 * 1) (Real.pi / 2) ∧
    ∀ omega, Admissible (2 * 1) omega →
      utility (Measure.map (fun r : ℝ => 2 * r) (Measure.dirac 1)) omega ≤
        utility (Measure.map (fun r : ℝ => 2 * r) (Measure.dirac 1)) (Real.pi / 2) := by
  have h := unique_optimal_frequency (Measure.dirac 1) (by norm_num : (0 : ℝ) < 1)
    (by simp) (by simp)
  exact maximizer_dilation (Measure.dirac 1) (by norm_num) (by simpa using h.1)
    (fun omega homega => by simpa using (h.2 omega homega).1)

/-- §4.2: scaling frequencies without dilating the task can hurt. For a
dependency fixed at distance one, halving the optimal frequency lowers utility. -/
theorem unchanged_dependency_scaling_hurts :
    utility (Measure.dirac 1) (Real.pi / 2) < utility (Measure.dirac 1) Real.pi := by
  norm_num [utility, contrast]

/-- §3.2, a derived consequence for unknown unbounded dependency width:
no positive frequency has a half-period covering every possible width. -/
theorem no_unbounded_admissible_frequency {omega : ℝ} (homega : 0 < omega) :
    ∃ width, 0 < width ∧ ¬ Admissible width omega := by
  refine ⟨2 * Real.pi / omega, by positivity, ?_⟩
  intro h
  have hphase : omega * (2 * Real.pi / omega) = 2 * Real.pi := by
    field_simp
  have hbound := h.2
  rw [hphase] at hbound
  linarith [Real.pi_pos]

example : ∃ width, 0 < width ∧ ¬ Admissible width 1 :=
  no_unbounded_admissible_frequency (by norm_num)

end Transformer.RoPE.FrequencyMatching
