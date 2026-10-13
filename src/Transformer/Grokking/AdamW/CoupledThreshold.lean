import Transformer.Grokking.AdamW.CoupledFirstStep

/-!
# Small-component growth under the actual fresh AdamW update

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition. This is the explicitly specialized bilinear binary CE,
combined with native lab AdamW at 91bb895, not the paper's claimed model.
The denominator comes from actual first-step adaptive normalization.

Continuity supplies a positive interval of component amplitudes that grow
exactly below the epsilon-dependent decay threshold. A positive growing
amplitude exists iff decay is below that threshold. Exactly zero remains
fixed, and these growing states were already correctly classified. The
gain can be confidence alone. This is a fresh-buffer finite-dimensional
criterion, not delayed generalization or a thermodynamic limit; persistent
moments in the actual GPTMini trajectory remain a separate question.

The CE here has unit weight. The actual lab objective averages answer
and EOS losses and shares parameters between those predictions. Its
gradient scale and coupling are therefore not identified with this
binary specialization, and the derived coefficient is not asserted
for that training objective or its archived moment states.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.Composition Transformer.Grokking.NaiveLoss

/-- Differentiate the denominator actually obtained from the fresh
adaptive update. Source: native AdamW at 91bb895 and the explicit
arXiv:2301.05217v1 appendix-inspired bilinear CE specialization. -/
theorem coupled_first_denominator_deriv (eps s : ℝ) :
    HasDerivAt (fun u : ℝ => u + eps * (Real.exp (u ^ 2) + 1))
      (1 + 2 * eps * s * Real.exp (s ^ 2)) s := by
  have h := (hasDerivAt_id s).add (((((hasDerivAt_id s).pow 2).exp).add_const 1).const_mul eps)
  convert h using 1
  · rfl
  · dsimp
    ring

/-- Actual continuity at the all-absent state, without supplying a
small-amplitude inequality as an all-time premise. Source: the denominator
derived from native AdamW in the arXiv:2301.05217v1-inspired model. -/
theorem coupled_first_denominator_continuous (eps : ℝ) :
    ContinuousAt (fun s : ℝ => s + eps * (Real.exp (s ^ 2) + 1)) 0 :=
  (coupled_first_denominator_deriv eps 0).continuousAt

/-- Strict decay threshold gives an actual neighborhood of positive
growing states. Source: native AdamW at 91bb895 applied to the scalar
composition model; the neighborhood follows from continuity and the
computed zero-amplitude denominator, not a prescribed favorable path. -/
theorem coupled_first_near_origin_growth (b1 b2 eps decay eta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta)
    (hd : decay < 1 / (2 * eps)) :
    ∃ radius : ℝ, 0 < radius ∧ ∀ s : ℝ, 0 < s → s < radius →
      s < (coupledFirstPoint b1 b2 eps decay eta s s).1 := by
  have hc := (coupled_first_denominator_continuous eps).const_mul decay
  have hh := (lt_div_iff₀ (show 0 < 2 * eps by positivity)).mp hd
  have hz : decay * (0 + eps * (Real.exp ((0 : ℝ) ^ 2) + 1)) < (1 : ℝ) := by
    norm_num
    nlinarith
  have hev := hc.eventually_lt (continuousAt_const : ContinuousAt (fun _ : ℝ => (1 : ℝ)) 0) hz
  obtain ⟨radius, hr, hball⟩ := Metric.eventually_nhds_iff.mp hev
  refine ⟨radius, hr, ?_⟩
  intro s hs hsmall
  apply (coupled_first_growth_iff b1 b2 eps decay eta s h1 h2 he heta hs).mpr
  apply hball
  rw [Real.dist_eq, sub_zero, abs_of_pos hs]
  exact hsmall

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧
    (1 / 10 : ℝ) < 1 / (2 * (1 / 100000000 : ℝ)) := by norm_num

/-- Existence of some positive growing amplitude is exactly the
epsilon-dependent decay criterion. Source: native AdamW at 91bb895 in
the explicit arXiv:2301.05217v1 composition specialization; amplitude
existence is not the existence of a newly correct task decision. -/
theorem coupled_first_positive_growth_exists_iff (b1 b2 eps decay eta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta) :
    (∃ s : ℝ, 0 < s ∧ s < (coupledFirstPoint b1 b2 eps decay eta s s).1) ↔
      decay < 1 / (2 * eps) := by
  constructor
  · rintro ⟨s, hs, hg⟩
    by_contra h
    have hd : 1 / (2 * eps) ≤ decay := by linarith
    have hn := coupled_first_no_growth_above_threshold b1 b2 eps decay eta s h1 h2 he heta hs hd
    linarith
  · intro hd
    obtain ⟨radius, hr, hg⟩ := coupled_first_near_origin_growth b1 b2 eps decay eta h1 h2 he heta hd
    refine ⟨radius / 2, by positivity, ?_⟩
    exact hg _ (by positivity) (by linarith)

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) := by norm_num

/-- The neighborhood also gives strictly lower actual CE for all its
positive components. Source: the appendix confidence/size discussion
of arXiv:2301.05217v1, with the fresh native AdamW deviation retained. -/
theorem coupled_first_near_origin_lowers_CE (b1 b2 eps decay eta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta)
    (hd : decay < 1 / (2 * eps)) :
    ∃ radius : ℝ, 0 < radius ∧ ∀ s : ℝ, 0 < s → s < radius →
      coupledLoss (coupledFirstPoint b1 b2 eps decay eta s s).1
        (coupledFirstPoint b1 b2 eps decay eta s s).2 < coupledLoss s s := by
  obtain ⟨radius, hr, hg⟩ := coupled_first_near_origin_growth b1 b2 eps decay eta h1 h2 he heta hd
  refine ⟨radius, hr, ?_⟩
  intro s hs hsmall
  apply coupled_first_growth_lowers_CE b1 b2 eps decay eta s h1 h2 he heta hs
  exact (coupled_first_growth_iff b1 b2 eps decay eta s h1 h2 he heta hs).mp (hg s hs hsmall)

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧
    (1 / 10 : ℝ) < 1 / (2 * (1 / 100000000 : ℝ)) := by norm_num

/-- Already correct binary decisions stay correct under growing positive
components. Source: Prieto et al., arXiv:2501.04697v1 §4.2, confidence
versus decisions, applied to the actual arXiv:2301.05217v1-inspired CE.
This explains why the amplitude threshold is not a grokking detector. -/
theorem coupled_first_growth_preserves_correct_decision (b1 b2 eps decay eta s : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < eta) (hs : 0 < s)
    (hg : decay * (s + eps * (Real.exp (s ^ 2) + 1)) < 1) :
    StrictCorrect (fun k : Fin 2 => if k = 0 then s * s else 0) 0 ∧
      StrictCorrect (fun k : Fin 2 => if k = 0 then
        (coupledFirstPoint b1 b2 eps decay eta s s).1 *
        (coupledFirstPoint b1 b2 eps decay eta s s).2 else 0) 0 := by
  have hi := (coupled_first_growth_iff b1 b2 eps decay eta s h1 h2 he heta hs).mpr hg
  rw [coupled_aligned_first_point b1 b2 eps decay eta s h1 h2 he hs] at hi ⊢
  dsimp only at hi ⊢
  constructor <;> intro k hk <;> fin_cases k <;> norm_num at *
  · nlinarith
  · nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧ 0 < (1 : ℝ) ∧
    (0 : ℝ) * (1 + 1 * (Real.exp (1 ^ 2) + 1)) < 1 := by norm_num

/-- The L2 origin-curvature coefficient one-half does not prevent native
fresh growth at the experimental epsilon/betas/rate. Source comparison:
the arXiv:2301.05217v1 appendix size argument, versus PyTorch 2.14.1
decoupled AdamW at 91bb895. The coefficient has different semantics. -/
theorem coupled_first_half_decay_growth :
    ∃ s : ℝ, 0 < s ∧
      s < (coupledFirstPoint (9 / 10) (49 / 50) (1 / 100000000) (1 / 2) (1 / 1000) s s).1 := by
  apply (coupled_first_positive_growth_exists_iff _ _ _ _ _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)).mpr
  norm_num

end Transformer.Grokking.AdamW
