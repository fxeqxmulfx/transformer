module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Duality

/-!
# Ordinary cross entropy and error-tolerant identification

Source: `review/convex/moments.md`, CM87. Ordinary full-vocabulary cross entropy
controls each observed margin. A certificate transfers imperfect margins with
their explicit weighted error, retaining every record and output class.
Neither statement assumes training convergence or observed real-valued logits.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification

open scoped BigOperators
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T N : ℕ}

/-- A per-record ordinary CE threshold implies every competing-object
margin, CM87. Its dependence on the desired margin is explicit. -/
theorem margin_of_crossEntropy_le (z : O → ℝ) (t c : O) (m : ℝ) (hc : c ≠ t)
    (hce : Landscape.Neural.crossEntropy z t ≤ Real.log (1 + Real.exp (-m))) :
    m ≤ z t - z c := by
  have hs : 1 + Real.exp (z c - z t) ≤ ∑ k, Real.exp (z k - z t) := by
    have h := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.subset_univ ({t, c} : Finset O))
      (fun k _ _ => (Real.exp_pos (z k - z t)).le)
    rw [Finset.sum_pair hc.symm] at h
    simpa only [sub_self, Real.exp_zero] using h
  have hlog := Real.log_le_log (by positivity : 0 < 1 + Real.exp (z c - z t)) hs
  rw [ConvexRecall.crossEntropy_relative] at hce
  have hexp := (Real.log_le_log_iff
    (by positivity : 0 < 1 + Real.exp (z c - z t))
    (by positivity : 0 < 1 + Real.exp (-m))).mp (hlog.trans hce)
  have he : Real.exp (z c - z t) ≤ Real.exp (-m) := by linarith
  have hm := Real.exp_le_exp.mp he
  linarith

/-- Actual raw next-object CE below the unit-margin threshold supplies
the observed consistency premise, CM87. A minibatch mean alone is insufficient. -/
theorem fits_of_record_losses (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (p : SequenceParameters O T)
    (h : ∀ i, sequenceLoss (tokens i) (labels i) p ≤ Real.log (1 + Real.exp (-1))) :
    FitsObserved tokens labels p := by
  intro i c hc
  exact margin_of_crossEntropy_le (sequenceLogits p (tokens i)) (labels i) c 1 hc (h i)

/-- Certificate transfer with explicit errors in every observed competing
margin, CM87. Self-comparison weights are zero, so no constraint is required there. -/
theorem certificate_robust (tokens : Fin N → Prefix O (T + 1)) (labels : Fin N → O)
    (query : Prefix O (T + 1)) (a b : O) (bound : ℝ) (w : Fin N → O → ℝ)
    (hw : Certificate tokens labels query a b bound w)
    (p : SequenceParameters O T) (err : Fin N → O → ℝ)
    (hp : ∀ i c, c ≠ labels i → 1 - err i c ≤ margin p (tokens i) (labels i) c) :
    bound - (∑ i, ∑ c, w i c * err i c) ≤ margin p query a b := by
  have hs : (∑ i, ∑ c, w i c * (1 - err i c)) ≤
      ∑ i, ∑ c, w i c * margin p (tokens i) (labels i) c := by
    apply Finset.sum_le_sum
    intro i hi
    apply Finset.sum_le_sum
    intro c hc
    by_cases he : c = labels i
    · subst c
      simp [hw.2.1]
    · exact mul_le_mul_of_nonneg_left (hp i c he) (hw.1 i c)
  rw [← certificate_identity tokens labels query a b w hw.2.2.2 p] at hs
  simp only [mul_sub, mul_one, Finset.sum_sub_distrib] at hs
  linarith [hw.2.2.1]

/-- A uniform observed-margin error is charged against the complete
certificate mass, CM87. This exposes amplification instead of hiding it. -/
theorem certificate_uniform_error (tokens : Fin N → Prefix O (T + 1))
    (labels : Fin N → O) (query : Prefix O (T + 1)) (a b : O) (bound : ℝ)
    (w : Fin N → O → ℝ) (hw : Certificate tokens labels query a b bound w)
    (p : SequenceParameters O T) (err : ℝ)
    (hp : ∀ i c, c ≠ labels i → 1 - err ≤ margin p (tokens i) (labels i) c) :
    bound - err * (∑ i, ∑ c, w i c) ≤ margin p query a b := by
  have h := certificate_robust tokens labels query a b bound w hw p (fun _ _ => err) hp
  simpa only [← Finset.sum_mul, mul_comm err] using h

example : (1 : ℝ) ≤ (fun c : Bool => if c then (1 : ℝ) else 0) true -
    (fun c : Bool => if c then (1 : ℝ) else 0) false := by
  apply margin_of_crossEntropy_le (fun c : Bool => if c then 1 else 0)
    true false 1 (by decide)
  simp [ConvexRecall.crossEntropy_relative]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification
