module

public import Transformer.SequenceModel.Attention.Stability.Periodic
public import Transformer.SequenceModel.Attention.Grid.Model

/-!
# Classification certificates for source RoPE approximations

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A4--A5.
The finite-context theorem preserves strict decisions of the complete
normalized source, with the same weights and depth. The quantitative readout
bound also applies when a separate numerical analysis supplies a hidden-state
error certificate. This file does not derive such a certificate for IEEE arithmetic.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini
open scoped Topology

/-- A5: a hidden-state error is amplified by at most 2/sqrt(eps) and the
actual tied embedding norm at the final normalized readout. -/
theorem readout_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x y e : EucSpace d) :
    |inner ℝ (rmsNormEps eps x) e - inner ℝ (rmsNormEps eps y) e| ≤
      (2 / Real.sqrt eps * ‖x - y‖) * ‖e‖ := by
  rw [← inner_sub_left]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_right (rmsNormEps_lipschitz eps heps hd x y) (norm_nonneg _))

/-- A5: target and competitor readout errors add. Their individual
embedding norms are retained rather than silently bounded by RMSNorm. -/
theorem classification_of_hidden_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x y target competitor : EucSpace d) (E : ℝ) (hE : ‖x - y‖ ≤ E)
    (hmargin : (2 / Real.sqrt eps * E) * (‖target‖ + ‖competitor‖) <
      inner ℝ (rmsNormEps eps x) target - inner ℝ (rmsNormEps eps x) competitor) :
    inner ℝ (rmsNormEps eps y) competitor < inner ℝ (rmsNormEps eps y) target := by
  have ht := readout_error hd heps x y target
  have hc := readout_error hd heps x y competitor
  have h0 : 0 ≤ 2 / Real.sqrt eps := by positivity
  have h1 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hE h0) (norm_nonneg target)
  have h2 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hE h0) (norm_nonneg competitor)
  have ht' := (abs_le.mp ht).2
  have hc' := (abs_le.mp hc).1
  nlinarith

/-- A4: a strict finite-context source classifier has a nonzero open
frequency tolerance simultaneously for every input tuple. The tolerance is
existential, not a numerically computed lower bound for a trained model. -/
theorem classifier_frequency_neighborhood (cfg : Config) (w : Parameters cfg)
    {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (enabled : Fin cfg.n_layers → Bool) (positions : Fin T → ℝ)
    (f : Fin (cfg.head_dim / 2) → ℝ) (i : Fin T)
    (target : (Fin T → Fin cfg.vocab_size) → Fin cfg.vocab_size)
    (hcorrect : ∀ x v, v ≠ target x →
      score cfg eps xeps (frequencyAngles cfg enabled positions f) i w x v <
        score cfg eps xeps (frequencyAngles cfg enabled positions f) i w x (target x)) :
    ∃ δ > 0, ∀ g, dist g f < δ → ∀ x v, v ≠ target x →
      score cfg eps xeps (frequencyAngles cfg enabled positions g) i w x v <
        score cfg eps xeps (frequencyAngles cfg enabled positions g) i w x (target x) := by
  apply Metric.eventually_nhds_iff.mp
  apply Filter.eventually_all.mpr
  intro x
  apply Filter.eventually_all.mpr
  intro v
  by_cases hv : v = target x
  · exact Filter.Eventually.of_forall (fun _ h => (h hv).elim)
  · have he := (continuous_frequency_score cfg w heps hxeps enabled positions x i v).continuousAt.eventually_lt
      (continuous_frequency_score cfg w heps hxeps enabled positions x i (target x)).continuousAt
      (hcorrect x v hv)
    exact he.mono (fun _ h _ => h)

/-- A4: on a fixed finite context every strict exact-source classifier
has a rational-pi frequency replacement with the same weights, width, depth
and strict decisions. These frequencies need not remain a geometric grid.
Their common rotation period is `rational_rotation_period`; the replacement
is chosen after T, so this is not a uniform language or depth-lower-bound transfer. -/
theorem source_classifier_periodic_approximation (cfg : Config) (w : Parameters cfg)
    {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (enabled : Fin cfg.n_layers → Bool) (i : Fin T)
    (target : (Fin T → Fin cfg.vocab_size) → Fin cfg.vocab_size)
    (hcorrect : ∀ x v, v ≠ target x →
      score cfg eps xeps (sourceAngles cfg enabled (fun j => j.val)) i w x v <
        score cfg eps xeps (sourceAngles cfg enabled (fun j => j.val)) i w x (target x)) :
    ∃ q : Fin (cfg.head_dim / 2) → ℚ, ∀ x v, v ≠ target x →
      score cfg eps xeps (frequencyAngles cfg enabled (fun j => j.val) (fun c => Real.pi * q c)) i w x v <
        score cfg eps xeps (frequencyAngles cfg enabled (fun j => j.val) (fun c => Real.pi * q c))
          i w x (target x) := by
  obtain ⟨δ, hδ, h⟩ := classifier_frequency_neighborhood cfg w heps hxeps enabled (fun j => j.val)
    (invFreq cfg.head_dim cfg.rope_theta) i target hcorrect
  obtain ⟨q, hq⟩ := (dense_rational_frequencies (Fin (cfg.head_dim / 2))).exists_dist_lt
    (invFreq cfg.head_dim cfg.rope_theta) hδ
  exact ⟨q, h _ (by simpa only [dist_comm] using hq)⟩

example := readout_error (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![1, 2, 3, 5]) (WithLp.toLp 2 ![1, 0, 0, 0])
example := classification_of_hidden_error (d := 1) (by decide) (eps := 1) (by norm_num)
  (WithLp.toLp 2 ![1]) (WithLp.toLp 2 ![1]) (WithLp.toLp 2 ![1]) 0 0 (by simp)
  (by norm_num [rmsNormEps, EuclideanSpace.inner_eq_star_dotProduct, dotProduct,
    EuclideanSpace.norm_sq_eq, Fin.sum_univ_succ])

private abbrev witnessCfg := Grid.sequenceConfig 2 (by decide) 1 0 1

/-- The frequency-neighborhood hypotheses hold for a nonconstant task
on every one of the four binary input pairs. -/
example : ∃ w : Parameters witnessCfg, ∃ δ > 0,
    ∀ g, dist g (invFreq witnessCfg.head_dim witnessCfg.rope_theta) < δ →
    ∀ (x : Fin 2 → Fin 2) v, v ≠ x 0 + x 1 →
      score witnessCfg (1 / 8388608) (1 / 1000000000000)
        (frequencyAngles witnessCfg (fun _ => true) (fun j => j.val) g) 1 w x v <
      score witnessCfg (1 / 8388608) (1 / 1000000000000)
        (frequencyAngles witnessCfg (fun _ => true) (fun j => j.val) g) 1 w x (x 0 + x 1) := by
  obtain ⟨w, hw⟩ := Grid.exists_finite_sequence_classifier 2 (by decide) 1 0 1
    (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
    (fun x => x 0 + x 1)
  exact ⟨w, classifier_frequency_neighborhood witnessCfg w (by norm_num) (by norm_num)
    (fun _ => true) (fun j => j.val) (invFreq witnessCfg.head_dim witnessCfg.rope_theta)
    1 (fun x => x 0 + x 1) hw⟩

/-- A nonconstant two-token task witnesses the strict-classification
hypotheses, using the existing exact-source finite-capacity construction. -/
example : ∃ w : Parameters witnessCfg, ∃ q : Fin (witnessCfg.head_dim / 2) → ℚ,
    ∀ (x : Fin 2 → Fin 2) v, v ≠ x 0 + x 1 →
      score witnessCfg (1 / 8388608) (1 / 1000000000000)
        (frequencyAngles witnessCfg (fun _ => true) (fun j => j.val) (fun c => Real.pi * q c))
        1 w x v <
      score witnessCfg (1 / 8388608) (1 / 1000000000000)
        (frequencyAngles witnessCfg (fun _ => true) (fun j => j.val) (fun c => Real.pi * q c))
        1 w x (x 0 + x 1) := by
  obtain ⟨w, hw⟩ := Grid.exists_finite_sequence_classifier 2 (by decide) 1 0 1
    (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
    (fun x => x 0 + x 1)
  exact ⟨w, source_classifier_periodic_approximation witnessCfg w (by norm_num) (by norm_num)
    (fun _ => true) 1 (fun x => x 0 + x 1) hw⟩

end Transformer.SequenceModel.Attention.Stability
