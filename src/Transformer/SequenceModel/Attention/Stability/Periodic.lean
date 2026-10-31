module

public import Transformer.SequenceModel.Attention.Stability.Model
public import Transformer.CRASP.PositionalPeriod
public import Mathlib.Topology.Algebra.Order.Archimedean

/-!
# Periodic approximations on a fixed finite context

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A4;
arXiv:2506.16055v3, Appendix F, `sec:rope_pes`. Rational multiples of pi
give periodic rotations. The approximating frequencies below are selected
after the context length and accuracy, unlike a uniform language simulation.
All source weights, width, depth, normalizers and original tokens stay fixed.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini
open scoped BigOperators Topology

/-- A frequency-parameterized version of the actual source phase table,
with its original per-layer enabling flags and supplied positions. -/
def frequencyAngles (cfg : Config) {T : ℕ} (enabled : Fin cfg.n_layers → Bool)
    (positions : Fin T → ℝ) (f : Fin (cfg.head_dim / 2) → ℝ)
    (l : Fin cfg.n_layers) (i : Fin T) (c : Fin (cfg.head_dim / 2)) : ℝ :=
  if enabled l then positions i * f c else 0

/-- A4: choosing the geometric frequencies recovers the exact source phases. -/
theorem frequencyAngles_source (cfg : Config) {T : ℕ} (enabled : Fin cfg.n_layers → Bool)
    (positions : Fin T → ℝ) :
    frequencyAngles cfg enabled positions (invFreq cfg.head_dim cfg.rope_theta) =
      sourceAngles cfg enabled positions := rfl

/-- A4: finite vectors of rational multiples of pi are dense in real
frequency space. This does not claim that the geometric grid is rational. -/
theorem dense_rational_frequencies (K : Type*) :
    DenseRange (fun q : K → ℚ => fun c => Real.pi * (q c : ℝ)) := by
  have hs : Function.Surjective (fun x : ℝ => Real.pi * x) := by
    intro x
    refine ⟨x / Real.pi, ?_⟩
    field_simp
  have hd : DenseRange (fun q : ℚ => Real.pi * (q : ℝ)) :=
    hs.denseRange.comp Rat.denseRange_cast (continuous_const.mul continuous_id)
  exact DenseRange.piMap (fun _ => hd)

/-- A4: the actual half-split rotations at rational-pi frequencies have a
positive common integer period, valid at every natural position. -/
theorem rational_rotation_period (d : ℕ) (q : Fin (d / 2) → ℚ) :
    ∃ M : ℕ, 0 < M ∧ ∀ (i : ℕ) (x : EucSpace d),
      rotateHead d (fun c => (i : ℝ) * (Real.pi * q c)) x =
        rotateHead d (fun c => ((i % M : ℕ) : ℝ) * (Real.pi * q c)) x := by
  classical
  let D := ∏ c, (q c).den
  have hD : 0 < D := Finset.prod_pos (fun c _ => (q c).den_pos)
  refine ⟨2 * D, by omega, ?_⟩
  intro i x
  have ht (c : Fin (d / 2)) := CRASP.rational_full_turn (q c) D
    (Finset.dvd_prod_of_mem (fun c => (q c).den) (Finset.mem_univ c))
  have htrig (c : Fin (d / 2)) :
      Real.cos ((i : ℝ) * (Real.pi * q c)) = Real.cos (((i % (2 * D) : ℕ) : ℝ) * (Real.pi * q c)) ∧
      Real.sin ((i : ℝ) * (Real.pi * q c)) = Real.sin (((i % (2 * D) : ℕ) : ℝ) * (Real.pi * q c)) := by
    obtain ⟨z, hz⟩ := ht c
    exact CRASP.trig_mod_period _ _ z hz i
  ext c
  rcases h : ropeSplit d c with (k | k) | j
  · simp only [rotateHead, PiLp.toLp_apply, h, rotate, (htrig k).1, (htrig k).2]
  · simp only [rotateHead, PiLp.toLp_apply, h, rotate, (htrig k).1, (htrig k).2]
  · simp only [rotateHead, PiLp.toLp_apply, h]

/-- A4: complete source scores vary continuously with the shared frequency
vector, through every layer, for any fixed finite context. -/
theorem continuous_frequency_score (cfg : Config) (w : Parameters cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (enabled : Fin cfg.n_layers → Bool) (positions : Fin T → ℝ)
    (x : Fin T → Fin cfg.vocab_size) (i : Fin T) (v : Fin cfg.vocab_size) :
    Continuous (fun f => score cfg eps xeps (frequencyAngles cfg enabled positions f) i w x v) := by
  apply continuous_score cfg w heps hxeps _ _ x i v
  intro l j c
  unfold frequencyAngles
  split_ifs <;> fun_prop

/-- A4: a single rational-pi frequency vector approximates all logits,
all token tuples and all readout positions in a prescribed finite context.
The approximation changes frequencies only. Its denominator can depend on
T, the weights, epsilons and requested accuracy; no uniform period is asserted. -/
theorem exists_rational_logit_approximation (cfg : Config) (w : Parameters cfg)
    {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (enabled : Fin cfg.n_layers → Bool) (positions : Fin T → ℝ)
    (f : Fin (cfg.head_dim / 2) → ℝ) (η : ℝ) (hη : 0 < η) :
    ∃ q : Fin (cfg.head_dim / 2) → ℚ, ∀ x i v,
      |score cfg eps xeps (frequencyAngles cfg enabled positions (fun c => Real.pi * q c)) i w x v -
        score cfg eps xeps (frequencyAngles cfg enabled positions f) i w x v| < η := by
  have he : ∀ᶠ g in 𝓝 f, ∀ x i v,
      |score cfg eps xeps (frequencyAngles cfg enabled positions g) i w x v -
        score cfg eps xeps (frequencyAngles cfg enabled positions f) i w x v| < η := by
    apply Filter.eventually_all.mpr
    intro x
    apply Filter.eventually_all.mpr
    intro i
    apply Filter.eventually_all.mpr
    intro v
    have hc := continuous_frequency_score cfg w heps hxeps enabled positions x i v
    exact (hc.sub continuous_const).abs.continuousAt.eventually_lt continuousAt_const (by simpa using hη)
  obtain ⟨δ, hδ, h⟩ := Metric.eventually_nhds_iff.mp he
  obtain ⟨q, hq⟩ := (dense_rational_frequencies (Fin (cfg.head_dim / 2))).exists_dist_lt f hδ
  exact ⟨q, h (by simpa only [dist_comm] using hq)⟩

example := rational_rotation_period 4 (fun c => if c = 0 then 1 / 7 else 1 / 11)
example := continuous_frequency_score witnessConfig 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun _ => true) (fun i : Fin 5 => i.val) (fun _ => 1) 4 2
example := exists_rational_logit_approximation witnessConfig
  (WithLp.toLp 2 (fun _ => 1)) (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) (fun _ => true) (fun i : Fin 5 => i.val)
  (invFreq witnessConfig.head_dim witnessConfig.rope_theta) (1 / 1000) (by norm_num)

end Transformer.SequenceModel.Attention.Stability
