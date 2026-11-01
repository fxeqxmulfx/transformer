module

public import Transformer.SequenceModel.Attention.Spectral.Geometry

/-!
# Exact feature-score realization and its source margin

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J2--J3. The weights are constructed from the complete real score table,
without knowing or assuming its winning labels. Every tied source logit is
the requested score times one positive row scale. Thus spectral score
margins transfer quantitatively to the actual normalized transformer.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Expressivity
open scoped RealInnerProductSpace

/-- The actual normalized tied source score, at the last original input token. -/
def sourceLogit (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {N : ℕ}
    (x : Fin (N + 1) → Fin cfg.vocab_size) (v : Fin cfg.vocab_size) : ℝ :=
  score cfg eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
    (Fin.last N) (encode cfg θ) x v

/-- J2: compile all class scores into actual source matrices, preserving
the entire row up to a positive scalar. No target labels enter the construction. -/
theorem exists_score_model (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ (N + 1) ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (S : (Fin (N + 1) → Fin cfg.vocab_size) → Fin cfg.vocab_size → ℝ) :
    ∃ θ : ModelParams cfg, θ.embedding = taskEmbedding cfg h1 eps e ∧
      ∃ g : (Fin (N + 1) → Fin cfg.vocab_size) → ℝ,
        (∀ x, minimumGain cfg eps S ≤ g x) ∧
        ∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v := by
  let target := normalizedScores cfg h1 e S
  obtain ⟨θ, he, hh⟩ := exists_bounded_model cfg h1 hw N hwidth heps hx e target
    (normalizedScores_norm cfg h1 e S)
  let g := fun x => normScale cfg.d_model eps (‖target x‖ ^ 2) * amplitude cfg eps / tableMass S
  refine ⟨θ, he, g, ?_, ?_⟩
  · intro x
    exact div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_right
        (normScale_lower cfg.d_model_pos heps (target x) (normalizedScores_norm cfg h1 e S x))
        (amplitude_pos cfg (by omega) heps).le) (tableMass_pos S).le
  · intro x v
    rw [sourceLogit, score, decode_encode, hh, he]
    rw [rms_of_normSq cfg.d_model eps (target x) (‖target x‖ ^ 2) rfl]
    change ⟪normScale cfg.d_model eps (‖target x‖ ^ 2) •
      ((tableMass S)⁻¹ • scoreVector cfg h1 e (S x)),
      amplitude cfg eps • merge cfg h1 (unit (e (.inl v)))⟫ = _
    rw [real_inner_smul_left, real_inner_smul_left, real_inner_smul_right, scoreVector_inner]
    dsimp only [g]
    ring

/-- J3: a spectral margin becomes an explicit source margin after the
proved positive normalization scale; both class scores are retained. -/
theorem source_margin_of_scores (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {N : ℕ}
    (S : (Fin (N + 1) → Fin cfg.vocab_size) → Fin cfg.vocab_size → ℝ)
    (g : (Fin (N + 1) → Fin cfg.vocab_size) → ℝ) (c m : ℝ)
    (hc : 0 < c) (hm : 0 < m) (hg : ∀ x, c ≤ g x)
    (hs : ∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v)
    (x : Fin (N + 1) → Fin cfg.vocab_size) (target v : Fin cfg.vocab_size)
    (hgap : m ≤ S x target - S x v) :
    c * m ≤ sourceLogit cfg θ eps xeps x target - sourceLogit cfg θ eps xeps x v := by
  rw [hs, hs, ← mul_sub]
  exact mul_le_mul (hg x) hgap hm.le (le_trans hc.le (hg x))

/-- J2: a concrete finite source-shaped family realizes any score table.
The sufficient width is exponential in the fixed input length; arbitrary
positive depths are obtained by identity padding, not by a depth advantage. -/
theorem exists_finite_score_model (p : ℕ) (hp : 0 < p) (N padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (S : (Fin (N + 1) → Fin p) → Fin p → ℝ) :
    let cfg := Grid.sequenceConfig p hp N padding extraDepth
    ∃ θ : ModelParams cfg, ∃ g : (Fin (N + 1) → Fin p) → ℝ,
      (∀ v, ‖θ.embedding v‖ = amplitude cfg eps) ∧
      (∀ x, minimumGain cfg eps S ≤ g x) ∧
      ∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v := by
  let cfg := Grid.sequenceConfig p hp N padding extraDepth
  obtain ⟨θ, he, g, hg, hs⟩ := exists_score_model cfg rfl
    (Finite.sequence_room p hp (N + 1) padding).1 N
    (Finite.sequence_room p hp (N + 1) padding).2 heps hx
    (banks p hp (3 * p ^ (N + 1) + padding) extraDepth) S
  refine ⟨θ, g, ?_, hg, hs⟩
  intro v
  simp only [he, taskEmbedding, norm_smul, Real.norm_eq_abs, LinearIsometry.norm_map, norm_unit, mul_one]
  exact abs_of_pos (amplitude_pos cfg (by
    have := (Finite.sequence_room p hp (N + 1) padding).1
    exact lt_trans (by decide : 3 < 9) this) heps)

private abbrev witnessCfg := Grid.sequenceConfig 3 (by decide) 1 0 1
private abbrev witnessBanks := banks 3 (by decide) 27 1
private def witnessScores (x : Fin 2 → Fin 3) (v : Fin 3) : ℝ :=
  if v = x 0 + x 1 then 2 else -1

example := exists_score_model witnessCfg rfl (by decide) 1 (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  witnessBanks witnessScores
example := exists_finite_score_model 3 (by decide) 1 0 1
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) witnessScores

example : ∃ θ : ModelParams witnessCfg,
    minimumGain witnessCfg (1 / 8388608) witnessScores * 3 ≤
      sourceLogit witnessCfg θ (1 / 8388608) (1 / 1000000000000) ![1, 1] 2 -
      sourceLogit witnessCfg θ (1 / 8388608) (1 / 1000000000000) ![1, 1] 0 := by
  obtain ⟨θ, g, _, hg, hs⟩ := exists_finite_score_model 3 (by decide) 1 0 1
    (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) witnessScores
  exact ⟨θ, source_margin_of_scores witnessCfg θ _ _ witnessScores g _ 3
    (minimumGain_pos _ (by decide) (by norm_num) _) (by norm_num) hg hs ![1, 1] 2 0
    (by norm_num [witnessScores])⟩

end Transformer.SequenceModel.Attention.Spectral
