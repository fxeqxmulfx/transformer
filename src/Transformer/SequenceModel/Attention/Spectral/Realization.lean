module

public import Transformer.SequenceModel.Attention.Grid.Model

/-!
# Realize bounded vector tables in the normalized source transformer

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J1. This extends the prescribed-RoPE construction P5 from one-hot answers
to arbitrary vectors in the unit ball. It retains the actual attention,
XSA, SwiGLU, normalizers, original input tokens and source phase grid.
The construction interpolates a finite table; it does not claim efficient
character arithmetic or convergence of the source optimizer.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Expressivity Grid
open scoped RealInnerProductSpace

/-- J1: a bounded target is within the genuine normalized residual branch
radius, as in P5, without requiring a one-hot or unit-norm answer. -/
theorem reachable_bounded_target {d : ℕ} (hw : 9 < d)
    (base target context : EucSpace d) (hb : ‖base‖ = 1) (ht : ‖target‖ ≤ 1)
    (hc : ‖context‖ < 1) : ‖target - (base + context)‖ ^ 2 < (d : ℝ) := by
  have h := norm_sub_le target (base + context)
  have h' := norm_add_le base context
  have hn : ‖target - (base + context)‖ < 3 := by rw [hb] at h'; linarith
  have hw' : (9 : ℝ) < d := by exact_mod_cast hw
  nlinarith [norm_nonneg (target - (base + context))]

/-- J1: one actual source block realizes any bounded finite vector table
at the last original token, with the prescribed geometric RoPE phases. -/
theorem exists_bounded_block (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ (N + 1) ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (target : (Fin (N + 1) → Fin cfg.vocab_size) → EucSpace cfg.d_model)
    (ht : ∀ x, ‖target x‖ ≤ 1) :
    ∃ C : BlockParams cfg, ∀ x : Fin (N + 1) → Fin cfg.vocab_size,
      block cfg C eps xeps (integerAngles cfg.head_dim cfg.rope_theta)
        (fun j => merge cfg h1 (unit (e (.inl (x j))))) (Fin.last N) = target x := by
  let : NeZero cfg.vocab_size := ⟨by have := cfg.vocab_pos; omega⟩
  have hd : 1 < cfg.head_dim := by
    simpa only [Config.head_dim, h1, Nat.div_one] using (show 1 < cfg.d_model by omega)
  obtain ⟨s, hs, hs1, hcode⟩ := exists_separating_gain (sourceSlopes cfg.head_dim N)
    (sourceSlopes_injective cfg.head_dim (by omega) N) (upper := 1) (by norm_num)
  let w := exponentialWeights (sourceSlopes cfg.head_dim N) s
  obtain ⟨τ, hτ, hsmall⟩ := Finite.exists_small_normalized_context cfg.d_model cfg.d_model_pos heps
    (fun x : Fin (N + 1) → Fin cfg.vocab_size =>
      merge cfg h1 (Finite.contextCode e (x (Fin.last N)) w (fun j => x j.castSucc)))
  let D := codeDenominator cfg.head_dim s N
  have hD : D ≠ 0 := (codeDenominator_pos cfg.head_dim s N).ne'
  let B := encoderBlock cfg h1 eps s (τ * D) e ⟨0, by omega⟩
  let state := sequenceState cfg h1 eps τ e w
  let u := Finite.tokenSum cfg h1 e
  have hu (x) : u (state x) = 1 := context_common cfg h1 eps τ e _ w _
  have hi : Function.Injective (fun x => (u (state x))⁻¹ • state x) := by
    simp only [hu, inv_one, one_smul]
    exact sequenceState_injective cfg h1 heps hτ e w hcode
  have hr (x) : ‖target x - state x‖ ^ 2 < (cfg.d_model : ℝ) := by
    apply reachable_bounded_target hw
    · rw [LinearIsometry.norm_map, norm_unit]
    · exact ht x
    · exact hsmall x
  have hH : 3 * Fintype.card (Fin (N + 1) → Fin cfg.vocab_size) ≤ cfg.d_ff := by
    simpa only [Fintype.card_fun, Fintype.card_fin] using hwidth
  obtain ⟨C, hQ, hP, hff⟩ := Finite.exists_normalized_ffn cfg hH B heps state target u
    (fun x => by rw [hu]; norm_num) hi hr
  refine ⟨C, ?_⟩
  intro x
  have ha : attention cfg C eps xeps (integerAngles cfg.head_dim cfg.rope_theta)
      (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (x j)))))) (Fin.last N) =
      τ • merge cfg h1 (Finite.contextCode e (x (Fin.last N)) w (fun j => x j.castSucc)) := by
    have he : attention cfg C eps xeps (integerAngles cfg.head_dim cfg.rope_theta)
        (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (x j)))))) (Fin.last N) =
        attention cfg B eps xeps (integerAngles cfg.head_dim cfg.rope_theta)
        (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (x j)))))) (Fin.last N) := by
      simp only [attention, hQ, hP]
    rw [he]
    have henc := encoder_attention cfg h1 hd heps hx hs hs1 (τ * D) e
      (x (Fin.last N)) (fun j => x j.castSucc)
    rw [split_input] at henc
    rw [henc]
    change (τ * D) • merge cfg h1 (D⁻¹ • Finite.contextCode e (x (Fin.last N)) w _) = _
    rw [map_smul, smul_smul, mul_assoc, mul_inv_cancel₀ hD, mul_one]
  dsimp only [block]
  rw [ha]
  exact hff x

/-- J1: the actual embedding and identity residual padding preserve the
constructed vector, not only the identity of its largest coordinate. -/
theorem sequence_hidden_target (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) {N : ℕ}
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (angle : Fin (N + 1) → Fin (cfg.head_dim / 2) → ℝ) (C : BlockParams cfg)
    (target : (Fin (N + 1) → Fin cfg.vocab_size) → EucSpace cfg.d_model)
    (hC : ∀ x, block cfg C eps xeps angle
      (fun j => merge cfg h1 (unit (e (.inl (x j))))) (Fin.last N) = target x)
    (x : Fin (N + 1) → Fin cfg.vocab_size) (n : ℕ) :
    hidden cfg (Finite.sequenceModel cfg h1 eps e C) eps xeps (fun _ => angle)
      x (n + 1) (Fin.last N) = target x := by
  induction n with
  | zero =>
    simp only [hidden, cfg.n_layers_pos, ↓reduceDIte]
    change block cfg C eps xeps angle
      (fun j => rmsNormEps eps (taskEmbedding cfg h1 eps e (x j))) (Fin.last N) = _
    simp only [embedding_normalize cfg h1 hw heps]
    exact hC x
  | succ n ih =>
    rw [hidden]
    split_ifs with hn
    · have hz : n + 1 ≠ 0 := by omega
      simp only [Finite.sequenceModel, hz, ↓reduceIte, block_zero]
      exact ih
    · exact ih

/-- J1: bounded vector realization by the complete source stack, with
shared raw embeddings, arbitrary positive depth and active source RoPE. -/
theorem exists_bounded_model (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ (N + 1) ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (target : (Fin (N + 1) → Fin cfg.vocab_size) → EucSpace cfg.d_model)
    (ht : ∀ x, ‖target x‖ ≤ 1) :
    ∃ θ : ModelParams cfg, θ.embedding = taskEmbedding cfg h1 eps e ∧ ∀ x,
      hidden cfg θ eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
        x cfg.n_layers (Fin.last N) = target x := by
  obtain ⟨C, hC⟩ := exists_bounded_block cfg h1 hw N hwidth heps hx e target ht
  refine ⟨Finite.sequenceModel cfg h1 eps e C, rfl, ?_⟩
  intro x
  have hn : cfg.n_layers - 1 + 1 = cfg.n_layers := by have := cfg.n_layers_pos; omega
  have hh := sequence_hidden_target cfg h1 (by omega) heps e _ C target hC x (cfg.n_layers - 1)
  rw [hn] at hh
  simpa only [sourceAngles_integer] using hh

example := reachable_bounded_target (d := 10) (by decide)
  (EuclideanSpace.single 0 1) 0 (0 : EucSpace 10) (by simp) (by simp) (by simp)

private abbrev witnessCfg := Grid.sequenceConfig 3 (by decide) 1 0 1
private abbrev witnessBanks := banks 3 (by decide) 27 1

example := exists_bounded_block witnessCfg rfl (by decide) 1 (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  witnessBanks (fun _ => 0) (by intro x; simp)
example := exists_bounded_model witnessCfg rfl (by decide) 1 (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  witnessBanks (fun _ => 0) (by intro x; simp)

example := sequence_hidden_target witnessCfg rfl (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num)
  witnessBanks (integerAngles witnessCfg.head_dim witnessCfg.rope_theta)
  ((decode witnessCfg 0).blocks 0)
  (fun x : Fin 2 → Fin 3 => merge witnessCfg rfl (unit (witnessBanks (.inl (x (Fin.last 1))))))
  (by intro x; rw [block_zero]) ![1, 2] 4

end Transformer.SequenceModel.Attention.Spectral
