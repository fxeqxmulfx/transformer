module

public import Transformer.SequenceModel.Attention.Grid.Encoder
public import Transformer.SequenceModel.Attention.Finite.Existence

/-!
# Finite full-model capacity with prescribed source RoPE and original inputs

Source: `review/flat_spectrum.md`, *Prescribed RoPE capacity*, P5--P6.
Fixed integer RoPE encodes the input, and a finite actual SwiGLU branch
decodes it. The complete normalized model with tied readout strictly
classifies every finite table at its last original token. Additional
identity blocks allow any positive finite depth. This is exact-real
capacity, not optimizer convergence or a floating-point error guarantee.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Grid

open GPTMini Expressivity
open scoped RealInnerProductSpace

/-- P5: all original input tuples are classified by one actual block, with
shared weights, fixed source phases and every source normalizer retained.
The explicit width bound is sufficient and makes no efficiency claim. -/
theorem exists_sequence_block (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ (N + 1) ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (f : (Fin (N + 1) → Fin cfg.vocab_size) → Fin cfg.vocab_size) :
    ∃ C : BlockParams cfg, ∀ x : Fin (N + 1) → Fin cfg.vocab_size,
      block cfg C eps xeps (integerAngles cfg.head_dim cfg.rope_theta)
        (fun j => merge cfg h1 (unit (e (.inl (x j))))) (Fin.last N) =
        merge cfg h1 (unit (e (.inl (f x)))) := by
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
  let target := fun x => merge cfg h1 (unit (e (.inl (f x))))
  have hu (x) : u (state x) = 1 := context_common cfg h1 eps τ e _ w _
  have hi : Function.Injective (fun x => (u (state x))⁻¹ • state x) := by
    simp only [hu, inv_one, one_smul]
    exact sequenceState_injective cfg h1 heps hτ e w hcode
  have hr (x) : ‖target x - state x‖ ^ 2 < (cfg.d_model : ℝ) := by
    apply Finite.reachable_unit_answer hw
    · rw [LinearIsometry.norm_map, norm_unit]
    · rw [LinearIsometry.norm_map, norm_unit]
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

/-- P5: the fixed integer phase table is exactly the source's enabled RoPE. -/
theorem sourceAngles_integer (cfg : Config) (T : ℕ) :
    sourceAngles cfg (fun _ => true) (fun j : Fin T => (j.val : ℝ)) =
      fun _ => integerAngles cfg.head_dim cfg.rope_theta := rfl

/-- P5: every finite table is a strict classifier of the complete source
forward pass with prescribed RoPE, all normalizers and tied readout.
Only model weights are existential; input tokens and positions are fixed. -/
theorem exists_classifier (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 9 < cfg.d_model)
    (N : ℕ) (hwidth : 3 * cfg.vocab_size ^ (N + 1) ≤ cfg.d_ff)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (f : (Fin (N + 1) → Fin cfg.vocab_size) → Fin cfg.vocab_size) :
    ∃ weights : Parameters cfg, ∀ x q, q ≠ f x →
      score cfg eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
        (Fin.last N) weights x q <
      score cfg eps xeps (sourceAngles cfg (fun _ => true) (fun j => (j.val : ℝ)))
        (Fin.last N) weights x (f x) := by
  rw [sourceAngles_integer]
  obtain ⟨C, hC⟩ := exists_sequence_block cfg h1 hw N hwidth heps hx e f
  refine ⟨encode cfg (Finite.sequenceModel cfg h1 eps e C), ?_⟩
  intro x q hq
  have hhidden := Finite.sequence_hidden cfg h1 (by omega) heps e (x (Fin.last N))
    (integerAngles cfg.head_dim cfg.rope_theta) C
    (fun z => f (Finite.withReadout (x (Fin.last N)) z))
    (fun z => hC (Finite.withReadout (x (Fin.last N)) z))
    (fun j => x j.castSucc) (cfg.n_layers - 1)
  have hn : cfg.n_layers - 1 + 1 = cfg.n_layers := by have := cfg.n_layers_pos; omega
  rw [hn, split_input] at hhidden
  have hscore (v : Fin cfg.vocab_size) :
      score cfg eps xeps (fun _ => integerAngles cfg.head_dim cfg.rope_theta) (Fin.last N)
        (encode cfg (Finite.sequenceModel cfg h1 eps e C)) x v =
      (normScale cfg.d_model eps 1 * amplitude cfg eps) * (if v = f x then 1 else 0) := by
    rw [score, decode_encode, hhidden, rms_merge_unit]
    change ⟪normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl (f x)))),
      amplitude cfg eps • merge cfg h1 (unit (e (.inl v)))⟫ = _
    rw [real_inner_smul_left, real_inner_smul_right, LinearIsometry.inner_map_map, inner_unit]
    simp only [e.injective.eq_iff, Sum.inl.injEq]
    by_cases hv : v = f x
    · simp [hv]
    · simp [hv, Ne.symm hv]
  rw [hscore, hscore]
  simp only [hq, ↓reduceIte, mul_zero, mul_one]
  exact mul_pos (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num))
    (amplitude_pos cfg (by omega) heps)

/-- Finite source FFN ratio and an exact context window for N+1 original tokens. -/
abbrev sequenceConfig (p : ℕ) (hp : 0 < p) (N padding extraDepth : ℕ) : Config :=
  { Expressivity.config p hp (3 * p ^ (N + 1) + padding) extraDepth with
    max_seq_len := N + 1
    max_seq_len_pos := by omega }

/-- P6: every task on any positive finite arity has finite weights for
the source integer-position geometric RoPE model. Width is
2(p+1)^2 + 6p^(N+1) + 2 padding, FFN width is floor((8W+1)/3), and depth
is extraDepth+1. No trained phase table or appended token is required. -/
theorem exists_finite_sequence_classifier (p : ℕ) (hp : 0 < p) (N padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (f : (Fin (N + 1) → Fin p) → Fin p) :
    ∃ weights : Parameters (sequenceConfig p hp N padding extraDepth), ∀ x q, q ≠ f x →
      score (sequenceConfig p hp N padding extraDepth) eps xeps
        (sourceAngles (sequenceConfig p hp N padding extraDepth) (fun _ => true) (fun j => (j.val : ℝ)))
        (Fin.last N) weights x q <
      score (sequenceConfig p hp N padding extraDepth) eps xeps
        (sourceAngles (sequenceConfig p hp N padding extraDepth) (fun _ => true) (fun j => (j.val : ℝ)))
        (Fin.last N) weights x (f x) := by
  exact exists_classifier (sequenceConfig p hp N padding extraDepth) rfl
    (Finite.sequence_room p hp (N + 1) padding).1 N (Finite.sequence_room p hp (N + 1) padding).2 heps hx
    (banks p hp (3 * p ^ (N + 1) + padding) extraDepth) f

example := exists_finite_sequence_classifier 2 (by decide) 0 0 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) (fun x => x 0)
example := exists_finite_sequence_classifier 3 (by decide) 2 0 0
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => x 0 * x 1 + x 2)
example := exists_finite_sequence_classifier 3 (by decide) 3 100 7
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => if x 0 = x 2 then x 1 else x 3)
example := exists_finite_sequence_classifier 97 (by decide) 2 0 4
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => x 0 ^ 5 + x 1 * x 2)
example := exists_finite_sequence_classifier 193 (by decide) 3 256 2
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun x => (x 0 + x 1) * (x 2 + x 3))

private abbrev witnessCfg := sequenceConfig 2 (by decide) 1 0 1
private abbrev witnessBanks := banks 2 (by decide) 12 1
example := exists_sequence_block witnessCfg rfl (by decide) 1 (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  witnessBanks (fun x => x 0 + x 1)
example := exists_classifier witnessCfg rfl (by decide) 1 (by decide)
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  witnessBanks (fun x => x 0 + x 1)

end Transformer.SequenceModel.Attention.Grid
