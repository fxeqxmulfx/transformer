module

public import Transformer.SequenceModel.Attention.Grid.Head
public import Transformer.SequenceModel.Attention.Finite.Encoder
public import Transformer.SequenceModel.Attention.Expressivity.Dimensions

/-!
# Fixed source matrices encode every original input tuple injectively

Source: `review/flat_spectrum.md`, *Prescribed RoPE capacity*, P3--P4.
The bounded gain is compiled into fused QKV. Every source normalizer stays
in the forward pass. The residual identifies the last original token and
the disjoint normalized context identifies its prefix, with a common
positive FFN channel even though the final label varies.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Grid

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

/-- A fixed raw query whose normalized first-pair amplitude is the chosen gain. -/
def gainQuery (cfg : Config) (h1 : cfg.n_heads = 1) (eps s : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model :=
  (Finite.tokenSum cfg h1 e).smulRight
    (merge cfg h1 (unnormalize cfg.head_dim eps (s • unit (ropeFst cfg.head_dim k))))

/-- P3: the same matrix produces the query at every token and position. -/
theorem gainQuery_unit (cfg : Config) (h1 : cfg.n_heads = 1) (eps s : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) (a : Fin cfg.vocab_size) :
    gainQuery cfg h1 eps s e k (merge cfg h1 (unit (e (.inl a)))) =
      merge cfg h1 (unnormalize cfg.head_dim eps (s • unit (ropeFst cfg.head_dim k))) := by
  change Finite.tokenSum cfg h1 e (merge cfg h1 (unit (e (.inl a)))) •
    merge cfg h1 (unnormalize cfg.head_dim eps (s • unit (ropeFst cfg.head_dim k))) = _
  rw [Finite.tokenSum_unit, one_smul]

/-- Genuine fused QKV and output matrices; no trainable positional table. -/
def encoderBlock (cfg : Config) (h1 : cfg.n_heads = 1) (eps s τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) : BlockParams cfg := {
  qkv := (normScale cfg.d_model eps 1)⁻¹ • Finite.packMaps cfg
    (gainQuery cfg h1 eps s e k) (Finite.tokenQuery cfg h1 eps e k) (ContinuousLinearMap.id ℝ _)
  projection := τ • ((merge cfg h1).toContinuousLinearMap.comp ((shift e).comp (slice cfg)))
  gate := 0
  value := 0
  output := 0 }

/-- P3: fixed QKV weights compensate the actual pre-attention RMSNorm. -/
theorem encoder_qkv (cfg : Config) (h1 : cfg.n_heads = 1) {eps : ℝ} (heps : 0 < eps)
    (s τ : ℝ) (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) (a : Fin cfg.vocab_size) :
    (encoderBlock cfg h1 eps s τ e k).qkv
      (rmsNormEps eps (merge cfg h1 (unit (e (.inl a))))) =
      Finite.packMaps cfg (gainQuery cfg h1 eps s e k) (Finite.tokenQuery cfg h1 eps e k)
        (ContinuousLinearMap.id ℝ _) (merge cfg h1 (unit (e (.inl a)))) := by
  rw [rms_merge_unit]
  change (normScale cfg.d_model eps 1)⁻¹ • Finite.packMaps cfg _ _ _
    (normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl a)))) = _
  rw [map_smul, smul_smul, inv_mul_cancel₀
    (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num)).ne', one_smul]

/-- P3: the complete source attention pipeline computes the ordered code
at the original last token, for every possible last label with shared weights. -/
theorem encoder_attention (cfg : Config) (h1 : cfg.n_heads = 1) (hd : 1 < cfg.head_dim)
    {eps xeps s : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) (hs : 0 < s) (hs1 : s < 1) (τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {N : ℕ} (last : Fin cfg.vocab_size) (x : Fin N → Fin cfg.vocab_size) :
    attention cfg (encoderBlock cfg h1 eps s τ e ⟨0, by omega⟩) eps xeps
      (integerAngles cfg.head_dim cfg.rope_theta)
      (fun j => rmsNormEps eps (merge cfg h1 (unit (e (.inl (Finite.withReadout last x j))))))
      (Fin.last N) =
      τ • merge cfg h1 ((codeDenominator cfg.head_dim s N)⁻¹ •
        Finite.contextCode e last (exponentialWeights (sourceSlopes cfg.head_dim N) s) x) := by
  have hq (j : Fin (N + 1)) := encoder_qkv cfg h1 heps s τ e ⟨0, by omega⟩ (Finite.withReadout last x j)
  simp only [attention, hq, Finite.qkvQ_packMaps, Finite.qkvK_packMaps, Finite.qkvV_packMaps,
    gainQuery_unit, Finite.tokenQuery_unit, ContinuousLinearMap.id_apply, headSlice_merge]
  change τ • merge cfg h1 (shift e (slice cfg (merge cfg h1
    (headOutput eps xeps (integerAngles cfg.head_dim cfg.rope_theta) _ _ _ (Fin.last N))))) = _
  rw [slice_merge, shifted_head_code hd cfg.rope_theta heps hx hs hs1 e last x]

/-- P4: splitting at the last token and reassembling changes no input token. -/
theorem split_input {N p : ℕ} (x : Fin (N + 1) → Fin p) :
    Finite.withReadout (x (Fin.last N)) (fun j => x j.castSucc) = x := by
  funext j
  refine Fin.lastCases ?_ (fun i => ?_) j <;>
    simp only [Finite.withReadout, Fin.lastCases_last, Fin.lastCases_castSucc]

/-- P4: every residual token coordinate records precisely the final label. -/
theorem context_coordinate (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {N : ℕ} (last q : Fin cfg.vocab_size) (w : EucSpace N) (x : Fin N → Fin cfg.vocab_size) :
    ⟪merge cfg h1 (unit (e (.inl q))), Finite.contextState cfg h1 eps τ e last w x⟫ =
      if q = last then 1 else 0 := by
  change ⟪merge cfg h1 (unit (e (.inl q))),
    merge cfg h1 (unit (e (.inl last))) + rmsNormEps eps (τ • merge cfg h1 (Finite.contextCode e last w x))⟫ = _
  rw [inner_add_right, LinearIsometry.inner_map_map, inner_unit]
  simp only [e.injective.eq_iff, Sum.inl.injEq, rmsNormEps, real_inner_smul_right,
    LinearIsometry.inner_map_map, Finite.context_orthogonal, mul_zero, add_zero]

/-- P4: summing the residual token bank is a common positive FFN channel,
uniformly across all last labels and every prefix. -/
theorem context_common (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {N : ℕ} (last : Fin cfg.vocab_size) (w : EucSpace N) (x : Fin N → Fin cfg.vocab_size) :
    Finite.tokenSum cfg h1 e (Finite.contextState cfg h1 eps τ e last w x) = 1 := by
  rw [Finite.tokenSum, sum_apply]
  change (∑ q, ⟪merge cfg h1 (unit (e (.inl q))), Finite.contextState cfg h1 eps τ e last w x⟫) = 1
  simp [context_coordinate]

/-- State at the last original token, without adding an input or a vocabulary symbol. -/
def sequenceState (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {N : ℕ} (w : EucSpace N) (x : Fin (N + 1) → Fin cfg.vocab_size) : EucSpace cfg.d_model :=
  Finite.contextState cfg h1 eps τ e (x (Fin.last N)) w (fun j => x j.castSucc)

/-- P4: the full original token tuple is recoverable after genuine positive-
epsilon RMSNorm and the attention residual, including the final label. -/
theorem sequenceState_injective (cfg : Config) (h1 : cfg.n_heads = 1)
    {eps τ : ℝ} (heps : 0 < eps) (hτ : 0 < τ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {N : ℕ} (w : EucSpace N)
    (hw : Function.Injective (fun b : Fin N → Bool => ⟪w, Finite.bits N b⟫)) :
    Function.Injective (sequenceState cfg h1 eps τ e w) := by
  intro x y h
  have hl : x (Fin.last N) = y (Fin.last N) := by
    have hc := congrArg (fun v => ⟪merge cfg h1 (unit (e (.inl (x (Fin.last N))))), v⟫) h
    change ⟪_, Finite.contextState cfg h1 eps τ e (x (Fin.last N)) w _⟫ =
      ⟪_, Finite.contextState cfg h1 eps τ e (y (Fin.last N)) w _⟫ at hc
    rw [context_coordinate, context_coordinate] at hc
    by_contra hn
    simp [hn] at hc
  have hp : (fun j : Fin N => x j.castSucc) = (fun j : Fin N => y j.castSucc) := by
    apply Finite.contextState_injective cfg h1 heps hτ e (x (Fin.last N)) w
      (Finite.context_injective e _ w hw)
    simpa only [sequenceState, hl] using h
  funext j
  exact Fin.lastCases hl (fun i => congrFun hp i) j

example := split_input ![(1 : Fin 3), 0, 2, 1]

private abbrev witnessCfg := Expressivity.config 2 (by decide) 12 1
private abbrev witnessBanks := banks 2 (by decide) 12 1
private def eps : ℝ := 1 / 8388608
private def xeps : ℝ := 1 / 1000000000000
private def keyIndex : Fin (witnessCfg.head_dim / 2) := ⟨0, by decide⟩
private def weights : EucSpace 1 := exponentialWeights (sourceSlopes witnessCfg.head_dim 1) (1 / 2)

private theorem weight_injective :
    Function.Injective (fun b : Fin 1 → Bool => ⟪weights, Finite.bits 1 b⟫) := by
  intro a b h
  have hc : weights 0 ≠ 0 := (Real.exp_pos _).ne'
  have he : (if a 0 then weights 0 else 0) = (if b 0 then weights 0 else 0) := by
    simpa only [Finite.inner_bits, Fin.sum_univ_one] using h
  funext j
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  cases ha : a 0 <;> cases hb : b 0 <;> simp_all

example := gainQuery_unit witnessCfg rfl eps (1 / 2) witnessBanks keyIndex 1
example := encoder_qkv witnessCfg rfl (eps := eps) (by norm_num [eps])
  (1 / 2) (1 / 10) witnessBanks keyIndex 1
example := encoder_attention witnessCfg rfl (by decide) (eps := eps) (xeps := xeps) (s := 1 / 2)
  (by norm_num [eps]) (by norm_num [xeps]) (by norm_num) (by norm_num) (1 / 10) witnessBanks 1 ![0]
example := context_coordinate witnessCfg rfl eps (1 / 10) witnessBanks 1 0 weights ![0]
example := context_coordinate witnessCfg rfl eps (1 / 10) witnessBanks 1 1 weights ![0]
example := context_common witnessCfg rfl eps (1 / 10) witnessBanks 1 weights ![0]
example := sequenceState_injective witnessCfg rfl (eps := eps) (τ := 1 / 10)
  (by norm_num [eps]) (by norm_num) witnessBanks weights weight_injective

end Transformer.SequenceModel.Attention.Grid
