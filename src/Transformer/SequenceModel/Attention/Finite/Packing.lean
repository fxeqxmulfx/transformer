module

public import Transformer.SequenceModel.Attention.Finite.Phases
public import Transformer.SequenceModel.Attention.Expressivity.Context

/-!
# Fixed matrices realize nonzero rotary Q/K and token values

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F5.
Every token embedding is a unit vector after the actual embedding RMSNorm.
A fixed sum of token coordinates supplies the same nonzero query and key
at every position. The fused QKV and single-head reshape use source layouts.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

/-- Pack three genuine residual-space linear maps in the source fused QKV order. -/
def packMaps (cfg : Config) (Q K V : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model) :
    EucSpace cfg.d_model →L[ℝ] EucSpace (3 * cfg.d_model) :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 (fun i => match qkvSplit cfg i with
      | .inl (.inl c) => Q x c | .inl (.inr c) => K x c | .inr c => V x c)
    map_add' := by
      intro x y; ext i
      rcases h : qkvSplit cfg i with (c | c) | c <;> simp [h]
    map_smul' := by
      intro a x; ext i
      rcases h : qkvSplit cfg i with (c | c) | c <;> simp [h] }

/-- F5: the actual fused query slice recovers its fixed linear map. -/
theorem qkvQ_packMaps (cfg : Config) (Q K V : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model)
    (x : EucSpace cfg.d_model) : qkvSlice cfg (qkvQ cfg) (packMaps cfg Q K V x) = Q x := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inl (.inl i))) with
    | .inl (.inl c) => Q x c | .inl (.inr c) => K x c | .inr c => V x c) = Q x i
  rw [Equiv.apply_symm_apply]

/-- F5: the actual fused key slice recovers its fixed linear map. -/
theorem qkvK_packMaps (cfg : Config) (Q K V : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model)
    (x : EucSpace cfg.d_model) : qkvSlice cfg (qkvK cfg) (packMaps cfg Q K V x) = K x := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inl (.inr i))) with
    | .inl (.inl c) => Q x c | .inl (.inr c) => K x c | .inr c => V x c) = K x i
  rw [Equiv.apply_symm_apply]

/-- F5: the actual fused value slice recovers its fixed linear map. -/
theorem qkvV_packMaps (cfg : Config) (Q K V : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model)
    (x : EucSpace cfg.d_model) : qkvSlice cfg (qkvV cfg) (packMaps cfg Q K V x) = V x := by
  ext i
  change (match qkvSplit cfg ((qkvSplit cfg).symm (.inr i)) with
    | .inl (.inl c) => Q x c | .inl (.inr c) => K x c | .inr c => V x c) = V x i
  rw [Equiv.apply_symm_apply]

/-- Sum of the token-bank coordinates, one on every normalized token embedding. -/
def tokenSum (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) :
    EucSpace cfg.d_model →L[ℝ] ℝ :=
  ∑ a, innerSL ℝ (merge cfg h1 (unit (e (.inl a))))

/-- F5: the common scalar channel is constant across token identity. -/
theorem tokenSum_unit (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (a : Fin cfg.vocab_size) :
    tokenSum cfg h1 e (merge cfg h1 (unit (e (.inl a)))) = 1 := by
  rw [tokenSum, sum_apply]
  change (∑ b, ⟪merge cfg h1 (unit (e (.inl b))), merge cfg h1 (unit (e (.inl a)))⟫) = 1
  simp [LinearIsometry.inner_map_map, inner_unit, e.injective.eq_iff]

/-- Fixed Q/K map whose output is nonzero and independent of the token. -/
def tokenQuery (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model :=
  (tokenSum cfg h1 e).smulRight (merge cfg h1 (unnormalize cfg.head_dim eps (unit (ropeFst cfg.head_dim k))))

/-- Attention matrices with source normalization compensation and disjoint context bank.
The positive projection scale is chosen once for the whole finite input table. -/
def encoderBlock (cfg : Config) (h1 : cfg.n_heads = 1) (eps τ : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) : BlockParams cfg := {
  qkv := (normScale cfg.d_model eps 1)⁻¹ • packMaps cfg
    (tokenQuery cfg h1 eps e k) (tokenQuery cfg h1 eps e k) (ContinuousLinearMap.id ℝ _)
  projection := τ • ((merge cfg h1).toContinuousLinearMap.comp ((shift e).comp (slice cfg)))
  gate := 0
  value := 0
  output := 0 }

/-- F5: pre-attention RMSNorm is exactly compensated by fixed QKV weights. -/
theorem encoder_qkv (cfg : Config) (h1 : cfg.n_heads = 1) {eps : ℝ} (heps : 0 < eps)
    (τ : ℝ) (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) (a : Fin cfg.vocab_size) :
    (encoderBlock cfg h1 eps τ e k).qkv
      (rmsNormEps eps (merge cfg h1 (unit (e (.inl a))))) =
      packMaps cfg (tokenQuery cfg h1 eps e k) (tokenQuery cfg h1 eps e k)
        (ContinuousLinearMap.id ℝ _) (merge cfg h1 (unit (e (.inl a)))) := by
  rw [rms_merge_unit]
  change (normScale cfg.d_model eps 1)⁻¹ • packMaps cfg _ _ _
    (normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl a)))) = _
  rw [map_smul, smul_smul, inv_mul_cancel₀
    (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num)).ne', one_smul]

/-- F5: every actual token supplies the same raw nonzero rotary query/key. -/
theorem tokenQuery_unit (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (k : Fin (cfg.head_dim / 2)) (a : Fin cfg.vocab_size) :
    tokenQuery cfg h1 eps e k (merge cfg h1 (unit (e (.inl a)))) =
      merge cfg h1 (unnormalize cfg.head_dim eps (unit (ropeFst cfg.head_dim k))) := by
  change tokenSum cfg h1 e (merge cfg h1 (unit (e (.inl a)))) •
    merge cfg h1 (unnormalize cfg.head_dim eps (unit (ropeFst cfg.head_dim k))) = _
  rw [tokenSum_unit, one_smul]

end Transformer.SequenceModel.Attention.Finite
