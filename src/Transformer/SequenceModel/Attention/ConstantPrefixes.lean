module

public import Transformer.SequenceModel.Attention.ConstantTokens

/-!
# The stable repeated-token path applies to causal prefixes

Source: `review/residual_connections.md`, *Repeated-input numerical audit*,
C11--C13. Future tokens need not repeat. Their causal weights are zero, so
every still-constant prefix has the same block output as its first position
while the value heads remain above clipping. The implementation carries the
token-derived prefix mask across blocks and permanently disables it after a
clipped head. This is an equality of source states, not a rule for arbitrary
independent-position gradients or a floating-point clipping certificate.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini
open scoped BigOperators

/-- C11: only the visible prefix must have constant values; future values
and all normalized rotary scores are unrestricted. -/
theorem headOutput_constant_prefix {T d : ℕ} (eps xeps : ℝ)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k v : Fin T → EucSpace d)
    (i : Fin T) (c : EucSpace d) (hvalue : ∀ j ≤ i, v j = c)
    (hclip : xeps ≤ ‖c‖) : headOutput eps xeps angle q k v i = 0 := by
  let scores := fun a b => headScore d eps (angle a) (angle b) (q a) (k b)
  have hs : ∑ j, causalWeight scores i j • (v j - c) = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    by_cases hji : j ≤ i
    · rw [hvalue j hji]
      simp
    · simp [causalWeight, hji]
  rw [headOutput_centered eps xeps angle q k v i c]
  split_ifs with hi
  · rfl
  · change centeredXSA xeps (causalWeight scores i) v c (v i) = 0
    unfold centeredXSA
    rw [hs, hvalue i le_rfl]
    simp [ownRemainder, hclip]

/-- C12: source attention vanishes at the end of a constant input prefix
when its actual pre-normalized value heads are above clipping. -/
theorem attention_constant_prefix (cfg : Config) (B : BlockParams cfg)
    (eps xeps : ℝ) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) (c : EucSpace cfg.d_model)
    (hstate : ∀ j ≤ i, x j = c)
    (hclip : ∀ h, xeps ≤ ‖normalizedValueHead cfg B eps c h‖) :
    attention cfg B eps xeps angle (fun j => rmsNormEps eps (x j)) i = 0 := by
  unfold attention
  have hz (h : Fin cfg.n_heads) := headOutput_constant_prefix eps xeps angle
    (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg) (B.qkv (rmsNormEps eps (x j)))) h)
    (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg) (B.qkv (rmsNormEps eps (x j)))) h)
    (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg) (B.qkv (rmsNormEps eps (x j)))) h)
    i (normalizedValueHead cfg B eps c h)
    (by intro j hj; rw [hstate j hj]; rfl) (hclip h)
  simp_rw [hz]
  have hm : headMerge cfg (fun _ => 0) = 0 := by ext j; simp
  rw [hm, map_zero]

/-- C13: a complete Peri-RMSNorm block preserves that common prefix state
via its shared FFN path. This state equality justifies the common path;
floating-point parameter-gradient accuracy is measured separately. -/
theorem block_constant_prefix (cfg : Config) (B : BlockParams cfg)
    (eps xeps : ℝ) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) (c : EucSpace cfg.d_model)
    (hstate : ∀ j ≤ i, x j = c)
    (hclip : ∀ h, xeps ≤ ‖normalizedValueHead cfg B eps c h‖) :
    block cfg B eps xeps angle x i =
      c + rmsNormEps eps (feedForward cfg B (rmsNormEps eps c)) := by
  unfold block
  dsimp only
  rw [attention_constant_prefix cfg B eps xeps angle x i c hstate hclip,
    hstate i le_rfl]
  have hz : rmsNormEps eps (0 : EucSpace cfg.d_model) = 0 := by simp [rmsNormEps]
  simp only [hz, add_zero]

/-- C13: copying the already computed first-position result is therefore
exact on any such prefix. Other positions are outside this theorem. -/
theorem block_prefix_eq_first (cfg : Config) (B : BlockParams cfg)
    (eps xeps : ℝ) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (first i : Fin T) (hfirst : first ≤ i)
    (hstate : ∀ j ≤ i, x j = x first)
    (hclip : ∀ h, xeps ≤ ‖normalizedValueHead cfg B eps (x first) h‖) :
    block cfg B eps xeps angle x i = block cfg B eps xeps angle x first := by
  rw [block_constant_prefix cfg B eps xeps angle x i (x first) hstate hclip,
    block_constant_prefix cfg B eps xeps angle x first (x first)
      (fun j hj => hstate j (le_trans hj hfirst)) hclip]

example := headOutput_constant_prefix (T := 3) (d := 2) (1 / 8388608)
  (1 / 1000000000000) (fun j _ => (j.val : ℝ) / 7)
  (fun j => EuclideanSpace.single 0 j.val) (fun j => EuclideanSpace.single 1 j.val)
  (fun j => EuclideanSpace.single 0 (if j ≤ 1 then 1 else 7)) 1
  (EuclideanSpace.single 0 1) (by intro j hj; simp [hj]) (by norm_num)

example (B : BlockParams witnessConfig) (c : EucSpace 4) :=
  attention_constant_prefix witnessConfig B (1 / 8388608) 0 (T := 3)
    (fun j k => (j.val * (k.val + 1) : ℝ) / 7)
    (fun j => if j ≤ 1 then c else 2 • c) 1 c
    (by intro j hj; simp [hj]) (fun _ => norm_nonneg _)

example (B : BlockParams witnessConfig) (c : EucSpace 4) :=
  block_constant_prefix witnessConfig B (1 / 8388608) 0 (T := 3)
    (fun j k => (j.val * (k.val + 1) : ℝ) / 7)
    (fun j => if j ≤ 1 then c else 2 • c) 1 c
    (by intro j hj; simp [hj]) (fun _ => norm_nonneg _)

example (B : BlockParams witnessConfig) (c : EucSpace 4) :=
  block_prefix_eq_first witnessConfig B (1 / 8388608) 0 (T := 3)
    (fun j k => (j.val * (k.val + 1) : ℝ) / 7)
    (fun j => if j ≤ 1 then c else 2 • c) 0 1 (by decide)
    (by intro j hj; simp [hj]) (fun _ => norm_nonneg _)

end Transformer.SequenceModel.Attention
