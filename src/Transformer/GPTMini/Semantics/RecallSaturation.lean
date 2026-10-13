import Transformer.GPTMini.Semantics.RecallRotaryInsert
import Transformer.GPTMini.QKNormLipschitz

/-!
# Clipped QKNorm removes genuine record amplitudes after saturation

Source: F.normalize and ordinary shared QKV in CausalMHA.forward at
f11b6e2. The actual first block has position-dependent positive RMS
and gate amplitudes. Clipped normalization removes these amplitudes
only when their projected norms reach epsilon; positivity alone is
insufficient. These local operator laws keep that condition explicit.

Above clipping, the actual normalization is exactly normL2 1 on a
base vector of norm at least one. Its error constant is then two,
independently of the implementation's small epsilon or the record's
amplitude. Actual linear insertion and RoPE preserve the base copy
error. Later raw-state bounds must discharge saturation for one
finite shared projection gain; no saturation or prepared normalized
key is yet assumed for a complete model correctness theorem.
-/

namespace Transformer.GPTMini.Semantics

/-- Actual clipped normalization cancels a positive scale only above its genuine clipping threshold.
Source: the original normL2 definition, with both actual and unit-reference divisors evaluated. -/
theorem recallNormL2_scaled {d : ℕ} (eps s : ℝ) (x : EucSpace d)
    (hs : 0 < s) (hx : 1 ≤ ‖x‖) (hclip : eps ≤ s * ‖x‖) :
    normL2 eps (s • x) = normL2 1 x := by
  have hn : ‖x‖ ≠ 0 := by linarith
  have hc : (1 / (s * ‖x‖)) * s = 1 / ‖x‖ := by
    field_simp [ne_of_gt hs, hn]
  rw [normL2, normL2, norm_smul_of_nonneg hs.le, max_eq_left hclip,
    max_eq_left hx, smul_smul, hc]

example : (0 : ℝ) < 1 ∧ 1 ≤ ‖recallRotaryCode (fun _ => 0)‖ ∧
    (1 : ℝ) ≤ 1 * ‖recallRotaryCode (fun _ => 0)‖ := by
  rw [recallRotaryCode_norm]
  norm_num

/-- Saturated queries and keys may have different positive amplitudes without changing their true scores.
Source: both independently evaluated actual QKNorm factors and the original score formula. -/
theorem recallScore_scaled {d : ℕ} (alpha eps a b : ℝ) (q k : EucSpace d)
    (ha : 0 < a) (hb : 0 < b) (hq : 1 ≤ ‖q‖) (hk : 1 ≤ ‖k‖)
    (hqa : eps ≤ a * ‖q‖) (hkb : eps ≤ b * ‖k‖) :
    score alpha eps (a • q) (b • k) = score alpha 1 q k := by
  unfold score
  rw [recallNormL2_scaled eps a q ha hq hqa, recallNormL2_scaled eps b k hb hk hkb]

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧
    1 ≤ ‖recallRotaryCode (fun _ => 0)‖ ∧ 1 ≤ ‖recallRotaryCode (fun _ => 1)‖ ∧
    (1 : ℝ) ≤ 1 * ‖recallRotaryCode (fun _ => 0)‖ ∧
    (1 : ℝ) ≤ 2 * ‖recallRotaryCode (fun _ => 1)‖ := by
  rw [recallRotaryCode_norm, recallRotaryCode_norm]
  norm_num

/-- The original real rotary operator commutes with every actual projection amplitude.
Source: apply_rope's two linear coordinate formulas in all original sixteen coordinates. -/
theorem recallRotary_smul (position s : ℝ) (x : EucSpace 16) :
    applyRope 16 10000 position (s • x) = s • applyRope 16 10000 position x := by
  ext i
  rcases h : ropeSplit 16 i with (p | p) | p <;>
    simp only [PiLp.smul_apply, applyRope_apply, ropeCoord, h, Sum.elim_inl,
      Sum.elim_inr, smul_eq_mul] <;> ring

/-- A compact softmax copy within distance one of its true norm-two symbol has norm at least one.
Source: the genuine Euclidean reverse triangle inequality, without an assumed nonzero copied direction. -/
theorem recallCopy_norm_lower (x : EucSpace 8) (digits : Fin 4 → Fin 4)
    (herr : ‖x - recallCode digits‖ ≤ 1) : 1 ≤ ‖x‖ := by
  have h := norm_sub_norm_le (recallCode digits) x
  rw [recallCode_norm, norm_sub_rev] at h
  linarith

example : ‖recallCode (fun _ => 0) - recallCode (fun _ => 0)‖ ≤ (1 : ℝ) := by
  rw [sub_self, norm_zero]
  norm_num

/-- Above clipping, different positive record gains cannot magnify the base-vector normalization error.
Source: the actual normL2 scaling law and its proved Lipschitz bound at unit clipping. -/
theorem recallNormL2_scaled_dist {d : ℕ} (eps a b : ℝ) (x y : EucSpace d)
    (ha : 0 < a) (hb : 0 < b) (hx : 1 ≤ ‖x‖) (hy : 1 ≤ ‖y‖)
    (hxa : eps ≤ a * ‖x‖) (hyb : eps ≤ b * ‖y‖) :
    ‖normL2 eps (a • x) - normL2 eps (b • y)‖ ≤ 2 * ‖x - y‖ := by
  rw [recallNormL2_scaled eps a x ha hx hxa, recallNormL2_scaled eps b y hb hy hyb]
  simpa only [div_one] using normL2_lipschitz 1 (by norm_num) x y

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧
    1 ≤ ‖recallRotaryCode (fun _ => 0)‖ ∧ 1 ≤ ‖recallRotaryCode (fun _ => 1)‖ ∧
    (1 : ℝ) ≤ 1 * ‖recallRotaryCode (fun _ => 0)‖ ∧
    (1 : ℝ) ≤ 2 * ‖recallRotaryCode (fun _ => 1)‖ := by
  rw [recallRotaryCode_norm, recallRotaryCode_norm]
  norm_num

/-- One saturated real vector is close to a unit-clipped reference with an epsilon-independent bound.
Source: the actual scaling identity followed by the generic unit-clipped Lipschitz theorem. -/
theorem recallNormL2_base_dist {d : ℕ} (eps a : ℝ) (x y : EucSpace d)
    (ha : 0 < a) (hx : 1 ≤ ‖x‖) (hclip : eps ≤ a * ‖x‖) :
    ‖normL2 eps (a • x) - normL2 1 y‖ ≤ 2 * ‖x - y‖ := by
  rw [recallNormL2_scaled eps a x ha hx hclip]
  simpa only [div_one] using normL2_lipschitz 1 (by norm_num) x y

example : (0 : ℝ) < 1 ∧ 1 ≤ ‖recallRotaryCode (fun _ => 0)‖ ∧
    (1 : ℝ) ≤ 1 * ‖recallRotaryCode (fun _ => 0)‖ := by
  rw [recallRotaryCode_norm]
  norm_num

/-- The true inserted and rotated copied key has a normalized error at most twice its unscaled compact error.
Source: faithful ordinary matrix, exact original RoPE, derived base norm and actual saturated QKNorm. -/
theorem recallRotary_normalized_error (eps a position eta : ℝ) (x : EucSpace 8)
    (digits : Fin 4 → Fin 4) (ha : 0 < a) (heta : eta ≤ 1)
    (herr : ‖x - recallCode digits‖ ≤ eta) (hclip : eps ≤ a * ‖x‖) :
    ‖normL2 eps (a • applyRope 16 10000 position (recallRotaryInsert x)) -
      normL2 1 (applyRope 16 10000 position (recallRotaryCode digits))‖ ≤ 2 * eta := by
  have hx := recallCopy_norm_lower x digits (herr.trans heta)
  have hr : 1 ≤ ‖applyRope 16 10000 position (recallRotaryInsert x)‖ := by
    rw [applyRope_isometry, recallRotaryInsert_norm]
    exact hx
  have hc : eps ≤ a * ‖applyRope 16 10000 position (recallRotaryInsert x)‖ := by
    rw [applyRope_isometry, recallRotaryInsert_norm]
    exact hclip
  have h := recallNormL2_base_dist eps a
    (applyRope 16 10000 position (recallRotaryInsert x))
    (applyRope 16 10000 position (recallRotaryCode digits)) ha hr hc
  have heq := congrArg (fun y : EucSpace 16 =>
    ‖applyRope 16 10000 position (recallRotaryInsert x) - applyRope 16 10000 position y‖)
    (recallRotaryInsert_code digits).symm
  have hd : ‖applyRope 16 10000 position (recallRotaryInsert x) -
      applyRope 16 10000 position (recallRotaryCode digits)‖ ≤ eta := by
    calc _ = ‖applyRope 16 10000 position (recallRotaryInsert x) -
          applyRope 16 10000 position (recallRotaryInsert (recallCode digits))‖ := heq
      _ = ‖x - recallCode digits‖ := recallRotaryInsert_rotated_dist position x _
      _ ≤ eta := herr
  exact h.trans (mul_le_mul_of_nonneg_left hd (by norm_num))

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    ‖recallCode (fun _ => 0) - recallCode (fun _ => 0)‖ ≤ (0 : ℝ) ∧
    (1 : ℝ) ≤ 1 * ‖recallCode (fun _ => 0)‖ := by
  rw [sub_self, norm_zero, recallCode_norm]
  norm_num

/-- The same error holds when the actual ordinary projection is applied before actual RoPE.
Source: the real operator ordering, proved rotary homogeneity and genuine clipping condition. -/
theorem recallRotary_projected_error (eps a position eta : ℝ) (x : EucSpace 8)
    (digits : Fin 4 → Fin 4) (ha : 0 < a) (heta : eta ≤ 1)
    (herr : ‖x - recallCode digits‖ ≤ eta) (hclip : eps ≤ a * ‖x‖) :
    ‖normL2 eps (applyRope 16 10000 position (a • recallRotaryInsert x)) -
      normL2 1 (applyRope 16 10000 position (recallRotaryCode digits))‖ ≤ 2 * eta := by
  rw [recallRotary_smul]
  exact recallRotary_normalized_error eps a position eta x digits ha heta herr hclip

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    ‖recallCode (fun _ => 0) - recallCode (fun _ => 0)‖ ≤ (0 : ℝ) ∧
    (1 : ℝ) ≤ 1 * ‖recallCode (fun _ => 0)‖ := by
  rw [sub_self, norm_zero, recallCode_norm]
  norm_num

/-- An exact saturated raw query retains its previously verified normalized rotary matching direction.
Source: faithful ordinary insertion, real rotary homogeneity and evaluated original QKNorm divisor. -/
theorem recallRotary_query_normalized (eps a position : ℝ) (digits : Fin 4 → Fin 4)
    (ha : 0 < a) (hclip : eps ≤ 2 * a) :
    normL2 eps (applyRope 16 10000 position (a • recallRotaryInsert (recallCode digits))) =
      normL2 1 (applyRope 16 10000 position (recallRotaryCode digits)) := by
  rw [recallRotaryInsert_code, recallRotary_smul]
  apply recallNormL2_scaled eps a _ ha
  · rw [recallRotaryCode_rotated_norm]
    norm_num
  · rw [recallRotaryCode_rotated_norm]
    linarith

example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 * 1 := by norm_num

end Transformer.GPTMini.Semantics
