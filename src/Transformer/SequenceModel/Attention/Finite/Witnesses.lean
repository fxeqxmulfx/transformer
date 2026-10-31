module

public import Transformer.SequenceModel.Attention.Finite.Existence

/-!
# Concrete witnesses for the full finite-sequence construction

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F1--F6.
These instances use the source epsilons, nonzero Q/K, a positive context
scale, an actual disjoint-bank layout and a nontrivial operand. The direct
FFN witness uses distinct input rows and a reachable unit answer.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped RealInnerProductSpace

private abbrev cfg := sequenceConfig 2 (by decide) 1 0 1
private abbrev e := banks 2 (by decide) (3 * 2 ^ 1) 1
private def eps : ℝ := 1 / 8388608
private def xeps : ℝ := 1 / 1000000000000
private def coeff : ℝ := Real.exp (1 / (2 * Real.sqrt (cfg.head_dim : ℝ)))
private def weights : EucSpace 1 := WithLp.toLp 2 (fun _ => coeff)
private def keyIndex : Fin (cfg.head_dim / 2) := ⟨0, by decide⟩

private theorem coeff_bounds : 1 < coeff ∧ coeff < Real.exp (1 / Real.sqrt (cfg.head_dim : ℝ)) := by
  have hs : 0 < Real.sqrt (cfg.head_dim : ℝ) := Real.sqrt_pos.mpr (by norm_num [Config.head_dim, modelWidth])
  constructor
  · exact Real.one_lt_exp_iff.mpr (div_pos (by norm_num) (mul_pos (by norm_num) hs))
  · apply Real.exp_lt_exp.mpr
    apply (div_lt_div_iff₀ (mul_pos (by norm_num) hs) hs).mpr
    nlinarith

private theorem weight_bounds (j : Fin 1) :
    1 < weights j ∧ weights j < Real.exp (1 / Real.sqrt (cfg.head_dim : ℝ)) := coeff_bounds

private theorem weight_injective : Function.Injective (fun b : Fin 1 → Bool => ⟪weights, bits 1 b⟫) := by
  intro a b h
  have hc : coeff ≠ 0 := ne_of_gt (lt_trans (by norm_num) coeff_bounds.1)
  have he : (if a 0 then coeff else 0) = (if b 0 then coeff else 0) := by
    simpa only [inner_bits, Fin.sum_univ_one, weights, WithLp.ofLp_toLp] using h
  funext j
  have hj : j = 0 := Subsingleton.elim _ _
  subst j
  cases ha : a 0 <;> cases hb : b 0 <;> simp_all

example := last_exp_score (d := cfg.head_dim) (by decide) (eps := eps) (by norm_num [eps])
  keyIndex weights weight_bounds 0
example := last_weight (d := cfg.head_dim) (by decide) (eps := eps) (by norm_num [eps])
  keyIndex weights weight_bounds 0
example := shifted_head_code (d := cfg.head_dim) (by decide) (eps := eps) (xeps := xeps)
  (by norm_num [eps]) (by norm_num [xeps]) keyIndex e 0 weights weight_bounds ![1]

example := qkvQ_packMaps cfg (ContinuousLinearMap.id ℝ _) (2 • ContinuousLinearMap.id ℝ _)
  (3 • ContinuousLinearMap.id ℝ _) (unit ⟨0, cfg.d_model_pos⟩)
example := qkvK_packMaps cfg (ContinuousLinearMap.id ℝ _) (2 • ContinuousLinearMap.id ℝ _)
  (3 • ContinuousLinearMap.id ℝ _) (unit ⟨0, cfg.d_model_pos⟩)
example := qkvV_packMaps cfg (ContinuousLinearMap.id ℝ _) (2 • ContinuousLinearMap.id ℝ _)
  (3 • ContinuousLinearMap.id ℝ _) (unit ⟨0, cfg.d_model_pos⟩)
example := tokenSum_unit cfg rfl e 1
example := tokenQuery_unit cfg rfl eps e keyIndex 1
example := encoder_qkv cfg rfl (eps := eps) (by norm_num [eps]) (1 / 10) e keyIndex 1
example := encoder_attention cfg rfl (by decide) (eps := eps) (xeps := xeps)
  (by norm_num [eps]) (by norm_num [xeps]) (1 / 10) e keyIndex 0 weights weight_bounds ![1]
example := context_common cfg rfl eps (1 / 10) e 0 weights ![1]
example := contextState_injective cfg rfl (eps := eps) (τ := 1 / 10)
  (by norm_num [eps]) (by norm_num) e 0 weights (context_injective e 0 weights weight_injective)

example := exists_sequence_block cfg rfl (by decide) 1 (by decide) (eps := eps) (xeps := xeps)
  (by norm_num [eps]) (by norm_num [xeps]) e 0 (fun x => x 0 + 1)
example := exists_classifier cfg rfl (by decide) 1 (by decide) (eps := eps) (xeps := xeps)
  (by norm_num [eps]) (by norm_num [xeps]) e 0 (fun x => x 0 + 1)
example := sequence_hidden cfg rfl (by decide) (eps := eps) (xeps := xeps)
  (by norm_num [eps]) (N := 1) e 0 (fun j k => j.val * (k.val + 1 : ℝ) / 7)
  ((decode cfg 0).blocks 0) (fun _ => 0)
  (by intro x; simp only [block_zero, withReadout, Fin.lastCases_last]) ![1] 7

private def ffRows (a : Fin 2) : EucSpace 4 := WithLp.toLp 2 ![(a.val : ℝ) / 10, 1, 0, 0]
private def ffCommon : EucSpace 4 →L[ℝ] ℝ := innerSL ℝ (unit 1)
private theorem ffCommon_rows (a : Fin 2) : ffCommon (ffRows a) = 1 := by
  simp [ffCommon, unit, EuclideanSpace.inner_single_left, ffRows]

example := exists_normalized_ffn (A := Fin 2) witnessConfig (by decide)
  ((decode witnessConfig 0).blocks 0) (eps := eps) (by norm_num [eps])
  ffRows (fun _ => WithLp.toLp 2 ![0, 0, 1, 0]) ffCommon (by intro a; rw [ffCommon_rows]; norm_num)
  (by
    intro a b h
    simp only [ffCommon_rows, inv_one, one_smul] at h
    have he : (a.val : ℝ) / 10 = b.val / 10 := congrArg (fun x : EucSpace 4 => x 0) h
    have hv : (a.val : ℝ) = b.val := by linarith
    apply Fin.ext
    exact_mod_cast hv)
  (by
    intro a
    fin_cases a <;> norm_num [EuclideanSpace.norm_sq_eq, ffRows, Fin.sum_univ_succ])

example := feedForward_withRidges witnessConfig ((decode witnessConfig 0).blocks 0)
  (interpolationLayout (A := Fin 2) witnessConfig (by decide))
  (fun r => ((r.1.val + r.2.val + 1 : ℝ)) • ffCommon) (fun _ => ffCommon)
  (fun r => ((r.1.val + r.2.val + 1 : ℝ)) • unit 2) (ffRows 1)

end Transformer.SequenceModel.Attention.Finite
