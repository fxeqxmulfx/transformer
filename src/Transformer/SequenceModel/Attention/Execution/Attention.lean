module

public import Transformer.SequenceModel.Attention.Execution.Scores

/-!
# Full attention-branch execution error

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C4;
diff_gpt `b98c665b`, `mha.py` and `block.py`. Changing scores and values,
the SDPA kernel contract, non-unit computed XSA directions, head merging,
learned projection and post-normalization are all composed explicitly.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini CRASP Stability
open scoped BigOperators

/-- C4: all recorded attention operations yield a quantitative bound against
the actual real source branch, with no assumed whole-head or whole-branch error. -/
theorem attention_error (cfg : Config) (B : BlockParams cfg) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (t : AttentionTrace cfg T) (δ τ : ℝ) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (ht : AttentionAccurate cfg B eps xeps a x t δ τ) (i : Fin T) :
    ‖t.output i - rmsNormEps eps
      (attention cfg B eps xeps a (fun j => rmsNormEps eps (x j)) i)‖ ≤
        (attentionBudget cfg B eps xeps δ τ).output := by
  have hscore := attention_scores_error cfg B heps xeps a x t δ τ hτ ht
  rcases ht with ⟨hpre, hfused, hqn, hkn, hc, hs, hq, hk, hsc, hav, hdir, hheads, hproj, hout⟩
  let b := attentionBudget cfg B eps xeps δ τ
  let avg := Normalized.sourceAverage cfg B eps a x
  let dir := fun j h => normL2 xeps (Normalized.component cfg B eps (qkvV cfg) (x j) h)
  let heads := fun j h => if j.val = 0 then 0 else avg j h - inner ℝ (avg j h) (dir j h) • dir j h
  have hve j h : ‖t.component (qkvV cfg) j h -
      Normalized.component cfg B eps (qkvV cfg) (x j) h‖ ≤ b.qkv :=
    trace_component_error cfg B eps x t δ hpre hfused _ (qkvV_injective cfg) j h
  have hae j h : ‖t.average j h - avg j h‖ ≤ b.average := by
    have hsum := average_joint_error (t.scores h)
      (fun k l => headScore cfg.head_dim eps (a k) (a l)
        (Normalized.component cfg B eps (qkvQ cfg) (x k) h)
        (Normalized.component cfg B eps (qkvK cfg) (x l) h))
      (fun k => t.component (qkvV cfg) k h)
      (fun k => Normalized.component cfg B eps (qkvV cfg) (x k) h)
      j b.score (valueBound cfg B) b.qkv (fun k hk => hscore h j k) (fun k => hve k h)
      (fun k => component_norm cfg B heps _ (qkvV_injective cfg) (x k) h)
    exact (norm_sub_le_norm_sub_add_norm_sub (t.average j h)
      (∑ k, causalWeight (t.scores h) j k • t.component (qkvV cfg) k h) (avg j h)).trans
      ((add_le_add (hav j h) hsum).trans_eq (by dsimp [b, attentionBudget]; ring))
  have hde j h : ‖t.direction j h - dir j h‖ ≤ b.direction := by
    have hn := (normL2_lipschitz xeps hxeps (t.component (qkvV cfg) j h)
      (Normalized.component cfg B eps (qkvV cfg) (x j) h)).trans
      (mul_le_mul_of_nonneg_left (hve j h) (by positivity))
    exact (norm_sub_le_norm_sub_add_norm_sub (t.direction j h)
      (normL2 xeps (t.component (qkvV cfg) j h)) (dir j h)).trans
      ((add_le_add (hdir j h) hn).trans_eq (by dsimp [b, attentionBudget]; ring))
  have hbA : 0 ≤ b.average := by dsimp [b, attentionBudget, valueBound, rmsLip]; positivity
  have hbD : 0 ≤ b.direction := by dsimp [b, attentionBudget]; positivity
  have hM : 0 ≤ valueBound cfg B := by unfold valueBound; positivity
  have hhe j h : ‖t.heads j h - heads j h‖ ≤ b.heads := by
    have hr : ‖(if j.val = 0 then 0 else t.average j h -
        inner ℝ (t.average j h) (t.direction j h) • t.direction j h) - heads j h‖ ≤
        (1 + (1 + b.direction) ^ 2) * b.average + valueBound cfg B * (2 + b.direction) * b.direction := by
      dsimp only [heads]
      split_ifs with hj
      · simp only [sub_self, norm_zero]
        positivity
      · exact xsa_pair_error (t.average j h) (avg j h) (t.direction j h) (dir j h)
          b.average b.direction (valueBound cfg B) (hae j h) (hde j h)
          (sourceAverage_norm cfg B heps a x j h) (normL2_norm_le xeps hxeps.le _)
    exact (norm_sub_le_norm_sub_add_norm_sub (t.heads j h)
      (if j.val = 0 then 0 else t.average j h -
        inner ℝ (t.average j h) (t.direction j h) • t.direction j h) (heads j h)).trans
      ((add_le_add (hheads j h) hr).trans_eq (by dsimp [b, attentionBudget]; ring))
  have hm := headMerge_dist_le cfg (t.heads i) (heads i) b.heads (hhe i)
  have hp : ‖t.projected i - B.projection (headMerge cfg (heads i))‖ ≤ b.projected :=
    linear_error B.projection (headMerge cfg (heads i)) (headMerge cfg (t.heads i))
      (t.projected i) _ δ hm (hproj i)
  exact rms_error cfg.d_model_pos heps (B.projection (headMerge cfg (heads i)))
    (t.projected i) (t.output i) b.projected δ hp (hout i)

/-- Record the exact real source attention at every operation boundary. -/
def exactAttention (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) : AttentionTrace cfg T :=
  let pre := fun i => rmsNormEps eps (x i)
  let fused := fun i => B.qkv (pre i)
  let component := fun part i h => headSlice cfg (qkvSlice cfg part (fused i)) h
  let qnorm := fun h i => rmsNormEps eps (component (qkvQ cfg) i h)
  let knorm := fun h i => rmsNormEps eps (component (qkvK cfg) i h)
  let query := fun h i => rotateHead cfg.head_dim (a i) (qnorm h i)
  let key := fun h i => rotateHead cfg.head_dim (a i) (knorm h i)
  let scores := fun h i j => inner ℝ (query h i) (key h j) / Real.sqrt (cfg.head_dim : ℝ)
  let average := fun i h => ∑ j, causalWeight (scores h) i j • component (qkvV cfg) j h
  let direction := fun i h => normL2 xeps (component (qkvV cfg) i h)
  let heads := fun i h => if i.val = 0 then 0 else average i h - inner ℝ (average i h) (direction i h) • direction i h
  let projected := fun i => B.projection (headMerge cfg (heads i))
  ⟨pre, fused, qnorm, knorm, (fun i c => Real.cos (a i c)), (fun i c => Real.sin (a i c)),
    query, key, scores, average, direction, heads, projected, fun i => rmsNormEps eps (projected i)⟩

/-- C4: arbitrary real source weights admit a trace satisfying every operation check. -/
theorem exactAttention_accurate (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    AttentionAccurate cfg B eps xeps a x (exactAttention cfg B eps xeps a x) 0 0 := by
  simp [AttentionAccurate, exactAttention, AttentionTrace.component, tableRotate_exact]

example (B : BlockParams witnessConfig) := attention_error witnessConfig B
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  (exactAttention witnessConfig B (1 / 8388608) (1 / 1000000000000)
    (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]))
  0 0 (by norm_num) (by norm_num) (exactAttention_accurate _ _ _ _ _ _) 2

example (B : BlockParams witnessConfig) := attention_scores_error witnessConfig B
  (eps := 1 / 8388608) (by norm_num) (1 / 1000000000000)
  (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4])
  (exactAttention witnessConfig B (1 / 8388608) (1 / 1000000000000)
    (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]))
  0 0 (by norm_num) (exactAttention_accurate _ _ _ _ _ _) 0 2 1

example (B : BlockParams witnessConfig) := trace_component_error witnessConfig B (1 / 8388608)
  (fun _ : Fin 3 => WithLp.toLp 2 ![1, 2, 3, 4])
  (exactAttention witnessConfig B (1 / 8388608) (1 / 1000000000000)
    (fun i : Fin 3 => fun _ => (i.val : ℝ)) (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]))
  0 (by intro i; simp [exactAttention]) (by intro i; simp [exactAttention])
  (qkvQ witnessConfig) (qkvQ_injective witnessConfig) 2 0

end Transformer.SequenceModel.Attention.Execution
