import Transformer.GPTMini.Convex.Structured.MarkovTeacher
import Transformer.GPTMini.Convex.Structured.MarkovEmissions

/-!
# Finite joint state and output confidence

Source: the actual causal path confidence at 842cc59, normalized joint
emissions at 33e99c8 and finite sharp-row laws at d68381b. All tables in
the learned inference model remain free. Explicit finite assignments
selected from data semantics favor a reference state path and its output
channels, with a quantified uniform error depending linearly on length,
state count and group/channel counts. No history table is materialized.

The full joint initial/path/output probability is bounded directly, as
is the actual compact encoder's reference endpoint. Reference rules and
labels select parameter assignments only; inference still propagates
all learned state rows and mixes every learned emission. Raw Basis
semantic rules, compact shared-slot realization and genuine tied integer
readout must still connect these local capacity bounds to a full solver.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

/-- A finite product of actual probability factors loses at most the sum of their deficits.
Source: induction using (1-p)*(1-q) >= 0 on true factors in [0,1], without assuming a deterministic joint path. -/
theorem probability_product_lower {I : Type*} (s : Finset I) (p : I → ℝ)
    (h : ∀ i ∈ s, 0 ≤ p i ∧ p i ≤ 1) : 1 - (∑ i ∈ s, (1 - p i)) ≤ ∏ i ∈ s, p i := by
  induction s using Finset.induction_on with
  | empty => simp only [Finset.sum_empty, Finset.prod_empty, sub_zero, le_rfl]
  | @insert i s hi ih =>
      have hs : ∀ j ∈ s, 0 ≤ p j ∧ p j ≤ 1 := fun j hj => h j (Finset.mem_insert_of_mem hj)
      have hp := h i (Finset.mem_insert_self i s)
      have hprod := Finset.prod_le_one₀ (fun j hj => (hs j hj).1) (fun j hj => (hs j hj).2)
      have hcross := mul_nonneg (sub_nonneg.mpr hp.2) (sub_nonneg.mpr hprod)
      have htail := ih hs
      rw [Finset.sum_insert hi, Finset.prod_insert hi]
      nlinarith

example : ∀ i : Fin 2, i ∈ Finset.univ → 0 ≤ (fun _ => (1 / 2 : ℝ)) i ∧
    (fun _ => (1 / 2 : ℝ)) i ≤ 1 := by
  intro i hi
  norm_num

variable {S A H D : Type*} [Fintype S] [Nonempty S] [Fintype H] [Fintype D] [Nonempty D]

omit [Nonempty D] in
/-- The actual complete value assignment probability is exactly a product of its small normalized channel rows.
Source: genuine factorial energy/normalizer and finite product-of-ratios identity, with no independent substitute value model. -/
theorem channelProbability_rows (potential : H → D → ℝ) (assignment : H → D) :
    channelProbability potential assignment = ∏ h, stateRow (potential h) (assignment h) := by
  unfold channelProbability
  rw [channelEnergy_exp]
  unfold channelPartition
  rw [← Finset.prod_div_distrib]
  rfl

/-- Finite freely trainable conditional emission logits favor the data state's output-channel labels.
Source: an explicit capacity weight assignment; labels are not arguments of learned inference at prediction time. -/
def sharpReferenceEmission (labels : S → H → D) (gain : ℝ) (state : S) (h : H) : D → ℝ :=
  sharpRowLogits (labels state h) gain

omit [Fintype S] [Nonempty S] in
/-- The actual learned conditional output distribution retains a quantified amount of the data-channel assignment's mass.
Source: true positive normalized rows, finite sharp-logit assignments and exact compact product contraction. -/
theorem sharpReference_value (labels : S → H → D) (state : S) (gain : ℝ) :
    1 - (Fintype.card H : ℝ) * ((Fintype.card D : ℝ) * Real.exp (-gain)) ≤
      channelProbability (sharpReferenceEmission labels gain state) (labels state) := by
  rw [channelProbability_rows]
  have hfactor : ∀ h : H, 0 ≤ stateRow (sharpReferenceEmission labels gain state h) (labels state h) ∧
      stateRow (sharpReferenceEmission labels gain state h) (labels state h) ≤ 1 :=
    fun _ => ⟨(stateRow_pos _ _).le, stateRow_le_one _ _⟩
  have hlower := probability_product_lower Finset.univ
    (fun h => stateRow (sharpReferenceEmission labels gain state h) (labels state h)) (fun h _ => hfactor h)
  have hdeficit := Finset.sum_le_sum (fun h (_ : h ∈ Finset.univ) =>
    show 1 - stateRow (sharpReferenceEmission labels gain state h) (labels state h) ≤
      (Fintype.card D : ℝ) * Real.exp (-gain) from by
        have hrow := stateRow_sharp (labels state h) gain
        change 1 - stateRow (sharpRowLogits (labels state h) gain) (labels state h) ≤ _
        linarith)
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at hdeficit
  linarith

/-- Finite actual learned initial/transition tables give a quantitative compact encoder reference-state mass bound.
Source: genuine initial/path probability inclusion, derived chronological reference endpoints and finite confidence propagation. -/
theorem sharpReference_run (rule : A → S → S) (start : S) (tokens : List A) (gain : ℝ) :
    1 - ((tokens.length : ℝ) + 1) * ((Fintype.card S : ℝ) * Real.exp (-gain)) ≤
      markovRun (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens (referenceRun rule start tokens) := by
  have hmodel := markovRun_path_le (sharpRowLogits start gain) (sharpReferenceTable rule gain) start tokens
    (referencePath rule start tokens)
  rw [referencePath_end] at hmodel
  have hi := stateRow_sharp start gain
  have ht := sharpReference_path rule start tokens gain
  have hiu := stateRow_le_one (sharpRowLogits start gain) start
  have htu := conditionalStatePath_le_one (sharpReferenceTable rule gain) start tokens (referencePath rule start tokens)
  have hcross := mul_nonneg (sub_nonneg.mpr hiu) (sub_nonneg.mpr htu)
  nlinarith

/-- The same actual full initial/path/output configuration keeps mass with a linear finite error budget.
Source: true causal path and channel contraction, independently derived finite initial/transition/value confidence, and positive normalized factor inequalities. -/
theorem sharpReference_joint (rule : A → S → S) (labels : S → H → D) (start : S)
    (tokens : List A) (gain : ℝ) :
    1 - (((tokens.length : ℝ) + 1) * (Fintype.card S : ℝ) +
      (Fintype.card H : ℝ) * (Fintype.card D : ℝ)) * Real.exp (-gain) ≤
      markovJoint (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
        (sharpReferenceEmission labels gain) (start, (referencePath rule start tokens, labels (referenceRun rule start tokens))) := by
  unfold markovJoint
  rw [referencePath_end]
  have hi := stateRow_sharp start gain
  have ht := sharpReference_path rule start tokens gain
  have hv := sharpReference_value labels (referenceRun rule start tokens) gain
  have hiu := stateRow_le_one (sharpRowLogits start gain) start
  have htu := conditionalStatePath_le_one (sharpReferenceTable rule gain) start tokens (referencePath rule start tokens)
  have hcu := mul_nonneg (sub_nonneg.mpr hiu) (sub_nonneg.mpr htu)
  have hpu : stateRow (sharpRowLogits start gain) start *
      conditionalStatePath (sharpReferenceTable rule gain) start tokens (referencePath rule start tokens) ≤ 1 := by
    have h := mul_le_mul_of_nonneg_left htu (stateRow_pos (sharpRowLogits start gain) start).le
    rw [mul_one] at h
    exact h.trans hiu
  have hvu : channelProbability (sharpReferenceEmission labels gain (referenceRun rule start tokens))
      (labels (referenceRun rule start tokens)) ≤ 1 := by
    rw [channelProbability_rows]
    exact Finset.prod_le_one₀ (fun _ _ => (stateRow_pos _ _).le) (fun _ _ => stateRow_le_one _ _)
  have hcv := mul_nonneg (sub_nonneg.mpr hpu) (sub_nonneg.mpr hvu)
  nlinarith

/-- The proposed compact dimensions have one finite joint-confidence budget throughout the largest Basis context.
Source: actual six-state/five-group/four-channel model and the raw length cap 128, with reference semantics still independently required. -/
theorem sharpReference_context_bound (rule : A → Fin 6 → Fin 6) (labels : Fin 6 → Fin 5 → Fin 4)
    (start : Fin 6) (tokens : List A) (gain : ℝ) (hT : tokens.length ≤ 128) :
    1 - 794 * Real.exp (-gain) ≤ markovJoint (sharpRowLogits start gain) (sharpReferenceTable rule gain)
      tokens (sharpReferenceEmission labels gain)
      (start, (referencePath rule start tokens, labels (referenceRun rule start tokens))) := by
  have h := sharpReference_joint rule labels start tokens gain
  norm_num only [Fintype.card_fin] at h
  have hTr : (tokens.length : ℝ) ≤ 128 := by exact_mod_cast hT
  have hcoef : ((tokens.length : ℝ) + 1) * 6 + 20 ≤ 794 := by linarith
  have herr := mul_le_mul_of_nonneg_right hcoef (Real.exp_pos (-gain)).le
  linarith

example : ([0, 1, 2] : List (Fin 3)).length ≤ 128 := by decide

/-- Actual six-state/five-group/four-channel inference admits finite joint confidence on a real three-token word.
Source: the proposed compact dimensions and the same actual freely learned causal value model, not hard reference inference. -/
example (rule : Fin 3 → Fin 6 → Fin 6) (labels : Fin 6 → Fin 5 → Fin 4) :
    1 - 44 * Real.exp (-(12 : ℝ)) ≤ markovJoint (sharpRowLogits (0 : Fin 6) 12)
      (sharpReferenceTable rule 12) [0, 1, 2] (sharpReferenceEmission labels 12)
      (0, (referencePath rule 0 [0, 1, 2], labels (referenceRun rule 0 [0, 1, 2]))) := by
  have h := sharpReference_joint rule labels 0 [0, 1, 2] 12
  norm_num only [List.length_cons, List.length_nil, Fintype.card_fin] at h
  exact h

end
end Transformer.GPTMini.Convex.Structured
