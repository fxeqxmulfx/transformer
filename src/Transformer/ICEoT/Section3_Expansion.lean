/-
# IC-EoT: signed affine expressivity of the expanded input

arXiv:2603.22095v2, §2.2.2 and §3.2, Eq. (13). A negative copy of an
original coordinate allows its affine coefficient to have either sign,
while every coefficient acting on expanded coordinates is non-negative.
This explains decreasing physical relationships without claiming arbitrary
non-convex expressivity.
-/

import Transformer.ICEoT.Section3_Encoder

noncomputable section

namespace Transformer.ICEoT

/-- A signed weight represented by two non-negative coefficients;
§3.2, Eqs. (13), (14). -/
def signedWeights {I O : Type*} (W : O → I → ℝ) : O → (Bool × I) → ℝ :=
  fun o r => if r.1 then relu (-W o r.2) else relu (W o r.2)

/-- The lifted weights meet the non-negative embedding restriction;
§3.2, Eq. (14). -/
theorem signedWeights_nonnegative {I O : Type*} (W : O → I → ℝ) :
    Nonnegative (signedWeights W) := by
  intro o r
  unfold signedWeights
  split <;> exact relu_conditions.2.2 _

/-- A scalar coefficient decomposes into its positive and negative parts;
§3.2, Eq. (13), explaining the expanded representation. -/
theorem relu_signed_difference (w : ℝ) : relu w - relu (-w) = w := by
  by_cases hw : 0 ≤ w
  · rw [relu, max_eq_right hw, relu, max_eq_left (by linarith)]
    ring
  · rw [relu, max_eq_left (by linarith), relu, max_eq_right (by linarith)]
    ring

/-- Every signed affine map can be realized by a non-negative affine map
on `[x,-x]`; §2.2.2 and §3.2, Eqs. (13), (14). -/
theorem signed_affine_recovery {d : ℕ} {O : Type*} (W : O → Fin d → ℝ)
    (b : O → ℝ) (x : Fin d → ℝ) :
    affine (signedWeights W) b
      (fun r : Bool × Fin d => if r.1 then -x r.2 else x r.2) = affine W b x := by
  funext o
  simp only [affine, Fintype.sum_prod_type, Fintype.sum_bool,
    signedWeights, Bool.false_eq_true, ↓reduceIte]
  rw [← Finset.sum_add_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro r hr
  have h := relu_signed_difference (W o r)
  have hs := congrArg (fun c => c * x r) h
  nlinarith only [hs]

/-- An admissible encoder whose original-coordinate output decreases;
§3.3, Corollary 1, expressly does not claim original-input monotonicity. -/
def decreasingEncoder : Encoder 0 (Bool × Fin 1) 1 1 1 1 1 where
  embedding := fun _ r => if r.1 then 1 else 0
  embeddingBias := fun _ => 0
  pos := fun _ => 0
  blocks := []
  readout := fun _ _ => 1
  readoutBias := fun _ => 0

/-- The decreasing original-input example still satisfies Assumption 1;
§3.3, Corollary 1. -/
theorem decreasingEncoder_conditions : EncoderConditions decreasingEncoder := by
  refine ⟨?_, ?_, fun _ _ => zero_le_one⟩
  · intro r s
    simp only [decreasingEncoder]
    split <;> norm_num
  · intro b hb
    exact False.elim (List.not_mem_nil hb)

/-- The model truly computes a decreasing coordinate rather than merely
having a suggestive name; §3.3, Corollary 1. -/
theorem decreasingEncoder_value (X : Sequence 1 1) :
    predictOriginal decreasingEncoder X 0 = -X (0, 0) := by
  simp [predictOriginal, predict, embed, expand, decreasingEncoder, encoderStack,
    affine, Fintype.sum_prod_type]

/-- Original-input monotonicity cannot be added to Corollary 1:
the expanded-coordinate monotonicity theorem remains valid;
§3.3, Theorem 1 and Corollary 1. -/
theorem original_monotonicity_not_guaranteed : EncoderConditions decreasingEncoder ∧
    ¬ Monotone (predictOriginal decreasingEncoder) := by
  refine ⟨decreasingEncoder_conditions, ?_⟩
  intro h
  have hj := h (a := (0 : Sequence 1 1)) (b := (fun _ => (1 : ℝ))) (fun _ => zero_le_one) 0
  rw [decreasingEncoder_value, decreasingEncoder_value] at hj
  norm_num at hj

end Transformer.ICEoT
