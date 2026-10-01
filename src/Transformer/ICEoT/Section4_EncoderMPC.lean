/-
# IC-EoT: connecting the proved encoder to the MPC hypotheses

arXiv:2603.22095v2, §4.1, Eqs. (33)–(36), and §4.3.1, Eq. (48).
The selective history representation is only a coordinate reindexing.
This bridge derives predictor convexity and predicted-variable monotonicity
from the encoder's parameters, rather than assuming the conclusion of §3.3.
-/

import Transformer.ICEoT.Section4_Global

noncomputable section

namespace Transformer.ICEoT

/-- Concatenate predicted variables and the two signed control copies;
§4.1, Eq. (34), and §4.3.1, Eq. (48). -/
def joinHistory {n y u : ℕ} (H : History n y u) :
    (Fin (n + 1) × (Fin y ⊕ (Bool × Fin u))) → ℝ :=
  fun ir => match ir.2 with
    | Sum.inl r => H.1 (ir.1, r)
    | Sum.inr r => H.2 (ir.1, r)

/-- The history concatenation is linear; §4.1, Eqs. (34), (35). -/
theorem joinHistory_combination {n y u : ℕ} (H K : History n y u) (a b : ℝ) :
    joinHistory (a • H + b • K) = a • joinHistory H + b • joinHistory K := by
  funext ir
  rcases ir with ⟨i, r | r⟩ <;> rfl

/-- The selective IC-EoT predictor used in MPC; §4.1, Eqs. (34), (36).
Its output contains all variables to be recursively propagated. -/
def selectivePredict {n y u m h heads ff : ℕ}
    (p : Encoder n (Fin y ⊕ (Bool × Fin u)) m h heads ff y)
    (H : History n y u) : Fin y → ℝ := predict p (joinHistory H)

/-- Assumption 1 implies exactly the predictor hypotheses of Corollary 2
for the selectively extended IC-EoT; §3.3 and §4.1. -/
theorem selectivePredict_conditions {n y u m h heads ff : ℕ}
    (p : Encoder n (Fin y ⊕ (Bool × Fin u)) m h heads ff y)
    (hp : EncoderConditions p) : PredictorConditions (selectivePredict p) := by
  have hf := predict_properties p hp
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro H hH K hK a b ha hb hab
    unfold selectivePredict
    rw [joinHistory_combination]
    exact hf.1.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
  · intro H K C hHK
    apply hf.2
    intro ir
    rcases ir with ⟨i, r | r⟩
    · exact hHK (i, r)
    · exact le_rfl

/-- Nonzero selective encoder parameters for the hypotheses of §4.1,
Corollary 2; one attention/residual block is actually present. -/
def selectiveWitness : Encoder 0 (Fin 1 ⊕ (Bool × Fin 1)) 1 1 1 1 1 where
  embedding := fun _ _ => 1
  embeddingBias := fun _ => -1
  pos := fun _ => 2
  blocks := [unitBlock]
  readout := fun _ _ => 1
  readoutBias := fun _ => -1

example : EncoderConditions selectiveWitness := by
  refine ⟨fun _ _ => zero_le_one, ?_, fun _ _ => zero_le_one⟩
  intro b hb
  simp only [selectiveWitness, List.mem_singleton] at hb
  rw [hb]
  exact unitBlock_conditions

end Transformer.ICEoT
