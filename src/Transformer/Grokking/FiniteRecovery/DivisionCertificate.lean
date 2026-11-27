module

public import Transformer.Grokking.FlatSpectrum.PatternGraph
public import Transformer.Grokking.FiniteRecovery.AdamWTrials

/-! A training-only certificate for a group-valued table with an additional
constant row. On the original division corpus the unit rows have unknown
row and column phases. A connected unit mask and one constant-row label
determine every answer, including the zero numerator. The verifier then
checks all competing vocabulary tokens in a trained Argand checkpoint. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set FlatSpectrum

variable {I J G Y : Type*} [AddCommGroup G]

/-- Output vocabulary may contain symbols outside the image of the unit code. -/
def factorTable (code : G ↪ Y) (r : I → G) (c : J → G) (zero : Y) : Option I × J → Y
  | (none, _) => zero
  | (some i, j) => code (r i + c j)

/-- Both factors and the constant row are unknown to the learner. -/
def factorClass (code : G ↪ Y) : Set (Option I × J → Y) :=
  {f | ∃ r c zero, f = factorTable code r c zero}

def unitMask (S : Set (Option I × J)) : Set (I × J) := {ij | (some ij.1, ij.2) ∈ S}

/-- A compact spanning witness suits bit masks: each row shares an observed
column with the anchor and each column is incident to some row. -/
theorem preconnected_of_row_bridges (Ω : Set (I × J)) (anchor : I)
    (rows : ∀ i, ∃ j, (i,j) ∈ Ω ∧ (anchor,j) ∈ Ω) (cols : ∀ j, ∃ i, (i,j) ∈ Ω) :
    (patternGraph Ω).Preconnected := by
  have hr i : (patternGraph Ω).Reachable (.inl anchor) (.inl i) := by
    obtain ⟨j,hi,ha⟩ := rows i
    exact (patternGraph_adj_inl_inr.mpr ha).reachable.trans (patternGraph_adj_inr_inl.mpr hi).reachable
  have all (u : I ⊕ J) : (patternGraph Ω).Reachable (.inl anchor) u := by
    cases u with
    | inl i => exact hr i
    | inr j =>
      obtain ⟨i,h⟩ := cols j
      exact (hr i).trans (patternGraph_adj_inl_inr.mpr h).reachable
  exact fun u v => (all u).symm.trans (all v)

/-- Differences of two fitted factorizations vanish on the unit graph and
therefore on every cell. An observed constant-row answer fixes that row. -/
theorem factorClass_identified (code : G ↪ Y) (S : Set (Option I × J))
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S) :
    InjOn (fun f : Option I × J → Y => fun x : S => f x) (factorClass code) := by
  rintro _ ⟨r,c,z,rfl⟩ _ ⟨r',c',z',rfl⟩ he
  obtain ⟨j₀,hj₀⟩ := hz
  have h0 : z = z' := congrFun he ⟨(none,j₀),hj₀⟩
  have localEq (ij : I × J) (h : ij ∈ unitMask S) :
      (r ij.1-r' ij.1) + (c ij.2-c' ij.2) = 0 := by
    rw [sub_add_sub_comm]
    exact sub_eq_zero.mpr (code.injective (congrFun he ⟨(some ij.1,ij.2),h⟩))
  funext ⟨i,j⟩
  cases i with
  | none => exact h0
  | some i =>
    have h := rigid_of_preconnected hc (fun k => r k-r' k) (fun l => c l-c' l) localEq i j
    rw [sub_add_sub_comm, sub_eq_zero] at h
    exact congrArg code h

private def triangle : Set (Option (Fin 2) × Fin 2) :=
  {ij | match ij.1 with | none => ij.2 = 0 | some i => i = 0 ∨ ij.2 = 1}

example : InjOn (fun f : Option (Fin 2) × Fin 2 → ZMod 2 => fun x : triangle => f x)
    (factorClass (Function.Embedding.refl (ZMod 2))) :=
  factorClass_identified _ _ (preconnected_of_row_bridges _ 0
    (fun _ => ⟨1,Or.inr rfl,Or.inl rfl⟩) (fun _ => ⟨0,Or.inl rfl⟩)) ⟨0,rfl⟩

namespace Argand

open scoped Classical

abbrev DivisionUnit := (ZMod 97)ˣ
abbrev DivisionInput := Option DivisionUnit × DivisionUnit

/-- Offsets are those of the 239-token OpenAI/grok vocabulary. -/
def residueToken (v : ZMod 97) : Fin 239 :=
  ⟨22 + v.val, by have := v.val_lt; omega⟩

theorem residueToken_injective : Function.Injective residueToken := by
  intro a b h
  apply ZMod.val_injective 97
  have hv := congrArg Fin.val h
  simpa [residueToken] using hv

def unitToken : Additive DivisionUnit ↪ Fin 239 :=
  ⟨fun a => residueToken a.toMul.val,
    fun _ _ h => Units.val_injective (residueToken_injective h)⟩

/-- Only the input prefix is supplied to the decoder; every one of the
239 tokens is eligible as the prediction at the equals sign. -/
def divisionPrompt (x : DivisionInput) : Word (Fin 239) 5 :=
  ![0, residueToken (x.1.elim 0 Units.val), 10, residueToken x.2.val, 1]

def divisionAnswer : DivisionInput → Fin 239
  | (none, _) => residueToken 0
  | (some a, b) => residueToken ((a / b : DivisionUnit).val)

theorem divisionAnswer_mem : divisionAnswer ∈ factorClass unitToken := by
  refine ⟨Additive.ofMul, fun b => Additive.ofMul b⁻¹, residueToken 0,?_⟩
  funext ⟨i,j⟩
  cases i with
  | none => rfl
  | some i => exact congrArg (fun v : DivisionUnit => residueToken v.val) (div_eq_mul_inv i j)

/-- Exact characterization of the certificate on every legal prompt.
The scores can be real AdamW logits or real values read from finite
hardware at one terminal checkpoint. -/
theorem division_certificate_iff (S : Set DivisionInput) (hc : (patternGraph (unitMask S)).Preconnected)
    (hz : ∃ j, (none,j) ∈ S) (scores : DivisionInput → Fin 239 → ℝ) :
    classCertificate S (factorClass unitToken) (fun x => divisionAnswer x) scores ↔
      StrictOn scores divisionAnswer :=
  classCertificate_iff _ _ (factorClass_identified _ _ hc hz) _ divisionAnswer_mem _

/-- A score table on the hidden equations alone. -/
def divisionHiddenCorrect (S : Set DivisionInput) (score : DivisionInput → Fin 239 → ℝ)
    (g : DivisionInput → Fin 239) : Prop :=
  StrictOn (fun x : (Sᶜ : Set DivisionInput) => score x) (fun x => g x)

/-- The actual full tensor state and chosen horizon determine these scores. -/
noncomputable def divisionAdamWEndpoint (K : ℕ) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) (x : DivisionInput) :=
  (Model.fromWeights (adamWRun 1 batch (wsdRate K) draw K).weights).logits 1 (divisionPrompt x) (Fin.last 4)

private theorem division_seed_probability {D : Type*} [MeasurableSpace D] [MeasurableSingletonClass D]
    (seeds : Finset D) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (scores : D → DivisionInput → Fin 239 → ℝ) :
    MeasureTheory.Measure.uniformOfFinset seeds {z | divisionHiddenCorrect S (scores z) divisionAnswer} =
      (observedSuccessCount seeds S (factorClass unitToken) (fun x => divisionAnswer x)
        (fun z => divisionHiddenCorrect S (scores z)) : ENNReal) / seeds.card :=
  finite_seed_checked_probability seeds S (factorClass unitToken) (factorClass_identified unitToken S hc hz)
    divisionAnswer divisionAnswer_mem (fun z => divisionHiddenCorrect S (scores z))

/-- A seed may drive the draw *and* the shuffling of every epoch. Count
only terminal weights at K; previous successes do not enter the formula. -/
theorem division_adamW_terminal_probability {D : Type*} [MeasurableSpace D] [MeasurableSingletonClass D]
    (seeds : Finset D) (K : ℕ) (batch : D → ℕ → List (LabeledPosition (Fin 239)))
    (draw : D → ComplexParameter (Fin 239) 4 16 6 → ℂ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S) :
    let scores z := divisionAdamWEndpoint K (batch z) (draw z)
    MeasureTheory.Measure.uniformOfFinset seeds {z | divisionHiddenCorrect S (scores z) divisionAnswer} =
      (observedSuccessCount seeds S (factorClass unitToken) (fun x => divisionAnswer x)
        (fun z => divisionHiddenCorrect S (scores z)) : ENNReal) / seeds.card :=
  division_seed_probability seeds S hc hz (fun z => divisionAdamWEndpoint K (batch z) (draw z))

example : divisionPrompt (none,1) (Fin.last 4) = 1 := rfl
example : divisionAnswer (some 1,1) = residueToken 1 := by simp [divisionAnswer]

end Argand
end Transformer.Grokking.FiniteRecovery
