module

public import Transformer.Grokking.FiniteRecovery.AdamWReachability
public import Transformer.Grokking.FiniteRecovery.AdamWRelabeling

/-! Reachability with the actual Argand initializer. Only embeddings and
projection entries are sampled. Scores, gate coefficients and frequencies
have fixed initial values and are updated along with the sampled entries.
The endpoint theorem controls exactly the sampled block and bounds the
other coordinates by a box known from the schedule alone. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set
open scoped NNReal

variable {A X : Type*} {H d L T : ℕ}

def ComplexParameter.sampled : ComplexParameter A H d L → Bool
  | .score _ | .residualGate .. => false
  | _ => true

def coordinateSampled : Parameter A H d L → Bool
  | .inl (p,_) => p.sampled
  | .inr _ => false

abbrev SampledParameter (A : Type*) (H d L : ℕ) := {i : Parameter A H d L // coordinateSampled i = true}

def drawOnSampled (u : SampledParameter A H d L → ℝ) (p : ComplexParameter A H d L) : ℂ :=
  if h : p.sampled = true then ⟨u ⟨.inl (p,false),h⟩,u ⟨.inl (p,true),h⟩⟩ else 0

theorem continuous_drawOnSampled : Continuous (drawOnSampled (A := A) (H := H) (d := d) (L := L)) := by
  apply continuous_pi
  intro p
  by_cases hp : p.sampled = true
  · simp only [drawOnSampled,dite_eq_left hp]
    let pair (u : SampledParameter A H d L → ℝ) := (u ⟨.inl (p,false),hp⟩,u ⟨.inl (p,true),hp⟩)
    have hc : Continuous pair := (continuous_apply _).prodMk (continuous_apply _)
    apply (Complex.equivRealProdCLM.symm.continuous.comp hc).congr
    intro u
    apply Complex.ext <;> simp [Function.comp_apply,pair]
  · simp only [drawOnSampled,dite_eq_right hp]
    exact continuous_const

theorem initial_sampled (u : SampledParameter A H d L → ℝ) (j : SampledParameter A H d L) :
    sgdInitialWeights true (drawOnSampled u) j.val = u j := by
  rcases j with ⟨⟨p,b⟩ | i,h⟩
  · cases p <;> cases b <;> first | rfl | cases h
  · cases h

theorem initial_unsampled (z : ComplexParameter A H d L → ℂ) (i : Parameter A H d L)
    (hi : coordinateSampled i = false) :
    sgdInitialWeights true z i = sgdInitialWeights true (fun _ => 0) i := by
  rcases i with ⟨p,b⟩ | i
  · cases p <;> cases b <;> first | rfl | cases hi
  · rfl

/-- A static condition on endpoint weights: the freely drawn block takes
the requested values, and each remaining coordinate lies within a known
distance of its initializer multiplied by pure decay. -/
def endpointBox (rate : ℕ → ℝ) (K : ℕ) (u : SampledParameter A H d L → ℝ)
    (w : Weights A H d L) : Prop :=
  (∀ j : SampledParameter A H d L, w j = u j) ∧ ∀ i, coordinateSampled i = false →
    |w i - AdamW.decayFactor rate K * sgdInitialWeights true (fun _ => 0) i| ≤
      120 * (1 - AdamW.decayFactor rate K)

variable [Fintype A]

/-- At any finite horizon, including stock WSD with its original peak,
every choice of all embedding and projection coordinates occurs at some
trained endpoint. The other trainable entries obey the stated box bound. -/
theorem adamW_reaches_sampled (r K : ℕ) (rate : ℕ → ℝ) (hr : ∀ t < K, 0 ≤ rate t ∧ rate t < 10)
    (batch : ℕ → List (LabeledPosition A)) (u : SampledParameter A H d L → ℝ) :
    ∃ z, endpointBox rate K u (adamWRun r batch rate z K).weights := by
  obtain ⟨v,hv⟩ := AdamW.reachable_coordinates rate (fun t => batchGradient r (batch t))
    (fun t => continuous_batchGradient r (batch t)) Subtype.val
    (sgdInitialWeights true ∘ drawOnSampled) ((continuous_sgdInitialWeights true).comp continuous_drawOnSampled)
    initial_sampled K hr u
  let z := drawOnSampled v
  refine ⟨z,fun i => congrFun hv i,fun i hi => ?_⟩
  have he := AdamW.decay_displacement rate (fun t => batchGradient r (batch t))
    (sgdInitialWeights true z) K hr i
  rwa [initial_unsampled z i hi] at he

/-- A static robust realization suffices for a positive single-trial
probability at the original learning rates; no successful trained draw
is an input. The corpus can consist solely of the hidden prompts. -/
theorem adamW_box_positive [Finite X] (r K : ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t < K, 0 ≤ rate t ∧ rate t < 10) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (u : SampledParameter A H d L → ℝ)
    (hu : ∀ w, endpointBox rate K u w → StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    0 < independentGaussianDraws variance (terminalStrictDraws r K rate (fun _ => batch) prompt f) := by
  obtain ⟨z,hz⟩ := adamW_reaches_sampled r K rate hr batch u
  let := independentGaussianDraws_openPos variance hv
  exact (adamW_strict_open r K rate batch prompt f).measure_pos _ ⟨z,hu _ hz⟩

/-- All rates of the ordinary uncapped WSD prefix satisfy the strict
decay condition. The theorem also covers the 200,000-step repeat budget. -/
theorem adamW_wsd_reaches_sampled (r K : ℕ) (batch : ℕ → List (LabeledPosition A))
    (u : SampledParameter A H d L → ℝ) :
    ∃ z, endpointBox (wsdRate K) K u (adamWRun r batch (wsdRate K) z K).weights :=
  adamW_reaches_sampled r K _ (fun t _ => ⟨(wsdRate_bounds K t).1,
    lt_of_le_of_lt (wsdRate_bounds K t).2 (by norm_num)⟩) batch u

example (batch : ℕ → List (LabeledPosition (Fin 239))) (u : SampledParameter (Fin 239) 4 16 6 → ℝ) :=
  adamW_wsd_reaches_sampled 1 100000 batch u

end Transformer.Grokking.FiniteRecovery.Argand
