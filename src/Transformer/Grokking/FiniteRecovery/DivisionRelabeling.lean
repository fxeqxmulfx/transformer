module

public import Transformer.Grokking.FiniteRecovery.AdamWRelabeling
public import Transformer.Grokking.FiniteRecovery.FactorNative

/-! Alphabet invariance for the native modular division run. The
experiments/argand initializer samples real and imaginary coordinates with
standard deviation 0.02. The same law and WSD schedule are used on both
sides. A seed-dependent minibatch order is transported with its data. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open MeasureTheory Set FlatSpectrum
open scoped NNReal

local notation "Draw" => (ComplexParameter (Fin 239) 4 16 6 → ℂ)

/-- On any held-out mask, the single-trial probability of answering all
prompts strictly at the specified step is the same for all 239-token
encodings. The answer function may be any table, including division. -/
theorem hidden_probability_relabel (e : Equiv.Perm (Fin 239)) (K : ℕ) (S : Set DivisionInput)
    (batch : Draw → ℕ → List (LabeledPosition (Fin 239))) (f : DivisionInput → Fin 239) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2)
    let prompt (x : (Sᶜ : Set DivisionInput)) := divisionPrompt x
    let answer (x : (Sᶜ : Set DivisionInput)) := f x
    μ (terminalStrictDraws 1 K (wsdRate K)
        (fun y t => (batch (relabelDraw e.symm y) t).map (relabelPosition e))
        (fun x => e ∘ prompt x) (e ∘ answer)) =
      μ (terminalStrictDraws 1 K (wsdRate K) batch prompt answer) :=
  adamW_relabel_probability e 1 K (wsdRate K) batch _ _ (fun _ => (1/50 : ℝ≥0)^2)

example (e : Equiv.Perm (Fin 239)) (S : Set DivisionInput) (batch : Draw → ℕ → List (LabeledPosition (Fin 239))) :=
  hidden_probability_relabel e 100000 S batch divisionAnswer

example (e : Equiv.Perm (Fin 239)) (S : Set DivisionInput) (batch : Draw → ℕ → List (LabeledPosition (Fin 239))) :=
  hidden_probability_relabel e 200000 S batch divisionAnswer

/-- For a member of the unknown-factor class, one strictly correct trained
real trajectory at stock WSD gives almost-sure correct retention under
independent restarts in every encoding. Its original single-run probability
determines the exact miss probability for all encodings and restart counts. -/
theorem factor_relabel_wsd_restarts (e : Equiv.Perm (Fin 239)) (K : ℕ) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (witness : Draw)
    (hw : witness ∈ terminalStrictDraws 1 K (wsdRate K) (fun _ => batch) divisionPrompt f) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2)
    let p := μ (terminalStrictDraws 1 K (wsdRate K) (fun _ => batch) divisionPrompt f)
    let scores (z : Draw) x := (Model.fromWeights (adamWRun 1 (fun t => (batch t).map (relabelPosition e))
      (wsdRate K) z K).weights).logits 1 (e ∘ divisionPrompt x) (Fin.last 4)
    let ok z := classCertificate S (relabelFamily e (factorClass unitToken)) (fun x => e (f x)) (scores z)
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) (e ∘ f) ∧
        ∀ k, retained ok z (j+1+k) = some (z j) :=
  adamW_relabel_certified_restarts e 1 K (wsdRate K) batch divisionPrompt S (factorClass unitToken)
    (factorClass_identified _ _ hc hz) f hf ((1/50 : ℝ≥0)^2) (by norm_num) witness hw

/-- The first actual update of the 100,000-step schedule has zero rate.
The native realizations still give strict logits at that step for every
table in the factor class, in every encoding. -/
example (e : Equiv.Perm (Fin 239)) (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) :
    ∃ witness : Draw, witness ∈ terminalStrictDraws 1 1 (wsdRate 100000)
      (fun _ t => (batch t).map (relabelPosition e)) (fun x => e ∘ divisionPrompt x) (e ∘ f) := by
  obtain ⟨z,hz⟩ := Factors.class_native_realization f hf
  refine ⟨relabelDraw e z,(mem_terminalStrictDraws_relabel e 1 1 (wsdRate 100000)
    (fun _ => batch) divisionPrompt f z).2 ?_⟩
  have hrate : wsdRate 100000 0 = 0 := by norm_num [wsdRate]
  simpa only [terminalStrictDraws,Set.mem_ofPred_eq,adamWRun,AdamW.run,hrate,zero_div,
    sub_zero,one_mul,zero_mul] using hz

end Transformer.Grokking.FiniteRecovery.Argand.Division
