module

public import Transformer.Grokking.FiniteRecovery.NormalizedAdam
public import Transformer.Grokking.FiniteRecovery.EqualMomentConfidence

/-! Strict hidden recovery for recursively normalized AdamW. The first-hit
time agrees pathwise with stock AdamW even when each seed chooses its own
batch stream. Independent attempts give almost-sure correct retention on
any identifying mask admitting a strict real training witness. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set
open scoped NNReal

variable {A X Z : Type*} [Fintype A] [Finite Z] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

noncomputable def normalizedAdamRun (r : ℕ) (batch : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (z : Draw) := NormalizedAdam.run rate (fun t => batchGradient r (batch t)) (sgdInitialWeights true z)

theorem normalizedAdam_weights (r : ℕ) (batch : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (z : Draw) (t : ℕ) : (normalizedAdamRun r batch rate z t).weights = (adamWRun r batch rate z t).weights :=
  (NormalizedAdam.agrees_with_stock rate _ _ t).1

/-- Hit probabilities are identical for any law of initial draws and
any set of possible reporting times, including the event of no hit. -/
theorem normalizedAdam_hit_law (μ : Measure Draw) (r : ℕ) (rate : ℕ → ℝ) (grid : Finset ℕ)
    (batch : Draw → ℕ → List (LabeledPosition A)) (accept : ℕ → Weights A H d L → Prop)
    (times : Set (Option ℕ)) :
    μ {z | firstPerfect grid (fun t => accept t (normalizedAdamRun r (batch z) rate z t).weights) ∈ times} =
      μ {z | firstPerfect grid (fun t => accept t (adamWRun r (batch z) rate z t).weights) ∈ times} := by
  simp_rw [normalizedAdam_weights]

/-- A successful real training path on the grid gives positive one-run
mass and almost-sure success with ordinary independent restarts. Every
retained checkpoint is certified using the known family and observed labels. -/
theorem normalizedAdam_certified_restarts (r : ℕ) (grid : Finset ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : Z → Word A (T+1)) (hidden : Z → X)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let score (z : Draw) t x := (Model.fromWeights (normalizedAdamRun r batch rate z t).weights).logits r (prompt x) (Fin.last T)
    let μ := independentGaussianDraws variance
    let p := μ {z | ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)}
    let first z := firstPerfect grid (fun t => hiddenCertificate S family (fun x => f x) hidden (score z t))
    let ok z := first z ≠ none
    (∃ z, ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)) →
      0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {ω | retained ok ω m = none} = (1-p)^m) ∧
        ∀ᵐ ω ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j t, first (ω j) = some t ∧
          StrictOn (score (ω j) t) (f ∘ hidden) ∧ ∀ k, retained ok ω (j+1+k) = some (ω j) := by
  intro score μ p first ok witness
  have ho (t : ℕ) : IsOpen {z | StrictOn (score z t) (f ∘ hidden)} := by
    simpa only [score,normalizedAdam_weights] using
      (adamW_strict_open (H := H) (d := d) (L := L) r t rate batch prompt (f ∘ hidden))
  let := independentGaussianDraws_openPos variance hv
  obtain ⟨z,t,ht,hz⟩ := witness
  have hp : 0 < p := ((ho t).measure_pos μ ⟨z,hz⟩).trans_le (measure_mono fun w hw => ⟨t,ht,hw⟩)
  exact ⟨hp,certified_grid_restarts μ grid S family hid f hf hidden score (fun t _ => (ho t).measurableSet) hp⟩

open Division.Factors

/-- Every quotient-factor table in the native width-64, six-layer model
meets the recovery theorem at peak 0.003. Including the native start on
the grid supplies a witness for every table, independently of minibatches. -/
theorem factor_normalizedAdam_restarts (cap : ℕ) (S : Set DivisionInput)
    (hc : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (hidden : Z → DivisionInput) (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) :
    let score (z : ComplexParameter (Fin 239) 4 16 6 → ℂ) t x := (Model.fromWeights
      (normalizedAdamRun 1 batch (fun k => 3*wsdRate 20000 k) z t).weights).logits
      1 (divisionPrompt (hidden x)) (Fin.last 4)
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => (1/50 : ℝ≥0)^2)
    let grid := evaluationGrid 256 cap
    let first z := firstPerfect grid (fun t => hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden (score z t))
    let ok z := first z ≠ none
    let p := μ {z | ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)}
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {ω | retained ok ω m = none} = (1-p)^m) ∧
      ∀ᵐ ω ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j t, first (ω j) = some t ∧
        StrictOn (score (ω j) t) (f ∘ hidden) ∧ ∀ k, retained ok ω (j+1+k) = some (ω j) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply normalizedAdam_certified_restarts 1 _ _ batch _ hidden S _ (factorClass_identified _ _ hc hz) f hf _ (by intro; norm_num)
  exact ⟨w,0,by simp [evaluationGrid],fun z => hw (hidden z)⟩

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_normalizedAdam_restarts 20000 (factorMask (1 : DivisionUnit))
    (by rw [unitMask_factorMask]; exact (FlatSpectrum.diagRow_connected _).preconnected)
    ⟨1,mem_insert _ _⟩ id divisionAnswer divisionAnswer_mem batch

end Transformer.Grokking.FiniteRecovery.Argand
