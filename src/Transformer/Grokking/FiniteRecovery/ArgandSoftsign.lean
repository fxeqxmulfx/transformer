module

public import Transformer.Grokking.FiniteRecovery.SoftsignAlignment
public import Transformer.Grokking.FiniteRecovery.ArgandCheckpointProposals
public import Transformer.Grokking.FiniteRecovery.NativeFirstPerfect

/-! Plain and certified softsign Lion on the full Argand network. The ordinary
update uses only the observed minibatches. The optional independent proposals
use a verifier based on the identifying observations. Both use native trainable
parameters and the same initial draw, including the score and gate overrides. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory ProbabilityTheory Metric Set
open scoped NNReal ENNReal Topology

variable {A X Z : Type*} [Fintype A] {H d L T : ℕ}

noncomputable def softsignRun (r : ℕ) (decay : ℝ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  SoftsignMomentum.run decay rate (fun t => batchGradient r (batch t)) (sgdInitialWeights true draw)

/-- For ordinary training, a positive mass of strictly successful native starts
at a prescribed step is equivalent to one successful real trajectory. An open
restriction on the draw may require a wrong prediction before training. -/
theorem softsign_hit_mass_iff [Finite Z] (r K : ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : Z → Word A (T+1)) (target : Z → A)
    (I : Set (ComplexParameter A H d L → ℂ)) (hI : IsOpen I)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let U := {z | z ∈ I ∧ StrictOn
      (fun x => (Model.fromWeights (softsignRun r decay batch rate z K).weights).logits r (prompt x) (Fin.last T)) target}
    0 < independentGaussianDraws variance U ↔ U.Nonempty := by
  intro U
  let := independentGaussianDraws_openPos variance hv
  have h : IsOpen U := hI.inter ((strict_prompt_weights_open r prompt target).preimage
    ((SoftsignMomentum.continuous_run decay rate (fun t => batchGradient r (batch t))
      (fun t => continuous_batchGradient r (batch t)) K).1.comp (continuous_sgdInitialWeights true)))
  exact ⟨fun hp => nonempty_of_measure_ne_zero hp.ne', h.measure_pos _⟩

open SoftsignMomentum in
/-- A strictly realizing representative supplies a positive radius. Ordinary
training eventually gets every hidden token right if its observed update
directions obey one-point alignment and the WSD residual fits inside that
radius. No successful iterate is a hypothesis. -/
theorem softsign_aligned_hidden [Finite Z] (r : ℕ) (prompt : Z → Word A (T+1)) (target : Z → A)
    (u : Weights A H d L) (hu : StrictOn
      (fun x => (Model.fromWeights u).logits r (prompt x) (Fin.last T)) target) :
    ∃ ρ > (0:ℝ), ∀ (decay : ℝ) (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A))
      (draw : ComplexParameter A H d L → ℂ) (B : ℕ) (η μ M : ℝ),
      let fit := softsignRun r decay batch rate draw
      let g t := batchGradient r (batch t) (fit t).weights
      0 < η → 0 < μ → 0 ≤ M → 2*η*μ ≤ 1 → (∀ k, rate (B+k) = η) →
      (∀ k, let s := fit (B+k)
        μ*squaredError u s.weights ≤ ∑ i, (s.weights i-u i)*force decay s (g (B+k)) i) →
      (∀ k, let s := fit (B+k); ∑ i, (force decay s (g (B+k)) i)^2 ≤ M) →
      η*M/(2*μ) < ρ → ∀ᶠ k in atTop, StrictOn
        (fun x => (Model.fromWeights (fit (B+k)).weights).logits r (prompt x) (Fin.last T)) target := by
  obtain ⟨ρ,hρ,hball⟩ := open_squaredError_radius (strict_prompt_weights_open r prompt target) hu
  refine ⟨ρ,hρ,?_⟩
  intro decay rate batch draw B η μ M fit g hη hμ hM hc hr ha hf hres
  exact aligned_eventually_mem decay rate (fun t => batchGradient r (batch t)) (sgdInitialWeights true draw) u
    B hη hμ hM hc hr ha hf hres hball

/-- From each fixed native start and arbitrary observed batches, the certified
variant almost surely retains correct hidden predictions in every identifiable,
strictly representable finite task. A miss after m proposals has a geometric
upper bound. Ordinary successes are preserved for every proposal sequence. -/
theorem softsign_checkpoint_recovers [Finite Z] (r : ℕ) (decay : ℝ) (hdc : 0 < decay) (schedule : ℕ → ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) (prompt : Z → Word A (T+1)) (hidden : Z → X)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden))
    (v : ℝ≥0) (hv : 0 < v) :
    PathCheckpointRecovery r (fun j => (softsignRun r decay batch rate draw (schedule j)).weights) prompt
      (f ∘ hidden) (hiddenCertificate S family (fun x => f x) hidden) v := by
  let x₀ := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  apply bounded_path_checkpoint_recovers r _ (R := max ‖x₀‖ decay⁻¹) _ prompt hidden S family hid f hf hrep v hv
  intro j
  rw [mem_closedBall,dist_zero_right]
  apply (pi_norm_le_iff_of_nonneg ((norm_nonneg x₀).trans (le_max_left _ _))).2
  exact fun i => SoftsignMomentum.weight_bound decay rate (fun t => batchGradient r (batch t)) x₀ hdc hr i (schedule j)

open Division.Factors FlatSpectrum

/-- All quotient-factor tables on any identifying mask, with the experimental
width 64, FFN 256, six layers, one pass and a freely scaled WSD schedule.
The first eligible perturbation can be delayed by any number of checkpoints. -/
theorem factor_softsign_checkpoint_recovers [Finite Z] (cap first : ℕ) (decay scale : ℝ) (hdc : 0 < decay)
    (hs : 0 ≤ scale) (hcap : scale*decay ≤ 1000) (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    PathCheckpointRecovery 1 (fun j => (softsignRun 1 decay batch (fun t => scale*wsdRate cap t) draw (256*(first+j))).weights)
      (divisionPrompt ∘ hidden) (f ∘ hidden) (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden)
      ((1/1000000 : ℝ≥0)^2) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply softsign_checkpoint_recovers 1 decay hdc _ _ _ batch draw _ hidden S _ (factorClass_identified _ _ hc hz)
    f hf ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩ _ (by norm_num)
  intro t
  obtain ⟨h₀,h₁⟩ := wsdRate_bounds cap t
  refine ⟨mul_nonneg hs h₀, ?_⟩
  calc
    scale*wsdRate cap t*decay = scale*decay*wsdRate cap t := by ring
    _ ≤ 1000*(1/1000) := mul_le_mul hcap h₁ h₀ (by norm_num)
    _ = 1 := by norm_num

example := factor_softsign_checkpoint_recovers 20000 32 (1/10) 1 (by norm_num) (by norm_num) (by norm_num) id
  (factorMask (1 : DivisionUnit)) (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected)
  ⟨1,mem_insert _ _⟩ divisionAnswer divisionAnswer_mem

example := softsign_aligned_hidden 1 divisionPrompt divisionAnswer _
  (Division.division_native_realization.choose_spec)

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  softsign_hit_mass_iff 1 4096 (1/10) (wsdRate 20000) batch divisionPrompt divisionAnswer
    _ (isOpen_wrong_native_start 1 divisionPrompt divisionAnswer) (fun _ : ComplexParameter (Fin 239) 4 16 6 => 1)
    (by intro; norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
