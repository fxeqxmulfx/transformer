module

public import Transformer.Grokking.FiniteRecovery.EqualMoment
public import Transformer.Grokking.FiniteRecovery.ArgandCheckpointProposals
public import Transformer.Grokking.FiniteRecovery.NativeFirstPerfect

/-! Equal-decay uncorrected Adam on the entire Argand model. Ordinary first
hits across independent native starts and the independent checkpoint search
are two distinct probability experiments. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set Metric FlatSpectrum
open scoped ENNReal NNReal

variable {A X Z : Type*} [Fintype A] {H d L T : ℕ}

noncomputable def uncorrectedAdamRun (r : ℕ) (firstDecay secondDecay ε decay : ℝ)
    (batch : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  EqualMoment.uncorrectedRun firstDecay secondDecay ε decay rate (fun t => batchGradient r (batch t))
    (sgdInitialWeights true draw)

noncomputable def equalMomentRun (r : ℕ) (decay : ℝ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  EqualMoment.run (9/10) (1/100000000) decay rate (fun t => batchGradient r (batch t))
    (sgdInitialWeights true draw)

/-- For ordinary training, a single strictly successful real trajectory on
any scheduled step is equivalent to positive success mass. The open start
condition can require an initially wrong answer. Independent starts give the
exact finite miss probability, with no independence between time steps. -/
theorem equalMoment_first_hit_mass [Finite Z] (r : ℕ) (grid : Finset ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : Z → Word A (T+1)) (f : Z → A)
    (I : Set (ComplexParameter A H d L → ℂ)) (hI : IsOpen I)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let ok z t := z ∈ I ∧ StrictOn (fun x => (Model.fromWeights (equalMomentRun r decay batch rate z t).weights).logits
      r (prompt x) (Fin.last T)) f
    let μ := independentGaussianDraws variance
    let p := μ {z | ∃ t ∈ grid, ok z t}
    (0 < p ↔ ∃ z, ∃ t ∈ grid, ok z t) ∧ ∀ m, Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∀ j ∈ Finset.range m, firstPerfect grid (ok (z j)) = none} = (1-p)^m := by
  intro ok μ p
  let := independentGaussianDraws_openPos variance hv
  have ho (t : ℕ) : IsOpen {z | ok z t} := hI.inter ((strict_prompt_weights_open r prompt f).preimage
    ((EqualMoment.continuous_run decay rate (fun t => batchGradient r (batch t)) (9/10)
      (by norm_num : (0:ℝ) < 1/100000000) (fun t => continuous_batchGradient r (batch t)) t).1.comp
      (continuous_sgdInitialWeights true)))
  have hp := firstPerfect_probability μ grid (fun t => {z | ok z t}) (fun t _ => (ho t).measurableSet)
  refine ⟨⟨fun h => nonempty_of_measure_ne_zero h.ne',fun ⟨z,t,ht,hz⟩ => ?_⟩,hp.2.2⟩
  exact ((ho t).measure_pos μ ⟨z,hz⟩).trans_le (hp.2.1 t ht)

/-- For every native start, the bounded ordinary path supports independent
Gaussian copies and an observed-label verifier on every identifiable and
strictly representable finite task. -/
theorem equalMoment_checkpoint_recovers [Finite Z] (r : ℕ) (decay : ℝ) (hd : 0 < decay)
    (schedule : ℕ → ℕ) (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (prompt : Z → Word A (T+1)) (hidden : Z → X) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden))
    (v : ℝ≥0) (hv : 0 < v) :
    PathCheckpointRecovery r (fun j => (equalMomentRun r decay batch rate draw (schedule j)).weights)
      prompt (f ∘ hidden) (hiddenCertificate S family (fun x => f x) hidden) v := by
  let w := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  apply bounded_path_checkpoint_recovers r _ (R := max ‖w‖ (1/decay)) _ prompt hidden S family hid f hf hrep v hv
  intro j
  rw [mem_closedBall,dist_zero_right]
  apply (pi_norm_le_iff_of_nonneg ((norm_nonneg w).trans (le_max_left _ _))).2
  exact fun i => EqualMoment.weight_bound rate (fun t => batchGradient r (batch t)) w
    (by norm_num [Set.mem_Icc]) (by norm_num) hd hr i (schedule j)

open Division.Factors

/-- All quotient-factor tables on each identifying observation mask at the
actual 4 × 16, six-layer native architecture and scaled 20,000-update WSD. -/
theorem factor_equalMoment_checkpoint_recovers [Finite Z] (scale : ℝ) (hs : 0 ≤ scale) (hcap : scale ≤ 10000)
    (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    PathCheckpointRecovery 1 (fun j => (equalMomentRun 1 (1/10) batch (fun t => scale*wsdRate 20000 t)
      draw (256*(32+j))).weights) (divisionPrompt ∘ hidden) (f ∘ hidden)
      (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden) ((1/1000000 : ℝ≥0)^2) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply equalMoment_checkpoint_recovers 1 (1/10) (by norm_num) _ _ _ batch draw _ hidden S _
    (factorClass_identified _ _ hc hz) f hf ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩ _ (by norm_num)
  intro t
  obtain ⟨h₀,h₁⟩ := wsdRate_bounds 20000 t
  exact ⟨mul_nonneg hs h₀,by linarith [mul_le_mul hcap h₁ h₀ (by norm_num : (0:ℝ) ≤ 10000)]⟩

example := factor_equalMoment_checkpoint_recovers 2 (by norm_num) (by norm_num)
  id (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
  divisionAnswer divisionAnswer_mem

end Transformer.Grokking.FiniteRecovery.Argand
