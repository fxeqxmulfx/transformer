module

public import Transformer.Grokking.FiniteRecovery.RmsProp
public import Transformer.Grokking.FiniteRecovery.ArgandCheckpointProposals

/-! The single mean-square buffer of RMSPropW on full Argand coordinates.
Certified recovery uses the same observed-label verifier as the other optimizers. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set Metric FlatSpectrum
open scoped NNReal

variable {A X Z : Type*} [Fintype A] {H d L T : ℕ}

noncomputable def rmsPropRun (r : ℕ) (decay : ℝ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  RmsProp.run decay rate (fun t => batchGradient r (batch t)) (sgdInitialWeights true draw)

/-- Arbitrary observed minibatches and native starts. Positive decoupled decay
bounds the RMSPropW centers. Independent real Gaussian copies and a verifier
constructed from identifying observations then give almost-sure hidden recovery. -/
theorem rmsProp_checkpoint_recovers [Finite Z] (r : ℕ) (decay : ℝ) (hdc : 0 < decay)
    (schedule : ℕ → ℕ) (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (prompt : Z → Word A (T+1)) (hidden : Z → X) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden))
    (v : ℝ≥0) (hv : 0 < v) :
    PathCheckpointRecovery r (fun j => (rmsPropRun r decay batch rate draw (schedule j)).weights)
      prompt (f ∘ hidden) (hiddenCertificate S family (fun x => f x) hidden) v := by
  let w := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  apply bounded_path_checkpoint_recovers r _ (R := max ‖w‖ (5/decay)) _ prompt hidden S family hid f hf hrep v hv
  intro j
  rw [mem_closedBall,dist_zero_right]
  apply (pi_norm_le_iff_of_nonneg ((norm_nonneg w).trans (le_max_left _ _))).2
  exact fun i => RmsProp.weight_bound decay rate (fun t => batchGradient r (batch t)) w hdc hr i (schedule j)

open Division.Factors

/-- The numerical division pilot uses decay 0.2, ordinary WSD and the first
eligible proposal at 8,192. The result covers the entire quotient-factor class
on any identifying observation mask and any finite collection of hidden inputs. -/
theorem factor_rmsProp_checkpoint_recovers [Finite Z] (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    PathCheckpointRecovery 1 (fun j => (rmsPropRun 1 (1/5) batch (wsdRate 20000) draw (256*(32+j))).weights)
      (divisionPrompt ∘ hidden) (f ∘ hidden) (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden)
      ((1/1000000 : ℝ≥0)^2) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply rmsProp_checkpoint_recovers 1 (1/5) (by norm_num) _ _ _ batch draw _ hidden S _
    (factorClass_identified _ _ hc hz) f hf ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩ _ (by norm_num)
  exact fun t => ⟨(wsdRate_bounds 20000 t).1,by linarith [(wsdRate_bounds 20000 t).2]⟩

example := factor_rmsProp_checkpoint_recovers id (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
  divisionAnswer divisionAnswer_mem

end Transformer.Grokking.FiniteRecovery.Argand
