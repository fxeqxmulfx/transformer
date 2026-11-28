module

public import Transformer.Grokking.FiniteRecovery.CheckpointProposals
public import Transformer.Grokking.FiniteRecovery.FactorNative
public import Transformer.Grokking.FiniteRecovery.FlatFirstNoise

/-! Certified perturbations of ordinary AdamW checkpoints. A rejected
copy leaves the native weights, both moments and the minibatch sequence
available for the next update. Every statement conditions on one start. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set ProbabilityTheory Metric FlatSpectrum Matrix
open scoped NNReal ENNReal

variable {A X Z : Type*} [Fintype A] {H d L T : ℕ}

/-- The verifier can reconstruct the restriction of the unknown target
to the hidden prompts from the observed labels in an identifying class. -/
def hiddenCertificate (S : Set X) (family : Set (X → A)) (labels : S → A)
    (hidden : Z → X) (score : Z → A → ℝ) : Prop :=
  ∃ g ∈ family, (∀ x : S, g x = labels x) ∧ StrictOn score (g ∘ hidden)

/-- A finite-time miss bound, arbitrary prescribed confidence, and almost
sure stabilization of the returned weights on a single native trajectory. -/
def CheckpointRecovery (r : ℕ) (schedule : ℕ → ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (prompt : Z → Word A (T+1)) (target : Z → A) (verify : (Z → A → ℝ) → Prop) (v : ℝ≥0) : Prop :=
  let score (w : Weights A H d L) x := (Model.fromWeights w).logits r (prompt x) (Fin.last T)
  let ok w := verify (score w)
  let center j := (adamWRun r batch rate draw (schedule j)).weights
  let trial (z : ℕ → Weights A H d L) j := checkpointCandidate ok (center j) (z j)
  let P := Measure.infinitePi (fun _ : ℕ => Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v))
  (∀ m z, (∃ j ∈ Finset.range m, StrictOn (score (center j)) target) →
    retained ok (trial z) m ≠ none) ∧
  ∃ p > (0 : ℝ≥0∞), (∀ m, P {z | retained ok (trial z) m = none} ≤ (1-p)^m) ∧
    (∀ q > (0 : ℝ≥0∞), ∃ m, P {z | retained ok (trial z) m = none} < q) ∧
    ∀ᵐ z ∂P, ∃ j, StrictOn (score (trial z j)) target ∧
      ∀ k, retained ok (trial z) (j+1+k) = some (trial z j)

/-- Any strictly representable hidden target in an identifiable class.
The 0.1 weight decay and clipped AdamW moments bound *every* ordinary
weight at *every* schedule index, independently of minibatch order.
This supplies the compactness needed for a single continuing path. -/
theorem adamW_checkpoint_recovers [Finite Z] (r : ℕ) (schedule : ℕ → ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) (prompt : Z → Word A (T+1)) (hidden : Z → X)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden))
    (v : ℝ≥0) (hv : 0 < v) :
    CheckpointRecovery r schedule rate batch draw prompt (f ∘ hidden)
      (hiddenCertificate S family (fun x => f x) hidden) v := by
  let score (w : Weights A H d L) x := (Model.fromWeights w).logits r (prompt x) (Fin.last T)
  let U := {w | hiddenCertificate S family (fun x => f x) hidden (score w)}
  have he w : w ∈ U ↔ StrictOn (score w) (f ∘ hidden) :=
    identified_property_iff S family hid f hf (fun g => StrictOn (score w) (g ∘ hidden))
  have hopen : IsOpen U := by
    rw [show U = {w | StrictOn (score w) (f ∘ hidden)} from Set.ext he]
    exact strict_prompt_weights_open r prompt (f ∘ hidden)
  obtain ⟨w,hw⟩ := hrep
  let μ := Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v)
  let : (gaussianReal 0 v).IsOpenPosMeasure :=
    (gaussianReal_absolutelyContinuous' 0 hv.ne').isOpenPosMeasure
  let x₀ := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  let R := max ‖x₀‖ 120
  let center j := (adamWRun r batch rate draw (schedule j)).weights
  have hC j : center j ∈ closedBall 0 R := by
    rw [mem_closedBall,dist_zero_right]
    apply (pi_norm_le_iff_of_nonneg (le_trans (norm_nonneg x₀) (le_max_left _ _))).2
    exact fun i => AdamW.weight_bound rate (fun t => batchGradient r (batch t)) x₀ hr i (schedule j)
  obtain ⟨p,hp,h⟩ := compact_checkpoint_convergence μ (isCompact_closedBall (0 : Weights A H d L) R)
    hopen ⟨w,(he w).2 hw⟩
  obtain ⟨hb,hq,ha⟩ := h center hC
  refine ⟨fun m z ⟨j,hj,hs⟩ => checkpoint_preserves_hit (· ∈ U) center z m ⟨j,hj,(he _).2 hs⟩,
    p,hp,hb,hq,?_⟩
  filter_upwards [ha] with z hz
  obtain ⟨j,hj,hk⟩ := hz
  exact ⟨j,(he _).1 hj,hk⟩

open Division.Factors

/-- The entire division-factor class, including arbitrary zero-row tokens,
on any identifying training mask and any subset of hidden division inputs. -/
theorem factor_checkpoint_recovers [Finite Z] (cap : ℕ) (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    CheckpointRecovery 1 (fun j => 256*j) (wsdRate cap) batch draw (divisionPrompt ∘ hidden) (f ∘ hidden)
      (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden) ((1/1000000 : ℝ≥0)^2) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  exact adamW_checkpoint_recovers 1 _ _ (fun t => ⟨(wsdRate_bounds cap t).1,
    (wsdRate_bounds cap t).2.trans (by norm_num)⟩) batch draw _ hidden S _ (factorClass_identified _ _ hc hz)
    f hf ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩ _ (by norm_num)

/-- Binary modular sums on their minimum staircase masks. Each trial
uses the same seed and advances the same WSD AdamW state. -/
theorem modularSum_checkpoint_recovers {k s : ℕ} [Finite Z] (cap : ℕ)
    (e : Fin (s+1) ≃ Word (ZMod 2) (k+1)) (hk : (k+1)+(k+1) ≤ 32)
    (hidden : Z → Word (ZMod 2) ((k+1)+k+1)) (batch : ℕ → List (LabeledPosition (ZMod 2)))
    (draw : ComplexParameter (ZMod 2) 4 16 6 → ℂ) :
    let cut := enumeratedCut e
    let S := inputsAtCells cut (staircase s)
    let family := unitRankOneFamily cut ZMod.stdAddChar
    (observe S (modularSum e)).card = 2*s+1 ∧
    CheckpointRecovery 1 (fun j => 256*j) (wsdRate cap) batch draw hidden (modularSum e ∘ hidden)
      (hiddenCertificate (S : Set _) family (fun x => modularSum e x) hidden) ((1/1000000 : ℝ≥0)^2) := by
  intro cut S family
  have hf : modularSum e ∈ family := ⟨_,_,targetKernel_modularSum e,fun _ _ => by simp [vecMulVec_apply]⟩
  have hn : ∀ f ∈ family, NuclearRecovers (observedCells cut S) (targetKernel cut ZMod.stdAddChar f) := by
    rintro f ⟨u,v,hu,hv⟩
    rw [show S = inputsAtCells cut _ from rfl,observedCells_inputsAtCells,hu]
    exact forall_isMinOn_vecMulVec_staircase hv
  have hi := (exists_learner_iff_injOn S family).mp
    (exists_learner_of_nuclear_recovers cut ZMod.stdAddChar ZMod.injective_stdAddChar S family hn)
  refine ⟨by rw [card_observe,card_inputsAtCells,ncard_staircase],?_⟩
  exact adamW_checkpoint_recovers 1 _ _ (fun t => ⟨(wsdRate_bounds cap t).1,
    (wsdRate_bounds cap t).2.trans (by norm_num)⟩) batch draw hidden hidden S family
    (fun a ha b hb he => hi ha hb (observe_eq_iff_eqOn.mpr fun x hx => congrFun he ⟨x,hx⟩))
    _ hf ⟨_,fun z => modularSum_native_realizes e hk (hidden z)⟩ _ (by norm_num)

example := factor_checkpoint_recovers 100000 id (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
  divisionAnswer divisionAnswer_mem

example := modularSum_checkpoint_recovers 20480 (wordEnumeration 2 4) (by decide) id
example := modularSum_checkpoint_recovers 20480 (wordEnumeration 2 8) (by decide) id
example := modularSum_checkpoint_recovers 20480 (wordEnumeration 2 16) (by decide) id

end Transformer.Grokking.FiniteRecovery.Argand
