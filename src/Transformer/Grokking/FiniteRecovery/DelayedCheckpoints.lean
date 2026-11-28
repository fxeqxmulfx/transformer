module

public import Transformer.Grokking.FiniteRecovery.NormalizedRecovery
public import Transformer.Grokking.FiniteRecovery.NormalizedMomentBound

/-! Certify the ordinary optimizer first and delay extra forward evaluations
until a chosen checkpoint. The number of updates to the first success can
only decrease. Independent proposals eventually recover every identifiable,
strictly representable target, from each fixed initial state. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set MeasureTheory Metric ProbabilityTheory
open scoped NNReal ENNReal

variable {E : Type*} [NormedAddCommGroup E]

/-- Rejected proposals are discarded. The next center always belongs to the
original training trajectory. The first random draw is used at index B. -/
noncomputable def delayedCheckpoint (B : ℕ) (ok : E → Prop) (center noise : ℕ → E) (j : ℕ) : E :=
  if j < B then center j else checkpointCandidate ok (center j) (noise (j-B))

/-- Any successful ordinary first hit occurs at the same or a later index
than a verified hit. Hits strictly before the first proposal are identical. -/
theorem delayed_first_hit (B : ℕ) (ok : E → Prop) (center noise : ℕ → E) (grid : Finset ℕ) {t : ℕ}
    (h : firstPerfect grid (ok ∘ center) = some t) :
    ∃ u ≤ t, firstPerfect grid (ok ∘ delayedCheckpoint B ok center noise) = some u ∧ (t < B → u = t) := by
  classical
  let trial := delayedCheckpoint B ok center noise
  obtain ⟨ht,success,earliest⟩ := (firstPerfect_eq_some ..).mp h
  have ht' : ok (trial t) := by
    by_cases he : t < B
    · simpa [trial,delayedCheckpoint,he] using success
    · simpa only [trial,delayedCheckpoint,ite_eq_right he] using
        (checkpointCandidate_pass ok (center t) (noise (t-B))).2 (Or.inl success)
  have ex : ∃ j ∈ grid, (ok ∘ trial) j := ⟨t,ht,ht'⟩
  have hu := Nat.find_spec ex
  refine ⟨Nat.find ex,Nat.find_min' ex ⟨ht,ht'⟩,dite_eq_left ex,?_⟩
  intro hB
  apply le_antisymm (Nat.find_min' ex ⟨ht,ht'⟩)
  by_contra! hl
  apply earliest _ hu.1 hl
  have before (j : ℕ) (hj : j < B) : trial j = center j := by simp [trial,delayedCheckpoint,hj]
  exact (congrArg ok (before _ (hl.trans hB))).mp hu.2

variable [MeasurableSpace E]

/-- A fixed ordinary start, a delayed proposal schedule, and retained
weights. The verified event is compared with the true hidden predictions. -/
def DelayedRecovery (B : ℕ) (P : Measure (ℕ → E)) (center : ℕ → E) (ok right : E → Prop) : Prop :=
  let trial := delayedCheckpoint B ok center
  (∀ noise grid t, firstPerfect grid (right ∘ center) = some t →
    ∃ u ≤ t, firstPerfect grid (ok ∘ trial noise) = some u ∧ (t < B → u = t)) ∧
  ∃ p > (0 : ℝ≥0∞), (∀ m, P {z | retained ok (trial z) (B+m) = none} ≤ (1-p)^m) ∧
    (∀ q > (0 : ℝ≥0∞), ∃ m, P {z | retained ok (trial z) (B+m) = none} < q) ∧
    ∀ᵐ z ∂P, ∃ j, right (trial z j) ∧ ∀ k, retained ok (trial z) (j+1+k) = some (trial z j)

/-- The ordinary prefix may include a perfect model, or fail throughout.
After it, m independent proposals give a geometric miss bound either way. -/
theorem delayed_compact_recovery [BorelSpace E] (μ : Measure E) [IsProbabilityMeasure μ] [μ.IsOpenPosMeasure]
    (B : ℕ) (center : ℕ → E) {C : Set E} (hc : IsCompact C) (hC : ∀ j, center j ∈ C)
    (ok right : E → Prop) (he : ∀ w, ok w ↔ right w) (ho : IsOpen {w | right w}) (hw : ∃ w, right w) :
    DelayedRecovery B (Measure.infinitePi (fun _ : ℕ => μ)) center ok right := by
  have eq : ok = right := funext fun w => propext (he w)
  subst right
  let tail j := center (B+j)
  let trial := delayedCheckpoint B ok center
  let next (z : ℕ → E) j := checkpointCandidate ok (tail j) (z j)
  have shifted (z : ℕ → E) (j : ℕ) : trial z (B+j) = next z j := by
    simp [trial,next,tail,delayedCheckpoint]
  obtain ⟨p,hp,h⟩ := compact_checkpoint_convergence μ hc ho hw
  obtain ⟨bound,confidence,almost⟩ := h tail (fun j => hC (B+j))
  have incl (m : ℕ) : {z | retained ok (trial z) (B+m) = none} ⊆
      {z | retained ok (next z) m = none} := by
    intro z miss
    apply (retained_none_iff ..).2
    intro j hj
    rw [← shifted]
    exact (retained_none_iff ..).1 miss _ (Finset.mem_range.mpr (Nat.add_lt_add_left (Finset.mem_range.mp hj) B))
  refine ⟨fun z grid _ hhit => delayed_first_hit B ok center z grid hhit,p,hp,
    fun m => (measure_mono (incl m)).trans (bound m),?_,?_⟩
  · intro q hq
    obtain ⟨m,hm⟩ := confidence q hq
    exact ⟨m,(measure_mono (incl m)).trans_lt hm⟩
  · filter_upwards [almost] with z hz
    obtain ⟨j,hj,_⟩ := hz
    have success : ok (trial z (B+j)) := (shifted z j).symm ▸ hj
    obtain ⟨k,hk,_,keep⟩ := retained_settles ok (trial z) ⟨B+j,success⟩
    exact ⟨k,hk,keep⟩

namespace Argand

variable {A X Z : Type*} [Fintype A] [Finite Z] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

/-- From an arbitrary fixed initialization, all hidden answers are eventually
certified on a single continuing normalized AdamW path. Each ordinary
center agrees with stock AdamW under the same updates. -/
theorem normalizedAdam_delayed_recovery (r B : ℕ) (schedule : ℕ → ℕ) (rate : ℕ → ℝ)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (batch : ℕ → List (LabeledPosition A)) (draw : Draw)
    (prompt : Z → Word A (T+1)) (hidden : Z → X) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (v : ℝ≥0) (hv : 0 < v)
    (hrep : ∃ w : Weights A H d L, StrictOn (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden)) :
    let score (w : Weights A H d L) x := (Model.fromWeights w).logits r (prompt x) (Fin.last T)
    let center j := (normalizedAdamRun r batch rate draw (schedule j)).weights
    (∀ j, center j = (adamWRun r batch rate draw (schedule j)).weights) ∧
      DelayedRecovery B (Measure.infinitePi (fun _ : ℕ => Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v)))
        center (fun w => hiddenCertificate S family (fun x => f x) hidden (score w)) (fun w => StrictOn (score w) (f ∘ hidden)) := by
  intro score center
  refine ⟨fun j => normalizedAdam_weights ..,?_⟩
  let w := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  let R := max ‖w‖ (10*Real.sqrt (19/14))
  let : (gaussianReal 0 v).IsOpenPosMeasure := (gaussianReal_absolutelyContinuous' 0 hv.ne').isOpenPosMeasure
  apply delayed_compact_recovery _ B center (isCompact_closedBall 0 R) _ _ _ _ (strict_prompt_weights_open r prompt _) hrep
  · intro j
    rw [mem_closedBall,dist_zero_right]
    apply (pi_norm_le_iff_of_nonneg ((norm_nonneg w).trans (le_max_left _ _))).2
    exact fun i => NormalizedAdam.corrected_weight_bound rate hr (fun t => batchGradient r (batch t)) w (schedule j) i
  · intro w
    exact identified_property_iff S family hid f hf (fun g => StrictOn (score w) (g ∘ hidden))

open Division.Factors

/-- Native 4 × 16 width, six layers, peak 0.003, decay 0.1, 256 updates
per check. Starting proposals at check 32 permits 8,192 ordinary updates. -/
theorem factor_delayed_normalizedAdam (B : ℕ) (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    let score (w : Weights (Fin 239) 4 16 6) x := (Model.fromWeights w).logits 1 (divisionPrompt (hidden x)) (Fin.last 4)
    let center j := (normalizedAdamRun 1 batch (fun k => 3*wsdRate 20000 k) draw (256*j)).weights
    DelayedRecovery B (Measure.infinitePi (fun _ : ℕ => Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 =>
      gaussianReal 0 ((1/1000000 : ℝ≥0)^2)))) center
      (fun w => hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden (score w)) (fun w => StrictOn (score w) (f ∘ hidden)) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply (normalizedAdam_delayed_recovery 1 B (fun j => 256*j) _ _ batch draw _ hidden S _
    (factorClass_identified _ _ hc hz) f hf _ (by norm_num) ⟨sgdInitialWeights true w,fun x => hw (hidden x)⟩).2
  intro t
  obtain ⟨lo,hi⟩ := wsdRate_bounds 20000 t
  exact ⟨by linarith,by linarith⟩

example := factor_delayed_normalizedAdam 32 id (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (FlatSpectrum.diagRow_connected _).preconnected)
  ⟨1,mem_insert _ _⟩ divisionAnswer divisionAnswer_mem

end Argand
end Transformer.Grokking.FiniteRecovery
