module

public import Transformer.Grokking.FiniteRecovery.ArgandEqualMoment

/-! Independent ordinary training runs with a first-hit stopping rule.
The observations certify the hidden labels on an identifying mask.
The quantitative statement distinguishes one-run success mass from
confidence after multiple fresh initializations. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set
open scoped ENNReal NNReal

/-- A verified first hit anywhere on the evaluation grid suffices. Its
original seed and first successful update determine a fixed, strictly
correct checkpoint for every later restart. -/
theorem certified_grid_restarts {D X Z A : Type*} [MeasurableSpace D] (μ : Measure D)
    [IsProbabilityMeasure μ] (grid : Finset ℕ) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hidden : Z → X) (scores : D → ℕ → Z → A → ℝ)
    (hu : ∀ t ∈ grid, MeasurableSet {z | StrictOn (scores z t) (f ∘ hidden)})
    (hp : 0 < μ {z | ∃ t ∈ grid, StrictOn (scores z t) (f ∘ hidden)}) :
    let first z := firstPerfect grid (fun t => Argand.hiddenCertificate S family (fun x => f x) hidden (scores z t))
    let ok z := first z ≠ none
    let p := μ {z | ∃ t ∈ grid, StrictOn (scores z t) (f ∘ hidden)}
    (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {ω | retained ok ω m = none} = (1-p)^m) ∧
      ∀ᵐ ω ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j t, first (ω j) = some t ∧
        StrictOn (scores (ω j) t) (f ∘ hidden) ∧ ∀ k, retained ok ω (j+1+k) = some (ω j) := by
  intro first ok p
  let U := {z | ∃ t ∈ grid, StrictOn (scores z t) (f ∘ hidden)}
  have he (z : D) : first z = firstPerfect grid (fun t => StrictOn (scores z t) (f ∘ hidden)) :=
    firstPerfect_hidden_certificate grid S family hid f hf hidden (scores z)
  have hnone (z : D) : first z = none ↔ z ∉ U := by rw [he,firstPerfect_eq_none]; rfl
  have hU := (firstPerfect_probability μ grid _ hu).1
  constructor
  · intro m
    have hset : {ω | retained ok ω m = none} = {ω | ∀ j ∈ Finset.Ico 0 (0+m), ω j ∉ U} := by
      ext ω
      simp only [Set.mem_ofPred_eq,retained_none_iff,ok,not_ne_iff,hnone,zero_add,Nat.Ico_zero_eq_range]
    exact (congrArg (Measure.infinitePi (fun _ : ℕ => μ)) hset).trans (iid_miss_window μ hU 0 m)
  · filter_upwards [iid_hits_after μ hU hp 0] with ω h
    obtain ⟨l,_,hl⟩ := h
    obtain ⟨j,hj,_,hs⟩ := retained_settles ok ω ⟨l,fun h => (hnone _).mp h hl⟩
    cases ht : first (ω j) with
    | none => exact (hj ht).elim
    | some t => exact ⟨j,t,ht,((firstPerfect_eq_some _ _ _).1 ((he _).symm.trans ht)).2.1,hs⟩

namespace Argand

variable {A X Z : Type*} [Fintype A] [Finite X] [Finite Z] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

omit [Finite X] in
/-- One successful ordinary trajectory in the real native draw space gives
almost-sure strictly correct retention. The witness may be at a positive
step after a wrong initialization. The stopping test uses only observations
and the known class. No auxiliary noise is added to any optimizer step. -/
theorem uncorrectedAdam_ordinary_restarts (r : ℕ) (grid : Finset ℕ) (a b ε decay : ℝ) (hε : 0 < ε) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : Z → Word A (T+1)) (hidden : Z → X)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let score (z : Draw) t x := (Model.fromWeights (uncorrectedAdamRun r a b ε decay batch rate z t).weights).logits
      r (prompt x) (Fin.last T)
    let μ := independentGaussianDraws variance
    let first z := firstPerfect grid (fun t => hiddenCertificate S family (fun x => f x) hidden (score z t))
    let ok z := first z ≠ none
    let p := μ {z | ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)}
    (∃ z, ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)) →
      0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {ω | retained ok ω m = none} = (1-p)^m) ∧
      ∀ᵐ ω ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j t, first (ω j) = some t ∧
        StrictOn (score (ω j) t) (f ∘ hidden) ∧ ∀ k, retained ok ω (j+1+k) = some (ω j) := by
  intro score μ first ok p witness
  have ho (t : ℕ) : IsOpen {z | StrictOn (score z t) (f ∘ hidden)} :=
    (strict_prompt_weights_open r prompt (f ∘ hidden)).preimage
      ((EqualMoment.continuous_uncorrectedRun decay rate (fun t => batchGradient r (batch t)) a b
        hε (fun t => continuous_batchGradient r (batch t)) t).1.comp
        (continuous_sgdInitialWeights true))
  let := independentGaussianDraws_openPos variance hv
  obtain ⟨z,t,ht,hz⟩ := witness
  have hp : 0 < p := ((ho t).measure_pos μ ⟨z,hz⟩).trans_le (measure_mono fun w hw => ⟨t,ht,hw⟩)
  exact ⟨hp,certified_grid_restarts μ grid S family hid f hf hidden score (fun t _ => (ho t).measurableSet) hp⟩

omit [Finite X] in
theorem equalMoment_ordinary_restarts (r : ℕ) (grid : Finset ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : Z → Word A (T+1)) (hidden : Z → X)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    let score (z : Draw) t x := (Model.fromWeights (equalMomentRun r decay batch rate z t).weights).logits
      r (prompt x) (Fin.last T)
    let μ := independentGaussianDraws variance
    let first z := firstPerfect grid (fun t => hiddenCertificate S family (fun x => f x) hidden (score z t))
    let ok z := first z ≠ none
    let p := μ {z | ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)}
    (∃ z, ∃ t ∈ grid, StrictOn (score z t) (f ∘ hidden)) →
      0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {ω | retained ok ω m = none} = (1-p)^m) ∧
      ∀ᵐ ω ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j t, first (ω j) = some t ∧
        StrictOn (score (ω j) t) (f ∘ hidden) ∧ ∀ k, retained ok ω (j+1+k) = some (ω j) :=
  uncorrectedAdam_ordinary_restarts r grid (9/10) (9/10) (1/100000000) decay (by norm_num) rate batch
    prompt hidden S family hid f hf variance hv

/-- One restart count works for the whole finite class at any requested
confidence, even when the minibatch order depends on all draws. The lower
bound comes from the mass of strictly correct native starts; it gives no
numerical estimate of how often ordinary training learns from wrong starts. -/
theorem uncorrectedAdam_uniform_observed_confidence (r : ℕ) (grid : Finset ℕ) (h0 : 0 ∈ grid)
    (prompt : Z → Word A (T+1)) (hidden : Z → X) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (hnative : ∀ f ∈ family, ∃ z : Draw, StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits
      r (prompt x) (Fin.last T)) (f ∘ hidden)) {failure : ℝ≥0∞} (hfailure : 0 < failure)
    (secondDecay : ℝ := 9/10) :
    let μ := independentGaussianDraws variance
    ∃ n, ∀ m ≥ n, ∀ f ∈ family, ∀ decay : ℝ, ∀ rate : ℕ → ℝ,
      ∀ batch : (ℕ → Draw) → ℕ → ℕ → List (LabeledPosition A),
      let score (ω : ℕ → Draw) j t x := (Model.fromWeights
        (uncorrectedAdamRun r (9/10) secondDecay (1/100000000) decay (batch ω j) rate (ω j) t).weights).logits r (prompt x) (Fin.last T)
      let first ω j := firstPerfect grid (fun t => hiddenCertificate S family (fun x => f x) hidden (score ω j t))
      (∀ ω j t, first ω j = some t → StrictOn (score ω j t) (f ∘ hidden)) ∧
      Measure.infinitePi (fun _ : ℕ => μ) {ω | ∀ j ∈ Finset.range m, first ω j = none} < failure := by
  intro μ
  let U (f : family) := {z : Draw | StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits
    r (prompt x) (Fin.last T)) (f.val ∘ hidden)}
  have ho (f : family) : IsOpen (U f) := (strict_prompt_weights_open r prompt (f.val ∘ hidden)).preimage
    (continuous_sgdInitialWeights true)
  let := independentGaussianDraws_openPos variance hv
  obtain ⟨n,hn⟩ := uniform_restart_confidence (fun f : family => μ (U f)) (fun f =>
    (ho f).measure_pos μ (hnative f f.property)) hfailure
  refine ⟨n,fun m hm f hf decay rate batch => ?_⟩
  intro score first
  constructor
  · intro ω j t ht
    have he := firstPerfect_hidden_certificate grid S family hid f hf hidden (score ω j)
    exact ((firstPerfect_eq_some _ _ _).1 (he.symm.trans ht)).2.1
  let i : family := ⟨f,hf⟩
  apply lt_of_le_of_lt _ (hn m hm i)
  rw [← iid_miss_window μ (ho i).measurableSet 0 m]
  simp only [zero_add,Nat.Ico_zero_eq_range]
  apply measure_mono
  intro ω h j hj hwin
  exact (firstPerfect_eq_none _ _).1 (h j hj) ⟨0,h0,f,hf,fun _ => rfl,hwin⟩

alias equalMoment_uniform_observed_confidence := uncorrectedAdam_uniform_observed_confidence

open Division.Factors

/-- Every quotient-factor table at stock width 64, depth six and one pass
has a native strict realization. It therefore satisfies ordinary-restart
convergence on every identifying mask, for every fixed minibatch order
and the practical WSD peak 0.003. The grid checks the initialization. -/
theorem factor_uncorrectedAdam_ordinary_restarts (cap : ℕ) (S : Set DivisionInput)
    (hc : (FlatSpectrum.patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (hidden : Z → DivisionInput) (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (secondDecay : ℝ := 9/10) :
    let score (z : ComplexParameter (Fin 239) 4 16 6 → ℂ) t x := (Model.fromWeights
      (uncorrectedAdamRun 1 (9/10) secondDecay (1/100000000) (1/10) batch (fun k => 3*wsdRate 20000 k) z t).weights).logits
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
  apply uncorrectedAdam_ordinary_restarts 1 _ _ _ _ _ (by norm_num) _ batch _ hidden S _ (factorClass_identified _ _ hc hz)
    f hf _ (by intro; norm_num)
  exact ⟨w,0,by simp [evaluationGrid],fun z => hw (hidden z)⟩

alias factor_equalMoment_ordinary_restarts := factor_uncorrectedAdam_ordinary_restarts

example (batch : ℕ → List (LabeledPosition (Fin 239))) :=
  factor_equalMoment_ordinary_restarts 20000 (factorMask (1 : DivisionUnit))
    (by rw [unitMask_factorMask]; exact (FlatSpectrum.diagRow_connected _).preconnected)
    ⟨1,mem_insert _ _⟩ id divisionAnswer divisionAnswer_mem batch

end Argand
end Transformer.Grokking.FiniteRecovery
