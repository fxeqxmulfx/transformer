module

public import Transformer.Grokking.FiniteRecovery.OrdinaryDivision
public import Transformer.Grokking.FiniteRecovery.TerminalValidation

/-! Certify a stock AdamW transition from an initially imperfect table
to universal strict correctness on the hidden prompts. The statistical
test uses independent training executions as the experimental units. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.OrdinaryDivision

open Set MeasureTheory
open scoped NNReal ENNReal

def learnedBy {X A : Type*} (grid : Finset ℕ) (score : ℕ → X → A → ℝ) (f : X → A) : Prop :=
  ¬StrictOn (score 0) f ∧ ∃ t ∈ grid, StrictOn (score t) f

/-- Already correct before the first ordinary gradient update. All native
gates and frequency offsets take their stock initial values. -/
def initialSuccess (hidden : Set DivisionInput) : Set Draw :=
  {z | ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x →
    (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4) b <
    (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)}

/-- On the original mask all 73! strict assignments of the unused
quotient tokens to their selected prompts are equiprobable at step zero. -/
theorem initial_success_ceiling (hidden : Set DivisionInput) (probe : Fin 73 → DivisionInput)
    (inHidden : ∀ i, probe i ∈ hidden) (answers : ∀ i, (divisionAnswer (probe i)).val = 32+i.val)
    (unused : ∀ i j, divisionPrompt (probe i) j ∉ divisionProbeTokens) (variance : ℝ≥0) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => variance)
    μ.real (initialSuccess hidden) ≤ 1/(Nat.factorial 73 : ℝ) := by
  intro μ
  let score (z : Draw) x := (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4)
  apply division_probe_ceiling μ score hidden probe inHidden answers
  let prompt (x : divisionProbeTokens) := divisionPrompt (probe (divisionProbeIndex x))
  apply symmetric_assignment_mass μ divisionProbeTokens (fun z x => score z (probe (divisionProbeIndex x)))
    (fun f => ((strict_prompt_weights_open 1 prompt f).preimage (continuous_sgdInitialWeights true)).measurableSet)
    (fun σ => relabelDraw (Equiv.Perm.ofSubtype σ))
    (fun σ => gaussian_relabel_preimage (Equiv.Perm.ofSubtype σ) (fun _ => variance))
  intro σ z x a
  let e := Equiv.Perm.ofSubtype σ
  have fixed : e ∘ prompt x = prompt x := funext fun j =>
    Equiv.Perm.ofSubtype_apply_of_not_mem σ (unused (divisionProbeIndex x) j)
  change (Model.fromWeights (sgdInitialWeights true (relabelDraw e z))).logits 1 (prompt x) (Fin.last 4) (e a) = _
  rw [initialWeights_relabel,fromWeights_relabel]
  simpa only [fixed] using (Model.logits_relabel e (Model.fromWeights (sgdInitialWeights true z)) 1 (prompt x) (Fin.last 4) a)

/-- The exact mass of first hits that require learning is obtained by
subtracting the initially correct starts, for every ordinary batch policy. -/
theorem learned_mass (μ : Measure Draw) [IsProbabilityMeasure μ] (B : ℕ) (rate : ℕ → ℝ)
    (batch : Draw → Batches) (hidden : Set DivisionInput) :
    μ.real (hit B rate batch hidden \ initialSuccess hidden) =
      μ.real (hit B rate batch hidden) - μ.real (initialSuccess hidden) := by
  refine measureReal_sdiff ?_ ?_ (measure_ne_top μ _)
  · exact fun z hz => ⟨0,by simp [evaluationGrid],hz⟩
  · let prompt (x : hidden) := divisionPrompt x.val
    have ho := (strict_prompt_weights_open (H := 4) (d := 16) (L := 6) 1 prompt
      (fun x => divisionAnswer x.val)).preimage (continuous_sgdInitialWeights true)
    simpa only [initialSuccess,StrictOn,Set.preimage_ofPred_eq,Subtype.forall,prompt] using ho.measurableSet

/-- A function determined by the observations verifies both that the
initial table fails and that a later ordinary iterate strictly passes.
This statement also applies to recorded hardware scores. -/
theorem observed_learning_iff {X Z A : Type*} (S : Set X) (family : Set (X → A))
    (id : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hidden : Z → X) (grid : Finset ℕ) (score : ℕ → Z → A → ℝ) :
    (∃ g ∈ family, (∀ x : S, g x = f x) ∧ learnedBy grid score (g ∘ hidden)) ↔
      learnedBy grid score (f ∘ hidden) :=
  identified_property_iff S family id f hf _

/-- The probability of accepting eleven independent ordinary runs when
the true single-run learned-success probability is at most three quarters.
The full execution may include device arithmetic and an interruption. -/
theorem eleven_learned_null {D X A : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (grid : Finset ℕ) (score : D → ℕ → X → A → ℝ) (f : X → A) (completed : D → Prop)
    (meas : MeasurableSet {z | completed z ∧ learnedBy grid (score z) f})
    (null : μ {z | completed z ∧ learnedBy grid (score z) f} ≤ 3/4) :
    Measure.infinitePi (fun _ : ℕ => μ) (allPassWindow {z | completed z ∧ learnedBy grid (score z) f} 0 11) ≤
      (3/4 : ℝ≥0∞)^11 ∧ (3/4 : ℝ≥0∞)^11 < 1/20 := by
  refine ⟨(iid_allPassWindow μ meas 0 11).le.trans (pow_le_pow_left' null _),?_⟩
  apply (ENNReal.toReal_lt_toReal (by finiteness) (by finiteness)).mp
  norm_num

/-- With the actual six-layer model and division rate, the transition is
measurable under a fixed minibatch stream, including checks before the cap.
The numerical hypothesis in the null test concerns whole ordinary runs. -/
theorem stock_learned_measurable {Z : Type*} [Finite Z] (hidden : Z → DivisionInput)
    (batch : Batches) (cap : ℕ) :
    MeasurableSet {z : Draw | learnedBy (evaluationGrid 256 cap)
      (fun t x => (Model.fromWeights (adamWRun 1 batch (fun k => 3*wsdRate 20000 k) z t).weights).logits
        1 (divisionPrompt (hidden x)) (Fin.last 4)) (divisionAnswer ∘ hidden)} := by
  let U (t : ℕ) := {z : Draw | StrictOn (fun x =>
    (Model.fromWeights (adamWRun 1 batch (fun k => 3*wsdRate 20000 k) z t).weights).logits
      1 (divisionPrompt (hidden x)) (Fin.last 4)) (divisionAnswer ∘ hidden)}
  have hm t : MeasurableSet (U t) :=
    (adamW_strict_open 1 t _ batch (divisionPrompt ∘ hidden) (divisionAnswer ∘ hidden)).measurableSet
  change MeasurableSet ((U 0)ᶜ ∩ {z | ∃ t ∈ evaluationGrid 256 cap, z ∈ U t})
  rw [show {z | ∃ t ∈ evaluationGrid 256 cap, z ∈ U t} = ⋃ t ∈ evaluationGrid 256 cap, U t by ext; simp]
  exact (hm 0).compl.inter (MeasurableSet.biUnion (evaluationGrid 256 cap).countable_toSet (fun t _ => hm t))

example (batch : Batches) := stock_learned_measurable id batch 8192

/-- A shuffle selected by the same seed as the weights can be dependent
on the initialization. A measurable choice among countably many streams
still gives a measurable finite-budget hit event. -/
theorem hit_measurable_by_seed {S : Type*} [Countable S] (select : Draw → S)
    (fibers : ∀ s, MeasurableSet {z | select z = s}) (streams : S → Batches)
    (hidden : Set DivisionInput) (B : ℕ) (rate : ℕ → ℝ) :
    MeasurableSet (hit B rate (streams ∘ select) hidden) := by
  let U (s : S) (t : ℕ) := {z : Draw | StrictOn (fun x : hidden =>
    (Model.fromWeights (adamWRun 1 (streams s) rate z t).weights).logits 1 (divisionPrompt x.val) (Fin.last 4))
    (fun x => divisionAnswer x.val)}
  have hm s t : MeasurableSet (U s t) :=
    (adamW_strict_open 1 t rate (streams s) (fun x : hidden => divisionPrompt x.val)
      (fun x => divisionAnswer x.val)).measurableSet
  have eq : hit B rate (streams ∘ select) hidden =
      ⋃ s, {z | select z = s} ∩ ⋃ t ∈ evaluationGrid 256 B, U s t := by
    ext z
    simp [hit,U,StrictOn,Subtype.forall]
  rw [eq]
  exact MeasurableSet.iUnion fun s => (fibers s).inter
    (MeasurableSet.biUnion (evaluationGrid 256 B).countable_toSet (fun t _ => hm s t))

end Transformer.Grokking.FiniteRecovery.Argand.OrdinaryDivision
