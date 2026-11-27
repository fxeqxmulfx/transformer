module

public import Mathlib.Probability.ProductMeasure
public import Mathlib.Probability.Distributions.Uniform

/-! A fixed-horizon learner may propose any scores, using only the observations.
An identifying mask makes acceptance decidable from its visible labels and
the admissible function class. Retaining the first verified proposal turns
independent attempts into an eventually constant exact classifier. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set Filter
open scoped Topology ENNReal Classical

variable {X A D : Type*}

/-- Every competing answer has a strictly smaller score on each input. -/
def StrictOn (score : X → A → ℝ) (f : X → A) : Prop :=
  ∀ x a, a ≠ f x → score x a < score x (f x)

/-- Only labels of observed inputs are arguments to this verifier. -/
def classCertificate (S : Set X) (family : Set (X → A)) (labels : S → A)
    (score : X → A → ℝ) : Prop :=
  ∃ g ∈ family, (∀ x : S, g x = labels x) ∧ StrictOn score g

/-- On an identifying mask, any property of the unknown function may be
checked on a member of the class fitting the observed labels. -/
theorem identified_property_iff (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (test : (X → A) → Prop) :
    (∃ g ∈ family, (∀ x : S, g x = f x) ∧ test g) ↔ test f := by
  constructor
  · rintro ⟨g,hg,he,hs⟩
    exact hid hg hf (funext he) ▸ hs
  · intro hs
    exact ⟨f,hf,fun _ => rfl,hs⟩

/-- This verifier accepts exactly the strictly correct predictions on the
entire domain. A different member cannot agree on the identifying mask. -/
theorem classCertificate_iff (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (score : X → A → ℝ) :
    classCertificate S family (fun x => f x) score ↔ StrictOn score f :=
  identified_property_iff S family hid f hf (StrictOn score)

/-- The first accepted draw is retained for deployment. -/
noncomputable def retained (ok : D → Prop) (z : ℕ → D) : ℕ → Option D
  | 0 => none
  | n+1 => match retained ok z n with
    | some d => some d
    | none => if ok (z n) then some (z n) else none

theorem retained_none_iff (ok : D → Prop) (z : ℕ → D) (n : ℕ) :
    retained ok z n = none ↔ ∀ j ∈ Finset.range n, ¬ok (z j) := by
  induction n with
  | zero => simp [retained]
  | succ n ih =>
    rw [Finset.range_add_one, Finset.forall_mem_insert, ← ih]
    cases h : retained ok z n with
    | none => by_cases hj : ok (z n) <;> simp [retained,h,hj]
    | some d => simp [retained,h]

private theorem retained_persists {ok : D → Prop} {z : ℕ → D} {n : ℕ} {d : D}
    (h : retained ok z n = some d) (k : ℕ) : retained ok z (n+k) = some d := by
  induction k with
  | zero => exact h
  | succ k ih =>
    change retained ok z ((n+k)+1) = some d
    rw [retained,ih]

/-- Given any successful draw, find the first and retain precisely that
draw at every later stage, even if subsequent proposals fail. -/
theorem retained_settles (ok : D → Prop) (z : ℕ → D) (h : ∃ j, ok (z j)) :
    ∃ j, ok (z j) ∧ (∀ i < j, ¬ok (z i)) ∧ ∀ k, retained ok z (j+1+k) = some (z j) := by
  let j := Nat.find h
  have hj : ok (z j) := Nat.find_spec h
  have hmin (i) (hi : i < j) : ¬ok (z i) := Nat.find_min h hi
  have hn := (retained_none_iff ok z j).2 (fun i hi => hmin i (Finset.mem_range.mp hi))
  exact ⟨j,hj,hmin,retained_persists (by simp only [retained,hn,ite_eq_left hj])⟩

variable [MeasurableSpace D]

theorem iid_miss_window (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hu : MeasurableSet U) (J m : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | ∀ i ∈ Finset.Ico J (J+m), z i ∉ U} = (1-μ U)^m := by
  change Measure.infinitePi _ (Set.pi (Finset.Ico J (J+m) : Set ℕ) (fun _ => Uᶜ)) = _
  rw [Measure.infinitePi_pi _ (fun _ _ => hu.compl)]
  simp [prob_compl_eq_one_sub hu]

theorem iid_hits_after (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hu : MeasurableSet U) (hp : 0 < μ U) (J : ℕ) :
    ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j ≥ J, z j ∈ U := by
  rw [ae_iff]
  apply le_antisymm _ bot_le
  have hlt : 1-μ U < 1 := ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hp.ne'
  apply le_of_tendsto_of_tendsto' tendsto_const_nhds (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
  intro m
  rw [← iid_miss_window μ hu J m]
  apply measure_mono
  intro z hz i hi hiU
  exact hz ⟨i,(Finset.mem_Ico.mp hi).1,hiU⟩

/-- A quantitative guarantee for the entire finite prefix and almost-sure
finite stabilization. The correctness claim applies to every hidden input
of the unknown function. The inner learner and its optimizer are arbitrary. -/
theorem certified_restarts (μ : Measure D) [IsProbabilityMeasure μ]
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (scores : D → X → A → ℝ)
    (hu : MeasurableSet {d | StrictOn (scores d) f}) (hp : 0 < μ {d | StrictOn (scores d) f}) :
    let ok := fun d => classCertificate S family (fun x => f x) (scores d)
    let p := μ {d | StrictOn (scores d) f}
    (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
    ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
      ∀ k, retained ok z (j+1+k) = some (z j) := by
  let U := {d | StrictOn (scores d) f}
  let ok d := classCertificate S family (fun x => f x) (scores d)
  have he d : ok d ↔ d ∈ U := classCertificate_iff S family hid f hf _
  constructor
  · intro m
    have hset : {z | retained ok z m = none} = {z | ∀ i ∈ Finset.Ico 0 (0+m), z i ∉ U} := by
      ext z
      simp only [Set.mem_ofPred_eq, retained_none_iff, he, zero_add, Nat.Ico_zero_eq_range]
    exact (congrArg (Measure.infinitePi (fun _ : ℕ => μ)) hset).trans (iid_miss_window μ hu 0 m)
  · filter_upwards [iid_hits_after μ hu hp 0] with z hz
    obtain ⟨j,_,hj⟩ := hz
    obtain ⟨k,hk,_,hs⟩ := retained_settles ok z ⟨j,(he _).2 hj⟩
    exact ⟨k,(he _).1 hk,hs⟩

example : StrictOn (fun x a : Bool => if x = a then (1:ℝ) else 0) id := by
  intro x a ha
  cases x <;> cases a <;> simp_all

/-- A single finite number of independent proposals suffices for every
member of a finite task class whenever their individual success masses
are positive. Taking δ = 1/4 gives success greater than three quarters. -/
theorem uniform_restart_confidence {I : Type*} [Finite I] (p : I → ℝ≥0∞) (hp : ∀ i, 0 < p i)
    {δ : ℝ≥0∞} (hδ : 0 < δ) : ∃ m, ∀ n ≥ m, ∀ i, (1 - p i)^n < δ := by
  apply eventually_atTop.mp
  apply eventually_all.mpr
  intro i
  exact (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one
    (ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero (hp i).ne')).eventually (gt_mem_nhds hδ)

/-- Count seeds for which the observation-compatible member passes the
terminal score test. Reading the count requires no external hidden labels. -/
noncomputable def observedSuccessCount (seeds : Finset D) (S : Set X) (family : Set (X → A))
    (labels : S → A) (test : D → (X → A) → Prop) : ℕ :=
  (seeds.filter fun z => ∃ g ∈ family, (∀ x : S, g x = labels x) ∧ test z g).card

/-- The success event may check every prompt, or only the hidden prompts.
The seed may determine both initialization and minibatch order. -/
theorem finite_seed_checked_probability [MeasurableSingletonClass D] (seeds : Finset D)
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (test : D → (X → A) → Prop) :
    Measure.uniformOfFinset seeds {z | test z f} =
      (observedSuccessCount seeds S family (fun x => f x) test : ℝ≥0∞) / seeds.card := by
  classical
  unfold observedSuccessCount
  rw [Measure.uniformOfFinset_apply]
  congr 2
  apply congrArg Finset.card
  apply Finset.filter_congr
  intro z _
  exact (identified_property_iff S family hid f hf (test z)).symm

example := uniform_restart_confidence (fun b : Bool => if b then 1/4 else 1/2)
  (by intro b; cases b <;> norm_num) (δ := 1/4) (by norm_num)

end Transformer.Grokking.FiniteRecovery
