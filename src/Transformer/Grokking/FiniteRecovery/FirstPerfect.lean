module

public import Transformer.Grokking.FiniteRecovery.ArgandFirstNoise
public import Transformer.Grokking.FiniteRecovery.FactorFirstNoise
public import Transformer.Grokking.FiniteRecovery.TerminalTail

/-! Stop a training trajectory on the first scheduled strict success.
All evaluations within a trajectory can be dependent. For repeated
trials the independent units are complete training paths. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set ProbabilityTheory
open scoped Classical NNReal

noncomputable def firstPerfect (grid : Finset ℕ) (ok : ℕ → Prop) : Option ℕ :=
  if h : ∃ t ∈ grid, ok t then some (Nat.find h) else none

/-- The division trainer evaluates at zero, the multiples of its positive
observation interval, and the cap even when the cap lies between them. -/
def evaluationGrid (interval cap : ℕ) : Finset ℕ :=
  insert cap ((Finset.range (cap+1)).filter fun t => interval ∣ t)

/-- The checkpoint saved by the stopping rule passes, lies on the grid,
and every earlier scheduled evaluation fails. -/
theorem firstPerfect_eq_some (grid : Finset ℕ) (ok : ℕ → Prop) (t : ℕ) :
    firstPerfect grid ok = some t ↔ t ∈ grid ∧ ok t ∧ ∀ s ∈ grid, s < t → ¬ok s := by
  by_cases h : ∃ t ∈ grid, ok t
  · rw [firstPerfect,dite_eq_left h,Option.some.injEq]
    constructor
    · rintro rfl
      exact ⟨(Nat.find_spec h).1,(Nat.find_spec h).2,fun s hs hl hp => Nat.find_min h hl ⟨hs,hp⟩⟩
    · rintro ⟨ht,hok,hm⟩
      exact le_antisymm (Nat.find_min' h ⟨ht,hok⟩) (le_of_not_gt fun hl =>
        hm _ (Nat.find_spec h).1 hl (Nat.find_spec h).2)
  · rw [firstPerfect,dite_eq_right h]
    exact ⟨fun e => by contradiction,fun ⟨ht,hs,_⟩ => (h ⟨t,ht,hs⟩).elim⟩

theorem firstPerfect_eq_none (grid : Finset ℕ) (ok : ℕ → Prop) :
    firstPerfect grid ok = none ↔ ¬∃ t ∈ grid, ok t := by
  by_cases h : ∃ t ∈ grid, ok t <;> simp [firstPerfect,h]

/-- The observation-only certificate and the true full strict score
select exactly the same first checkpoint on an identifying mask. -/
theorem firstPerfect_certificate {X A : Type*} (grid : Finset ℕ) (S : Set X)
    (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (scores : ℕ → X → A → ℝ) :
    firstPerfect grid (fun t => classCertificate S family (fun x => f x) (scores t)) =
      firstPerfect grid (fun t => StrictOn (scores t) f) := by
  exact congrArg (firstPerfect grid) (funext fun t => propext (classCertificate_iff S family hid f hf _))

/-- The numerical stopping condition may concern only a held-out subset.
An identifying observation mask determines the target needed to check
every hidden margin and gives the same stopping time. -/
theorem firstPerfect_hidden_certificate {X Z A : Type*} (grid : Finset ℕ) (S : Set X)
    (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (hidden : Z → X) (scores : ℕ → Z → A → ℝ) :
    firstPerfect grid (fun t => ∃ g ∈ family, (∀ x : S, g x = f x) ∧ StrictOn (scores t) (g ∘ hidden)) =
      firstPerfect grid (fun t => StrictOn (scores t) (f ∘ hidden)) :=
  congrArg (firstPerfect grid) (funext fun t => propext
    (identified_property_iff S family hid f hf (fun g => StrictOn (scores t) (g ∘ hidden))))

/-- The hit probability dominates each scheduled endpoint probability.
The chance that all independent paths miss the entire grid is exact. -/
theorem firstPerfect_probability {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    (grid : Finset ℕ) (U : ℕ → Set D) (hu : ∀ t ∈ grid, MeasurableSet (U t)) :
    let H := {z | ∃ t ∈ grid, z ∈ U t}
    MeasurableSet H ∧ (∀ t ∈ grid, μ (U t) ≤ μ H) ∧
      ∀ m, Measure.infinitePi (fun _ : ℕ => μ)
        {z | ∀ j ∈ Finset.range m, firstPerfect grid (fun t => z j ∈ U t) = none} = (1-μ H)^m := by
  intro H
  have hh : MeasurableSet H := by
    rw [show H = ⋃ t ∈ grid, U t by ext; simp [H]]
    exact grid.measurableSet_biUnion hu
  refine ⟨hh,fun t ht => measure_mono fun z hz => ⟨t,ht,hz⟩,fun m => ?_⟩
  simpa only [firstPerfect_eq_none,Nat.zero_add,Nat.Ico_zero_eq_range,H,Set.mem_ofPred_eq] using
    iid_miss_window μ hh 0 m

namespace Argand

/-- First-update Gaussian AdamW has positive probability of already
having stopped by the WSD cap on every strictly representable finite
task. Every scheduled evaluation uses the same training trajectory. -/
theorem firstNoise_hit_probability {A X : Type*} [Fintype A] [Finite X] {H d L T : ℕ}
    (r n : ℕ) (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1))
    (f : X → A) (draw : ComplexParameter A H d L → ℂ) (grid : Finset ℕ) (hk : n+1 ∈ grid)
    (hrep : ∃ w : Weights A H d L, StrictOn (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f)
    (v : ℝ≥0) (hv : 0 < v) :
    let μ := Measure.pi (fun _ : Parameter A H d L => gaussianReal 0 v)
    let ok z t := StrictOn (firstNoiseScores r t (wsdRate (n+1)) batch prompt draw z) f
    let p := μ {z | ∃ t ∈ grid, ok z t}
    0 < p ∧ ∀ m, Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∀ j ∈ Finset.range m, firstPerfect grid (ok (z j)) = none} = (1-p)^m := by
  intro μ ok p
  have hr (t : ℕ) := (wsdRate_bounds (n+1) t)
  have pos (k : ℕ) := firstNoise_strict_mass r k (wsdRate (n+1)) (fun t _ =>
    ⟨(hr t).1,lt_of_le_of_lt (hr t).2 (by norm_num)⟩) batch prompt f draw hrep v hv
  have hu (t : ℕ) : MeasurableSet {z | ok z t} := by
    cases t with
    | zero =>
      change MeasurableSet {z : Weights A H d L | StrictOn (fun x => (Model.fromWeights
        (sgdInitialWeights true draw)).logits r (prompt x) (Fin.last T)) f}
      exact MeasurableSet.const _
    | succ k => exact (pos k).1.measurableSet
  have h := firstPerfect_probability μ grid (fun t => {z | ok z t}) (fun t _ => hu t)
  exact ⟨(pos n).2.trans_le (h.2.1 _ hk),h.2.2⟩

open Division.Factors

/-- All unknown group-factor tables on any hidden subset of division
inputs use the experimental width 64, six layers, one pass, evaluations
every 256 updates and the unscaled 100,000-update WSD. -/
example {Z : Type*} [Finite Z] (hidden : Z → DivisionInput) (f : DivisionInput → Fin 239)
    (hf : f ∈ factorClass unitToken) (batch : ℕ → List (LabeledPosition (Fin 239)))
    (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  firstNoise_hit_probability 1 99999 batch (divisionPrompt ∘ hidden) (f ∘ hidden) draw
    (evaluationGrid 256 100000) (Finset.mem_insert_self _ _)
    (by obtain ⟨w,hw⟩ := class_native_realization f hf
        exact ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩)
    ((1/1000000 : ℝ≥0)^2) (by norm_num)

end Argand

example : firstPerfect ({0,256,512} : Finset ℕ) (fun t => t = 256) = some 256 :=
  (firstPerfect_eq_some _ _ _).2 ⟨by decide,rfl,fun _ _ h => h.ne⟩

end Transformer.Grokking.FiniteRecovery
