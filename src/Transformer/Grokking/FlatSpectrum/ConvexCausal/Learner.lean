module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Optimality

/-!
# Finite learning from ordinary categorical answers

Source: `review/convex_causal.md`, CCa5. The actual finite filter receives a
fixed bank of complete causal rules, observed inputs and ordinary labels.
It has explicit inconsistent and ambiguous returns. Pairwise separation is
checked on the bank's observed-input signatures; it contains no unseen labels.
Under realizability and separation the returned rule equals the unknown target.
Even without separation, a nonempty survivor set constructs a global minimum
of the repaired CE. No optimizer convergence or target-specific feature is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {H X C S : Type*} [Fintype H] [DecidableEq H] [Fintype S] [DecidableEq C]

/-- Finite elimination of every rule contradicted by an observed answer (CCa5). -/
def survivors (rule : H → X → C) (input : S → X) (label : S → C) : Finset H :=
  Finset.univ.filter fun h => ∀ s, rule h (input s) = label s

/-- Explicit outcomes, including refusal to guess an ambiguous rule (CCa5). -/
inductive Verdict (H : Type*)
  | inconsistent
  | ambiguous
  | learned (rule : H)
  deriving DecidableEq

/-- The learner enumerates and filters the supplied bank; it receives no target rule (CCa5). -/
def learn [LinearOrder H] (rule : H → X → C) (input : S → X) (label : S → C) : Verdict H :=
  let s := survivors rule input label
  if hs : s.Nonempty then
    if s.card = 1 then .learned (s.min' hs) else .ambiguous
  else .inconsistent

/-- Distinct candidate signatures differ on an observed input (CCa5).
This is an input-only condition relative to the fixed bank. -/
def Separates (rule : H → X → C) (input : S → X) : Prop :=
  ∀ h k, h ≠ k → ∃ s, rule h (input s) ≠ rule k (input s)

omit [DecidableEq H] in
/-- A realizable rule survives actual finite elimination (CCa5). -/
theorem target_survives (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) : g ∈ survivors rule input label := by
  simp only [survivors, Finset.mem_filter, Finset.mem_univ, true_and]
  exact hfit

/-- Separation derives uniqueness by eliminating every incorrect rule (CCa5). -/
theorem survivors_singleton (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input) :
    survivors rule input label = {g} := by
  ext h
  simp only [survivors, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  constructor
  · intro hh
    by_contra hne
    obtain ⟨s, hs⟩ := hsep h g hne
    exact hs ((hh s).trans (hfit s).symm)
  · rintro rfl
    exact hfit

/-- The computed learner returns the unknown rule, not an assumed fitted checkpoint (CCa5). -/
theorem learn_target [LinearOrder H] (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input) :
    learn rule input label = .learned g := by
  unfold learn
  rw [survivors_singleton rule input label g hfit hsep]
  simp

/-- Processing two batches retains precisely the intersection of survivors (CCa5). -/
theorem survivors_batches {T : Type*} [Fintype T] (rule : H → X → C)
    (input : S → X) (label : S → C) (input' : T → X) (label' : T → C) :
    survivors rule (Sum.elim input input') (Sum.elim label label') =
      survivors rule input label ∩ survivors rule input' label' := by
  ext h
  simp [survivors]

omit [DecidableEq H] in
/-- Reordering or repeating observations does not change the retained candidates (CCa5). -/
theorem survivors_reindex {T : Type*} [Fintype T] (rule : H → X → C)
    (input : S → X) (label : S → C) (f : T → S) (hf : Function.Surjective f) :
    survivors rule (input ∘ f) (label ∘ f) = survivors rule input label := by
  ext h
  simp only [survivors, Finset.mem_filter, Finset.mem_univ, true_and, Function.comp_apply]
  constructor
  · intro ht s
    obtain ⟨t, rfl⟩ := hf s
    exact ht t
  · intro hs t
    exact hs (f t)

variable [Fintype C] [Nonempty C]

/-- A real-valued parameter vector constructed from the finite elimination output (CCa5). -/
noncomputable def trained (rule : H → X → C) (input : S → X) (label : S → C) : X → C → ℝ :=
  mix (fun h => point (rule h)) (uniformOn (survivors rule input label))

omit [Fintype C] [Nonempty C] in
/-- Every surviving mixture fits all original observations at unit unsmoothed mass (CCa5). -/
theorem trained_fit (rule : H → X → C) (input : S → X) (label : S → C)
    (hs : (survivors rule input label).Nonempty) (s : S) :
    trained rule input label (input s) (label s) = 1 := by
  have hw := simplex_uniformOn hs
  change (∑ h, uniformOn (survivors rule input label) h *
    point (rule h) (input s) (label s)) = 1
  calc
    _ = ∑ h, uniformOn (survivors rule input label) h := by
      apply Finset.sum_congr rfl
      intro h _
      by_cases hh : h ∈ survivors rule input label
      · have ht := (Finset.mem_filter.mp hh).2 s
        simp [point, ht]
      · simp [uniformOn, hh]
    _ = 1 := hw.2

/-- Finite training constructs a global minimum of the same ordinary CE (CCa5). -/
theorem trained_minimum (rule : H → X → C) (input : S → X) (label : S → C)
    (hs : (survivors rule input label).Nonempty) {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1) :
    Rows (trained rule input label) ∧
      IsMinOn (loss ε input label) {Q | Rows Q} (trained rule input label) := by
  have hr := mix_rows (fun h => rows_point (rule h)) (simplex_uniformOn hs)
  exact ⟨hr, fit_minimum hr hε hε1 input label (trained_fit rule input label hs)⟩

omit [Fintype C] [Nonempty C] in
/-- Under input separation the trained distribution recovers the rule on every input (CCa5). -/
theorem trained_target (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input) :
    trained rule input label = point (rule g) := by
  unfold trained
  rw [survivors_singleton rule input label g hfit hsep]
  have hw : uniformOn ({g} : Finset H) = vertex g := by
    funext h
    simp [uniformOn, vertex]
  rw [hw, mix_vertex]

/-! A nonconstant parity family with one observation and three unseen inputs. -/

/-- Two possible orientations of the same two-bit interaction (CCa5 witness). -/
def parityRule (h : Bool) (x : Bool × Bool) : Bool := xor h (xor x.1 x.2)

/-- One observed input separates the two complete rules (CCa5 witness). -/
theorem parity_separates : Separates parityRule (fun _ : Unit => (false, false)) := by
  intro h k hne
  refine ⟨(), ?_⟩
  simpa [parityRule] using hne

example := target_survives parityRule (fun _ : Unit => (false, false))
  (fun _ => false) false (by intro s; rfl)
example := survivors_singleton parityRule (fun _ : Unit => (false, false))
  (fun _ => false) false (by intro s; rfl) parity_separates
example := survivors_reindex parityRule (fun _ : Unit => (false, false))
  (fun _ => false) (fun _ : Bool => ()) (by intro s; cases s; exact ⟨false, rfl⟩)
example := trained_fit parityRule (fun _ : Unit => (false, false))
  (fun _ => false) ⟨false, by simp [survivors, parityRule]⟩ ()
example := learn_target parityRule (fun _ : Unit => (false, false))
  (fun _ => false) false (by intro s; rfl) parity_separates
example := trained_minimum parityRule (fun _ : Unit => (false, false))
  (fun _ => false) ⟨false, by simp [survivors, parityRule]⟩ (ε := 1 / 4)
  (by norm_num) (by norm_num)
example := trained_target parityRule (fun _ : Unit => (false, false))
  (fun _ => false) false (by intro s; rfl) parity_separates

end Transformer.Grokking.FlatSpectrum.ConvexCausal
