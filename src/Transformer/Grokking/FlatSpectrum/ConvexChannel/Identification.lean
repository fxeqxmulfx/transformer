module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Optimality

/-!+# Learning generated recall from ordinary answer labels

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC5.
Every wrong routing/output atom must encounter an answer that contradicts
it while its input value is present. This is an explicit finite-data
condition. Under realizability and this separation, every global CE
optimizer is the generated rule, hence works on every new dictionary.
The condition does not follow from flatness or from sufficient rank.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {K V C S : Type*} [Fintype K] [Fintype C] [Fintype S]
  [Nonempty V] [DecidableEq K] [DecidableEq C]

/-- Observed inputs and answer labels contradict every incorrect atom.
This predicate has no learned weights or correctness premise (CC5). -/
def SeparatesAtoms (query : S → K) (data : S → K → V) (label : S → C)
    (g : K → V → C) : Prop :=
  ∀ q k v c, (k ≠ q ∨ c ≠ g q v) →
    ∃ s, query s = q ∧ data s k = v ∧ label s ≠ c

/-- Ordinary CE jointly identifies routing and output generation when
the actual dataset separates the incorrect atoms (CC5). -/
theorem eq_generator_of_answerLoss_le [Nontrivial C] (g : K → V → C)
    (query : S → K) (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s)))
    (hsep : SeparatesAtoms query data label g) {W : Channel K K V C}
    (hW : Feasible W)
    (hmin : answerLoss query data label W ≤ answerLoss query data label (generator g)) :
    W = generator g := by
  have hoff (q k : K) (v : V) (c : C) (h : k ≠ q ∨ c ≠ g q v) : W q k v c = 0 := by
    obtain ⟨s, hq, hv, hc⟩ := hsep q k v c h
    have hm := sample_target_mass_one g query data label hlabel hW hmin s
    have hz := atom_eq_zero hW (query s) (data s) (label s) hm k c (Ne.symm hc)
    simpa only [hq, hv] using hz
  have hdiag (q : K) (v : V) : W q q v (g q v) = 1 := by
    have hrow : ∑ c, W q q v c = W q q v (g q v) :=
      Finset.sum_eq_single (g q v) (fun c _ hc => hoff q q v c (Or.inr hc)) (by simp)
    have hall : ∑ k, ∑ c, W q k v c = ∑ c, W q q v c := by
      refine Finset.sum_eq_single q (fun k _ hk => ?_) (by simp)
      exact Finset.sum_eq_zero fun c _ => hoff q k v c (Or.inl hk)
    rw [hrow] at hall
    exact hall.symm.trans (hW.2.2 q v)
  funext q k v c
  by_cases hk : k = q
  · subst k
    by_cases hc : c = g q v
    · subst c
      simpa [generator] using hdiag q v
    · simpa [generator, hc] using hoff q q v c (Or.inr hc)
  · simpa [generator, hk] using hoff q k v c (Or.inl hk)

/-- Every global optimizer generates the correct answer on every test
dictionary, including dictionaries absent from training (CC5). -/
theorem trained_logits [Nontrivial C] (g : K → V → C)
    (query : S → K) (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s)))
    (hsep : SeparatesAtoms query data label g) {W : Channel K K V C}
    (hW : Feasible W) (hmin : IsMinOn (answerLoss query data label) {B | Feasible B} W)
    (q : K) (D : K → V) (c : C) :
    logits W q D c = if c = g q (D q) then 1 else 0 := by
  rw [eq_generator_of_answerLoss_le g query data label hlabel hsep hW
    (hmin (feasible_generator g))]
  exact logits_generator g q D c

/-- Generated recall has a strict correct argmax on every dictionary,
not merely zero-one training error (CC5). -/
theorem trained_strict_argmax [Nontrivial C] (g : K → V → C)
    (query : S → K) (data : S → K → V) (label : S → C)
    (hlabel : ∀ s, label s = g (query s) (data s (query s)))
    (hsep : SeparatesAtoms query data label g) {W : Channel K K V C}
    (hW : Feasible W) (hmin : IsMinOn (answerLoss query data label) {B | Feasible B} W)
    (q : K) (D : K → V) (c : C) (hc : c ≠ g q (D q)) :
    logits W q D c < logits W q D (g q (D q)) := by
  rw [trained_logits g query data label hlabel hsep hW hmin,
    trained_logits g query data label hlabel hsep hW hmin]
  simp [hc]

/-- All two-key Boolean dictionaries, queried at both keys with negated
answer labels, satisfy the separation condition (CC5). -/
theorem separatesAtoms_bool_not :
    SeparatesAtoms (Prod.fst : Bool × (Bool → Bool) → Bool) Prod.snd
      (fun s => !(s.2 s.1)) (fun _ : Bool => Bool.not) := by
  unfold SeparatesAtoms
  decide

example := eq_generator_of_answerLoss_le (fun _ : Bool => Bool.not) Prod.fst Prod.snd
  (fun s : Bool × (Bool → Bool) => !(s.2 s.1)) (fun _ => rfl)
  separatesAtoms_bool_not (feasible_generator (fun _ : Bool => Bool.not)) le_rfl

example := trained_logits (fun _ : Bool => Bool.not) Prod.fst Prod.snd
  (fun s : Bool × (Bool → Bool) => !(s.2 s.1)) (fun _ => rfl)
  separatesAtoms_bool_not (feasible_generator (fun _ : Bool => Bool.not))
  (isMinOn_generator _ _ _ _ (fun _ => rfl)) true (fun _ => false) true

example := trained_strict_argmax (fun _ : Bool => Bool.not) Prod.fst Prod.snd
  (fun s : Bool × (Bool → Bool) => !(s.2 s.1)) (fun _ => rfl)
  separatesAtoms_bool_not (feasible_generator (fun _ : Bool => Bool.not))
  (isMinOn_generator _ _ _ _ (fun _ => rfl)) true (fun _ => false) false (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
