module

public import Transformer.Phasor.TripleTransport.OverlappingEvidence

/-!
# Reading learned prototype relations under role permutations

Source: “Evidence from permutations of hidden roles” in the triple-transport
experiment. Partial observations can support permutations of the context,
key and output. The native search also compares the fraction of shared
pairs when it joins contexts in a symmetry-augmented bank.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.ObservedSymmetry

variable {A I : Type*}

/-- A partial observation of the relation between three learned prototype roles. -/
abbrev Triple (A : Type*) := A × A × A

def graph (f : A → A → A) : Set (Triple A) := {p | f p.1 p.2.1 = p.2.2}

def reachable (actions : I → Triple A → Triple A) (selected : I → Prop)
    (p q : Triple A) : Prop :=
  Relation.ReflTransGen (fun a b => ∃ i, selected i ∧ actions i a = b) p q

/-- Following any number of supported role permutations preserves the true
graph when each selected permutation preserves that graph. The initial
bank may leave arbitrarily many pairs unanswered. -/
theorem reached_in_graph (f : A → A → A) (bank : Set (Triple A)) (sound : bank ⊆ graph f)
    (actions : I → Triple A → Triple A) (selected : I → Prop)
    (valid : ∀ i, selected i → ∀ p ∈ graph f, actions i p ∈ graph f)
    {p q : Triple A} (seen : p ∈ bank) (path : reachable actions selected p q) :
    q ∈ graph f := by
  induction path with
  | refl => exact sound seen
  | tail _ step ih =>
    obtain ⟨i, hs, rfl⟩ := step
    exact valid i hs _ ih

/-- Two proposed outputs for a single cell obtained from sound role
permutations agree, even when reached via different witnesses. -/
theorem proposed_unique (f : A → A → A) (bank : Set (Triple A)) (sound : bank ⊆ graph f)
    (actions : I → Triple A → Triple A) (selected : I → Prop)
    (valid : ∀ i, selected i → ∀ p ∈ graph f, actions i p ∈ graph f)
    {p q : Triple A} {a b y z : A} (hp : p ∈ bank) (hq : q ∈ bank)
    (left : reachable actions selected p (a,b,y)) (right : reachable actions selected q (a,b,z)) :
    y = z := by
  have h : f a b = y := reached_in_graph f bank sound actions selected valid hp left
  have k : f a b = z := reached_in_graph f bank sound actions selected valid hq right
  exact h.symm.trans k

def exchange (p : Triple A) : Triple A := (p.2.1, p.1, p.2.2)

private def action (i : Bool) (p : Triple ℕ) : Triple ℕ := if i then exchange p else p
private def observed : Set (Triple ℕ) := {(2, 3, 5), (3, 2, 5)}

private theorem observed_sound : observed ⊆ graph Nat.add := by
  intro p h
  rcases (show p = (2, 3, 5) ∨ p = (3, 2, 5) from h) with rfl | rfl <;> norm_num [graph, Set.mem_ofPred_eq]

private theorem add_actions (i : Bool) (p : Triple ℕ) (h : p ∈ graph Nat.add) :
    action i p ∈ graph Nat.add := by
  rcases p with ⟨a,b,y⟩
  cases i
  · exact h
  · exact Nat.add_comm b a |>.trans h

private theorem swap_path : reachable action (fun i => i = true) (2,3,5) (3,2,5) :=
  .single ⟨true, rfl, rfl⟩

example : (3,2,5) ∈ graph Nat.add :=
  reached_in_graph _ observed observed_sound action (fun i => i = true)
    (fun i _ => add_actions i) (show (2,3,5) ∈ observed from .inl rfl) swap_path

example (y : ℕ) (path : reachable action (fun i => i = true) (3,2,5) (3,2,y)) :
    5 = y :=
  proposed_unique _ observed observed_sound action (fun i => i = true)
    (fun i _ => add_actions i) (show (2,3,5) ∈ observed from .inl rfl)
    (show (3,2,5) ∈ observed from .inr rfl) swap_path path

/-- Observed symmetry can coexist with different outputs on unseen keys. Both orientations of one off-diagonal pair are present in this
example, but there is a different, missing pair on which exchange fails. -/
example : (∀ p ∈ observed, exchange p ∈ observed) ∧
    ∃ f : ℕ → ℕ → ℕ, observed ⊆ graph f ∧
      (0,1,1) ∈ graph f ∧ exchange (0,1,1) ∉ graph f := by
  constructor
  · intro p h
    rcases (show p = (2,3,5) ∨ p = (3,2,5) from h) with rfl | rfl
    · exact .inr rfl
    · exact .inl rfl
  · let f := fun a b : ℕ => if a = 1 ∧ b = 0 then 0 else a + b
    refine ⟨f, ?_, by norm_num [f, graph, Set.mem_ofPred_eq], by norm_num [f, exchange, graph, Set.mem_ofPred_eq]⟩
    intro p h
    rcases (show p = (2,3,5) ∨ p = (3,2,5) from h) with rfl | rfl <;> norm_num [f, graph, Set.mem_ofPred_eq]

end Transformer.Phasor.TripleTransport.ObservedSymmetry

namespace Transformer.Phasor.TripleTransport.RelativeWitnesses

variable {R Y M : Type*} [DecidableEq Y]

/-- Two partial observations must share k distinct witnesses and at least
one q-th of the smaller observation. -/
def linked (k q : ℕ) (a b : Finset Y) : Prop :=
  k ≤ (a ∩ b).card ∧ min a.card b.card ≤ q * (a ∩ b).card

/-- Relations may overlap in enough absolute witnesses to defeat a fixed
cutoff. A relative cutoff still separates them whenever their full overlap
is too small compared to the observed evidence. This guarantee concerns
connected components, not just direct comparisons. -/
theorem same_mode (k q : ℕ) (seen : R → Finset Y) (truth : M → Finset Y) (mode : R → M)
    (sound : ∀ r, seen r ⊆ truth (mode r))
    (distinct : ∀ i j, mode i ≠ mode j →
      (truth (mode i) ∩ truth (mode j)).card < k ∨
      q * (truth (mode i) ∩ truth (mode j)).card < min (seen i).card (seen j).card)
    {r s : R} (path : Relation.EqvGen (fun i j => linked k q (seen i) (seen j)) r s) :
    mode r = mode s := by
  induction path with
  | rel i j link =>
    by_contra diff
    have sub := Finset.card_le_card (Finset.inter_subset_inter (sound i) (sound j))
    rcases distinct i j diff with few | sparse
    · exact (Nat.not_lt_of_ge (link.1.trans sub)) few
    · exact (Nat.not_lt_of_ge (link.2.trans (Nat.mul_le_mul_left q sub))) sparse
  | refl => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ a b => exact a.trans b

private def samples (m : Bool) : Finset ℕ :=
  if m then (Finset.range 16).image (· + 10) else Finset.range 16

/-- Six distinct common outputs need not identify the same relation. -/
example : 6 ≤ (samples true ∩ samples false).card ∧ ¬ linked 6 2 (samples true) (samples false) := by
  change 6 ≤ _ ∧ ¬ (6 ≤ _ ∧ _ ≤ _)
  decide +kernel

example {a b : Bool × Bool}
    (path : Relation.EqvGen (fun i j : Bool × Bool => linked 6 2 (samples i.1) (samples j.1)) a b) :
    a.1 = b.1 := by
  apply same_mode 6 2 (fun i : Bool × Bool => samples i.1) samples Prod.fst (fun _ => by rfl) _ path
  intro i j different
  rcases i with ⟨i,_⟩
  rcases j with ⟨j,_⟩
  cases i <;> cases j <;> simp_all [samples] <;> decide

end Transformer.Phasor.TripleTransport.RelativeWitnesses
