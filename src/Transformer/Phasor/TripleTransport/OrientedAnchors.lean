module

public import Transformer.Phasor.TripleTransport.Incremental

/-!
# Both orientations of one unordered anchor pair

Source: “Representing the two anchor orders” in the triple-transport
experiment. Every trusted triple match also matches its simultaneous swap.
The proof relates the original ordered graph to the signed component test.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.OrientedAnchors
variable {I : Type*}

inductive Walk (edge : I → I → Bool → Prop) : I → I → Bool → Prop where
  | edge {a b g} : edge a b g → Walk edge a b g
  | refl (a) : Walk edge a a false
  | symm {a b g} : Walk edge a b g → Walk edge b a g
  | trans {a b c f g} : Walk edge a b f → Walk edge b c g → Walk edge a c (f ^^ g)

/-- Swapping both anchors at both ends gives a second ordinary edge for
each signed correspondence, even for two different dictionary roles. -/
def lift (edge : I → I → Bool → Prop) (x y : I × Bool) : Prop :=
  edge x.1 y.1 (x.2 ^^ y.2)

theorem project {e : I → I → Bool → Prop} {x y : I × Bool}
    (h : Relation.EqvGen (lift e) x y) : Walk e x.1 y.1 (x.2 ^^ y.2) := by
  induction h with
  | rel _ _ h => exact .edge h
  | refl p => simpa only [Bool.xor_self] using Walk.refl (edge := e) p.1
  | symm p q _ ih => simpa only [Bool.xor_comm q.2 p.2] using ih.symm
  | trans p q r _ _ ih ih' =>
    have cancel : ∀ a b c : Bool, ((a ^^ b) ^^ (b ^^ c)) = (a ^^ c) := by decide
    simpa only [cancel] using ih.trans ih'

theorem realize {e : I → I → Bool → Prop} {a b : I} {g : Bool}
    (h : Walk e a b g) (s : Bool) : Relation.EqvGen (lift e) (a,s) (b,s ^^ g) := by
  induction h generalizing s with
  | @edge a b g h =>
    apply Relation.EqvGen.rel
    change e a b (s ^^ (s ^^ g))
    cases s <;> cases g <;> exact h
  | refl a => cases s <;> exact .refl _
  | @symm a b g _ ih =>
    have rev := Relation.EqvGen.symm _ _ (ih (s ^^ g))
    have cancel : ∀ u v : Bool, ((u ^^ v) ^^ v) = u := by decide
    simpa only [cancel] using rev
  | @trans a b c f g _ _ ih ih' =>
    simpa only [Bool.xor_assoc] using Relation.EqvGen.trans _ _ _ (ih s) (ih' (s ^^ f))

theorem connected {e : I → I → Bool → Prop} (a b : I) (s t : Bool) :
    Relation.EqvGen (lift e) (a,s) (b,t) ↔ Walk e a b (s ^^ t) := by
  constructor
  · exact project
  · intro h
    have cancel : ∀ u v : Bool, (u ^^ (u ^^ v)) = v := by decide
    simpa only [cancel] using realize h s

/-- Relative signs at a common root completely decide equality of the
two lifted components. Either both have the same root orientation, or
there is a closed walk reversing that orientation. -/
theorem at_root {e : I → I → Bool → Prop} {a b r : I} {p q : Bool}
    (ha : Walk e a r p) (hb : Walk e b r q) (s t : Bool) :
    Relation.EqvGen (lift e) (a,s) (b,t) ↔ (s ^^ p) = (t ^^ q) ∨ Walk e r r true := by
  rw [connected]
  constructor
  · intro h
    by_cases align : (s ^^ p) = (t ^^ q)
    · exact Or.inl align
    · right
      have rev : ∀ a b c d : Bool, (a ^^ c) ≠ (b ^^ d) → (c ^^ ((a ^^ b) ^^ d)) = true := by decide
      simpa only [rev _ _ _ _ align] using ha.symm.trans (h.trans hb)
  · rintro (align | twist)
    · have parity : ∀ a b c d : Bool, (a ^^ c) = (b ^^ d) → (c ^^ d) = (a ^^ b) := by decide
      simpa only [parity _ _ _ _ align] using ha.trans hb.symm
    · by_cases align : (s ^^ p) = (t ^^ q)
      · have parity : ∀ a b c d : Bool, (a ^^ c) = (b ^^ d) → (c ^^ d) = (a ^^ b) := by decide
        simpa only [parity _ _ _ _ align] using ha.trans hb.symm
      · have parity : ∀ a b c d : Bool, (a ^^ c) ≠ (b ^^ d) → (c ^^ (true ^^ d)) = (a ^^ b) := by decide
        simpa only [parity _ _ _ _ align] using ha.trans (twist.trans hb.symm)

/-- A new edge whose sign disagrees with the paths to a common root
establishes a reversing walk at that root. -/
theorem closing {e : I → I → Bool → Prop} {a b r : I} {p q g : Bool}
    (ha : Walk e a r p) (hb : Walk e b r q) (edge : e a b g) (different : p ≠ (g ^^ q)) :
    Walk e r r true := by
  have path : Relation.EqvGen (lift e) (a,false) (b,g) := by
    apply (connected a b false g).2
    simpa using Walk.edge edge
  rcases (at_root ha hb false g).1 path with aligned | twisted
  · exact (different (by simpa using aligned)).elim
  · exact twisted

/-- Reversal is a property of the whole connected component. -/
theorem carry {e : I → I → Bool → Prop} {r s : I} {g : Bool} (path : Walk e r s g)
    (twisted : Walk e r r true) : Walk e s s true := by
  have cancel : ∀ k : Bool, (k ^^ (true ^^ k)) = true := by decide
  simpa only [cancel] using path.symm.trans (twisted.trans path)

private def flip (a b g : Bool) := a = false ∧ b = true ∧ g = true
private def parallel (a b g : Bool) := (a = false ∧ b = true) ∨ (a = true ∧ b = false ∧ g = false)

example : Relation.EqvGen (lift flip) (false,true) (true,false) := by
  exact realize (.edge (by simp [flip] : flip false true true)) true

example : Walk flip true false true := project (Relation.EqvGen.symm _ _
  (Relation.EqvGen.rel (false,false) (true,true) (by simp [lift,flip] : lift flip (false,false) (true,true))))

example : Walk parallel true true true :=
  (Walk.symm (.edge (by simp [parallel] : parallel false true false))).trans
    (.edge (by simp [parallel] : parallel false true true))

example (s t : Bool) : Relation.EqvGen (lift parallel) (false,s) (true,t) := by
  apply (at_root (r := true) (.edge (by simp [parallel] : parallel false true false)) (.refl _) s t).2
  exact Or.inr ((Walk.symm (.edge (by simp [parallel] : parallel false true false))).trans
    (.edge (by simp [parallel] : parallel false true true)))


/-- The same learned partial map witnesses the reversed anchor pair. -/
def exchange {X : Type*} (p : Triple X) : Triple X := ⟨p.second,p.first,p.third⟩

theorem exchanged {X J : Type*} (P : J → X → Option X) (a b : Triple X) :
    Edge P (exchange a) (exchange b) ↔ Edge P a b := by
  have turn (x y : Triple X) : Edge P x y → Edge P (exchange x) (exchange y) := by
    rintro ⟨⟨ha,hc,hd⟩, ⟨hb,he,hf⟩, j, u, v, t⟩
    exact ⟨⟨Ne.symm ha,hd,hc⟩, ⟨Ne.symm hb,hf,he⟩, j,v,u,t⟩
  exact ⟨turn (exchange a) (exchange b), turn a b⟩

abbrev Anchors (X : Type*) [LE X] := {v : Triple X // v.first ≤ v.second}

def chart {X : Type*} [LinearOrder X] (v : Triple X) : Anchors X × Bool :=
  if h : v.first ≤ v.second then (⟨v,h⟩,false)
  else (⟨exchange v,le_of_not_ge h⟩,true)

def vertex {X : Type*} [LE X] (p : Anchors X × Bool) : Triple X :=
  if p.2 then exchange p.1.val else p.1.val

theorem vertex_chart {X : Type*} [LinearOrder X] (v : Triple X) : vertex (chart v) = v := by
  unfold chart
  split <;> rfl

def signed {X J : Type*} [LE X] (P : J → X → Option X) (a b : Anchors X) (g : Bool) :=
  Edge P a.val (vertex (b,g))

theorem displayed {X J : Type*} [LE X] (P : J → X → Option X) (p q : Anchors X × Bool) :
    lift (signed P) p q ↔ Edge P (vertex p) (vertex q) := by
  obtain ⟨a,s⟩ := p
  obtain ⟨b,t⟩ := q
  cases s <;> cases t
  · rfl
  · rfl
  · exact (exchanged P a.val (exchange b.val)).symm
  · exact (exchanged P a.val b.val).symm

/-- On the original triple graph, one sorted anchor pair represents both
orders. The path sign distinguishes the two orientations. -/
theorem original {X J : Type*} [LinearOrder X] (P : J → X → Option X) (a b : Triple X) :
    Connected P a b ↔ Walk (signed P) (chart a).1 (chart b).1 ((chart a).2 ^^ (chart b).2) := by
  constructor
  · intro h
    have travel {x y : Triple X} (path : Connected P x y) :
        Relation.EqvGen (lift (signed P)) (chart x) (chart y) := by
      induction path with
      | rel i j step =>
        apply Relation.EqvGen.rel
        exact (displayed P _ _).2 (by simpa only [vertex_chart] using step)
      | refl _ => exact .refl _
      | symm _ _ _ ih => exact .symm _ _ ih
      | trans _ _ _ _ _ ih ih' => exact .trans _ _ _ ih ih'
    exact project (travel h)
  · intro h
    have travel {p q : Anchors X × Bool} (path : Relation.EqvGen (lift (signed P)) p q) :
        Connected P (vertex p) (vertex q) := by
      induction path with
      | rel i j step => exact .rel _ _ ((displayed P i j).1 step)
      | refl _ => exact .refl _
      | symm _ _ _ ih => exact .symm _ _ ih
      | trans _ _ _ _ _ ih ih' => exact .trans _ _ _ ih ih'
    have path : Relation.EqvGen (lift (signed P)) (chart a) (chart b) :=
      (connected _ _ _ _).2 h
    simpa only [vertex_chart] using travel path

private def shifts (r : Bool) (x : Fin 3) := some (if r then x+1 else x)

example : Walk (signed shifts) (chart ⟨2,0,1⟩).1 (chart ⟨0,1,2⟩).1 true := by
  apply (original shifts ⟨2,0,1⟩ ⟨0,1,2⟩).1
  apply Relation.EqvGen.rel
  exact ⟨by unfold Triple.distinct; decide, by unfold Triple.distinct; decide, true, rfl, rfl, rfl⟩

example : Walk parallel false false true := by
  have common : Walk parallel true true true := closing
    (.edge (by simp [parallel] : parallel false true false)) (.refl true)
    (by simp [parallel] : parallel false true true) (by decide)
  exact carry (Walk.symm (.edge (by simp [parallel] : parallel false true false))) common

end Transformer.Phasor.TripleTransport.OrientedAnchors
