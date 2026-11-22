module

public import Transformer.Phasor.TripleTransport.EvidenceRounds

/-!
# Candidate output classes from pairs of learned correspondences

Source: “Pair classes on the Fibonacci orbit split” in the triple-transport
experiment. The graph uses ordered pairs from three separate roles. Different
output slots can represent a common value. The meaning map and the complete
table appear only in the soundness hypotheses.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.PairClasses
variable {X Y C : Type*}

inductive Node (X : Type*) where
  | keys (a b : X)
  | rows (a b : X)
  | images (a b : X)

inductive Step (bank : X → X → Option X) : Node X → Node X → Prop where
  | row (r a b u v) (ha : bank r a = some u) (hb : bank r b = some v) : Step bank (.keys a b) (.images u v)
  | col (k a b u v) (ha : bank a k = some u) (hb : bank b k = some v) : Step bank (.rows a b) (.images u v)

def Sound (bank : X → X → Option X) (meaning : X → Y) (task : X → X → Y) : Prop :=
  ∀ r k y, bank r k = some y → meaning y = task r k

def color (keys rows : X → X → C) (images : Y → Y → C) (meaning : X → Y) : Node X → C
  | .keys a b => keys a b
  | .rows a b => rows a b
  | .images a b => images (meaning a) (meaning b)

/-- Coarse correctness of the evidence suffices: two proposed slots may be
distinct even when they express the same abstract value. -/
theorem paths (bank : X → X → Option X) (meaning : X → Y) (task : X → X → Y)
    (good : Sound bank meaning task) (keys rows : X → X → C) (images : Y → Y → C)
    (rowlaw : ∀ r a b, keys a b = images (task r a) (task r b))
    (collaw : ∀ k a b, rows a b = images (task a k) (task b k))
    {p q} (w : Relation.EqvGen (Step bank) p q) :
    color keys rows images meaning p = color keys rows images meaning q := by
  induction w with
  | rel _ _ h =>
    cases h with
    | row r a b u v ha hb =>
      change keys a b = images (meaning u) (meaning v)
      rw [good r a u ha, good r b v hb, rowlaw]
    | col k a b u v ha hb =>
      change rows a b = images (meaning u) (meaning v)
      rw [good a k u ha, good b k v hb, collaw]
  | refl => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ ih ih' => exact ih.trans ih'

/-- Candidates can have distinct slot names. When a row anchor distinguishes
abstract outputs, every connected candidate represents the desired value. -/
theorem row_candidate (bank : X → X → Option X) (meaning : X → Y) (task : X → X → Y)
    (good : Sound bank meaning task) (keys rows : X → X → C) (images : Y → Y → C)
    (rowlaw : ∀ r a b, keys a b = images (task r a) (task r b))
    (collaw : ∀ k a b, rows a b = images (task a k) (task b k))
    (r a k u v : X) (anchor : bank r a = some u)
    (separates : Function.Injective (images (meaning u)))
    (path : Relation.EqvGen (Step bank) (.keys a k) (.images u v)) :
    meaning v = task r k := by
  apply separates
  have h : keys a k = images (meaning u) (meaning v) :=
    paths bank meaning task good keys rows images rowlaw collaw path
  have ref := rowlaw r a k
  rw [← good r a u anchor] at ref
  exact h.symm.trans ref

/-- The column version uses the observed image in the same hidden key. -/
theorem column_candidate (bank : X → X → Option X) (meaning : X → Y) (task : X → X → Y)
    (good : Sound bank meaning task) (keys rows : X → X → C) (images : Y → Y → C)
    (rowlaw : ∀ r a b, keys a b = images (task r a) (task r b))
    (collaw : ∀ k a b, rows a b = images (task a k) (task b k))
    (a r k u v : X) (anchor : bank a k = some u)
    (separates : Function.Injective (images (meaning u)))
    (path : Relation.EqvGen (Step bank) (.rows a r) (.images u v)) :
    meaning v = task r k := by
  apply separates
  have h : rows a r = images (meaning u) (meaning v) :=
    paths bank meaning task good keys rows images rowlaw collaw path
  have ref := collaw k a r
  rw [← good a k u anchor] at ref
  exact h.symm.trans ref

/-- Repeated completion needs correctness only up to the same meaning map.
Each new proposal must satisfy the proof obligation at that round. -/
theorem rounds (base : X → X → Option X) (update : (X → X → Option X) → X → X → Option X)
    (meaning : X → Y) (task : X → X → Y) (good : Sound base meaning task)
    (safe : ∀ bank, Sound bank meaning task → Sound (update bank) meaning task) (n : ℕ) :
    Sound (update^[n] base) meaning task := by
  induction n with
  | zero => exact good
  | succ n ih => simpa only [Function.iterate_succ_apply'] using safe _ ih

private def hidden (r k : Fin 3) : Option (Fin 3 × Bool) := some (r+k,false)
private def bank (r k : Fin 3 × Bool) := hidden r.1 k.1

private abbrev Slot := Fin 3 × Bool
private def aliases (r k : Slot) : Option Slot := some (r.1 + k.1, if k.1 = 0 then false else r.2)
private def offset (x y : Slot) := y.1 - x.1

private theorem aliases_sound : Sound aliases Prod.fst (fun r k => r.1 + k.1) := by
  intro r k y h
  exact congrArg Prod.fst (Option.some.inj h).symm

private theorem same_row (r a k : Slot) : offset a k = (r.1+k.1) - (r.1+a.1) := by
  unfold offset
  abel

private theorem same_column (k a r : Slot) : offset a r = (r.1+k.1) - (a.1+k.1) := by
  unfold offset
  abel

/-- Two different names for the output at a row query are both admitted by
observations. Either one has the right meaning for the query. -/
example (b : Bool) : ((0, b) : Slot).1 = ((1, false) : Slot).1 + ((2, false) : Slot).1 := by
  apply row_candidate aliases Prod.fst (fun r k => r.1 + k.1) aliases_sound offset offset
    (fun x y : Fin 3 => y - x) same_row same_column (1, false) (0, false) (2, false)
    (1, false) (0, b) (by decide) (fun _ _ h => sub_left_inj.mp h)
  cases b
  · exact .rel _ _ (.row (1, false) _ _ _ _ (by decide) (by decide))
  · exact .rel _ _ (.row (1, true) _ _ _ _ (by decide) (by decide))

example : ((0, true) : Slot).1 = ((2, true) : Slot).1 + ((1, true) : Slot).1 := by
  apply column_candidate aliases Prod.fst (fun r k => r.1 + k.1) aliases_sound offset offset
    (fun x y : Fin 3 => y - x) same_row same_column (0, false) (2, true) (1, true)
    (1, false) (0, true) (by decide) (fun _ _ h => sub_left_inj.mp h)
  exact .rel _ _ (.col (1, true) _ _ _ _ (by decide) (by decide))

example : color (fun a b : Fin 3 × Bool => b.1-a.1) (fun a b : Fin 3 × Bool => b.1-a.1)
    (fun a b : Fin 3 => b-a) Prod.fst (.keys (0,true) (2,false)) =
    color (fun a b : Fin 3 × Bool => b.1-a.1) (fun a b : Fin 3 × Bool => b.1-a.1)
    (fun a b : Fin 3 => b-a) Prod.fst (.images (1,false) (0,false)) := by
  have good : Sound bank Prod.fst (fun r k => r.1+k.1) := by
    intro r k y h
    exact congrArg Prod.fst (Option.some.inj h).symm
  exact paths bank Prod.fst (fun r k => r.1+k.1) good _ _ _
    (by intros; abel) (by intros; abel) (.rel _ _ (.row (1,true) (0,true) (2,false) (1,false) (0,false) rfl rfl))

example : Sound ((fun b r k => (b r k).map (fun y : Fin 3 × Bool => (y.1,!y.2)))^[3] bank)
    Prod.fst (fun r k => r.1+k.1) := by
  apply rounds
  · intro r k y h; exact congrArg Prod.fst (Option.some.inj h).symm
  · intro b h r k y found
    cases obs : b r k with
    | none => simp [obs] at found
    | some z =>
      simp [obs] at found
      exact (congrArg Prod.fst found).symm.trans (h r k z obs)

end Transformer.Phasor.TripleTransport.PairClasses
