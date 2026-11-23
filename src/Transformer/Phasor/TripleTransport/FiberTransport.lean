module

public import Transformer.Phasor.TripleTransport

/-!
# Equal-value partners in a learned partial memory

Source: the Phasor experiment section “Interpolation in a learned cyclic
order”. Each two-element observed fiber gives a partial correspondence.
Existing triple transport can complete a missing partner when the learned
relations distinguish the third key.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.FiberTransport
variable {R K V Z : Type*}

/-- The other key must be distinct and have the same observed value. -/
def Other (bank : R → K → Option V) (r : R) (k b : K) : Prop :=
  k ≠ b ∧ ∃ v, bank r k = some v ∧ bank r b = some v

/-- A fiber with exactly two observed keys gives one directed correspondence
in each direction. Larger fibers and unwritten values abstain. -/
noncomputable def mate (bank : R → K → Option V) (r : R) (k : K) : Option K := by
  classical
  exact if h : ∃! b, Other bank r k b then some h.exists.choose else none

theorem mate_iff (bank : R → K → Option V) (r : R) (k b : K) :
    mate bank r k = some b ↔ Other bank r k b ∧ ∀ c, Other bank r k c → c = b := by
  classical
  unfold mate
  split
  · rename_i h
    constructor
    · intro eq
      obtain rfl := Option.some.inj eq
      exact ⟨h.exists.choose_spec, fun c hc => h.unique hc h.exists.choose_spec⟩
    · rintro ⟨_, unique⟩
      exact congrArg some (unique _ h.exists.choose_spec)
  · rename_i none
    constructor
    · intro h
      contradiction
    · rintro ⟨found, unique⟩
      exact (none ⟨b, found, unique⟩).elim

/-- Two observed members cannot determine a unique partner when the completed
value has three possible keys. The fiber assumption below concerns the
completed map and ranges over all its keys. -/
def SeparatesPartners (value : R → K → V) (partner : R → K → K) : Prop :=
  ∀ r a b, a ≠ b → value r a = value r b → partner r a = b

/-- Applying the existing triple-transport read to equal-value partners gives
a reusable key. Reading its observed image recovers the missing value if
the true row partners preserve the triple invariant and the fixed anchors
separate third keys. All partners and the invariant occur only in the
soundness assumptions. -/
theorem recover (bank : R → K → Option V) (value : R → K → V) (partner : R → K → K)
    (observed : ∀ r k v, bank r k = some v → value r k = v)
    (identifies : SeparatesPartners value partner)
    (same : ∀ r k, value r (partner r k) = value r k)
    (color : Triple K → Z) (stable : ∀ r t, color t = color (t.map (partner r)))
    (r : R) (a b q u v candidate : K) (y : V)
    (ha : mate bank r a = some u) (hb : mate bank r b = some v)
    (sep : Function.Injective fun z => color ⟨u,v,z⟩)
    (connected : Connected (mate bank) ⟨a,b,q⟩ ⟨u,v,candidate⟩)
    (available : bank r candidate = some y) : value r q = y := by
  have agrees (i : R) (x z : K) (h : mate bank i x = some z) : partner i x = z := by
    obtain ⟨ne, w, p, p'⟩ := ((mate_iff bank i x z).1 h).1
    exact identifies i x z ne ((observed i x w p).trans (observed i z w p').symm)
  have edges (s t : Triple K) (h : Edge (mate bank) s t) : color s = color t := by
    obtain ⟨_, _, i, hx, hy, hz⟩ := h
    rcases s with ⟨x,b,k⟩
    rcases t with ⟨u,v,w⟩
    simpa only [Triple.map, agrees i _ _ hx, agrees i _ _ hy, agrees i _ _ hz] using
      stable i ⟨x,b,k⟩
  have image : color ⟨a,b,q⟩ = color ⟨u,v,partner r q⟩ := by
    simpa only [Triple.map, agrees r a u ha, agrees r b v hb] using stable r ⟨a,b,q⟩
  have found := third_eq edges _ _ _ _ _ sep image connected
  rw [← same r q, ← found]
  exact observed r candidate y available

private abbrev Key := Bool × Fin 3
private def truth (r : Bool) (k : Key) : Key := (r,k.2)
private def table (r : Bool) (k : Key) : Option Key :=
  if r = false ∧ k = (false,2) then none else some (truth r k)
private def flip (k : Key) : Key := (!k.1,k.2)
private def shade (s : Triple Key) : Key := (s.first.1 == s.third.1, s.third.2)

private theorem full_mate (k : Key) : mate table true k = some (flip k) := by
  apply (mate_iff _ _ _ _).2
  unfold Other
  rcases k with ⟨b,i⟩
  cases b <;> fin_cases i <;> decide +revert

private theorem anchors (a : Fin 3) (ne : a ≠ 2) :
    mate table false (false,a) = some (true,a) := by
  apply (mate_iff _ _ _ _).2
  unfold Other
  fin_cases a <;> first | contradiction | decide +revert

private theorem shade_stable (s : Triple Key) : shade s = shade (s.map flip) := by
  rcases s with ⟨⟨b,i⟩, second, ⟨c,k⟩⟩
  cases b <;> cases c <;> rfl

/-- The target row has a genuine missing cell. One other row supplies the
connected triple, and the target row supplies the two anchors and final value. -/
example : table false (false,2) = none ∧ truth false (false,2) = (false,2) := by
  refine ⟨rfl, ?_⟩
  apply recover table truth (fun _ => flip) (by decide +revert)
    (by unfold SeparatesPartners; decide +revert) (fun _ _ => rfl)
    shade (fun _ s => shade_stable s) false (false,0) (false,1) (false,2)
    (true,0) (true,1) (true,2) (false,2) (anchors 0 (by decide)) (anchors 1 (by decide))
  · rintro ⟨b,i⟩ ⟨c,j⟩ h
    cases b <;> cases c <;> simp_all [shade]
  · apply Relation.EqvGen.rel
    exact ⟨by unfold Triple.distinct; decide, by unfold Triple.distinct; decide, true,
      full_mate (false,0), full_mate (false,1), full_mate (false,2)⟩
  · rfl

end Transformer.Phasor.TripleTransport.FiberTransport
