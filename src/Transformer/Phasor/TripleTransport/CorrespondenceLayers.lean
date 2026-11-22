module

public import Transformer.Phasor.TripleTransport

/-!
# Reading outputs from learned row agreement and graph distance

Source: “Hidden row agreement and graph distance” in the triple-transport
experiment. Common outputs suggest edges between hidden rows. Other observed
pairs at the same finite graph distance support a categorical output. The
conditional correctness result allows arbitrary graphs and output palettes.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.RowAgreement

open Finset
variable {K Y Z : Type*} [DecidableEq Y]

/-- The caller supplies keys on which both hidden rows have a trusted output. -/
def coincident (S : Finset K) (u v : K → Y) : ℕ := ∑ i ∈ S, if u i = v i then 1 else 0

def independent (S : Finset K) (u v : K → Y) : ℕ :=
  ∑ i ∈ S, ∑ j ∈ S, if u i = v j then 1 else 0

def histogram (S : Finset K) (u : K → Y) (y : Y) : ℕ :=
  ∑ i ∈ S, if u i = y then 1 else 0

/-- Computing independent collisions from two histograms counts each equal
ordered pair of observations exactly once. The finite output alphabet may
include arbitrarily many values absent from either row. -/
theorem histogram_collisions (S : Finset K) (alphabet : Finset Y) (u v : K → Y)
    (covers : ∀ i ∈ S, u i ∈ alphabet) :
    (∑ y ∈ alphabet, histogram S u y * histogram S v y) = independent S u v := by
  simp only [histogram, independent, sum_mul]
  simp only [mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i hi
  rw [sum_comm]
  apply sum_congr rfl
  intro j _
  rw [sum_eq_single (u i)]
  · simp only [ite_true, one_mul, eq_self]
    by_cases h : u i = v j
    · simp [h]
    · simp [h, Ne.symm h]
  · intro b _ h
    simp [Ne.symm h]
  · simp [covers i hi]

/-- The learned row-neighbour test compares the agreement on aligned keys
with twice the number expected from independent draws over the same keys. -/
def adjacent (S : Finset K) (u v : K → Y) : Prop :=
  6 ≤ coincident S u v ∧ 2 * independent S u v < coincident S u v * S.card

/-- Row-neighbour discovery is invariant under any injective renaming of the
observed outputs, including embeddings into larger dictionaries. -/
theorem adjacent_rename [DecidableEq Z] (rename : Y → Z) (inj : Function.Injective rename)
    (S : Finset K) (u v : K → Y) :
    adjacent S (rename ∘ u) (rename ∘ v) ↔ adjacent S u v := by
  simp only [adjacent, coincident, independent, Function.comp_apply, inj.eq_iff]

example : (∑ y ∈ (univ : Finset Bool),
    histogram (range 6) (fun i => decide (i < 3)) y *
      histogram (range 6) (fun i => decide (i % 2 = 0)) y) = 18 := by decide

example (u v : Fin 13 → Bool) :
    adjacent univ (id ∘ u) (id ∘ v) ↔ adjacent univ u v :=
  adjacent_rename id Function.injective_id univ u v

example : adjacent (univ : Finset (Fin 8)) id id := by
  unfold adjacent
  decide

example (u v : Fin 8 → Bool) :
    (∑ y ∈ (univ : Finset Bool), histogram univ u y * histogram univ v y) = independent univ u v :=
  histogram_collisions _ _ _ _ (fun _ _ => mem_univ _)

end Transformer.Phasor.TripleTransport.RowAgreement

namespace Transformer.Phasor.TripleTransport.CorrespondenceGraph

variable {A B Y : Type*}

/-- Length layers of an arbitrary learned adjacency relation. Disconnected
vertices have no first layer. -/
def walks (edge : A → A → Prop) : ℕ → A → A → Prop
  | 0, a, b => a = b
  | n + 1, a, b => ∃ c, walks edge n a c ∧ edge c b

def first (edge : A → A → Prop) (n : ℕ) (a b : A) : Prop :=
  walks edge n a b ∧ ∀ i < n, ¬ walks edge i a b

/-- A permutation of prototype names preserving learned edges preserves
every finite walk length. -/
theorem walks_rename (e : A ≃ B) (left : A → A → Prop) (right : B → B → Prop)
    (same : ∀ a b, left a b ↔ right (e a) (e b)) (n : ℕ) (a b : A) :
    walks left n a b ↔ walks right n (e a) (e b) := by
  induction n generalizing a b with
  | zero => exact e.injective.eq_iff.symm
  | succ n ih =>
    constructor
    · rintro ⟨c, p, h⟩
      exact ⟨e c, (ih a c).1 p, (same c b).1 h⟩
    · rintro ⟨c, p, h⟩
      obtain ⟨c, rfl⟩ := e.surjective c
      exact ⟨c, (ih a c).2 p, (same c b).2 h⟩

theorem first_rename (e : A ≃ B) (left : A → A → Prop) (right : B → B → Prop)
    (same : ∀ a b, left a b ↔ right (e a) (e b)) (n : ℕ) (a b : A) :
    first left n a b ↔ first right n (e a) (e b) := by
  simp only [first, ← walks_rename e left right same]

/-- The first nonempty layer is unique even if longer walks also exist. -/
theorem first_unique (edge : A → A → Prop) {a b : A} {i j : ℕ}
    (hi : first edge i a b) (hj : first edge j a b) : i = j := by
  rcases lt_trichotomy i j with lt | eq | gt
  · exact False.elim (hj.2 i lt hi.1)
  · exact eq
  · exact False.elim (hi.2 j gt hj.1)

/-- A supported observation can answer another key in the same layer of
an unnamed graph. The unknown response can be any function of the layer.
No class labels, numeric geometry or palette participate in computing it. -/
theorem read_sound (e : A ≃ B) (learned : A → A → Prop) (latent : B → B → Prop)
    (same : ∀ a b, learned a b ↔ latent (e a) (e b)) (response : B → B → Y)
    (palette : ℕ → Y) (onLayer : ∀ n a b, first latent n a b → response a b = palette n)
    (bank : A → A → Option Y)
    (sound : ∀ a b y, bank a b = some y → response (e a) (e b) = y)
    {a b c d : A} {y : Y} {n : ℕ} (observation : bank c d = some y)
    (witness : first learned n c d) (query : first learned n a b) :
    response (e a) (e b) = y := by
  have known := (first_rename e learned latent same n c d).1 witness
  have new := (first_rename e learned latent same n a b).1 query
  exact (onLayer _ _ _ new).trans ((onLayer _ _ _ known).symm.trans (sound _ _ _ observation))

private def star (a b : Fin 4) : Prop := (a = 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ b = 0)

private theorem leaves {a b : Fin 4} (ha : a ≠ 0) (hb : b ≠ 0) (diff : a ≠ b) :
    first star 2 a b := by
  refine ⟨⟨0, ⟨a, rfl, Or.inr ⟨ha, rfl⟩⟩, Or.inl ⟨rfl, hb⟩⟩, ?_⟩
  intro i lt
  interval_cases i
  · exact diff
  · rintro ⟨c, rfl, edge⟩
    rcases edge with h | h
    · exact ha h.1
    · exact hb h.2

example : first star 2 1 2 ∧ first star 2 2 3 :=
  ⟨leaves (by decide) (by decide) (by decide), leaves (by decide) (by decide) (by decide)⟩

example (n : ℕ) (h : first star n 2 3) : n = 2 :=
  first_unique star h (leaves (by decide) (by decide) (by decide))

private noncomputable def paint (edge : A → A → Prop) (a b : A) : Bool := by
  classical
  exact decide (first edge 2 a b)

private theorem palette (edge : A → A → Prop) (n : ℕ) (a b : A) (h : first edge n a b) :
    paint edge a b = decide (n = 2) := by
  classical
  have eq : first edge 2 a b ↔ n = 2 := ⟨first_unique edge h, fun e => e ▸ h⟩
  simp [paint, eq]

private def sample (a b : Fin 4) : Option Bool :=
  if a = 1 ∧ b = 2 then some true else none

example : paint star 2 3 = true := by
  apply read_sound (Equiv.refl _) star star (fun _ _ => Iff.rfl) (paint star)
    (fun n => decide (n = 2)) (palette star) sample _ (a := 2) (b := 3) (c := 1) (d := 2)
    (n := 2) (by decide) (leaves (by decide) (by decide) (by decide))
    (leaves (by decide) (by decide) (by decide))
  intro a b y seen
  by_cases h : a = 1 ∧ b = 2
  · obtain ⟨rfl, rfl⟩ := h
    simp [sample] at seen
    subst y
    exact palette star 2 1 2 (leaves (by decide) (by decide) (by decide))
  · simp [sample, h] at seen

end Transformer.Phasor.TripleTransport.CorrespondenceGraph
