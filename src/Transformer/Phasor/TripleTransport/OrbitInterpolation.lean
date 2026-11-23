module

public import Transformer.Phasor.TripleTransport.FiberTransport
public import Mathlib.LinearAlgebra.Lagrange
public import Mathlib.Algebra.Field.ZMod

/-!
# Interpolation in a learned cyclic order

Source: the Phasor experiment section “Interpolation in a learned cyclic
order”. Two learned permutations can generate an orbit covering the
dictionary. In its coordinates, bounded-degree rows are identifiable from
finitely many anchors and their predictions commute with slot renaming.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.OrbitInterpolation
open Polynomial
variable {X Y F : Type*} [Field F]

/-- Walking a composition of two learned slot permutations commutes with every
renaming. A single orbit covering the dictionary therefore supplies a coordinate
system relative only to the chosen starting prototype and the learned maps. -/
theorem walk_rename (a b : X → X) (origin : X) (rename : X ≃ Y) (n : ℕ) :
    ((fun y => rename (b (a (rename.symm y))))^[n]) (rename origin) =
      rename ((b ∘ a)^[n] origin) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih, Equiv.symm_apply_apply]
    rfl

/-- Interpolate observed prototype values in a chart obtained from a full
orbit. Transport the prediction back to the same hidden dictionary. -/
noncomputable def read [DecidableEq X] (chart : X ≃ F) (anchors : Finset X)
    (observed : X → X) (q : X) : X :=
  chart.symm ((Lagrange.interpolate anchors chart (chart ∘ observed)).eval (chart q))

/-- For the class of responses of degree less than the number of distinct
anchors in the learned chart, observations determine every novel output. -/
theorem recover [DecidableEq X] (chart : X ≃ F) (anchors : Finset X)
    (observed truth : X → X) (p : F[X]) (degree : p.degree < anchors.card)
    (law : ∀ x, chart (truth x) = p.eval (chart x))
    (agrees : ∀ x ∈ anchors, observed x = truth x) (q : X) :
    read chart anchors observed q = truth q := by
  have h : p = Lagrange.interpolate anchors chart (chart ∘ observed) :=
    Lagrange.eq_interpolate_of_eval_eq _ (fun _ _ _ _ e => chart.injective e) degree
      (fun x hx => (law x).symm.trans (congrArg chart (agrees x hx).symm))
  apply chart.injective
  simpa [read, ← h] using (law q).symm

/-- With one fewer anchor the degree budget admits two responses that agree
on every observation and differ on the specified missing prototype. -/
theorem insufficient (chart : X ≃ F) (anchors : Finset X) (q : X) (novel : q ∉ anchors) :
    ∃ p : F[X], p.degree = anchors.card ∧
      (∀ x ∈ anchors, p.eval (chart x) = 0) ∧ p.eval (chart q) ≠ 0 := by
  refine ⟨Lagrange.nodal anchors chart, Lagrange.degree_nodal .., ?_, ?_⟩
  · exact fun x hx => Lagrange.eval_nodal_at_node hx
  · exact Lagrange.eval_nodal_not_at_node (fun _ hx e => novel (chart.injective e ▸ hx))

/-- One Newton correction fits a newly chosen mismatch while leaving every
previously chosen witness fixed. This is the incremental native fit. -/
theorem update (anchors : Finset F) (p : F[X]) (x y : F) (fresh : x ∉ anchors) :
    let basis := Lagrange.nodal anchors id
    let next := p + C ((y - p.eval x) / basis.eval x) * basis
    next.eval x = y ∧ ∀ a ∈ anchors, next.eval a = p.eval a := by
  dsimp only
  have nonzero : (Lagrange.nodal anchors id).eval x ≠ 0 :=
    Lagrange.eval_nodal_not_at_node (fun a ha e => fresh (e ▸ ha))
  constructor
  · simp [nonzero]
  · intro a ha
    have root : (Lagrange.nodal anchors id).eval a = 0 :=
      Lagrange.eval_nodal_at_node (v := id) ha
    simp [root]

example : ((fun b : Bool => !b)^[3]) false = true := by decide

example : read (Equiv.refl ℚ) ({0,1,2} : Finset ℚ) (fun x => x*x+1) 3 = 10 := by
  have recovered := recover (Equiv.refl ℚ) ({0,1,2} : Finset ℚ)
    (fun x => x*x+1) (fun x => x*x+1) (Polynomial.X^2+1)
    (by apply lt_of_le_of_lt (Polynomial.degree_add_le ..); norm_num [Polynomial.degree_X_pow]) (by simp [pow_two]) (by simp) 3
  norm_num at recovered ⊢
  exact recovered

example : ∃ p : ℚ[X], p.degree = ↑({0,1} : Finset ℚ).card ∧
    (∀ x ∈ ({0,1} : Finset ℚ), p.eval x = 0) ∧ p.eval 2 ≠ 0 :=
  insufficient (Equiv.refl ℚ) _ _ (by norm_num)

example : ∃ p : Polynomial (ZMod 7), p.eval 2 = 6 ∧ p.eval 0 = 1 := by
  let _ : Fact (Nat.Prime 7) := ⟨by decide⟩
  let basis := Lagrange.nodal ({0} : Finset (ZMod 7)) id
  let p : Polynomial (ZMod 7) := 1
  refine ⟨p + C ((6 - p.eval 2) / basis.eval 2) * basis, ?_⟩
  have h := update ({0} : Finset (ZMod 7)) p 2 6 (by decide)
  exact ⟨h.1, (h.2 0 (by simp)).trans (by simp [p])⟩

end Transformer.Phasor.TripleTransport.OrbitInterpolation

namespace Transformer.Phasor.TripleTransport.OrbitInterpolation
open Polynomial
variable {X F : Type*} [Field F] [DecidableEq X]

/-- In a fixed chart a successful redundant polynomial fit supplies the
same in-sample proposal using only other, distinct observed coordinates.
The stored value at the address is unrestricted in the punctured read. -/
theorem read_after_erasing (c : X ≃ F) (s : Finset X) (q : X)
    (stored replacement : X → X) (p : F[X]) (member : q ∈ s)
    (redundant : p.degree < ↑(s.card - 1))
    (fit : ∀ i ∈ s, c (stored i) = p.eval (c i))
    (other : ∀ i ∈ s.erase q, replacement i = stored i) :
    read c (s.erase q) replacement q = stored q := by
  have eq : p = Lagrange.interpolate (s.erase q) c (c ∘ replacement) := by
    apply Lagrange.eq_interpolate_of_eval_eq _ (fun _ _ _ _ h => c.injective h)
    · rwa [Finset.card_erase_of_mem member]
    · intro i hi
      change p.eval (c i) = c (replacement i)
      rw [other i hi, fit i (Finset.mem_of_mem_erase hi)]
  apply c.injective
  simpa [read, ← eq] using (fit q member).symm

example : read (Equiv.refl ℚ) (({0,1,2,3} : Finset ℚ).erase 2)
    (fun x => if x=2 then 99 else x*x+1) 2 = 5 := by
  rw [show (5 : ℚ) = (fun x : ℚ => x*x+1) 2 by norm_num]
  apply read_after_erasing (Equiv.refl ℚ) _ 2 (fun x => x*x+1) _ (Polynomial.X^2+1)
  · decide
  · apply lt_of_le_of_lt (Polynomial.degree_add_le ..)
    norm_num [Polynomial.degree_X_pow]
  · intro i hi
    simp [pow_two]
  · intro i hi
    have different : i ≠ 2 := (Finset.mem_erase.mp hi).1
    simp [different]

end Transformer.Phasor.TripleTransport.OrbitInterpolation
namespace Transformer.Phasor.TripleTransport.OrbitInterpolation
open Polynomial
variable {X Y F : Type*} [Field F] [DecidableEq X] [DecidableEq Y]

/-- An arbitrary renaming of the prototypes and a correspondingly transported
chart rename the entire prediction, even when the sampled row has no simple law. -/
theorem read_rename (chart : X ≃ F) (s : Finset X) (observed : X → X) (q : X)
    (rename : X ≃ Y) :
    read (rename.symm.trans chart) (s.map rename.toEmbedding)
      (fun y => rename (observed (rename.symm y))) (rename q) = rename (read chart s observed q) := by
  let p := Lagrange.interpolate s chart (chart ∘ observed)
  have inject : Set.InjOn chart s := fun _ _ _ _ h => chart.injective h
  have eq : p = Lagrange.interpolate (s.map rename.toEmbedding) (rename.symm.trans chart)
      ((rename.symm.trans chart) ∘ (fun y => rename (observed (rename.symm y)))) := by
    apply Lagrange.eq_interpolate_of_eval_eq _ (fun _ _ _ _ e => (rename.symm.trans chart).injective e)
    · simpa [p] using Lagrange.degree_interpolate_lt (chart ∘ observed) inject
    · intro y hy
      obtain ⟨x,hx,rfl⟩ := Finset.mem_map.mp hy
      simpa [p] using Lagrange.eval_interpolate_at_node (chart ∘ observed) inject hx
  unfold read
  rw [← eq]
  simp only [Equiv.trans_apply, Equiv.symm_apply_apply, Equiv.symm_trans_apply, Equiv.symm_symm, p]

example (q : ℚ) :
    read ((Equiv.swap (0 : ℚ) 1).symm.trans (Equiv.refl ℚ))
      (({0,1,3} : Finset ℚ).map (Equiv.swap (0 : ℚ) 1).toEmbedding)
      (fun y => (Equiv.swap (0 : ℚ) 1) (((Equiv.swap (0 : ℚ) 1).symm y)^3+1))
      ((Equiv.swap (0 : ℚ) 1) q) =
        (Equiv.swap (0 : ℚ) 1) (read (Equiv.refl ℚ) {0,1,3} (fun x => x^3+1) q) :=
  read_rename (Equiv.refl ℚ) ({0,1,3} : Finset ℚ) (fun x : ℚ => x^3+1) q (Equiv.swap 0 1)

end Transformer.Phasor.TripleTransport.OrbitInterpolation
namespace Transformer.Phasor.TripleTransport.OrbitInterpolation
open Polynomial

/-- A fallback chart subsumes a partner read on the positions where the
chart supplies the same prediction whenever a partner is available. -/
theorem drop_partner_iff {A : Type*} (partner chart : Option A) :
    partner.or chart = chart ↔ ∀ value, partner = some value → chart = some value := by
  cases partner with
  | none => simp
  | some v => simp [eq_comm]

/-- Two members of an actual equal-output fiber have the same response in
a valid chart. With enough distinct interpolation anchors, returning the
partner output is redundant even on a novel query coordinate. -/
theorem chart_covers_partner {I F : Type*} [Field F] [DecidableEq I]
    (chart : I ≃ F) (s : Finset I) (seen truth : I → I) (p : F[X])
    (degree : p.degree < s.card) (law : ∀ i, chart (truth i) = p.eval (chart i))
    (trusted : ∀ i ∈ s, seen i = truth i) (q partner : I)
    (same : truth partner = truth q) (copied : seen partner = truth partner) :
    (some (seen partner)).or (some (read chart s seen q)) = some (read chart s seen q) := by
  rw [recover chart s seen truth p degree law trusted q]
  simp [copied,same]

example : (some (((-3 : ℚ)^2))).or (some (read (Equiv.refl ℚ) {0,1,2}
    (fun x => x^2) 3)) = some (read (Equiv.refl ℚ) {0,1,2} (fun x => x^2) 3) := by
  exact chart_covers_partner (Equiv.refl ℚ) _ (fun x => x^2) (fun x => x^2) (Polynomial.X^2)
    (by norm_num [Polynomial.degree_X_pow]) (by simp) (by simp) 3 (-3) (by norm_num) rfl

/-- An abstaining chart can leave an available partner as the only proposal. -/
example : (some (7 : ℕ)).or none ≠ none := by decide

end Transformer.Phasor.TripleTransport.OrbitInterpolation
