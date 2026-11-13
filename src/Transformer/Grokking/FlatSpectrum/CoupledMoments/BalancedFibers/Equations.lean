module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers.Basic

/-!
# Actual phase-sum equations from complete decoder fibers

Source: `review/flat_spectrum.md`, CM26. Complete categorical fibers yield
linear equations in additive phase coordinates. The decoder need not be
injective. Fiber cardinality and mass are explicit assumptions about the
actual decoder; solving guessed equations alone does not identify labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers

variable {I G C : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G] [DecidableEq C]

/-- Sum the actual distinct shifted phases, then separate the common shift.
CM26: this is an equation for the decoder's full fiber, not a guessed target. -/
theorem fiber_sum {r : I → G} {b : G} {F : Finset I} {d : G → C} {c : C}
    (hr : Function.Injective r) (hobs : ∀ i ∈ F, d (r i + b) = c)
    (hcard : F.card = (fiber d c).card) :
    (∑ i ∈ F, r i) + F.card • b = ∑ z ∈ fiber d c, z := by
  have hsum : (∑ z ∈ F.image (fun i => r i + b), z) = ∑ i ∈ F, (r i + b) :=
    Finset.sum_image (shift_injective hr b).injOn
  rw [complete_image hr hobs hcard] at hsum
  rw [Finset.sum_add_distrib, Finset.sum_const] at hsum
  exact hsum.symm

/-- A complete size-k observed label group in one column gives its actual
row-phase equation, CM26. The supplied mass may be shared across labels only
when each relevant decoder fiber really has that mass. -/
theorem row_equation {r s : I → G} {j : I} {F : Finset I} {d : G → C} {c : C}
    {k : ℕ} {μ : G} (hr : Function.Injective r)
    (hobs : ∀ i ∈ F, d (r i + s j) = c) (hsize : F.card = k)
    (hfull : (fiber d c).card = k) (hmass : ∑ z ∈ fiber d c, z = μ) :
    (∑ i ∈ F, r i) + k • s j = μ := by
  have h := fiber_sum hr hobs (hsize.trans hfull.symm)
  simpa only [hsize, hmass] using h

/-- The transposed complete group gives the column-phase equation, CM26.
Both orientations can constrain the same unknown phases and fiber mass. -/
theorem column_equation {r s : I → G} {i : I} {F : Finset I} {d : G → C} {c : C}
    {k : ℕ} {μ : G} (hs : Function.Injective s)
    (hobs : ∀ j ∈ F, d (r i + s j) = c) (hsize : F.card = k)
    (hfull : (fiber d c).card = k) (hmass : ∑ z ∈ fiber d c, z = μ) :
    k • r i + (∑ j ∈ F, s j) = μ := by
  have hobs' : ∀ j ∈ F, d (s j + r i) = c := by
    simpa only [add_comm] using hobs
  have h := fiber_sum hs hobs' (hsize.trans hfull.symm)
  simpa only [hsize, hmass, add_comm] using h

/-- Three phases, nonzero shift, and a many-to-one decoder witness the sum. -/
example : (∑ i : ZMod 3, i) + 3 • (1 : ZMod 3) =
    ∑ z ∈ fiber (fun (_ : ZMod 3) => ()) (), z := by
  simpa using fiber_sum (r := fun i : ZMod 3 => i) (b := 1) (F := Finset.univ)
    (d := fun _ => ()) (c := ()) (fun _ _ h => h) (by intros; rfl) (by simp [fiber])

example : (∑ i : ZMod 3, i) + 3 • (1 : ZMod 3) = 0 := by
  apply row_equation (r := fun i : ZMod 3 => i) (s := fun _ => 1)
    (j := 0) (F := Finset.univ) (d := fun _ => ()) (c := ())
    (fun _ _ h => h) (by intros; rfl)
  · decide
  · simp [fiber]
  · decide

example : 3 • (1 : ZMod 3) + (∑ j : ZMod 3, j) = 0 := by
  apply column_equation (r := fun _ : ZMod 3 => 1) (s := fun j => j)
    (i := 0) (F := Finset.univ) (d := fun _ => ()) (c := ())
    (fun _ _ h => h) (by intros; rfl)
  · decide
  · simp [fiber]
  · decide

/-- A nonconstant decoder and two nonconstant phase axes also satisfy the
complete-fiber premises: its four-element nonzero fiber has mass zero. -/
example : (∑ i ∈ ({0, 1, 2, 4} : Finset (ZMod 5)), i) + 4 • (2 : ZMod 5) = 0 := by
  apply row_equation (r := fun i : ZMod 5 => i) (s := fun j => j)
    (j := 2) (F := {0, 1, 2, 4})
    (d := fun z : ZMod 5 => if z = 0 then false else true) (c := true)
    (fun _ _ h => h)
  · decide
  · decide
  · decide
  · decide

/-- CM26 counterexample to dropping completeness: an observed singleton lies
inside a genuine three-element fiber but has the wrong shifted phase sum. -/
theorem incomplete_fiber_counterexample :
    ({0} : Finset (ZMod 3)).image (fun i => i + 1) ⊆
      fiber (fun (_ : ZMod 3) => ()) () ∧
    (∑ i ∈ ({0} : Finset (ZMod 3)), i) + 1 • (1 : ZMod 3) ≠
      ∑ z ∈ fiber (fun (_ : ZMod 3) => ()) (), z := by
  constructor
  · exact observed_subset (by intros; rfl)
  · decide

end Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers
