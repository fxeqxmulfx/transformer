module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Certificate
public import Mathlib.LinearAlgebra.Basis.Defs

/-!
# The complete observed coefficient kernel

Source: `review/flat_spectrum.md`, CM34. Row names, column names and observed
class names have separate types, as in native categorical compaction. The
kernel contains the actual observed equations and exactly two shift gauges.
No missing input coordinate or unseen answer is pinned. Normalization retains
every query action and every class-code fiber.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

variable {R I J C S : Type*} [CommRing R]

/-- Independent categorical row, column and observed-class coordinates, CM34. -/
abbrev Coordinates (I J C : Type*) := I ⊕ (J ⊕ C)

/-- The actual three signed coefficients of a proposed class, CM34. -/
def queryAction (i : I) (j : J) (c : C) : (Coordinates I J C → R) →ₗ[R] R where
  toFun w := w (.inl i) + w (.inr (.inl j)) - w (.inr (.inr c))
  map_add' a b := by simp only [Pi.add_apply]; ring
  map_smul' a w := by
    change a * w (.inl i) + a * w (.inr (.inl j)) - a * w (.inr (.inr c)) =
      a * (w (.inl i) + w (.inr (.inl j)) - w (.inr (.inr c)))
    ring

/-- The corresponding explicit matrix row, with no target oracle, CM34. -/
noncomputable def queryRow (i : I) (j : J) (c : C) : Coordinates I J C → R := by
  classical
  exact fun v => CompositeDiscovery.coordinate (.inl i) v +
    CompositeDiscovery.coordinate (.inr (.inl j)) v -
    CompositeDiscovery.coordinate (.inr (.inr c)) v

/-- The linear functional is exactly the action of the original signed
matrix coefficients, including for unequal row/column dictionaries, CM34. -/
theorem queryRow_action [Fintype I] [Fintype J] [Fintype C]
    (i : I) (j : J) (c : C) (w : Coordinates I J C → R) :
    (∑ v, queryRow i j c v * w v) = queryAction i j c w := by
  classical
  simp [queryRow, CompositeDiscovery.coordinate, queryAction, sub_mul, add_mul,
    Finset.sum_sub_distrib, Finset.sum_add_distrib, ite_mul]

/-- The complete compatible space: all observed equations plus row-first
and class-first shift gauges, CM34. This is not a desired-answer predicate. -/
def observedKernel (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C) :
    Submodule R (Coordinates I J C → R) where
  carrier := {w | (∀ t, queryAction (cell t).1 (cell t).2 (labels t) w = 0) ∧
    w (.inl i0) = 0 ∧ w (.inr (.inr c0)) = 0}
  zero_mem' := by simp [queryAction]
  add_mem' := by
    intro a b ha hb
    refine ⟨?_, ?_, ?_⟩
    · intro t
      rw [map_add, ha.1 t, hb.1 t, add_zero]
    · change a (.inl i0) + b (.inl i0) = 0
      rw [ha.2.1, hb.2.1, add_zero]
    · change a (.inr (.inr c0)) + b (.inr (.inr c0)) = 0
      rw [ha.2.2, hb.2.2, add_zero]
  smul_mem' := by
    intro a w hw
    refine ⟨?_, ?_, ?_⟩
    · intro t
      rw [map_smul, hw.1 t, smul_zero]
    · change a * w (.inl i0) = 0
      rw [hw.2.1, mul_zero]
    · change a * w (.inr (.inr c0)) = 0
      rw [hw.2.2, mul_zero]

/-- The finite matrix contains only the supplied observed cell/class rows,
CM34. The two gauges remain separate, explicit equations. -/
noncomputable def observedMatrix (cell : S → I × J) (labels : S → C) :
    S → Coordinates I J C → R :=
  fun t => queryRow (cell t).1 (cell t).2 (labels t)

/-- Kernel membership is exactly the original matrix equations and the
two stated gauges, CM34; no rank or decoder conclusion is built into it. -/
theorem observedKernel_matrix_iff [Fintype I] [Fintype J] [Fintype C]
    (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (w : Coordinates I J C → R) : w ∈ observedKernel cell labels i0 c0 ↔
      (∀ t, (∑ v, observedMatrix cell labels t v * w v) = 0) ∧
      w (.inl i0) = 0 ∧ w (.inr (.inr c0)) = 0 := by
  constructor
  · intro h
    refine ⟨?_, h.2⟩
    intro t
    rw [observedMatrix, queryRow_action]
    exact h.1 t
  · intro h
    refine ⟨?_, h.2⟩
    intro t
    rw [← queryRow_action]
    exact h.1 t

/-- Store the three actual families in the native coordinate order, CM34. -/
def pack (r : I → R) (s : J → R) (e : C → R) : Coordinates I J C → R :=
  Sum.elim r (Sum.elim s e)

/-- Remove exactly the two harmless shifts, CM34. All coordinates remain
free apart from the declared gauges and original observed equations. -/
def normalize (i0 : I) (c0 : C) (w : Coordinates I J C → R) :
    Coordinates I J C → R :=
  pack (fun i => w (.inl i) - w (.inl i0))
    (fun j => w (.inr (.inl j)) + w (.inl i0) - w (.inr (.inr c0)))
    (fun c => w (.inr (.inr c)) - w (.inr (.inr c0)))

/-- Normalization changes neither an observed nor an unobserved query's
coefficient action, CM34. No missing target value enters the calculation. -/
theorem normalize_query (i0 : I) (c0 : C) (w : Coordinates I J C → R)
    (i : I) (j : J) (c : C) :
    queryAction i j c (normalize i0 c0 w) = queryAction i j c w := by
  simp only [normalize, pack, queryAction, LinearMap.coe_mk, AddHom.coe_mk,
    Sum.elim_inl, Sum.elim_inr]
  ring

/-- Equalities between raw class codes survive both shifts, CM34. Thus
normalization cannot invent or discard a decoder fiber. -/
theorem normalize_alias_iff (i0 : I) (c0 : C) (w : Coordinates I J C → R) (a b : C) :
    normalize i0 c0 w (.inr (.inr a)) = normalize i0 c0 w (.inr (.inr b)) ↔
      w (.inr (.inr a)) = w (.inr (.inr b)) := by
  simp only [normalize, pack, Sum.elim_inr, sub_left_inj]

/-- Every fitting coefficient vector has a gauge-normalized representative
in the actual observed kernel, CM34. The observed fit premise contains no
unseen label or query conclusion. -/
theorem normalize_mem (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (w : Coordinates I J C → R)
    (hw : ∀ t, queryAction (cell t).1 (cell t).2 (labels t) w = 0) :
    normalize i0 c0 w ∈ observedKernel cell labels i0 c0 := by
  refine ⟨?_, ?_, ?_⟩
  · intro t
    rw [normalize_query, hw t]
  · simp [normalize, pack]
  · simp [normalize, pack]

/-- A concrete nonzero fit with two nonzero shifts, used to check CM34's
normalization independently of any discovered basis. -/
def shiftedExample : Coordinates Bool Bool Bool → ℚ :=
  pack (fun b => if b then 4 else 2) (fun b => if b then 5 else 3)
    (fun b => if b then 9 else 5)

example := queryRow_action (R := ℚ) true false true shiftedExample

example := observedKernel_matrix_iff (fun b : Bool => (b, b)) id false false
  (normalize false false shiftedExample)

example := normalize_query false false shiftedExample true false true

example := normalize_alias_iff false false shiftedExample false true

example : normalize false false shiftedExample ∈
    observedKernel (fun b : Bool => (b, b)) id false false :=
  normalize_mem _ _ _ _ shiftedExample (by
    intro b
    cases b <;> norm_num [shiftedExample, pack, queryAction])

example : normalize false false shiftedExample (.inl true) = 2 ∧
    normalize false false shiftedExample (.inr (.inl true)) = 2 ∧
    normalize false false shiftedExample (.inr (.inr true)) = 4 := by
  norm_num [normalize, shiftedExample, pack]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes
