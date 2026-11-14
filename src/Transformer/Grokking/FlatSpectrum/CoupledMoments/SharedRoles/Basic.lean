module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Basic

/-!
# Actual shared-role equations, independent of gauge

Source: `review/flat_spectrum.md`, CM40. An alignment lists occurrences of
the same categorical token in the two position dictionaries. Row-minus-column
differences agree across alignments; their common offset remains free. These
homogeneous linear equations intersect the actual observed kernel. They are
an architectural restriction, not a conclusion of flatness or observed fit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles

open KernelCodes

variable {R I J C S T : Type*} [CommRing R]

/-- Compare two aligned row/column differences, CM40. The shared offset
cancels; no observed class or desired query answer enters this measurement. -/
def roleAction (i : I) (j : J) (i0 : I) (j0 : J) :
    (Coordinates I J C → R) →ₗ[R] R where
  toFun w := (w (.inl i) - w (.inr (.inl j))) -
    (w (.inl i0) - w (.inr (.inl j0)))
  map_add' a b := by simp only [Pi.add_apply]; ring
  map_smul' a w := by
    change (a * w (.inl i) - a * w (.inr (.inl j))) -
      (a * w (.inl i0) - a * w (.inr (.inl j0))) =
      a * ((w (.inl i) - w (.inr (.inl j))) - (w (.inl i0) - w (.inr (.inl j0))))
    ring

/-- The explicit four signed native matrix coefficients, CM40. -/
noncomputable def roleRow (i : I) (j : J) (i0 : I) (j0 : J) :
    Coordinates I J C → R := by
  classical
  exact fun v => (CompositeDiscovery.coordinate (.inl i) v -
    CompositeDiscovery.coordinate (.inr (.inl j)) v) -
    (CompositeDiscovery.coordinate (.inl i0) v -
      CompositeDiscovery.coordinate (.inr (.inl j0)) v)

/-- The shared-role functional is the action of its actual matrix row,
CM40, including cancellation when two coordinate positions coincide. -/
theorem roleRow_action [Fintype I] [Fintype J] [Fintype C]
    (i : I) (j : J) (i0 : I) (j0 : J) (w : Coordinates I J C → R) :
    (∑ v, roleRow i j i0 j0 v * w v) = roleAction i j i0 j0 w := by
  classical
  simp [roleRow, CompositeDiscovery.coordinate, roleAction, sub_mul,
    Finset.sum_sub_distrib, ite_mul]

/-- The linear space of one shared offset over the supplied alignments,
CM40. The list of aligned input identities is fixed independently of labels. -/
def roleKernel (align : T → I × J) (t0 : T) : Submodule R (Coordinates I J C → R) where
  carrier := {w | ∀ t, roleAction (align t).1 (align t).2 (align t0).1 (align t0).2 w = 0}
  zero_mem' := by intro t; simp [roleAction]
  add_mem' := by
    intro a b ha hb t
    rw [map_add, ha t, hb t, add_zero]
  smul_mem' := by
    intro a w hw t
    rw [map_smul, hw t, smul_zero]

/-- Original observations and both original gauges, intersected with the
additional linear shared-role equations, CM40. No missing answer is pinned. -/
def sharedKernel (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (align : T → I × J) (t0 : T) : Submodule R (Coordinates I J C → R) :=
  observedKernel cell labels i0 c0 ⊓ roleKernel align t0

/-- Both gauge shifts leave every shared-role matrix action unchanged,
CM40. They can change the common row-minus-column offset itself. -/
theorem normalize_roleAction (i0 : I) (c0 : C) (w : Coordinates I J C → R)
    (i : I) (j : J) (i' : I) (j' : J) :
    roleAction i j i' j' (normalize i0 c0 w) = roleAction i j i' j' w := by
  simp only [KernelCodes.normalize, pack, roleAction, LinearMap.coe_mk, AddHom.coe_mk,
    Sum.elim_inl, Sum.elim_inr]
  ring

/-- Shared-role membership survives the original normalization in both
directions, CM40; forcing the common offset to zero would not be equivalent. -/
theorem normalize_roleKernel_iff (align : T → I × J) (t0 : T) (i0 : I) (c0 : C)
    (w : Coordinates I J C → R) :
    normalize i0 c0 w ∈ roleKernel align t0 ↔ w ∈ roleKernel align t0 := by
  change (∀ t, roleAction _ _ _ _ (normalize i0 c0 w) = 0) ↔
    ∀ t, roleAction _ _ _ _ w = 0
  simp only [normalize_roleAction]

/-- A fitting shared-role vector normalizes into the actual restricted
kernel, CM40. The hypotheses involve only observations and role alignment. -/
theorem normalize_mem_shared (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (align : T → I × J) (t0 : T) (w : Coordinates I J C → R)
    (hobs : ∀ t, queryAction (cell t).1 (cell t).2 (labels t) w = 0)
    (hrole : w ∈ roleKernel align t0) :
    normalize i0 c0 w ∈ sharedKernel cell labels i0 c0 align t0 := by
  exact ⟨KernelCodes.normalize_mem cell labels i0 c0 w hobs,
    (normalize_roleKernel_iff align t0 i0 c0 w).mpr hrole⟩

/-- One common offset supplies precisely the shared-role equations on the
actual packed coefficient vector, CM40, without fixing its class coordinates. -/
theorem pack_mem_roleKernel (align : T → I × J) (t0 : T)
    (r : I → R) (s : J → R) (e : C → R)
    (hshare : ∀ t, r (align t).1 - s (align t).2 = r (align t0).1 - s (align t0).2) :
    pack r s e ∈ roleKernel align t0 := by
  intro t
  change (r (align t).1 - s (align t).2) - (r (align t0).1 - s (align t0).2) = 0
  rw [hshare t, sub_self]

/-- A single categorical encoder with independent position offsets
satisfies the actual shared-role constraints, CM40. Only aligned token
identities are compared; the token names have no numerical answer meaning. -/
theorem shared_encoder_mem {A : Type*} (align : T → I × J) (t0 : T)
    (rowToken : I → A) (columnToken : J → A)
    (halign : ∀ t, rowToken (align t).1 = columnToken (align t).2)
    (h : A → R) (a b : R) (e : C → R) :
    pack (fun i => h (rowToken i) + a) (fun j => h (columnToken j) + b) e ∈
      roleKernel align t0 := by
  apply pack_mem_roleKernel
  intro t
  rw [halign t, halign t0]
  ring

example := roleRow_action (R := ℚ) true true false false shiftedExample

example := normalize_roleAction false false shiftedExample true true false false

example := normalize_roleKernel_iff (fun b : Bool => (b, b)) false false false shiftedExample

example : shiftedExample ∈ roleKernel (fun b : Bool => (b, b)) false := by
  apply pack_mem_roleKernel
  intro b
  cases b <;> norm_num

example : normalize false false shiftedExample ∈
    sharedKernel (fun b : Bool => (b, b)) id false false (fun b : Bool => (b, b)) false := by
  apply normalize_mem_shared
  · intro b
    cases b <;> norm_num [shiftedExample, pack, queryAction]
  · apply pack_mem_roleKernel
    intro b
    cases b <;> norm_num

example := shared_encoder_mem (R := ℚ) (C := Bool) (fun b : Bool => (b, b)) false
  id id (fun _ => rfl) (fun b => if b then 2 else 0) 2 3 (fun b => if b then 9 else 5)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
