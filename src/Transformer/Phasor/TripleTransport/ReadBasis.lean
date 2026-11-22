module

public import Transformer.Phasor.TripleTransport.CalibratedRead

/-!
# Linear changes of the exposed dense values in a prefix-conditioned readout

Source: “A basis for mixing exposed values” in the triple-transport experiment.
The control is fixed when transforming the values. The width accommodates
an independent reference vector for each exposed read.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.ReadBasis
open scoped BigOperators

abbrev Vec (d : ℕ) := Fin d → ℝ

/-- A fixed number of exposed dense value reads and a separate prefix
control. Changing all value vectors by the same linear map acts on the
output by that map and leaves the input controlling the mixture fixed. -/
def Equivariant {C : Type*} {m d : ℕ} (f : C → (Fin m → Vec d) → Vec d) : Prop :=
  ∀ c (a : Vec d →ₗ[ℝ] Vec d) v, f c (fun i => a (v i)) = a (f c v)

def mix {m d : ℕ} (g : Fin m → ℝ) (v : Fin m → Vec d) : Vec d := ∑ i, g i • v i

def frame {m d : ℕ} (axis : Fin m → Fin d) (i : Fin m) (j : Fin d) : ℝ :=
  if j = axis i then 1 else 0

private def realize {m d : ℕ} (axis : Fin m → Fin d) (v : Fin m → Vec d) :
    Vec d →ₗ[ℝ] Vec d where
  toFun x := mix (fun i => x (axis i)) v
  map_add' a b := by simp [mix, add_smul, Finset.sum_add_distrib]
  map_smul' c a := by simp [mix, Finset.smul_sum, mul_smul, Pi.smul_apply, smul_eq_mul]

private theorem realize_frame {m d : ℕ} (axis : Fin m → Fin d) (inj : Function.Injective axis)
    (v : Fin m → Vec d) (i : Fin m) : realize axis v (frame axis i) = v i := by
  change (∑ j, (if axis j = axis i then (1 : ℝ) else 0) • v j) = v i
  simp [inj.eq_iff, ite_smul]

/-- Even a nonlinear readout must be a scalar mixture if it commutes with
every endomorphism of the value vectors, including singular ones. Each
coefficient is the readout at one independent reference frame. -/
theorem coordinates {C : Type*} {m d : ℕ} (axis : Fin m → Fin d) (inj : Function.Injective axis)
    (f : C → (Fin m → Vec d) → Vec d) (natural : Equivariant f) (c : C) (v : Fin m → Vec d) :
    f c v = mix (fun i => f c (frame axis) (axis i)) v := by
  have h := natural c (realize axis v) (frame axis)
  change f c v = realize axis v (f c (frame axis))
  simpa only [realize_frame axis inj] using h

/-- The computation class is specified by simultaneous linear changes of
the dense values. The learned scalar maps may depend arbitrarily on the
prefix control and have no access to an omitted value. -/
theorem complete {C : Type*} {m d : ℕ} (axis : Fin m → Fin d) (inj : Function.Injective axis)
    (f : C → (Fin m → Vec d) → Vec d) :
    Equivariant f ↔ ∃ g : C → Fin m → ℝ, ∀ c v, f c v = mix (g c) v := by
  constructor
  · intro natural
    exact ⟨fun c i => f c (frame axis) (axis i), coordinates axis inj f natural⟩
  · rintro ⟨g, fits⟩ c map v
    simp only [fits, mix, map_sum, map_smul]

/-- The frame coordinates also characterize the experiment's affine
coefficient budget. These weights and biases work unchanged on every
tuple of exposed value vectors. -/
theorem affine_complete {m d h : ℕ} (axis : Fin m → Fin d) (inj : Function.Injective axis)
    (f : Vec h → (Fin m → Vec d) → Vec d) (natural : Equivariant f)
    (bias : Fin m → ℝ) (w : Fin m → Fin h → ℝ)
    (affine : ∀ c i, f c (frame axis) (axis i) = bias i + ∑ j, w i j * c j)
    (c : Vec h) (v : Fin m → Vec d) :
    f c v = mix (fun i => bias i + ∑ j, w i j * c j) v := by
  rw [coordinates axis inj f natural c v]
  simp only [affine]

/-- Any readout given only the surviving primitives must return equal
outputs when their values agree. This allows arbitrary downstream work. -/
def Omits {m d : ℕ} (k : Fin m) (f : (Fin m → Vec d) → Vec d) : Prop :=
  ∀ u v, (∀ j, j ≠ k → u j = v j) → f u = f v

/-- Every exposed primitive is necessary on the full class of independent
dense input values: projecting a particular read is equivariant and ceases
to be expressible after that read is removed. -/
theorem necessary {m d : ℕ} (k : Fin m) (j : Fin d) :
    Equivariant (fun (_ : Unit) (v : Fin m → Vec d) => v k) ∧
      ¬ Omits k (fun v : Fin m → Vec d => v k) := by
  refine ⟨fun _ _ _ => rfl, ?_⟩
  intro missing
  let changed : Fin m → Vec d := fun i => if i = k then fun _ => 1 else 0
  have equal := missing (fun _ => 0) changed (by intro i ne; simp [changed, ne])
  have h := congrFun equal j
  norm_num [changed] at h

example (c : Bool) (v : Fin 3 → Vec 3) :
    v 2 = mix (fun i => (frame id 2 : Vec 3) i) v :=
  coordinates id Function.injective_id (fun (_ : Bool) x => x 2) (fun _ _ _ => rfl) c v

example : Equivariant (fun (_ : Bool) (v : Fin 3 → Vec 3) => v 1) :=
  (complete id Function.injective_id _).2
    ⟨fun _ i => frame id 1 i, fun c v => coordinates id Function.injective_id
      (fun (_ : Bool) x => x 1) (fun _ _ _ => rfl) c v⟩

example (c : Vec 2) (v : Fin 3 → Vec 3) : v 0 =
    mix (fun i => frame id 0 i + ∑ j : Fin 2, (0 : ℝ) * c j) v := by
  apply affine_complete id Function.injective_id (fun (_ : Vec 2) x => x 0) (fun _ _ _ => rfl)
  intro h i
  simp

example : ¬ Omits (2 : Fin 3) (fun v : Fin 3 → Vec 3 => v 2) := (necessary 2 (0 : Fin 3)).2

end Transformer.Phasor.TripleTransport.ReadBasis
