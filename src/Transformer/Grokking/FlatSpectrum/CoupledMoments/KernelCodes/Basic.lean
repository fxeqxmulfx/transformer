module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Defs

/-!
# Basis-independent full-kernel code comparisons

Source: `review/flat_spectrum.md`, CM34. Read every coefficient direction in
a complete basis of the actual compatible space. Equality of the resulting
vectors is equivalent to agreement of every compatible coefficient vector.
The comparison therefore does not depend on a particular basis. Searching
an observed class bank may still refuse a query. Concrete nonzero examples
are in `KernelCodes/Witness` and `KernelCodes/Identification`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

variable {R I J C K L S : Type*} [CommRing R]
variable {M : Submodule R (Coordinates I J C → R)}

/-- Sum the actual row and column coefficients in every basis direction,
CM34. This uses neither a selected coordinate nor a proposed target. -/
noncomputable def queryCode (b : Module.Basis K R M) (i : I) (j : J) : K → R :=
  fun k => (b k).val (.inl i) + (b k).val (.inr (.inl j))

/-- The vector of actual observed-class coefficients, CM34. Distinct raw
classes can have identical vectors, so this definition assumes no injectivity. -/
noncomputable def classCode (b : Module.Basis K R M) (c : C) : K → R :=
  fun k => (b k).val (.inr (.inr c))

/-- A full vector match is precisely agreement under every compatible
coefficient vector, CM34. Spanning the actual space is the crucial basis fact. -/
theorem code_match_iff (b : Module.Basis K R M) (i : I) (j : J) (c : C) :
    queryCode b i j = classCode b c ↔ ∀ w : M, queryAction i j c w.val = 0 := by
  constructor
  · intro h
    let f : M →ₗ[R] R := (queryAction i j c).comp M.subtype
    have hf : f = 0 := b.ext (by
      intro k
      change queryCode b i j k - classCode b c k = 0
      exact sub_eq_zero.mpr (congrFun h k))
    intro w
    exact congrArg (fun g : M →ₗ[R] R => g w) hf
  · intro h
    funext k
    exact sub_eq_zero.mp (h (b k))

/-- Changing the complete basis preserves every query/class comparison,
CM34. No coordinatewise equality between the two bases is assumed. -/
theorem code_match_basis_iff (b : Module.Basis K R M) (d : Module.Basis L R M)
    (i : I) (j : J) (c : C) :
    queryCode b i j = classCode b c ↔ queryCode d i j = classCode d c := by
  rw [code_match_iff, code_match_iff]

/-- Difference of two actual class coordinates, for checking aliases,
CM34. This is a linear measurement of the same coefficient space. -/
def classAction (a c : C) : (Coordinates I J C → R) →ₗ[R] R where
  toFun w := w (.inr (.inr a)) - w (.inr (.inr c))
  map_add' w v := by simp only [Pi.add_apply]; ring
  map_smul' r w := by
    change r * w (.inr (.inr a)) - r * w (.inr (.inr c)) =
      r * (w (.inr (.inr a)) - w (.inr (.inr c)))
    ring

/-- Full class-vector equality is exactly an alias forced by every
compatible coefficient vector, CM34. One chosen direction is insufficient. -/
theorem class_alias_iff (b : Module.Basis K R M) (a c : C) :
    classCode b a = classCode b c ↔ ∀ w : M, classAction a c w.val = 0 := by
  constructor
  · intro h
    let f : M →ₗ[R] R := (classAction a c).comp M.subtype
    have hf : f = 0 := b.ext (by
      intro k
      change classCode b a k - classCode b c k = 0
      exact sub_eq_zero.mpr (congrFun h k))
    intro w
    exact congrArg (fun g : M →ₗ[R] R => g w) hf
  · intro h
    funext k
    exact sub_eq_zero.mp (h (b k))

/-- The class-code fibers do not depend on the complete basis, CM34.
This is equality of decisions, not equality of chosen vector coordinates. -/
theorem class_alias_basis_iff (b : Module.Basis K R M) (d : Module.Basis L R M)
    (a c : C) : classCode b a = classCode b c ↔ classCode d a = classCode d c := by
  rw [class_alias_iff, class_alias_iff]

/-- Every original observed row matches its recorded class under the
full code, CM34. The kernel dimension can exceed one. -/
theorem observed_code_match (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (b : Module.Basis K R (observedKernel cell labels i0 c0)) (t : S) :
    queryCode b (cell t).1 (cell t).2 = classCode b (labels t) := by
  apply (code_match_iff b _ _ _).mpr
  intro w
  exact w.property.1 t

/-- Full-code alias classes transfer any matched query consistently,
CM34. This does not assert equality of the raw class names. -/
theorem matched_alias (b : Module.Basis K R M) (i : I) (j : J) (a c : C)
    (hmatch : queryCode b i j = classCode b a) (halias : classCode b a = classCode b c) :
    queryCode b i j = classCode b c := by
  calc
    queryCode b i j = classCode b a := hmatch
    _ = classCode b c := halias

/-- Search an ordered bank of observed class names by exact full-vector
comparison, CM34. Neither an unknown target nor a readout enters the search. -/
noncomputable def chooseClass (b : Module.Basis K R M) (z : K → R) : List C → Option C
  | [] => none
  | c :: rest => by
    classical
    exact if z = classCode b c then some c else chooseClass b z rest

/-- A successful search carries the actual vector equality, CM34. All
soundness conclusions below are derived from this checked equality. -/
theorem chooseClass_match (b : Module.Basis K R M) (z : K → R) (bank : List C) {c : C}
    (h : chooseClass b z bank = some c) : z = classCode b c := by
  induction bank with
  | nil => cases h
  | cons a rest ih =>
    classical
    by_cases ha : z = classCode b a
    · simp only [chooseClass, ha, ite_true] at h
      cases h
      exact ha
    · simp only [chooseClass, ha, ite_false] at h
      exact ih h

/-- A head-of-bank match is accepted before later alternatives, CM34.
This fixes the class-name priority without using a held-out score. -/
theorem chooseClass_first (b : Module.Basis K R M) (c : C) (rest : List C) :
    chooseClass b (classCode b c) (c :: rest) = some c := by
  classical
  simp only [chooseClass, ite_true]

/-- An unchanged class bank makes the same decision in every complete
basis, CM34. Only the vector representation changes. -/
theorem chooseClass_basis_eq (b : Module.Basis K R M) (d : Module.Basis L R M)
    (i : I) (j : J) (bank : List C) :
    chooseClass b (queryCode b i j) bank = chooseClass d (queryCode d i j) bank := by
  induction bank with
  | nil => rfl
  | cons c rest ih =>
    classical
    have h := code_match_basis_iff b d i j c
    by_cases hb : queryCode b i j = classCode b c
    · simp only [chooseClass, hb, h.mp hb, ite_true]
    · have hd : queryCode d i j ≠ classCode d c := fun hc => hb (h.mpr hc)
      simp only [chooseClass, hb, hd, ite_false, ih]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes
