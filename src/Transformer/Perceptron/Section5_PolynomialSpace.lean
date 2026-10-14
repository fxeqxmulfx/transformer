/-
# The actual closed space of bounded-degree spherical polynomials

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
Original bounded-degree polynomials are restrictions of actual
multivariate polynomials to the original sphere. Their actual evaluation
map takes values in the continuous functions with their uniform norm.

The original degree bound is a finite-dimensional polynomial submodule.
Its actual image is therefore finite-dimensional and closed. Membership
is proved equivalent to the existing original polynomial predicate;
no representation, kernel identity or inverse is assumed by this space.

Consequently uniform limits and actual derivatives of curves in the
same bounded-degree space remain in it. Derivative membership follows
from the actual difference quotients, each an actual real linear
combination of original polynomials with the same degree bound. These
facts allow the proved original power tests to extend to mixed products.
-/

import Transformer.Perceptron.Section5_QuadraticPolynomials
import Mathlib.RingTheory.MvPolynomial.Basic
import Mathlib.Topology.ContinuousMap.Compact
import Mathlib.Analysis.Calculus.Deriv.Slope

open Real Set Filter
open scoped Topology

namespace Transformer.Perceptron

/-- The actual coordinate-polynomial evaluation on the original sphere,
as an algebra homomorphism into its actual continuous functions.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), polynomial tests. -/
noncomputable def spherePolynomialEval (d : ℕ) :
    MvPolynomial (Idx d) ℝ →ₐ[ℝ] C(SSphere d, ℝ) :=
  MvPolynomial.aeval fun i => ⟨fun x : SSphere d => (x : EucSpace d) i,
    (PiLp.continuous_apply 2 (fun _ : Idx d => ℝ) i).comp continuous_subtype_val⟩

/-- The original continuous evaluation is exactly the original coordinate
evaluation at every actual sphere point. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol` (i). Both evaluations use the same actual polynomial. -/
theorem spherePolynomialEval_apply (d : ℕ) (P : MvPolynomial (Idx d) ℝ) (x : SSphere d) :
    spherePolynomialEval d P x = MvPolynomial.eval (fun i => (x : EucSpace d) i) P := by
  exact MvPolynomial.comp_aeval_apply _ (ContinuousMap.evalAlgHom ℝ ℝ x) P

/-- The actual image of the original bounded-degree polynomial submodule
under the actual original sphere evaluation. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol` (i). Its membership and finiteness are proved below. -/
noncomputable def spherePolynomialSpace (d k : ℕ) : Submodule ℝ C(SSphere d, ℝ) :=
  ((spherePolynomialEval d).toLinearMap.domRestrict (MvPolynomial.restrictTotalDegree (Idx d) ℝ k)).range

/-- Membership is equivalent to the actual original polynomial predicate,
with its actual total-degree bound. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). Both directions retain the same original witness. -/
theorem mem_spherePolynomialSpace_iff {d k : ℕ} (f : C(SSphere d, ℝ)) :
    f ∈ spherePolynomialSpace d k ↔ IsPolyOfDegreeLE k (f : SSphere d → ℝ) := by
  constructor
  · rintro ⟨P, hP⟩
    refine ⟨P.1, (MvPolynomial.mem_restrictTotalDegree (Idx d) k P.1).mp P.2, fun x => ?_⟩
    rw [← hP]
    exact spherePolynomialEval_apply d P.1 x
  · rintro ⟨P, hP, hval⟩
    refine ⟨⟨P, (MvPolynomial.mem_restrictTotalDegree (Idx d) k P).mpr hP⟩, ?_⟩
    apply ContinuousMap.ext
    intro x
    exact (spherePolynomialEval_apply d P x).trans (hval x).symm

/-- The actual original bounded-degree spherical polynomial space is
finite-dimensional. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
Finiteness comes from the actual bounded-degree multivariate domain and
its actual linear image, including the sphere's polynomial relations. -/
instance finiteDimensional_spherePolynomialSpace (d k : ℕ) :
    FiniteDimensional ℝ (spherePolynomialSpace d k) := by
  unfold spherePolynomialSpace
  infer_instance

/-- The actual original polynomial space is closed in the original
uniform norm. Source: arXiv:2601.21366v2, §5.1, the finite polynomial
spaces in `lem: quadpol` (i). Its actual finite-dimensional image gives
closedness; this is distinct from density of all polynomial degrees. -/
theorem isClosed_spherePolynomialSpace (d k : ℕ) :
    IsClosed (spherePolynomialSpace d k : Set C(SSphere d, ℝ)) :=
  (spherePolynomialSpace d k).closed_of_finiteDimensional

/-- Enlarging the original degree bound includes the actual original
polynomial space in the new one. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). The actual polynomial witness is unchanged. -/
theorem spherePolynomialSpace_mono {d k n : ℕ} (hkn : k ≤ n) :
    spherePolynomialSpace d k ≤ spherePolynomialSpace d n := by
  intro f hf
  exact (mem_spherePolynomialSpace_iff f).mpr
    (isPolyOfDegreeLE_mono ((mem_spherePolynomialSpace_iff f).mp hf) hkn)

/-- Every actual real constant is in each original polynomial space.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree zero. -/
theorem const_mem_spherePolynomialSpace (d k : ℕ) (c : ℝ) :
    ContinuousMap.const (SSphere d) c ∈ spherePolynomialSpace d k :=
  (mem_spherePolynomialSpace_iff _).mpr
    (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const c) (Nat.zero_le k))

/-- The original continuous linear coordinate function of an actual
original test vector. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
It uses the actual original inner product at each original sphere point. -/
noncomputable def sphereLinearFunction {d : ℕ} (b : EucSpace d) : C(SSphere d, ℝ) :=
  ⟨fun x => inner (𝕜 := ℝ) b (x : EucSpace d), continuous_const.inner continuous_subtype_val⟩

/-- Actual powers of original linear coordinate functions belong to the
original space at their actual degree bound. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, the homogeneous polynomial tests. -/
theorem sphereLinearFunction_pow_mem {d n k : ℕ} (b : EucSpace d) (hn : n ≤ k) :
    sphereLinearFunction b ^ n ∈ spherePolynomialSpace d k := by
  apply (mem_spherePolynomialSpace_iff _).mpr
  have h := isPolyOfDegreeLE_mono (isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner b) n)
    (show n * 1 ≤ k by simpa using hn)
  obtain ⟨P, hP, hval⟩ := h
  refine ⟨P, hP, fun x => ?_⟩
  rw [ContinuousMap.pow_apply]
  exact hval x

/-- An actual uniform limit of original polynomials with one fixed
original degree bound retains that bound. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol` (i), finite polynomial spaces.
Closedness of the actual uniform-norm space supplies the actual witness. -/
theorem isPolyOfDegreeLE_of_uniform_polynomial_limit {d k : ℕ}
    (f : ℕ → C(SSphere d, ℝ)) (g : C(SSphere d, ℝ))
    (hf : ∀ n, IsPolyOfDegreeLE k (f n : SSphere d → ℝ)) (hlim : Tendsto f atTop (𝓝 g)) :
    IsPolyOfDegreeLE k (g : SSphere d → ℝ) := by
  apply (mem_spherePolynomialSpace_iff g).mp
  exact (isClosed_spherePolynomialSpace d k).mem_of_tendsto hlim
    (Eventually.of_forall fun n => (mem_spherePolynomialSpace_iff _).mpr (hf n))

/-- The actual derivative of a curve of original spherical polynomials
with a fixed original degree bound retains that bound.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, mixed polynomial tests.
Every actual difference quotient lies in the same actual linear space;
its proved uniform-norm limit is the original derivative. -/
theorem isPolyOfDegreeLE_of_hasDerivAt_polynomial_curve {d k : ℕ}
    (f : ℝ → C(SSphere d, ℝ)) (f' : C(SSphere d, ℝ)) (t : ℝ)
    (hf : ∀ r, IsPolyOfDegreeLE k (f r : SSphere d → ℝ)) (hd : HasDerivAt f f' t) :
    IsPolyOfDegreeLE k (f' : SSphere d → ℝ) := by
  apply (mem_spherePolynomialSpace_iff f').mp
  exact (isClosed_spherePolynomialSpace d k).mem_of_tendsto hd.tendsto_slope_zero
    (Eventually.of_forall fun r => (spherePolynomialSpace d k).smul_mem r⁻¹
      ((spherePolynomialSpace d k).sub_mem ((mem_spherePolynomialSpace_iff _).mpr (hf (t + r)))
        ((mem_spherePolynomialSpace_iff _).mpr (hf t))))

/-- Every degree, uniform-limit and actual-derivative hypothesis is
jointly realized by original constant-one functions on the actual circle.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The derivative is the actual zero continuous function. -/
example : (0 : ℕ) ≤ 2 ∧ (2 : ℕ) ≤ 2 ∧
    (∀ r : ℝ, IsPolyOfDegreeLE 2
      (((fun _ : ℝ => ContinuousMap.const (SSphere 2) (1 : ℝ)) r) : SSphere 2 → ℝ)) ∧
    HasDerivAt (fun _ : ℝ => ContinuousMap.const (SSphere 2) (1 : ℝ)) 0 0 ∧
    Tendsto (fun _ : ℕ => ContinuousMap.const (SSphere 2) (1 : ℝ)) atTop
      (𝓝 (ContinuousMap.const (SSphere 2) (1 : ℝ))) :=
  ⟨Nat.zero_le _, le_rfl, fun _ => isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const 1) (Nat.zero_le 2),
    hasDerivAt_const 0 _, tendsto_const_nhds⟩

end Transformer.Perceptron
