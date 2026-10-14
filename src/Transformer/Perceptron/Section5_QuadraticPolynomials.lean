/-
# Original finite quadratic functions as actual spherical polynomials

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i), and §5.3,
`rem:genericity-vacuity`. The original finite sums of squared neuron
coordinates are restrictions of actual multivariate polynomials of
total degree at most two.
The bounded-degree predicate retains that actual polynomial witness;
only a bound is intrinsic, since `Σ_i x_i² = 1` on the original sphere.

The polynomial operations below retain the actual witnesses and prove
their actual degree bounds. A linear inner product is represented by
the actual sum of its coordinate coefficients times the original
polynomial variables. Squaring gives degree at most two, and actual
finite real weighted sums and a constant preserve that bound.

No polynomial-density statement or inverse theorem is assumed. These
lemmas establish the degree of the actual candidate produced by the
quadratic-kernel inverse, rather than declaring it polynomial by fiat.
-/

import Transformer.Perceptron.Transform
import Transformer.Metastability.Section4_UniformLaw
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Topology.Algebra.MvPolynomial

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Perceptron

/-- An actual multivariate polynomial of total degree at most `k`
restricts to the given function on the original sphere.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
def IsPolyOfDegreeLE {d : ℕ} (k : ℕ) (f : SSphere d → ℝ) : Prop :=
  ∃ P : MvPolynomial (Idx d) ℝ, P.totalDegree ≤ k ∧
    ∀ x : SSphere d, f x = MvPolynomial.eval (fun i => (x : EucSpace d) i) P

/-- An actual constant is a degree-zero original spherical polynomial.
Source: arXiv:2601.21366v2, §5.1, the degree-zero part of `lem: quadpol` (i). -/
theorem isPolyOfDegreeLE_const {d : ℕ} (c : ℝ) : IsPolyOfDegreeLE 0 (fun _ : SSphere d => c) :=
  ⟨MvPolynomial.C c, by simp [MvPolynomial.totalDegree_C], fun _ => by simp⟩

/-- A larger actual degree bound keeps the same actual polynomial witness.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), degree bounds. -/
theorem isPolyOfDegreeLE_mono {d k n : ℕ} {f : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) (hkn : k ≤ n) : IsPolyOfDegreeLE n f := by
  obtain ⟨P, hP, hval⟩ := hf
  exact ⟨P, hP.trans hkn, hval⟩

/-- Adding two original spherical polynomials preserves their actual
common degree bound. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The witness is the sum of their actual multivariate polynomial witnesses. -/
theorem isPolyOfDegreeLE_add {d k : ℕ} {f g : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) (hg : IsPolyOfDegreeLE k g) :
    IsPolyOfDegreeLE k (fun y => f y + g y) := by
  obtain ⟨P, hP, hPv⟩ := hf
  obtain ⟨Q, hQ, hQv⟩ := hg
  exact ⟨P + Q, (MvPolynomial.totalDegree_add P Q).trans (max_le hP hQ),
    fun y => by simpa only [MvPolynomial.eval_add] using congrArg₂ (fun a b : ℝ => a + b) (hPv y) (hQv y)⟩

/-- Scaling an original spherical polynomial by an actual real constant
preserves its actual degree bound. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). The actual witness is the original polynomial times
the constant polynomial, whose total degree is zero. -/
theorem isPolyOfDegreeLE_const_mul {d k : ℕ} (c : ℝ) {f : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) : IsPolyOfDegreeLE k (fun y => c * f y) := by
  obtain ⟨P, hP, hval⟩ := hf
  refine ⟨MvPolynomial.C c * P, ?_, fun y => by simp [hval y]⟩
  exact (MvPolynomial.totalDegree_mul _ _).trans (by simpa using hP)

/-- Taking the actual power multiplies the original degree bound by that
power. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The witness is the actual power of the original multivariate polynomial. -/
theorem isPolyOfDegreeLE_pow {d k : ℕ} {f : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) (n : ℕ) : IsPolyOfDegreeLE (n * k) (fun y => f y ^ n) := by
  obtain ⟨P, hP, hval⟩ := hf
  exact ⟨P ^ n, (MvPolynomial.totalDegree_pow P n).trans (Nat.mul_le_mul_left n hP),
    fun y => by simpa only [MvPolynomial.eval_pow] using congrArg (fun a : ℝ => a ^ n) (hval y)⟩

/-- The actual product of two original spherical polynomials has degree
at most the sum of their original bounds. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol` (i). Actual multiplication supplies the witness. -/
theorem isPolyOfDegreeLE_mul {d k n : ℕ} {f g : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) (hg : IsPolyOfDegreeLE n g) :
    IsPolyOfDegreeLE (k + n) (fun y => f y * g y) := by
  obtain ⟨P, hP, hPv⟩ := hf
  obtain ⟨Q, hQ, hQv⟩ := hg
  exact ⟨P * Q, (MvPolynomial.totalDegree_mul P Q).trans (Nat.add_le_add hP hQ),
    fun y => by simpa only [MvPolynomial.eval_mul] using congrArg₂ (fun a b : ℝ => a * b) (hPv y) (hQv y)⟩

/-- Every original spherical polynomial is continuous on the actual
sphere. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
This uses its actual coordinate polynomial and actual sphere inclusion. -/
theorem continuous_of_isPolyOfDegreeLE {d k : ℕ} {f : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) : Continuous f := by
  obtain ⟨P, -, hval⟩ := hf
  have hp : Continuous (fun y : SSphere d =>
      MvPolynomial.eval (fun i => (y : EucSpace d) i) P) :=
    (MvPolynomial.continuous_eval P).comp (continuous_pi fun i =>
      (PiLp.continuous_apply 2 (fun _ : Idx d => ℝ) i).comp continuous_subtype_val)
  rwa [← show f = (fun y : SSphere d => MvPolynomial.eval (fun i => (y : EucSpace d) i) P)
    from funext hval] at hp

/-- Every actual original spherical polynomial is integrable against
every actual finite original law. Source: arXiv:2601.21366v2, §5.1,
the original polynomial tests in the proof of `lem: quadpol` (i). -/
theorem integrable_of_isPolyOfDegreeLE {d k : ℕ} {f : SSphere d → ℝ}
    (hf : IsPolyOfDegreeLE k f) (ν : Measure (SSphere d)) [IsFiniteMeasure ν] : Integrable f ν :=
  Perspective.integrable_of_continuous_compact (continuous_of_isPolyOfDegreeLE hf) ν

/-- A finite sum of actual original spherical polynomials preserves their
common actual degree bound. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). The induction constructs actual sum witnesses. -/
theorem isPolyOfDegreeLE_finsetSum {d k : ℕ} {ι : Type*} (S : Finset ι)
    (f : ι → SSphere d → ℝ) (hf : ∀ i ∈ S, IsPolyOfDegreeLE k (f i)) :
    IsPolyOfDegreeLE k (fun y => ∑ i ∈ S, f i y) := by
  classical
  induction S using Finset.induction_on with
  | empty =>
    simpa using isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const (d := d) 0) (Nat.zero_le k)
  | @insert i S hi ih =>
    have hiP := hf i (Finset.mem_insert_self _ _)
    have hS := ih fun j hj => hf j (Finset.mem_insert_of_mem hj)
    simpa only [Finset.sum_insert hi] using isPolyOfDegreeLE_add hiP hS

/-- Every original ambient linear function is an actual degree-one
spherical polynomial. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
Its witness uses the actual coordinate coefficients of the same vector. -/
theorem isPolyOfDegreeLE_inner {d : ℕ} (b : EucSpace d) :
    IsPolyOfDegreeLE 1 (fun y : SSphere d => inner (𝕜 := ℝ) b (y : EucSpace d)) := by
  classical
  let P : MvPolynomial (Idx d) ℝ := ∑ i, MvPolynomial.C (b i) * MvPolynomial.X i
  have hsum (S : Finset (Idx d)) :
      (∑ i ∈ S, MvPolynomial.C (b i) * MvPolynomial.X i).totalDegree ≤ 1 := by
    induction S using Finset.induction_on with
    | empty => simp
    | @insert i S hi ih =>
      rw [Finset.sum_insert hi]
      exact (MvPolynomial.totalDegree_add _ _).trans
        (max_le ((MvPolynomial.totalDegree_mul _ _).trans (by simp)) ih)
  refine ⟨P, hsum Finset.univ, fun y => ?_⟩
  simp only [P, MvPolynomial.eval_sum, MvPolynomial.eval_mul, MvPolynomial.eval_C,
    MvPolynomial.eval_X, PiLp.inner_apply, RCLike.inner_apply', starRingEnd_apply, star_trivial]

/-- A constant plus the original finite weighted squared neuron
coordinates is an actual spherical polynomial of degree at most two.
Source: arXiv:2601.21366v2, §5.3, `rem:genericity-vacuity`.
The actual original vectors, weights and constant determine its witness. -/
theorem isPolyOfDegreeLE_const_add_weighted_squares {d : ℕ} {ι : Type*} (S : Finset ι)
    (A : ℝ) (ω : ι → ℝ) (a : ι → EucSpace d) :
    IsPolyOfDegreeLE 2 (fun y : SSphere d => A + ∑ i ∈ S, ω i * inner (𝕜 := ℝ) (a i) (y : EucSpace d) ^ 2) := by
  apply isPolyOfDegreeLE_add (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const A) (by omega))
  apply isPolyOfDegreeLE_finsetSum
  intro i _
  exact isPolyOfDegreeLE_const_mul (ω i) (isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner (a i)) 2)

/-- Every actual polynomial hypothesis of the operations holds jointly
for the original constant one and a nonzero original coordinate function.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The actual finite family supplies the finite-sum inputs at degree two. -/
example : (0 : ℕ) ≤ 2 ∧ IsPolyOfDegreeLE (d := 2) 0 (fun _ => (1 : ℝ)) ∧
    IsPolyOfDegreeLE 1 (fun y : SSphere 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (y : EucSpace 2)) ∧
    ∀ i ∈ (Finset.univ : Finset (Fin 2)), IsPolyOfDegreeLE 2
      (fun y : SSphere 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (y : EucSpace 2) ^ 2) :=
  ⟨Nat.zero_le _, isPolyOfDegreeLE_const 1, isPolyOfDegreeLE_inner _,
    fun _ _ => isPolyOfDegreeLE_pow (isPolyOfDegreeLE_inner _) 2⟩

/-- The polynomial and finite-integrability inputs also hold jointly for
the actual original uniform probability and actual constant one.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
example : IsFiniteMeasure (MeanField.uniformLaw 2) ∧
    IsPolyOfDegreeLE (d := 2) 0 (fun _ => (1 : ℝ)) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let _ : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨inferInstance, isPolyOfDegreeLE_const 1⟩

end Transformer.Perceptron
