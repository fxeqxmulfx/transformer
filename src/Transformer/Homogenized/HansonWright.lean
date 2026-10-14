/-
# Homogenized Transformers — the Hanson–Wright inequality

Formalization of `lem:Hanson_wright` of arXiv:2604.01978v1, *Homogenized
Transformers*, which recalls `[vershynin2018high, Thm 6.2.1]`: a quadratic form
in a vector with independent subgaussian coordinates concentrates around its
mean, at the subgaussian rate in the Frobenius norm of the matrix and the
subexponential rate in its operator norm.

The source applies it to `X = Ṽu`, `B = Proj_{x_i}`, with `‖Proj‖_op = 1` and
`‖Proj‖_F = √(d-1)`, to bound the fluctuation of `‖Proj_{x_i} Ṽ u‖²`.

**What the source says and what is carried here.**

* `M = max_i ‖X_i‖_{ψ₂}` is the Orlicz `ψ₂` norm of the coordinates, which
  Mathlib does not have.  The hypothesis is written with the variance proxy of
  `ProbabilityTheory.HasSubgaussianMGF` instead — `mgf X_i(t) ≤ exp(M² t²/2)`,
  the parameter being `M2 = M²` — which is the same notion of subgaussianity up
  to a universal factor, and the universal factor is absorbed by the universal
  constant `c` the conclusion already carries.  The exponent's `M⁴‖B‖_F²` and
  `M²‖B‖_op` are therefore `M2² ‖B‖_F²` and `M2 ‖B‖_op`.

* "mean-zero" is not a separate hypothesis: `HasSubgaussianMGF X c` bounds the
  moment generating function at negative `t` as well, which already forces
  `E X = 0`.

* `‖B‖_F` is `√(Σ_{ij} B_{ij}²)`, so `‖B‖_F² = Σ_{ij} B_{ij}²`, and `‖B‖_op` is
  the `ℓ²` operator norm, i.e. the norm of `B` as a continuous linear map of
  `EuclideanSpace ℝ (Fin n)`.  Neither is replaced by a dominating constant:
  a dominating `opB` would make the bound larger, hence the statement weaker.

The proof combines the actual centered diagonal MGF with the off-diagonal
MGF obtained by random coordinate splitting and Gaussian comparison.
Chernoff optimization and the two-tail union bound then give the stated
rate, including zero variance and zero matrix norm.

Source: arXiv:2604.01978v1, `lem:Hanson_wright`; `[vershynin2018high, Thm 6.2.1]`.
-/

import Transformer.Homogenized.Section6_QuadraticChernoff
import Mathlib.Analysis.CStarAlgebra.Matrix

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Lemma (lem:Hanson_wright), the Hanson–Wright inequality.**  There is a
universal `c > 0` such that, for a random vector `Z ∈ ℝⁿ` with independent
subgaussian coordinates of variance proxy `M²`, every `B ∈ ℝ^{n×n}` and every
`t ≥ 0`,

  `P(|ZᵀBZ - E[ZᵀBZ]| ≥ t)
      ≤ 2 exp(-c min{t²/(M⁴‖B‖_F²), t/(M²‖B‖_op)})`.

See the module docstring for the `ψ₂`-norm substitution this statement makes.

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem hanson_wright :
    ∃ c : ℝ, 0 < c ∧
      ∀ (n : ℕ) (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω), IsProbabilityMeasure P →
      ∀ (Z : Ω → EucSpace n) (M2 : ℝ≥0),
        iIndepFun (fun (i : Fin n) (ω : Ω) => Z ω i) P →
        (∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) M2 P) →
      ∀ (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ), 0 ≤ t →
        P {ω | t ≤ |(∑ i : Fin n, ∑ j : Fin n, B i j * Z ω i * Z ω j)
              - ∫ ω', ∑ i : Fin n, ∑ j : Fin n, B i j * Z ω' i * Z ω' j ∂P|}
          ≤ ENNReal.ofReal (2 * Real.exp (-(c *
              min (t ^ 2 / ((M2 : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2))
                (t / ((M2 : ℝ) * matOpNorm B))))) := by
  let K : ℝ := 8192 * (Real.exp 1) ^ 4
  let r : ℝ := 1 / (64 * Real.exp 1)
  have hK : 0 < K := by positivity
  have hr : 0 < r := by positivity
  refine ⟨min (1 / (4 * K)) (r / 2), quadratic_tail_constant_pos K r hK hr, ?_⟩
  intro n Ω m P hP Z M2 hindep hZ B t ht
  let : IsProbabilityMeasure P := hP
  let X : Ω → ℝ := fun ω => hansonWrightQuadraticForm B (Z ω) -
    ∫ η, hansonWrightQuadraticForm B (Z η) ∂P
  let v : ℝ := (M2 : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2
  let w : ℝ := (M2 : ℝ) * matOpNorm B
  have hv : 0 ≤ v := by positivity
  have hw : 0 ≤ w := mul_nonneg M2.coe_nonneg (matOpNorm_nonneg B)
  have hmgf : ∀ u : ℝ, |u| * w ≤ r →
      Integrable (fun ω => Real.exp (u * X ω)) P ∧
      (∫ ω, Real.exp (u * X ω) ∂P) ≤ Real.exp (K * u ^ 2 * v) := by
    intro u hu
    have hsmall : |u| * (M2 : ℝ) * matOpNorm B ≤ 1 / (64 * Real.exp 1) := by
      simpa only [w, r, mul_assoc] using hu
    obtain ⟨hi, hb⟩ := subgaussian_centered_quadratic_exp_integral_le
      P Z M2 hindep hZ B u hsmall
    refine ⟨hi, ?_⟩
    simpa only [K, v, mul_assoc] using hb
  have htail := abs_tail_of_local_quadratic_mgf P X K v w r hK hv hw hr hmgf t ht
  simpa only [X, v, w, hansonWrightQuadraticForm] using htail

/-- The hypotheses `hanson_wright` quantifies over are satisfiable, and not only
on the empty index: the one-point space carries the zero vector of `ℝⁿ`, whose
coordinates are independent and subgaussian with variance proxy `0`. -/
example (n : ℕ) :
    IsProbabilityMeasure (Measure.dirac ()) ∧
      iIndepFun (fun (i : Fin n) (_ : Unit) => (0 : EucSpace n) i) (Measure.dirac ()) ∧
      ∀ i : Fin n,
        HasSubgaussianMGF (fun _ : Unit => (0 : EucSpace n) i) 0 (Measure.dirac ()) := by
  refine ⟨inferInstance, iIndepFun_of_unit _, fun i => ?_⟩
  simp

/-- The same hypotheses also hold for a genuinely random vector with
two independent standard Gaussian coordinates and positive variance proxy.
This witnesses the nonempty, positive-variance case of the original
Hanson–Wright inequality on its actual product probability measure. -/
example : ∃ Z : (Fin 2 → ℝ) → EucSpace 2,
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    iIndepFun (fun i : Fin 2 => fun z => Z z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z => Z z i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) := by
  refine ⟨WithLp.toLp 2, inferInstance, ?_, hasSubgaussianMGF_standardProduct_coordinate⟩
  simpa only [WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
    (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id)
    (fun i => measurable_id.aemeasurable)

/-- A genuine Gaussian quadratic specialization of §6.3,
`lem:Hanson_wright`, keeps the actual expectation and gives Frobenius
denominator two and operator denominator one for the identity matrix.
This also checks that the conclusion applies to a nonconstant quadratic
form on the original, nonempty product probability space. -/
example : ∃ c : ℝ, 0 < c ∧ ∀ t : ℝ, 0 ≤ t →
    (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))
      {z | t ≤ |(∑ i : Fin 2, z i ^ 2) -
        ∫ z', ∑ i : Fin 2, z' i ^ 2 ∂(Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))|} ≤
      ENNReal.ofReal (2 * Real.exp (-(c * min (t ^ 2 / 2) t))) := by
  obtain ⟨c, hc, htail⟩ := hanson_wright
  refine ⟨c, hc, ?_⟩
  intro t ht
  have hindep : iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) := by
    simpa only [id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id)
      (fun i => measurable_id.aemeasurable)
  have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by
    rw [← Matrix.diagonal_one, matOpNorm_diagonal]
    simp
  have h := htail 2 (Fin 2 → ℝ) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))
    (by infer_instance) (WithLp.toLp 2) 1 hindep hasSubgaussianMGF_standardProduct_coordinate
    (1 : Matrix (Fin 2) (Fin 2) ℝ) t ht
  simpa [Matrix.one_apply, Fin.sum_univ_two, hn, pow_two] using h

end Homogenized
end Transformer
