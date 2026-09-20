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

Not proved here: this is an external theorem the source recalls verbatim, and
it is on the books because the source's own estimates are derived from it.

Source: arXiv:2604.01978v1, `lem:Hanson_wright`; `[vershynin2018high, Thm 6.2.1]`.
-/

import Transformer.Homogenized.RandomChain
import Mathlib.Analysis.CStarAlgebra.Matrix

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- The `ℓ²` operator norm of a real square matrix: its norm as a continuous
linear endomorphism of `EuclideanSpace ℝ (Fin n)`.

Source: arXiv:2604.01978v1, `lem:Hanson_wright` (`‖B‖_op`). -/
noncomputable def matOpNorm {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ‖(Matrix.toEuclideanCLM (𝕜 := ℝ) B : EucSpace n →L[ℝ] EucSpace n)‖

theorem matOpNorm_nonneg {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : 0 ≤ matOpNorm B :=
  norm_nonneg _

@[simp] theorem matOpNorm_zero {n : ℕ} : matOpNorm (0 : Matrix (Fin n) (Fin n) ℝ) = 0 := by
  simp [matOpNorm]

/-- **Lemma (lem:Hanson_wright), the Hanson–Wright inequality.**  There is a
universal `c > 0` such that, for a random vector `Z ∈ ℝⁿ` with independent
subgaussian coordinates of variance proxy `M²`, every `B ∈ ℝ^{n×n}` and every
`t ≥ 0`,

  `P(|ZᵀBZ - E[ZᵀBZ]| ≥ t)
      ≤ 2 exp(-c min{t²/(M⁴‖B‖_F²), t/(M²‖B‖_op)})`.

See the module docstring for the `ψ₂`-norm substitution this statement makes.

Not proved here.

Source: arXiv:2604.01978v1, `lem:Hanson_wright`. -/
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
  sorry

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

end Homogenized
end Transformer
