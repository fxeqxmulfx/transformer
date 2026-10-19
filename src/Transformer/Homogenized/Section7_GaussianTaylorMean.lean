module

public import Transformer.Homogenized.Section7_GaussianQueryCovariance
public import Transformer.Homogenized.Section7_CorrelationTaylorExpectation

/-!
# Exact mean of the genuine Gaussian correlation Taylor polynomial

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the contraction
of the displayed Hessian with the actual query covariance.
The polynomial used here is the genuine quadratic Taylor polynomial
of normalized correlation, with all its mixed terms retained.

First, linearity of expectation is justified for square-integrable
centered scalar deviations under an arbitrary probability measure.
All products in the polynomial are integrable by their actual second
moments. No probabilistic remainder estimate is assumed in this step.

For the original images of arbitrary two unit inputs under one Gaussian
matrix factor, the exact covariance formulas then give expected
polynomial `r + (r³-r)/(2d)`. This is the single-factor correction.
The source uses the same coefficient for its product of two factors;
the full product must instead be treated by composing the independent
factor laws, including the actual Taylor remainder at both stages.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The genuine quadratic polynomial is integrable whenever its
three actual deviations are square integrable. No centering or
third-moment hypothesis is needed for this assertion.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the quadratic expectation before estimating the cubic remainder. -/
theorem integrable_deltaCorrelationTaylor {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsFiniteMeasure P] (r : ℝ) (w u v : Ω → ℝ)
    (hw : MemLp w 2 P) (hu : MemLp u 2 P) (hv : MemLp v 2 P) :
    Integrable (fun ω => deltaCorrelationTaylor r (w ω) (u ω) (v ω)) P := by
  have iw := hw.integrable (by norm_num)
  have iu := hu.integrable (by norm_num)
  have iv := hv.integrable (by norm_num)
  have iu2 : Integrable (fun ω => u ω ^ 2) P := hu.integrable_sq
  have iv2 : Integrable (fun ω => v ω ^ 2) P := hv.integrable_sq
  have iuv : Integrable (fun ω => u ω * v ω) P := hu.integrable_mul hv
  have iadd : Integrable (fun ω => u ω + v ω) P := by convert iu.add iv using 1
  have iwadd : Integrable (fun ω => w ω * (u ω + v ω)) P := by
    convert hw.integrable_mul (hu.add hv) using 1
  have isq : Integrable (fun ω => u ω ^ 2 + v ω ^ 2) P := by convert iu2.add iv2 using 1
  have ib : Integrable (fun ω => 3 * (u ω ^ 2 + v ω ^ 2) / 8 + u ω * v ω / 4) P := by
    convert ((isq.const_mul 3).div_const 8).add (iuv.div_const 4) using 1
  unfold deltaCorrelationTaylor
  convert ((((integrable_const r).add iw).sub ((iadd.const_mul r).div_const 2)).sub
    (iwadd.div_const 2)).add (ib.const_mul r) using 1

/-- Exact integration of the genuine quadratic Taylor polynomial
for centered square-integrable deviations. Source: arXiv:2604.01978v1,
§7.3, the Hessian contraction in `lem:delta_method`. -/
theorem integral_centered_deltaCorrelationTaylor {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (r : ℝ) (w u v : Ω → ℝ)
    (hw : MemLp w 2 P) (hu : MemLp u 2 P) (hv : MemLp v 2 P)
    (hw0 : ∫ ω, w ω ∂P = 0) (hu0 : ∫ ω, u ω ∂P = 0) (hv0 : ∫ ω, v ω ∂P = 0) :
    (∫ ω, deltaCorrelationTaylor r (w ω) (u ω) (v ω) ∂P) =
      r - ((∫ ω, w ω * u ω ∂P) + ∫ ω, w ω * v ω ∂P) / 2 +
        r * (3 * ((∫ ω, u ω ^ 2 ∂P) + ∫ ω, v ω ^ 2 ∂P) / 8 +
          (∫ ω, u ω * v ω ∂P) / 4) := by
  have iw := hw.integrable (by norm_num)
  have iu := hu.integrable (by norm_num)
  have iv := hv.integrable (by norm_num)
  have iwu : Integrable (fun ω => w ω * u ω) P := hw.integrable_mul hu
  have iwv : Integrable (fun ω => w ω * v ω) P := hw.integrable_mul hv
  have iuv : Integrable (fun ω => u ω * v ω) P := hu.integrable_mul hv
  have iu2 : Integrable (fun ω => u ω ^ 2) P := hu.integrable_sq
  have iv2 : Integrable (fun ω => v ω ^ 2) P := hv.integrable_sq
  have iuvAdd : Integrable (fun ω => u ω + v ω) P := by convert iu.add iv using 1
  have iwuv : Integrable (fun ω => w ω * (u ω + v ω)) P := by
    convert hw.integrable_mul (hu.add hv) using 1
  have isq : Integrable (fun ω => u ω ^ 2 + v ω ^ 2) P := by convert iu2.add iv2 using 1
  have ib : Integrable (fun ω => 3 * (u ω ^ 2 + v ω ^ 2) / 8 + u ω * v ω / 4) P := by
    convert ((isq.const_mul 3).div_const 8).add (iuv.div_const 4) using 1
  have irw : Integrable (fun ω => r + w ω) P := by convert (integrable_const r).add iw using 1
  have ia : Integrable (fun ω => r + w ω - r * (u ω + v ω) / 2) P := by
    convert irw.sub ((iuvAdd.const_mul r).div_const 2) using 1
  have ia2 : Integrable (fun ω => r + w ω - r * (u ω + v ω) / 2 - w ω * (u ω + v ω) / 2) P := by
    convert ia.sub (iwuv.div_const 2) using 1
  have hwuv : (∫ ω, w ω * (u ω + v ω) ∂P) =
      (∫ ω, w ω * u ω ∂P) + ∫ ω, w ω * v ω ∂P := by
    simpa only [mul_add] using integral_add iwu iwv
  unfold deltaCorrelationTaylor
  rw [integral_add ia2 (ib.const_mul r), integral_sub ia (iwuv.div_const 2),
    integral_sub irw ((iuvAdd.const_mul r).div_const 2),
    integral_add (integrable_const r) iw, integral_div, integral_const_mul,
    integral_add iu iv, integral_div, hwuv, integral_const_mul,
    integral_add ((isq.const_mul 3).div_const 8) (iuv.div_const 4),
    integral_div, integral_const_mul, integral_add iu2 iv2, integral_div]
  simp only [integral_const, probReal_univ, one_smul, hw0, hu0, hv0]
  ring

/-- **Exact expected quadratic polynomial for one actual Gaussian factor.**
The coefficient is obtained from the original score and norm deviations,
uniformly for every original unit input pair and positive dimension.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the displayed Hessian contraction, applied to one factor only. -/
theorem integral_gaussianMatrix_unit_delta_taylor_mean {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      deltaCorrelationTaylor (inner ℝ x y)
        (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y)
        (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1)
        (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1) ∂gaussianMatrixLaw d) =
          inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (2 * (d : ℝ)) := by
  have hu : MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      ‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1) 2 (gaussianMatrixLaw d) := by
    simpa only [hx, one_pow] using memLp_gaussianMatrix_query_norm_deviation x
  have hv : MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      ‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1) 2 (gaussianMatrixLaw d) := by
    simpa only [hy, one_pow] using memLp_gaussianMatrix_query_norm_deviation y
  have hu0 : (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1) ∂gaussianMatrixLaw d) = 0 := by
    simpa only [hx, one_pow] using integral_gaussianMatrix_query_norm_deviation hd x
  have hv0 : (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1) ∂gaussianMatrixLaw d) = 0 := by
    simpa only [hy, one_pow] using integral_gaussianMatrix_query_norm_deviation hd y
  have hwu := integral_gaussianMatrix_query_score_square_deviation hd x y
  rw [hx, one_pow, mul_one] at hwu
  have hwv : (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y) *
        (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1) ∂gaussianMatrixLaw d) =
          2 * inner ℝ x y / (d : ℝ) := by
    simpa only [hy, one_pow, mul_one, real_inner_comm] using
      integral_gaussianMatrix_query_score_square_deviation hd y x
  have huv := integral_gaussianMatrix_query_square_deviation_product hd x y
  rw [hx, hy, one_pow] at huv
  rw [integral_centered_deltaCorrelationTaylor (gaussianMatrixLaw d) (inner ℝ x y) _ _ _
    (memLp_gaussianMatrix_query_score_deviation x y) hu hv
    (integral_gaussianMatrix_query_score_deviation hd x y) hu0 hv0,
    hwu, hwv, huv, integral_gaussianMatrix_unit_query_deviation_sq hd x hx,
    integral_gaussianMatrix_unit_query_deviation_sq hd y hy]
  ring

/-- The scalar standard Gaussian identity gives jointly centered,
square-integrable deviations for the general expectation formula. -/
example : MemLp (id : ℝ → ℝ) 2 (gaussianReal 0 1) ∧
    (∫ z : ℝ, z ∂gaussianReal 0 1) = 0 := by
  exact ⟨memLp_id_gaussianReal 2, by simp⟩

/-- Two original coordinate unit queries in positive dimension
jointly satisfy all the actual Gaussian polynomial hypotheses. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
