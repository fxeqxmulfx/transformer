/-
# Homogenized Transformers — the Gaussian interpolation lemma

Formalization of `lem:gaussian_interp` of arXiv:2604.01978v1, *Homogenized
Transformers*, which recalls `[vershynin2018high, Lemma 7.2.7]`: interpolating
between two independent centered Gaussian vectors by
`X_t = √t X + √(1-t) Y` moves the expectation of a test function at the rate

  `d/dt E φ(X_t) = ½ E[Tr((Σ^X - Σ^Y)ᵀ ∇²φ(X_t))]`.

The source uses it on `f(h,g)` a product of two softmax weights, to
differentiate `F(R) = E f(h,g)` in the Gram matrix `R` — this is the step that
turns `s_{μ_X}` into a function of `R` alone and produces `eq:steins_lemma`.

**What the source says and what is carried here.**

* The source states it for every `φ ∈ C²(ℝ^m)` with no growth condition, under
  which neither side need be finite: `φ(z) = e^{‖z‖²}` is `C²` and not
  integrable against the standard Gaussian. We prove the corrected version
  with bounded `φ` and derivatives of order `≤ 2`. These hypotheses make the
  original expectations finite and justify dominated differentiation.

* The source calls `X,Y` "random vectors". Their original measurability is
  restored explicitly as `AEMeasurable X P` and `AEMeasurable Y P`.
  `IsGaussian (P.map X)` alone does not record this: Mathlib's `Measure.map`
  can return a fallback Dirac measure for a nonmeasurable function. The
  explicit premises ensure that the given laws are the actual original laws.

* `t` ranges over `[0,1]`, where `X_t` is defined, and the derivative at the
  two endpoints is the one-sided one, i.e. `HasDerivWithinAt` on `Set.Icc 0 1`.
  The proof retains both original endpoint values of the covariance trace.

* The covariance is pinned by its entries, `Σ^X_{ij} = E[X_i X_j]`, alongside
  `IsGaussian` for the shape of the law and `E X_i = 0` for centering; Mathlib's
  `IsGaussian` is "every continuous linear form pushes to a real Gaussian" and
  carries no covariance matrix of its own.

* `Tr(Aᵀ H) = Σ_{ij} A_{ij} H_{ij}`, and the Hessian entries `H_{ij}` are the
  second derivative `D²φ(z)[e_i, e_j]`, as in `sphHess` of `Generator.lean`.

The proof uses genuine scalar Gaussian integration by parts, actual Gaussian
coordinate and residual independence, the covariance square-root representation,
Fubini on the original independent vectors and actual dominated differentiation.
Continuity of the original expectations extends the identity to both endpoints.
Singular covariance matrices are included throughout.

The source's softmax products are bounded. Their derivatives with respect to
`h,g` can grow because the scores `h_iᵀg_j` are bilinear. Applying interpolation
to that application additionally needs a version with derivative growth control;
bounded derivatives of the softmax in its scores do not give that hypothesis.

Source: arXiv:2604.01978v1, `lem:gaussian_interp`, `eq:interpolation_gaussian`;
`[vershynin2018high, Lemma 7.2.7]`.
-/

import Transformer.Homogenized.Basic
import Transformer.Homogenized.Section7_GaussianInterpolationCovariance

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- `eq:interpolation_gaussian`: the interpolation `X_t = √t X + √(1-t) Y`
between two random vectors of `ℝ^m`.

Source: arXiv:2604.01978v1, `eq:interpolation_gaussian`. -/
noncomputable def gaussInterp {m : ℕ} {Ω : Type*} (X Y : Ω → EucSpace m)
    (t : ℝ) (ω : Ω) : EucSpace m :=
  Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω

@[simp] theorem gaussInterp_zero_fun {m : ℕ} {Ω : Type*} (t : ℝ) (ω : Ω) :
    gaussInterp (0 : Ω → EucSpace m) 0 t ω = 0 := by
  simp [gaussInterp]

/-- **Lemma (lem:gaussian_interp), Gaussian interpolation.**  For independent
centered Gaussian vectors `X ∼ 𝒩(0, Σ^X)` and `Y ∼ 𝒩(0, Σ^Y)` on `ℝ^m` and a
`C²` test function `φ` with bounded derivatives of order `≤ 2`,

  `d/dt E φ(√t X + √(1-t) Y) = ½ E[Tr((Σ^X - Σ^Y)ᵀ ∇²φ(X_t))]`

on `[0,1]`.

The source states "for all `φ ∈ C²`" without growth control. The corrected
statement adds the original uniform test and derivative bounds, and restores
the actual measurability implicit in its term "random vectors". See the
module docstring for these changes and the one-sided endpoint reading.

Actual Gaussian integration by parts and dominated differentiation prove the
identity on the original probability space. Continuity of the actual covariance
trace then gives the original one-sided derivative at both endpoints.

Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`. -/
theorem gaussian_interp {m : ℕ} {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (X Y : Ω → EucSpace m)
    (SX SY : Matrix (Fin m) (Fin m) ℝ)
    (hXmeas : AEMeasurable X P) (hYmeas : AEMeasurable Y P)
    (hX : IsGaussian (P.map X)) (hY : IsGaussian (P.map Y))
    (hXc : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0) (hYc : ∀ i : Fin m, ∫ ω, Y ω i ∂P = 0)
    (hXcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = SX i j)
    (hYcov : ∀ i j : Fin m, ∫ ω, Y ω i * Y ω j ∂P = SY i j)
    (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (Cφ : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ Cφ)
    (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    HasDerivWithinAt (fun r => ∫ ω, φ (gaussInterp X Y r ω) ∂P)
      ((1 / 2 : ℝ) * ∫ ω, ∑ i : Fin m, ∑ j : Fin m, (SX i j - SY i j) *
        iteratedFDeriv ℝ 2 φ (gaussInterp X Y t ω)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] ∂P)
      (Set.Icc 0 1) t := by
  have hXlaw : HasGaussianLaw X P := ⟨hXmeas, hX⟩
  have hYlaw : HasGaussianLaw Y P := ⟨hYmeas, hY⟩
  have hf : Continuous (fun r : ℝ => ∫ ω, φ (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P) := by
    simpa only [gaussianInterpolationPoint] using continuous_expectation_gaussian_blend P X Y
      hXmeas hYmeas φ hφ.continuous Cφ (norm_gaussian_test_le φ Cφ hbdd)
  have hg : Continuous (fun r : ℝ => (1 / 2 : ℝ) * ∫ ω,
      gaussianCovarianceTrace φ (SX - SY) (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P) := by
    have h := continuous_expectation_gaussianCovarianceTrace P X Y hXmeas hYmeas φ hφ Cφ hbdd (SX - SY)
    simpa only [gaussianInterpolationPoint] using h.const_mul (1 / 2 : ℝ)
  have hd (r : ℝ) (hr : r ∈ Set.Ioo (0 : ℝ) 1) :
      HasDerivAt (fun q : ℝ => ∫ ω, φ (gaussianInterpolationPoint q (X ω) (Y ω)) ∂P)
        ((1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY)
          (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P) r := by
    have h := (gaussianInterpolation_differentiate_gaussian_integral P X Y hXlaw hYlaw φ hφ Cφ hbdd r hr).2
    rw [integral_gaussianInterpolation_velocity_covariance P X Y hXlaw hYlaw SX SY hXc hYc
      hXcov hYcov hindep φ hφ Cφ hbdd r hr] at h
    exact h
  have h := interpolation_hasDerivWithinAt_endpoints
    (fun r : ℝ => ∫ ω, φ (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P)
    (fun r : ℝ => (1 / 2 : ℝ) * ∫ ω, gaussianCovarianceTrace φ (SX - SY)
      (gaussianInterpolationPoint r (X ω) (Y ω)) ∂P) hf hg hd t ht
  simpa only [gaussInterp, gaussianInterpolationPoint, gaussianCovarianceTrace, Matrix.sub_apply] using h

/-- The hypotheses of `gaussian_interp` are satisfiable: the one-point space
carries the degenerate Gaussian `δ_0` of `ℝ^m`, twice, independent of itself,
with covariance `0`; and the zero test function is `C²` with all derivatives
bounded by `0`. -/
example (m : ℕ) :
    AEMeasurable (fun _ : Unit => (0 : EucSpace m)) (Measure.dirac ()) ∧
      IsGaussian ((Measure.dirac ()).map (fun _ : Unit => (0 : EucSpace m))) ∧
      (∀ i : Fin m, ∫ _ω, (0 : EucSpace m) i ∂(Measure.dirac ()) = 0) ∧
      (∀ i j : Fin m,
        ∫ _ω, (0 : EucSpace m) i * (0 : EucSpace m) j ∂(Measure.dirac ())
          = (0 : Matrix (Fin m) (Fin m) ℝ) i j) ∧
      IndepFun (fun _ : Unit => (0 : EucSpace m)) (fun _ : Unit => (0 : EucSpace m))
        (Measure.dirac ()) ∧
      ContDiff ℝ 2 (fun _ : EucSpace m => (0 : ℝ)) ∧
      (∀ k ≤ 2, ∀ z : EucSpace m,
        ‖iteratedFDeriv ℝ k (fun _ : EucSpace m => (0 : ℝ)) z‖ ≤ 0) ∧
      (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨measurable_const.aemeasurable, ?_, fun _ => by simp, fun _ _ => by simp, indepFun_const_left _ _,
    contDiff_const, fun k _ z => ?_, by norm_num⟩
  · rw [Measure.map_const]
    simp
    infer_instance
  · rcases Nat.eq_zero_or_pos k with rfl | hk
    · simp
    · rw [iteratedFDeriv_const_of_ne (by omega)]
      simp

end Homogenized
end Transformer
