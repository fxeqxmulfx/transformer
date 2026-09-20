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
  Gaussian-integrable.  Boundedness of `φ` and of its derivatives of order
  `≤ 2` is added, which is what the source's own application satisfies — a
  product of softmax weights lies in `(0,1)` and has bounded derivatives of
  every order.  This is a missing hypothesis, not a weakening: without it the
  displayed identity is an equation between undefined quantities.

* `t` ranges over `[0,1]`, where `X_t` is defined, and the derivative at the
  two endpoints is the one-sided one, i.e. `HasDerivWithinAt` on `Set.Icc 0 1`.
  Outside `[0,1]` the source's `√(1-t)` is not real.

* The covariance is pinned by its entries, `Σ^X_{ij} = E[X_i X_j]`, alongside
  `IsGaussian` for the shape of the law and `E X_i = 0` for centering; Mathlib's
  `IsGaussian` is "every continuous linear form pushes to a real Gaussian" and
  carries no covariance matrix of its own.

* `Tr(Aᵀ H) = Σ_{ij} A_{ij} H_{ij}`, and the Hessian entries `H_{ij}` are the
  second derivative `D²φ(z)[e_i, e_j]`, as in `sphHess` of `Generator.lean`.

Not proved here: this is an external theorem the source recalls verbatim, and
it is on the books because `eq:steins_lemma` is derived from it.

Source: arXiv:2604.01978v1, `lem:gaussian_interp`, `eq:interpolation_gaussian`;
`[vershynin2018high, Lemma 7.2.7]`.
-/

import Transformer.Homogenized.Basic
import Mathlib.Probability.Distributions.Gaussian.Basic

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

See the module docstring for the boundedness hypothesis the source omits and
for the reading of `t` at the endpoints.

Not proved here.

Source: arXiv:2604.01978v1, `lem:gaussian_interp`. -/
theorem gaussian_interp {m : ℕ} {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (X Y : Ω → EucSpace m)
    (SX SY : Matrix (Fin m) (Fin m) ℝ)
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
  sorry

/-- The hypotheses of `gaussian_interp` are satisfiable: the one-point space
carries the degenerate Gaussian `δ_0` of `ℝ^m`, twice, independent of itself,
with covariance `0`; and the zero test function is `C²` with all derivatives
bounded by `0`. -/
example (m : ℕ) :
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
  refine ⟨?_, fun _ => by simp, fun _ _ => by simp, indepFun_const_left _ _,
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
