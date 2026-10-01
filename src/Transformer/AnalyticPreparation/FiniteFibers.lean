/-
# Finite fibers of prepared zero sets

After preparation, projection of an analytic zero set along the distinguished
variable has finite fibers on a neighborhood. This is the finite-root input
needed to lift parameter curves; existence of ramified analytic root branches
is a further step.
-/

import Transformer.AnalyticPreparation.PreparationReconstruction
import Mathlib.Algebra.Polynomial.Monic
import Mathlib.Algebra.Polynomial.Roots

open Filter
open scoped BigOperators Topology

noncomputable section

namespace Transformer.AnalyticPreparation

/-- The actual polynomial in the distinguished variable at fixed
parameters. Auxiliary construction for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def distinguishedPolynomial {n : ℕ} (d : ℕ) (a : Fin d → Base n → ℝ)
    (z : Base n) : Polynomial ℝ :=
  Polynomial.X ^ d + ∑ i : Fin d, Polynomial.C (a i z) * Polynomial.X ^ (i : ℕ)

/-- Polynomial evaluation agrees with the prepared analytic expression.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem distinguishedPolynomial_eval {n : ℕ} (d : ℕ)
    (a : Fin d → Base n → ℝ) (z : Base n) (w : ℝ) :
    (distinguishedPolynomial d a z).eval w = preparedPolynomial d a (z, w) := by
  simp [distinguishedPolynomial, preparedPolynomial, Polynomial.eval_finsetSum]

/-- The distinguished polynomial is monic of the specified leading
degree and hence is nonzero for every parameter value. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem distinguishedPolynomial_monic {n : ℕ} (d : ℕ)
    (a : Fin d → Base n → ℝ) (z : Base n) :
    (distinguishedPolynomial d a z).Monic :=
  Polynomial.monic_X_pow_add (Polynomial.degree_sum_fin_lt (fun i => a i z))

/-- An actual analytic preparation has finite distinguished-variable
zero fibers on a neighborhood of the origin. The assertion uses the
nonvanishing analytic unit, so it applies to the original zero set.
Auxiliary for the singular curve-selection step in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem prepared_zero_fibers_finite {n d : ℕ} (f : Ambient n → ℝ)
    (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ)
    (hprep : IsWeierstrassPreparation f d a u) :
    ∃ r > 0, ∀ z : Base n,
      Set.Finite {w : ℝ | (z, w) ∈ Metric.ball (0 : Ambient n) r ∧ f (z, w) = 0} := by
  have hunit : ∀ᶠ x in nhds (0 : Ambient n), u x ≠ 0 :=
    hprep.2.2.1.continuousAt.eventually_ne hprep.2.2.2.1
  have hzero : ∀ᶠ x in nhds (0 : Ambient n),
      f x = 0 ↔ (distinguishedPolynomial d a x.1).eval x.2 = 0 := by
    filter_upwards [hprep.2.2.2.2, hunit] with x hx hu
    rw [hx, mul_eq_zero, or_iff_right hu, distinguishedPolynomial_eval]
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp hzero
  refine ⟨r, hr, fun z => ?_⟩
  have hfinite := Polynomial.finite_setOfPred_isRoot
    (distinguishedPolynomial_monic d a z).ne_zero
  apply hfinite.subset
  intro w hw
  exact (hball (Metric.mem_ball.mp hw.1)).mp hw.2

/-- The prepared quadratic `w² + z₀²` is an analytic energy with exact
distinguished order two. Its preparation exercises finite fibers of the
original analytic zero set. Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ r > 0, ∀ z : Base 1,
    Set.Finite {w : ℝ | (z, w) ∈ Metric.ball (0 : Ambient 1) r ∧
      w ^ 2 + (z 0) ^ 2 = 0} := by
  let f : Ambient 1 → ℝ := fun x => x.2 ^ 2 + (x.1 0) ^ 2
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hf : AnalyticAt ℝ f 0 :=
    (analyticAt_snd.fun_pow 2).add ((L.analyticAt 0).fun_pow 2)
  have hslice : lastSlice f = fun t : ℝ => t ^ 2 := by funext t; simp [lastSlice, f]
  have hord : analyticOrderAt (fun t : ℝ => t ^ 2) 0 = 2 := by
    simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := 0)) 2
  have horder : ExactOrderInLastVariable f 2 := by
    rw [ExactOrderInLastVariable, hslice]
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 2)).mp hord
  obtain ⟨a, u, hprep⟩ := exists_isWeierstrassPreparation hf horder
  exact prepared_zero_fibers_finite f a u hprep

end Transformer.AnalyticPreparation
