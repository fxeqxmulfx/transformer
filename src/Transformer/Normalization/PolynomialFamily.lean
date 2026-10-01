/-
# Real analytic monic polynomial families

A scalar parameter and a polynomial variable provide the local families
used in ramified curve lifting. Preparation at a central root replaces
the degree by the root's actual multiplicity.
-/

import Transformer.Normalization.PolynomialRootOrder
import Transformer.AnalyticPreparation

open Filter Polynomial
open scoped BigOperators

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- A real monic polynomial with analytic coefficient germs in a scalar
parameter. Auxiliary construction for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def polynomialFamily (d : ℕ) (a : Fin d → ℝ → ℝ) (x : ℝ × ℝ) : ℝ :=
  x.2 ^ d + ∑ i : Fin d, a i x.1 * x.2 ^ i.val

/-- The actual monic polynomial at a fixed parameter, using the already
verified distinguished-polynomial construction. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def parameterPolynomial (d : ℕ) (a : Fin d → ℝ → ℝ) (t : ℝ) : Polynomial ℝ :=
  distinguishedPolynomial d (fun i (z : Base 1) => a i (z 0)) (fun _ => t)

/-- Evaluation of the parameter polynomial is the polynomial family's
value. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_eval {d : ℕ} (a : Fin d → ℝ → ℝ) (t y : ℝ) :
    (parameterPolynomial d a t).eval y = polynomialFamily d a (t, y) := by
  simp [parameterPolynomial, distinguishedPolynomial, polynomialFamily, eval_finsetSum]

/-- The parameter polynomial is monic for every parameter. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem parameterPolynomial_monic {d : ℕ} (a : Fin d → ℝ → ℝ) (t : ℝ) :
    (parameterPolynomial d a t).Monic :=
  distinguishedPolynomial_monic d _ _

/-- Analytic coefficients give a jointly analytic shifted polynomial
family. The distinguished variable may be based at any central root.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomialFamily_analyticAt {d : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (r : ℝ) :
    AnalyticAt ℝ (fun x : ℝ × ℝ => polynomialFamily d a (x.1, r + x.2)) 0 := by
  have hs : AnalyticAt ℝ (fun x : ℝ × ℝ => r + x.2) 0 :=
    analyticAt_const.fun_add analyticAt_snd
  apply (hs.fun_pow d).fun_add
  apply Finset.analyticAt_fun_sum
  intro i hi
  have hai : AnalyticAt ℝ (fun x : ℝ × ℝ => a i x.1) 0 :=
    (ha i).comp (f := fun x : ℝ × ℝ => x.1) (x := 0) analyticAt_fst
  exact hai.fun_mul (hs.fun_pow i.val)

/-- Preparation at any central real root gives an analytic polynomial
of degree equal to that root's multiplicity, with all new coefficients
zero at the origin. The analytic nonvanishing unit is retained in the
exact local identity. Auxiliary for general ramified curve lifting in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_polynomial_preparation_at_root {d : ℕ}
    (a : Fin d → ℝ → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (r : ℝ) (hroot : (parameterPolynomial d a 0).IsRoot r) :
    let m := (parameterPolynomial d a 0).rootMultiplicity r
    ∃ (b : Fin m → ℝ → ℝ) (u : ℝ × ℝ → ℝ), 0 < m ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ x in nhds (0 : ℝ × ℝ),
        polynomialFamily d a (x.1, r + x.2) = u x * polynomialFamily m b x := by
  let P := parameterPolynomial d a 0
  let m := P.rootMultiplicity r
  let F : Ambient 1 → ℝ := fun x => polynomialFamily d a (x.1 0, r + x.2)
  let project : Ambient 1 →L[ℝ] ℝ × ℝ :=
    ((ContinuousLinearMap.proj (0 : Fin 1)).comp
      (ContinuousLinearMap.fst ℝ (Base 1) ℝ)).prod (ContinuousLinearMap.snd ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 :=
    (by simpa using polynomialFamily_analyticAt a ha r : AnalyticAt ℝ
      (fun x : ℝ × ℝ => polynomialFamily d a (x.1, r + x.2)) (project 0)).comp
        (f := project) (x := 0) (project.analyticAt 0)
  have hslice : lastSlice F = fun t : ℝ => P.eval (r + t) := by
    funext t
    simp only [lastSlice, F, Pi.zero_apply, P, parameterPolynomial_eval]
  have horder : ExactOrderInLastVariable F m := by
    rw [ExactOrderInLastVariable, hslice]
    exact polynomial_exact_order_at_root (n := 0) P (parameterPolynomial_monic a 0).ne_zero r
  obtain ⟨A, U, hprep⟩ := exists_isWeierstrassPreparation hF horder
  let diagonal : ℝ →L[ℝ] Base 1 :=
    ContinuousLinearMap.pi (fun _ => ContinuousLinearMap.id ℝ ℝ)
  let liftCoordinates : ℝ × ℝ →L[ℝ] Ambient 1 :=
    (diagonal.comp (ContinuousLinearMap.fst ℝ ℝ ℝ)).prod (ContinuousLinearMap.snd ℝ ℝ ℝ)
  let b : Fin m → ℝ → ℝ := fun i t => A i (diagonal t)
  let u : ℝ × ℝ → ℝ := fun x => U (liftCoordinates x)
  have hb (i : Fin m) : AnalyticAt ℝ (b i) 0 :=
    (by simpa using hprep.1 i : AnalyticAt ℝ (A i) (diagonal 0)).comp
      (f := diagonal) (x := 0) (diagonal.analyticAt 0)
  have hb0 (i : Fin m) : b i 0 = 0 := by simpa [b] using hprep.2.1 i
  have hu : AnalyticAt ℝ u 0 :=
    (by simpa using hprep.2.2.1 : AnalyticAt ℝ U (liftCoordinates 0)).comp
      (f := liftCoordinates) (x := 0) (liftCoordinates.analyticAt 0)
  have hu0 : u 0 ≠ 0 := by simpa [u] using hprep.2.2.2.1
  refine ⟨b, u, (rootMultiplicity_pos (parameterPolynomial_monic a 0).ne_zero).mpr hroot,
    hb, hb0, hu, hu0, ?_⟩
  have hnear : Tendsto liftCoordinates (nhds 0) (nhds 0) := by
    simpa only [map_zero] using liftCoordinates.continuous.tendsto (0 : ℝ × ℝ)
  filter_upwards [hnear.eventually hprep.2.2.2.2] with x hx
  exact hx

/-- The moving cubic `y³ - 3y + 2 - t` is analytic and has a multiple
central root at one. The preparation therefore exercises a singular
root rather than only a root with nonzero derivative. Auxiliary example
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then 2 - t else
    if i = 1 then -3 else 0
    let m := (parameterPolynomial 3 a 0).rootMultiplicity 1
    ∃ (b : Fin m → ℝ → ℝ) (u : ℝ × ℝ → ℝ), 0 < m ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, b i 0 = 0) ∧
      AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ x in nhds (0 : ℝ × ℝ),
        polynomialFamily 3 a (x.1, 1 + x.2) = u x * polynomialFamily m b x := by
  apply analytic_polynomial_preparation_at_root
  · intro i
    fin_cases i
    · exact analyticAt_const.fun_sub analyticAt_id
    · exact analyticAt_const
    · exact analyticAt_const
  · norm_num [IsRoot, parameterPolynomial_eval, polynomialFamily, Fin.sum_univ_succ]

end Transformer.Normalization
