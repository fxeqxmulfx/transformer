/-
# Real analytic germs: PolynomialGerms

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/GermDivision.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.Coordinates
import Transformer.AnalyticPreparation.AnalyticDivisionUniqueness

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- The prepared polynomial, written in the standard `Fin (n + 1) → ℝ`
coordinate model used by `AnalyticGerm`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedPolynomialFunction {n d : ℕ}
    (a : Fin d → Base n → ℝ) : Base (n + 1) → ℝ :=
  fun x ↦ preparedPolynomial d a (wptAmbientEquiv n x)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_preparedPolynomialFunction {n d : ℕ}
    (a : Fin d → Base n → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) :
    AnalyticAt ℝ (preparedPolynomialFunction a) 0 := by
  have hP : AnalyticAt ℝ (preparedPolynomial d a) 0 := by
    unfold preparedPolynomial
    apply AnalyticAt.add
    · exact analyticAt_snd.pow d
    · refine Finset.analyticAt_fun_sum Finset.univ fun i _ ↦ ?_
      have hai : AnalyticAt ℝ (fun x : Ambient n ↦ a i x.1) 0 :=
        AnalyticAt.comp (g := a i) (f := fun x : Ambient n ↦ x.1)
          (ha i) analyticAt_fst
      exact hai.mul (analyticAt_snd.pow (i : ℕ))
  change AnalyticAt ℝ (preparedPolynomial d a ∘ (wptAmbientEquiv n)) 0
  simpa using
    hP.compContinuousLinearMap (u := (wptAmbientEquiv n :
      Base (n + 1) →L[ℝ] Ambient n)) (x := 0)

/-- The germ of a fixed prepared polynomial.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def preparedPolynomialGerm {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0) :
    AnalyticGerm (n + 1) :=
  AnalyticGerm.ofFunction (preparedPolynomialFunction a)
    (analyticAt_preparedPolynomialFunction a ha)

/-- The degree-`< d` polynomial germ with a prescribed coefficient vector.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def remainderPolynomialGerm {n d : ℕ}
    (r : Fin d → AnalyticGerm n) : AnalyticGerm (n + 1) :=
  ∑ i : Fin d,
    lowerDimensionalInclusion n (r i) * lastCoordinateGerm n ^ (i : ℕ)

/-- A quotient and coefficient vector satisfy Weierstrass division at the
level of germs.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def IsPreparedGermDivision {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (h q : AnalyticGerm (n + 1)) (r : Fin d → AnalyticGerm n) : Prop :=
  h = q * preparedPolynomialGerm a ha + remainderPolynomialGerm r

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
private theorem functionGerm_coe_fin_sum {m d : ℕ}
    (f : Fin d → Base m → ℝ) :
    (((∑ i : Fin d, f i) : Base m → ℝ) : FunctionGerm m) =
      ∑ i : Fin d, (f i : FunctionGerm m) := by
  exact map_sum (Filter.Germ.coeRingHom
    (𝓝 (0 : Base m))) f Finset.univ

/-- Coercion of a polynomial assembled from analytic coefficient
representatives agrees with its pointwise polynomial function.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coe_remainderPolynomialGerm_ofFunction {n d : ℕ}
    (r : Fin d → Base n → ℝ) (hr : ∀ i, AnalyticAt ℝ (r i) 0) :
    (remainderPolynomialGerm
        (fun i ↦ AnalyticGerm.ofFunction (r i) (hr i)) :
      FunctionGerm (n + 1)) =
      ((fun x ↦ ∑ i : Fin d,
        r i (baseProjectionCLM n x) * lastCoordinateCLM n x ^ (i : ℕ)) :
        FunctionGerm (n + 1)) := by
  simp only [remainderPolynomialGerm, Subring.coe_mul, Subring.coe_pow,
    AddSubmonoidClass.coe_finsetSum, analyticGermPullbackHom_coe,
    lowerDimensionalInclusion]
  simp_rw [show ∀ i,
    ((AnalyticGerm.ofFunction (r i) (hr i) : AnalyticGerm n) :
      FunctionGerm n) = r i from fun _ ↦ rfl]
  simp only [functionGermPullbackHom_coe]
  rw [show (lastCoordinateGerm n : FunctionGerm (n + 1)) =
    lastCoordinateCLM n from rfl]
  let terms : Fin d → Base (n + 1) → ℝ :=
    fun i ↦ (r i ∘ baseProjectionCLM n) * (lastCoordinateCLM n) ^ (i : ℕ)
  calc
    ∑ i : Fin d,
        ((r i ∘ baseProjectionCLM n) : FunctionGerm (n + 1)) *
          (lastCoordinateCLM n : FunctionGerm (n + 1)) ^ (i : ℕ) =
      ∑ i : Fin d, (terms i : FunctionGerm (n + 1)) := by
        apply Finset.sum_congr rfl
        intro i _
        simp only [terms, Filter.Germ.coe_mul, Filter.Germ.coe_pow]
        congr 1
    _ = ((∑ i : Fin d, terms i) :
        Base (n + 1) → ℝ) :=
      (functionGerm_coe_fin_sum terms).symm
    _ = ((fun x ↦ ∑ i : Fin d,
        r i (baseProjectionCLM n x) * lastCoordinateCLM n x ^ (i : ℕ)) :
        FunctionGerm (n + 1)) := by
      apply congrArg Filter.Germ.ofFun
      funext x
      simp [terms]

/-- Coercion of the full quotient-plus-remainder expression assembled from
analytic representatives.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coe_preparedDivisionExpression_ofFunction {n d : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (q : Base (n + 1) → ℝ) (hq : AnalyticAt ℝ q 0)
    (r : Fin d → Base n → ℝ) (hr : ∀ i, AnalyticAt ℝ (r i) 0) :
    ((AnalyticGerm.ofFunction q hq * preparedPolynomialGerm a ha +
        remainderPolynomialGerm
          (fun i ↦ AnalyticGerm.ofFunction (r i) (hr i)) :
      AnalyticGerm (n + 1)) : FunctionGerm (n + 1)) =
      ((fun x ↦ q x * preparedPolynomialFunction a x +
        ∑ i : Fin d,
          r i (baseProjectionCLM n x) * lastCoordinateCLM n x ^ (i : ℕ)) :
        FunctionGerm (n + 1)) := by
  simp only [Subring.coe_add, Subring.coe_mul]
  rw [show ((AnalyticGerm.ofFunction q hq : AnalyticGerm (n + 1)) :
    FunctionGerm (n + 1)) = q from rfl]
  rw [show (preparedPolynomialGerm a ha : FunctionGerm (n + 1)) =
    preparedPolynomialFunction a from rfl]
  rw [coe_remainderPolynomialGerm_ofFunction r hr]
  rw [← Filter.Germ.coe_mul, ← Filter.Germ.coe_add]
  apply congrArg Filter.Germ.ofFun
  rfl


end Transformer.RealAnalyticGerms
