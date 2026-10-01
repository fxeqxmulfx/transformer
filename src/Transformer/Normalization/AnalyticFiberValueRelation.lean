/-
# Monic relations for the values of analytic functions on finite fibers

The finite power basis of the prepared analytic-germ quotient produces an
actual monic polynomial relation for every analytic fiber value. Coefficients
remain convergent real analytic functions of the base parameters.
-/

import Transformer.RealAnalyticGerms.IntegralRepresentatives
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.Normalization

open AnalyticPreparation RealAnalyticGerms

/-- The values of any real analytic function on the zeros of a prepared
equation satisfy a monic polynomial relation with analytic base coefficients.
The relation has the same positive degree as the prepared equation and
holds for every nearby zero, including singular and repeated roots.
Auxiliary for finite analytic projection in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_prepared_fiber_value_relation {n d : ℕ} (hd : 0 < d)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0)
    (G : Ambient n → ℝ) (hG : AnalyticAt ℝ G 0) :
    ∃ (k : ℕ) (b : Fin k → Base n → ℝ), 0 < k ∧ k = d ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧
      ∀ᶠ x in 𝓝 (0 : Ambient n), preparedPolynomial d a x = 0 →
        preparedPolynomial k b (x.1, G x) = 0 := by
  let GS : Base (n + 1) → ℝ := fun x => G (wptAmbientEquiv n x)
  have hGS : AnalyticAt ℝ GS 0 :=
    (analyticAt_comp_wptAmbientEquiv_iff n G).mpr hG
  let g := AnalyticGerm.ofFunction GS hGS
  obtain ⟨p, hp, hdegree, hmem⟩ := prepared_analytic_germ_relation_degree a ha ha0 g
  obtain ⟨b, hb, hrepresentative⟩ := monic_analytic_germ_relation_representative p hp GS hGS
  obtain ⟨q, hq⟩ := Ideal.mem_span_singleton'.mp hmem
  obtain ⟨Q, hQ, hQrep⟩ := AnalyticGerm.exists_rep q
  have hproduct : (Polynomial.aeval g p : FunctionGerm (n + 1)) =
      ((fun x : Base (n + 1) => Q x * preparedPolynomialFunction a x) :
        FunctionGerm (n + 1)) := by
    rw [← hq]
    change (q : FunctionGerm (n + 1)) *
      (preparedPolynomialFunction a : FunctionGerm (n + 1)) = _
    rw [← hQrep, ← Filter.Germ.coe_mul]
    rfl
  have heq : (fun x : Base (n + 1) => GS x ^ p.natDegree +
      ∑ i : Fin p.natDegree, b i (baseProjectionCLM n x) * GS x ^ (i : ℕ)) =ᶠ[𝓝 0]
        (fun x => Q x * preparedPolynomialFunction a x) :=
    Filter.Germ.coe_eq.mp (hrepresentative.symm.trans hproduct)
  have ht : Tendsto (wptAmbientEquiv n).symm (𝓝 0) (𝓝 0) := by
    simpa only [map_zero] using (wptAmbientEquiv n).symm.continuous.continuousAt.tendsto (x := 0)
  have hfactor : ∀ᶠ x in 𝓝 (0 : Ambient n),
      preparedPolynomial p.natDegree b (x.1, G x) =
        Q ((wptAmbientEquiv n).symm x) * preparedPolynomial d a x := by
    filter_upwards [heq.comp_tendsto ht] with x hx
    simpa [Function.comp_def, GS, preparedPolynomial, preparedPolynomialFunction,
      baseProjectionCLM] using hx
  refine ⟨p.natDegree, b, by rwa [hdegree], hdegree, hb, ?_⟩
  filter_upwards [hfactor] with x hx hzero
  simpa only [hzero, mul_zero] using hx

/-- An exponential fiber value over the singular equation `y² - z₀ = 0`
has an analytic monic relation. This exercises nonpolynomial analyticity
and all hypotheses of the value-elimination theorem together. Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then -z 0 else 0
    ∃ (k : ℕ) (b : Fin k → Base 1 → ℝ), 0 < k ∧ k = 2 ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧
      ∀ᶠ x in 𝓝 (0 : Ambient 1), preparedPolynomial 2 a x = 0 →
        preparedPolynomial k b (x.1, Real.exp x.2) = 0 := by
  intro a
  have ha : ∀ i, AnalyticAt ℝ (a i) 0 := by
    intro i
    dsimp only [a]
    split_ifs
    · exact ((ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0).neg
    · exact analyticAt_const
  have ha0 : ∀ i, a i 0 = 0 := by intro i; simp only [a]; split_ifs <;> simp
  exact analytic_prepared_fiber_value_relation (by norm_num) a ha ha0
    (fun x : Ambient 1 => Real.exp x.2) (analyticAt_rexp.comp analyticAt_snd)

end Transformer.Normalization
