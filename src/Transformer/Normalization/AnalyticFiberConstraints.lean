/-
# Polynomial representatives of analytic conditions on a prepared fiber

Weierstrass division gives finite-degree representatives of every analytic
function on the zero set of a prepared polynomial. The representatives retain
actual function values, so equality, strict signs, and weak signs are all
preserved together on a common neighborhood.
-/

import Transformer.AnalyticPreparation
import Transformer.Normalization.AnalyticSignStability
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Order.Filter.Finite

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.Normalization

open AnalyticPreparation

/-- A finite family of analytic functions has polynomial representatives
of degree below `d` on every nearby zero of the prepared divisor. All values
are preserved on one common neighborhood; no branch or witness is assumed.
Auxiliary for analytic projection in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_prepared_fiber_remainders {n d m : ℕ}
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0) :
    ∃ b : Fin m → Fin d → Base n → ℝ,
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧
      ∀ᶠ x in 𝓝 (0 : Ambient n), preparedPolynomial d a x = 0 →
        ∀ j, G j x = ∑ i : Fin d, b j i x.1 * x.2 ^ (i : ℕ) := by
  choose q b hq hb hfactor using
    fun j => exists_analyticWeierstrassDivision (G j) (hG j) a ha ha0
  refine ⟨b, hb, ?_⟩
  have hcommon : ∀ᶠ x in 𝓝 (0 : Ambient n), ∀ j,
      G j x = q j x * preparedPolynomial d a x +
        ∑ i : Fin d, b j i x.1 * x.2 ^ (i : ℕ) :=
    eventually_all.mpr hfactor
  filter_upwards [hcommon] with x hx hzero j
  simpa only [hzero, mul_zero, zero_add] using hx j

/-- Any analytic equation regular in the last variable, together with a
finite family of analytic sign conditions, is equivalent near the origin
to a distinguished polynomial equation and polynomial sign conditions in
that variable. The exact zero set and all signs are retained; the parameter
coefficients are analytic and the equation has the original exact order.
Auxiliary for general curve selection in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_fiber_sign_conditions_polynomial {n d m : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (horder : ExactOrderInLastVariable F d)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0)
    (requirement : Fin m → AnalyticSignRequirement) :
    ∃ (a : Fin d → Base n → ℝ) (b : Fin m → Fin d → Base n → ℝ),
      (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧
      ∀ᶠ x in 𝓝 (0 : Ambient n),
        (F x = 0 ∧ ∀ j, (requirement j).Holds (G j x)) ↔
          (preparedPolynomial d a x = 0 ∧ ∀ j,
            (requirement j).Holds (∑ i : Fin d, b j i x.1 * x.2 ^ (i : ℕ))) := by
  obtain ⟨a, u, ha, ha0, hu, hu0, hfactor⟩ := exists_isWeierstrassPreparation hF horder
  obtain ⟨b, hb, hvalues⟩ := analytic_prepared_fiber_remainders a ha ha0 G hG
  have hunit : ∀ᶠ x in 𝓝 (0 : Ambient n), u x ≠ 0 := hu.continuousAt.eventually_ne hu0
  refine ⟨a, b, ha, ha0, hb, ?_⟩
  filter_upwards [hfactor, hvalues, hunit] with x hx hval hne
  have hzero : F x = 0 ↔ preparedPolynomial d a x = 0 := by
    rw [hx, mul_eq_zero]
    simp only [hne, false_or]
  constructor
  · rintro ⟨hFx, hsign⟩
    have hPx := hzero.mp hFx
    exact ⟨hPx, fun j => by rw [← hval hPx j]; exact hsign j⟩
  · rintro ⟨hPx, hsign⟩
    exact ⟨hzero.mpr hPx, fun j => by rw [hval hPx j]; exact hsign j⟩

/-- A singular quadratic equation and two nonpolynomial analytic
constraints jointly satisfy both reduction theorems. Strict positivity of
the exponential and a weak sign of the sine are retained exactly on the
original fiber. Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let F : Ambient 1 → ℝ := fun x => x.2 ^ 2 - x.1 0
    let G : Fin 2 → Ambient 1 → ℝ := fun j x =>
      if j = 0 then Real.exp x.2 else Real.sin x.2
    let requirement : Fin 2 → AnalyticSignRequirement := fun j =>
      if j = 0 then .positive else .nonnegative
    ∃ (a : Fin 2 → Base 1 → ℝ) (b : Fin 2 → Fin 2 → Base 1 → ℝ),
      (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧
      ∀ᶠ x in 𝓝 (0 : Ambient 1),
        (F x = 0 ∧ ∀ j, (requirement j).Holds (G j x)) ↔
          (preparedPolynomial 2 a x = 0 ∧ ∀ j,
            (requirement j).Holds (∑ i : Fin 2, b j i x.1 * x.2 ^ (i : ℕ))) := by
  intro F G requirement
  have hz : AnalyticAt ℝ (fun x : Ambient 1 => x.1 0) 0 := by
    let L : Ambient 1 →L[ℝ] ℝ := (ContinuousLinearMap.proj (0 : Fin 1)).comp
      (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
    exact L.analyticAt 0
  have hF : AnalyticAt ℝ F 0 := (analyticAt_snd.fun_pow 2).sub hz
  have hslice : lastSlice F = fun t : ℝ => t ^ 2 := by funext t; simp [lastSlice, F]
  have horder : ExactOrderInLastVariable F 2 := by
    rw [ExactOrderInLastVariable, hslice]
    apply (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 2)).mp
    simpa [Pi.pow_def] using analyticOrderAt_pow
      (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 2
  have hG : ∀ j, AnalyticAt ℝ (G j) 0 := by
    intro j
    dsimp only [G]
    split_ifs
    · exact analyticAt_rexp.comp analyticAt_snd
    · exact Real.analyticAt_sin.comp analyticAt_snd
  exact analytic_fiber_sign_conditions_polynomial F hF horder G hG requirement

end Transformer.Normalization
