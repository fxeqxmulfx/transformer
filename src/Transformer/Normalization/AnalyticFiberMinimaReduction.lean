/-
# Polynomial formulas for all local analytic fiber minima

Real analytic division retains a finite constraint family and an arbitrary
analytic objective on every nearby root. On one fixed parameter-variable
box, the universal minimum comparison is therefore an exact polynomial
formula in the distinguished variable. No base curve is supplied.
-/

import Transformer.Normalization.AnalyticFiberPolynomialValues
import Transformer.Normalization.GradientEnergyImage

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.Normalization

open AnalyticPreparation

/-- Every constrained minimum on a regular analytic scalar fiber has an
exact polynomial description on one fixed box. The original equation,
all finite sign conditions, and comparison with every admissible root
are retained in both directions. The remaining base coordinates are
arbitrary. This is the reduction before eliminating quantified variables
in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_fiber_minima_polynomial {n d m : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (horder : ExactOrderInLastVariable F d)
    (G : Fin m → Ambient n → ℝ) (hG : ∀ j, AnalyticAt ℝ (G j) 0)
    (requirement : Fin m → AnalyticSignRequirement)
    (V : Ambient n → ℝ) (hV : AnalyticAt ℝ V 0) :
    ∃ (a : Fin d → Base n → ℝ) (b : Fin m → Fin d → Base n → ℝ)
      (v : Fin d → Base n → ℝ) (r : ℝ),
      0 < r ∧ (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ j i, AnalyticAt ℝ (b j i) 0) ∧ (∀ i, AnalyticAt ℝ (v i) 0) ∧
      ∀ z : Base n, ‖z‖ < r → ∀ y : ℝ, |y| < r →
        ((F (z, y) = 0 ∧ (∀ j, (requirement j).Holds (G j (z, y))) ∧
          ∀ t : ℝ, |t| < r → F (z, t) = 0 →
            (∀ j, (requirement j).Holds (G j (z, t))) → V (z, y) ≤ V (z, t)) ↔
        (preparedPolynomial d a (z, y) = 0 ∧
          (∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * y ^ (i : ℕ))) ∧
          ∀ t : ℝ, |t| < r → preparedPolynomial d a (z, t) = 0 →
            (∀ j, (requirement j).Holds (∑ i : Fin d, b j i z * t ^ (i : ℕ))) →
              (∑ i : Fin d, v i z * y ^ (i : ℕ)) ≤ ∑ i : Fin d, v i z * t ^ (i : ℕ))) := by
  obtain ⟨a, b, v, r, hr, ha, ha0, hb, hv, hbox⟩ :=
    analytic_fiber_polynomial_values F hF horder G hG V hV
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro z hz y hy
  have hxy := hbox z hz y hy
  constructor
  · rintro ⟨hFy, hGy, hminimum⟩
    have hPy := hxy.1.mp hFy
    refine ⟨hPy, fun j => ?_, ?_⟩
    · rw [← hxy.2.1 hPy j]
      exact hGy j
    · intro t ht hPt hGt
      have hxt := hbox z hz t ht
      have hFt := hxt.1.mpr hPt
      have hGt' : ∀ j, (requirement j).Holds (G j (z, t)) := by
        intro j
        rw [hxt.2.1 hPt j]
        exact hGt j
      have h := hminimum t ht hFt hGt'
      rwa [hxy.2.2 hPy, hxt.2.2 hPt] at h
  · rintro ⟨hPy, hGy, hminimum⟩
    refine ⟨hxy.1.mpr hPy, fun j => ?_, ?_⟩
    · rw [hxy.2.1 hPy j]
      exact hGy j
    · intro t ht hFt hGt
      have hxt := hbox z hz t ht
      have hPt := hxt.1.mp hFt
      have hGt' : ∀ j,
          (requirement j).Holds (∑ i : Fin d, b j i z * t ^ (i : ℕ)) := by
        intro j
        rw [← hxt.2.1 hPt j]
        exact hGt j
      rw [hxy.2.2 hPy, hxt.2.2 hPt]
      exact hminimum t ht hPt hGt'

/-- The degenerate energy `x⁴` on the moving level `z₀`, its actual
squared gradient norm, and an exponential sign constraint jointly
satisfy the polynomial minimum reduction hypotheses. Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => (x 0) ^ 4
    let phi : Ambient 1 → EucSpace 1 := fun x => x.2 • PiLp.single 2 (0 : Fin 1) 1
    let F : Ambient 1 → ℝ := fun x => E (phi x) - x.1 0
    AnalyticAt ℝ F 0 ∧ ExactOrderInLastVariable F 4 ∧
      AnalyticAt ℝ (fun x => squaredGradientNorm E (phi x)) 0 ∧
      (∀ j : Fin 1, AnalyticAt ℝ (fun x : Ambient 1 => Real.exp x.2 + (j : ℝ)) 0) := by
  intro E phi F
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt 0).fun_pow 4
  have hphi : AnalyticAt ℝ phi 0 := analyticAt_snd.smul analyticAt_const
  have hphi0 : phi 0 = 0 := by simp [phi]
  have hz : AnalyticAt ℝ (fun x : Ambient 1 => x.1 0) 0 :=
    ((ContinuousLinearMap.proj (0 : Fin 1)).comp
      (ContinuousLinearMap.fst ℝ (Base 1) ℝ)).analyticAt 0
  have hF : AnalyticAt ℝ F 0 :=
    ((by simpa only [hphi0] using hE : AnalyticAt ℝ E (phi 0)).comp hphi).sub hz
  have hslice : lastSlice F = fun t : ℝ => t ^ 4 := by
    funext t
    simp [lastSlice, F, E, phi]
  have horder : ExactOrderInLastVariable F 4 := by
    rw [ExactOrderInLastVariable, hslice]
    apply (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero
      (analyticAt_id.fun_pow 4)).mp
    simpa [Pi.pow_def] using analyticOrderAt_pow
      (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 4
  have hnorm : AnalyticAt ℝ (squaredGradientNorm E) (phi 0) := by
    simpa only [hphi0] using squared_gradient_norm_analyticAt E 0 hE
  exact ⟨hF, horder, hnorm.comp hphi, fun _ =>
    (analyticAt_rexp.comp analyticAt_snd).add analyticAt_const⟩

end Transformer.Normalization
