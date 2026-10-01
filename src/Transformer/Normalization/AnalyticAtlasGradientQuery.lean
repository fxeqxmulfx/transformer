/-
# Gradient-minimizer sets on a finite analytic energy atlas

The actual universal gradient-norm comparison on the entire atlas source
is retained, while all scalar competitor variables are eliminated.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticAtlasMinimumQuery
import Transformer.Normalization.GradientMinimumCluster

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- The genuine energy-level gradient-minimizer set on a finite analytic
energy atlas has exact arithmetic tests on every chart. The norm comparison
is with the entire source, including points outside the candidate's chart.
The analytic objective is constructed as the actual squared gradient norm;
no gradient-minimizing branch or curve is supplied. Only universal base
quantifiers remain. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_atlas_gradient_minima_arithmetic {k N : ℕ}
    (n d m : Fin k → ℕ) (E : EucSpace N → ℝ)
    (phi : ∀ j, Ambient (n j) → EucSpace N)
    (hphi : ∀ j, AnalyticAt ℝ (phi j) 0)
    (hE : ∀ j, AnalyticAt ℝ E (phi j 0))
    (level : ∀ j, Base (n j) → ℝ) (hlevel : ∀ j, AnalyticAt ℝ (level j) 0)
    (horder : ∀ j, ExactOrderInLastVariable
      (fun x => E (phi j x) - level j x.1) (d j))
    (G : ∀ j, Fin (m j) → Ambient (n j) → ℝ)
    (hG : ∀ j i, AnalyticAt ℝ (G j i) 0)
    (requirement : ∀ j, Fin (m j) → AnalyticSignRequirement) :
    ∃ (a : ∀ j, Fin (d j) → Base (n j) → ℝ)
      (b : ∀ j, Fin (m j) → Fin (d j) → Base (n j) → ℝ)
      (v : ∀ j, Fin (d j) → Base (n j) → ℝ) (r : Fin k → ℝ),
      (∀ j, 0 < r j) ∧ (∀ j i, AnalyticAt ℝ (a j i) 0) ∧
      (∀ j i, a j i 0 = 0) ∧ (∀ j i t, AnalyticAt ℝ (b j i t) 0) ∧
      (∀ j i, AnalyticAt ℝ (v j i) 0) ∧
      ∀ S : Set ℝ, ∀ w : EucSpace N,
        w ∈ energyFiberGradientMinima E
          (analyticEnergyAtlasSource n m E phi level G requirement r) S ↔
        w ∈ analyticEnergyAtlasSource n m E phi level G requirement r ∧ E w ∈ S ∧
          ∀ j, ∀ z : Base (n j), ‖z‖ < r j → level j z = E w →
            fiberLowerBoundQuery (a j) (b j) (v j) (requirement j)
              z (squaredGradientNorm E w) (r j) = 0 := by
  obtain ⟨a, b, v, r, hr, ha, ha0, hb, hv, heq⟩ :=
    analytic_atlas_lower_bounds_arithmetic n d m E (squaredGradientNorm E) phi hphi hE
      (fun j => squared_gradient_norm_analyticAt E (phi j 0) (hE j))
      level hlevel horder G hG requirement
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro S w
  constructor
  · rintro ⟨hw, hwE, hmin⟩
    refine ⟨hw, hwE, (heq (E w) (squaredGradientNorm E w)).mp ?_⟩
    intro x hx hxE
    exact pow_le_pow_left₀ (norm_nonneg _) (hmin x hx hxE) 2
  · rintro ⟨hw, hwE, hqueries⟩
    refine ⟨hw, hwE, ?_⟩
    intro x hx hxE
    exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
      ((heq (E w) (squaredGradientNorm E w)).mpr hqueries x hx hxE)

end Transformer.Normalization
