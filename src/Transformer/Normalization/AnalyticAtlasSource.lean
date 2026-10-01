/-
# Sources described by finite analytic coordinate families

Domains retain the original finite signs and both strict box bounds.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.AnalyticPreparation.Basic
import Transformer.Normalization.AnalyticSignStability
import Transformer.Basic

namespace Transformer.Normalization

open AnalyticPreparation Set

/-- The actual coordinate domain of one constrained analytic chart.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticChartDomain {n m : ℕ} (G : Fin m → Ambient n → ℝ)
    (requirement : Fin m → AnalyticSignRequirement) (r : ℝ) : Set (Ambient n) :=
  {x | ‖x.1‖ < r ∧ |x.2| < r ∧ ∀ i, (requirement i).Holds (G i x)}

/-- The union of constrained energy-graph chart images. Dimensions and
numbers of constraints may vary between charts. The level parameter must
equal the actual energy of the image point. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def analyticEnergyAtlasSource {k N : ℕ} (n m : Fin k → ℕ) (E : EucSpace N → ℝ)
    (phi : ∀ j, Ambient (n j) → EucSpace N)
    (level : ∀ j, Base (n j) → ℝ)
    (G : ∀ j, Fin (m j) → Ambient (n j) → ℝ)
    (requirement : ∀ j, Fin (m j) → AnalyticSignRequirement)
    (r : Fin k → ℝ) : Set (EucSpace N) :=
  ⋃ j, phi j '' {x | x ∈ analyticChartDomain (G j) (requirement j) (r j) ∧
    E (phi j x) = level j x.1}

/-- Membership supplies a chart and a genuine point in its constrained
box. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem mem_analyticEnergyAtlasSource_iff {k N : ℕ} (n m : Fin k → ℕ)
    (E : EucSpace N → ℝ)
    (phi : ∀ j, Ambient (n j) → EucSpace N)
    (level : ∀ j, Base (n j) → ℝ)
    (G : ∀ j, Fin (m j) → Ambient (n j) → ℝ)
    (requirement : ∀ j, Fin (m j) → AnalyticSignRequirement)
    (r : Fin k → ℝ) (w : EucSpace N) :
    w ∈ analyticEnergyAtlasSource n m E phi level G requirement r ↔
      ∃ j, ∃ x : Ambient (n j),
        ‖x.1‖ < r j ∧ |x.2| < r j ∧
          (∀ i, (requirement j i).Holds (G j i x)) ∧
            E (phi j x) = level j x.1 ∧ phi j x = w := by
  simp only [analyticEnergyAtlasSource, mem_iUnion, mem_image,
    analyticChartDomain, mem_ofPred_eq]
  constructor
  · rintro ⟨j, x, ⟨⟨hz, hy, hs⟩, hlevel⟩, heq⟩
    exact ⟨j, x, hz, hy, hs, hlevel, heq⟩
  · rintro ⟨j, x, hz, hy, hs, hlevel, heq⟩
    exact ⟨j, x, ⟨⟨hz, hy, hs⟩, hlevel⟩, heq⟩

end Transformer.Normalization
