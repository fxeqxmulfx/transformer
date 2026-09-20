/-
# Homogenized Transformers — the three scaling regimes

Formalization of `cor:ode1` (ballistic), `cor:ode2` (modified) and `cor:SDE`
(diffusive) of arXiv:2604.01978v1, *Homogenized Transformers*: the three
corollaries of `thm:weak_error_clean` that fill the phase diagram.

**On the scaling conditions.**  The hypotheses of the three corollaries are
asymptotic — `αηL = o(1)`, `η²L = o(1)`, `η³L = o(1)`, `αηL = O(1)` — and an
asymptotic condition is a statement about a family, not about one `(η, L)`.
The corollaries are therefore stated along a sequence `m ↦ (η_m, L_m)`, with
the scaling written as the convergence it is, and the displayed bound asserted
at every index.  Nothing is weakened: that is what "in the subcritical scaling
… the chain is weakly approximated by" says.

Source: arXiv:2604.01978v1, §4–§5, `cor:ode1`, `cor:ode2`, `cor:SDE`.
-/

import Transformer.Homogenized.WeakError

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory Filter

namespace Transformer
namespace Homogenized

/-- **Corollary (cor:ode1), ballistic regime.**  Under `ass:high_order_short`,
in the subcritical scaling `αηL = o(1)` and `η²L = o(1)`, the interpolated
chain is weakly approximated by the solution of `eq: deterministic`,
`Ẋ = b(X)`, started at `X⁰`: for every `φ ∈ C⁴`,

  `sup_{t∈[0,t_L]} |𝔼φ(X^η(t)) - φ(X(t))| ≤ C e^{C t_L} η (t_L+1) max(η,α)`.

Not proved here.

Source: arXiv:2604.01978v1, `cor:ode1`, `eq: deterministic`. -/
theorem ballistic_regime {d n H : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ)
    (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ) (hs : IsVarianceProxy d n β ρ s)
    (hη : ∀ m, 0 < η m)
    (hαL : Tendsto (fun m => alphaOf (η m) s H * η m * L m) atTop (nhds 0))
    (hηL : Tendsto (fun m => η m ^ 2 * L m) atTop (nhds 0)) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (m : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain (η m) β ρ P Θ Xd x₀ →
      ∀ X : ℝ → Idx n → EucSpace d, IsBallisticFlow β ρ X → X 0 = x₀ →
      ∀ t ∈ Set.Icc (0 : ℝ) (η m * L m),
        |(∫ ω, φ (interpChain (η m) (Xd ω) t) ∂P) - φ (X t)| ≤
          C * Real.exp (C * (η m * L m)) * η m * (η m * L m + 1) *
            max (η m) (alphaOf (η m) s H) := by
  sorry

/-- The hypotheses of `ballistic_regime` are satisfiable: `ρ* = δ_0` satisfies
`ass:high_order_short`, a constant `φ` is `C⁴`, `s = 0` is the variance proxy
of `δ_0`, and the scalings `η_m = 1/(m+1)`, `L_m = 0` are subcritical. -/
example (d n : ℕ) (β : ℝ) :
    HasHighOrderLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      ContDiff ℝ 4 (fun _ : Idx (n + 1) → EucSpace (d + 1) => (0 : ℝ)) ∧
      IsVarianceProxy (d + 1) (n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) 0 ∧
      (∀ m : ℕ, (0 : ℝ) < 1 / (m + 1)) ∧
      Tendsto (fun m : ℕ => alphaOf (1 / (m + 1)) 0 0 * (1 / (m + 1)) * ((0 : ℕ) : ℝ))
        atTop (nhds 0) ∧
      Tendsto (fun m : ℕ => (1 / ((m : ℝ) + 1)) ^ 2 * ((0 : ℕ) : ℝ)) atTop (nhds 0) := by
  refine ⟨hasHighOrderLaw_dirac_zero _, contDiff_const,
    isVarianceProxy_dirac_zero d n β (fun _ => (basePoint d : EucSpace (d + 1)))
      (fun _ => by simp [basePoint, PiLp.norm_single]),
    fun m => by positivity, ?_, ?_⟩ <;> simp

/-- **Corollary (cor:ode2), modified regime.**  Under `ass:high_order_short`,
in the refined deterministic scaling `αηL = o(1)` and `η³L = o(1)`, the
interpolated chain is weakly approximated by the solution of
`eq: deterministic.modified`, `Ẋ = b(X) - (η/2)∇_{b(X)}b(X)`, at the same rate.

Not proved here.

Source: arXiv:2604.01978v1, `cor:ode2`, `eq: deterministic.modified`. -/
theorem modified_regime {d n H : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ)
    (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ) (hs : IsVarianceProxy d n β ρ s)
    (hη : ∀ m, 0 < η m)
    (hαL : Tendsto (fun m => alphaOf (η m) s H * η m * L m) atTop (nhds 0))
    (hηL : Tendsto (fun m => η m ^ 3 * L m) atTop (nhds 0)) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (m : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain (η m) β ρ P Θ Xd x₀ →
      ∀ X : ℝ → Idx n → EucSpace d, IsModifiedFlow (η m) β ρ X → X 0 = x₀ →
      ∀ t ∈ Set.Icc (0 : ℝ) (η m * L m),
        |(∫ ω, φ (interpChain (η m) (Xd ω) t) ∂P) - φ (X t)| ≤
          C * Real.exp (C * (η m * L m)) * η m * (η m * L m + 1) *
            max (η m) (alphaOf (η m) s H) := by
  sorry

/-- The hypotheses of `modified_regime` are satisfiable, by the same witnesses
as `ballistic_regime`. -/
example (d n : ℕ) (β : ℝ) :
    HasHighOrderLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      IsVarianceProxy (d + 1) (n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) 0 ∧
      Tendsto (fun m : ℕ => (1 / ((m : ℝ) + 1)) ^ 3 * ((0 : ℕ) : ℝ)) atTop (nhds 0) := by
  refine ⟨hasHighOrderLaw_dirac_zero _,
    isVarianceProxy_dirac_zero d n β (fun _ => (basePoint d : EucSpace (d + 1)))
      (fun _ => by simp [basePoint, PiLp.norm_single]), ?_⟩
  simp

/-- **Corollary (cor:SDE), diffusive regime.**  In the diffusive scaling
`αηL = O(1)` and `η³L = o(1)`, the interpolated chain is weakly approximated
by the solution of `eq:SDE_ito_clean` at the rate
`C e^{C t_L} η (t_L+1) max(η,α)`.

Not proved here.

Source: arXiv:2604.01978v1, `cor:SDE`. -/
theorem diffusive_regime {d n H : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ)
    (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ) (hs : IsVarianceProxy d n β ρ s)
    (hη : ∀ m, 0 < η m)
    (hαL : ∃ M : ℝ, ∀ m, alphaOf (η m) s H * η m * L m ≤ M)
    (hηL : Tendsto (fun m => η m ^ 3 * L m) atTop (nhds 0)) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (m : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain (η m) β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde (η m) β (alphaOf (η m) s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ t ∈ Set.Icc (0 : ℝ) (η m * L m),
        |(∫ ω, φ (interpChain (η m) (Xd ω) t) ∂P) - ∫ ω', φ (X t ω') ∂P'| ≤
          C * Real.exp (C * (η m * L m)) * η m * (η m * L m + 1) *
            max (η m) (alphaOf (η m) s H) := by
  sorry

/-- The hypotheses of `diffusive_regime` are satisfiable: the same witnesses,
with `αηL ≤ 0` bounded. -/
example (d n : ℕ) (β : ℝ) :
    HasHighOrderLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      IsVarianceProxy (d + 1) (n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) 0 ∧
      ∃ M : ℝ, ∀ m : ℕ, alphaOf (1 / ((m : ℝ) + 1)) 0 0 * (1 / ((m : ℝ) + 1)) * ((0 : ℕ) : ℝ) ≤ M := by
  refine ⟨hasHighOrderLaw_dirac_zero _,
    isVarianceProxy_dirac_zero d n β (fun _ => (basePoint d : EucSpace (d + 1)))
      (fun _ => by simp [basePoint, PiLp.norm_single]), 0, ?_⟩
  intro m; simp

end Homogenized
end Transformer
