/-
# Homogenized Transformers — the conditional diffusive corollary

The finite grid-time estimate of arXiv:2604.01978v1, §5, `cor:SDE`, follows
from an improved weak estimate for the chosen model and observable.
The general estimate of §2.3.2 is false even at grid times, as proved by
`not_weak_error_modified`, so this corollary takes it as a hypothesis.
The same constant works for every member of the sequence `(η_m, L_m)`.

The source's off-grid improved rate is also refuted in `WeakErrorRefutation`.
The grid restriction, explicit initial condition, and probability condition
on the comparison measure are retained in this conditional statement.

A constant observable supplies a genuine witness for the weak-estimate
hypothesis; its expectation is one on either probability space.
-/

import Transformer.Homogenized.WeakError

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- For the observable `φ = 1`, the improved bound of arXiv:2604.01978v1,
§2.2.3, the paragraph after `thm:weak_error_clean`, holds with `C = 1`.
Both expectations equal one. No assumption on the head law is needed for
this special observable; the random-chain predicate supplies probability
normalization on the discrete space. -/
theorem weak_error_modified_const {d n H : ℕ} (β : ℝ)
    (ρ : Measure (HeadParam d)) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ l : ℕ, l ≤ L →
        |(∫ ω', (fun _ : Idx n → EucSpace d => (1 : ℝ)) (X ((l : ℝ) * η) ω') ∂P') -
            ∫ ω, (fun _ : Idx n → EucSpace d => (1 : ℝ))
              (interpChain η (Xd ω) ((l : ℝ) * η)) ∂P| ≤
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H) := by
  refine ⟨1, le_rfl, ?_⟩
  intro η s hη _ L x₀ _ Ω _ P Θ Xd hchain Ω' _ P' _ X _ _ l _
  let := hchain.prob
  simp only [integral_const, probReal_univ, one_smul, sub_self, abs_zero]
  have hηL : 0 ≤ η * (L : ℝ) := mul_nonneg hη.le (Nat.cast_nonneg L)
  have hmax : 0 ≤ max η (alphaOf η s H) := hη.le.trans (le_max_left _ _)
  positivity

/-- **Conditional corollary (`cor:SDE`).** Assuming the improved weak estimate,
the grid-time bound holds along any sequence of positive step sizes, with a
single constant independent of its index. The unproved input is stated in
full in `hweak`; using the still-open theorem would introduce `sorryAx`.

The source restricts to `αηL = O(1)` and `η³L = o(1)`. Those conditions
describe the diffusive scaling, but are unnecessary for this finite bound,
which follows from the estimate valid for each positive `η` and each `L`.
They are therefore omitted from the finite corollary. This does not assert
that the right-hand side tends to zero for every sequence.

The source states the bound uniformly on `[0,t_L]`; it is false away from
the grid (`not_diffusive_regime_printed`). Here `t = ℓη`, `ℓ ≤ L`.
The expectation on the continuous space explicitly requires a probability
measure, as the source's notation `E` does.

Source: arXiv:2604.01978v1, §5, `cor:SDE`, and §2.2.3, the improved estimate
following `thm:weak_error_clean`. -/
theorem diffusive_regime {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ)
    (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ) (hs : IsVarianceProxy d n β ρ s)
    (hη : ∀ m, 0 < η m)
    (hweak : 0 < H → HasHighOrderLaw d σV σA ρ → ContDiff ℝ 4 φ →
      ∃ C : ℝ, 1 ≤ C ∧
        ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
        ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
        ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
          (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
          IsRandomChain η β ρ P Θ Xd x₀ →
        ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
          (X : ℝ → Ω' → (Idx n → EucSpace d)),
          IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
        ∀ l : ℕ, l ≤ L →
          |(∫ ω', φ (X ((l : ℝ) * η) ω') ∂P') -
              ∫ ω, φ (interpChain η (Xd ω) ((l : ℝ) * η)) ∂P| ≤
            C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H)) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (m : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain (η m) β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsModifiedSde (η m) β (alphaOf (η m) s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ l : ℕ, l ≤ L m →
        |(∫ ω, φ (interpChain (η m) (Xd ω) ((l : ℝ) * η m)) ∂P) -
            ∫ ω', φ (X ((l : ℝ) * η m) ω') ∂P'| ≤
          C * Real.exp (C * (η m * L m)) * η m * (η m * L m + 1) *
            max (η m) (alphaOf (η m) s H) := by
  obtain ⟨C, hC, hbound⟩ := hweak hH hρ hφ
  refine ⟨C, hC, ?_⟩
  intro m x₀ hx₀ Ω _ P Θ Xd hchain Ω' _ P' _ X hSde hinit l hl
  rw [abs_sub_comm]
  exact hbound (η m) s (hη m) hs (L m) x₀ hx₀ Ω P Θ Xd hchain Ω' P' X hSde hinit l hl

/-- All hypotheses of the conditional corollary are realizable. The head law
is `δ_0`, the observable is one, `s = 0`, `η_m = 1/(m+1)` and `L_m = m+1`.
The explicit weak-estimate input is proved by `weak_error_modified_const`,
so this example uses no open theorem. The horizon `η_m L_m` equals one. -/
example (d n : ℕ) (β : ℝ) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (m : ℕ) (x₀ : Idx (n + 1) → EucSpace (d + 1)), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx 1 → Ω → HeadParam (d + 1))
        (Xd : Ω → ℕ → Idx (n + 1) → EucSpace (d + 1)),
        IsRandomChain (1 / ((m : ℝ) + 1)) β (Measure.dirac (0 : HeadParam (d + 1)))
          P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
        (X : ℝ → Ω' → (Idx (n + 1) → EucSpace (d + 1))),
        IsModifiedSde (1 / ((m : ℝ) + 1)) β (alphaOf (1 / ((m : ℝ) + 1)) 0 1) 0
          (Measure.dirac (0 : HeadParam (d + 1))) P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ l : ℕ, l ≤ m + 1 →
        |(∫ ω, (fun _ => (1 : ℝ))
            (interpChain (1 / ((m : ℝ) + 1)) (Xd ω) ((l : ℝ) / ((m : ℝ) + 1))) ∂P) -
          ∫ ω', (fun _ => (1 : ℝ)) (X ((l : ℝ) / ((m : ℝ) + 1)) ω') ∂P'| ≤
          C * Real.exp C * (1 / ((m : ℝ) + 1)) * 2 *
            max (1 / ((m : ℝ) + 1)) (alphaOf (1 / ((m : ℝ) + 1)) 0 1) := by
  have h := diffusive_regime (H := 1) Nat.one_pos β 0 0
    (Measure.dirac (0 : HeadParam (d + 1))) (hasHighOrderLaw_dirac_zero _)
    (fun _ : Idx (n + 1) → EucSpace (d + 1) => (1 : ℝ)) contDiff_const
    (fun m => 1 / ((m : ℝ) + 1)) (fun m => m + 1) 0
    (isVarianceProxy_dirac_zero d n β (fun _ => (basePoint d : EucSpace (d + 1)))
      (fun _ => by simp [basePoint, PiLp.norm_single])) (fun m => by positivity)
    (fun _ _ _ => weak_error_modified_const β _)
  have hden (m : ℕ) : (m : ℝ) + 1 ≠ 0 := by positivity
  simpa [Nat.cast_add, Nat.cast_one, hden, one_add_one_eq_two] using h

/-- The inner process hypotheses are also satisfiable: the zero head leaves
the unit configuration fixed in both the chain and the modified SDE.
Source: arXiv:2604.01978v1, §5, `eq:update_tokens`, `eq:SDE_ito_clean`. -/
example (d n : ℕ) (η β : ℝ) :
    IsRandomChain (H := 1) η β (Measure.dirac (0 : HeadParam (d + 1)))
      (Measure.dirac ()) (fun _ _ _ => 0) (fun _ _ _ => (basePoint d : EucSpace (d + 1)))
      (fun _ : Idx (n + 1) => (basePoint d : EucSpace (d + 1))) ∧
    IsModifiedSde η β 0 0 (Measure.dirac (0 : HeadParam (d + 1))) (Measure.dirac ())
      (fun _ _ (_ : Idx (n + 1)) => (basePoint d : EucSpace (d + 1))) := by
  have hx : ∀ _i : Idx (n + 1), ‖(basePoint d : EucSpace (d + 1))‖ = 1 :=
    fun _ => by simp [basePoint, PiLp.norm_single]
  constructor
  · exact isRandomChain_dirac_zero η β _ hx
  · have hB : bField (n := n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) =
        fun _ => 0 := by funext x i; exact bField_dirac_zero β x i
    have hCov : covDerivB (n := n + 1) β (Measure.dirac (0 : HeadParam (d + 1))) =
        fun _ _ => 0 := by funext x i; simp [covDerivB, covDeriv, hB, proj]
    have hnoise (θ : HeadParam (d + 1)) (x : Idx (n + 1) → EucSpace (d + 1)) :
        noiseField β 0 0 (Measure.dirac (0 : HeadParam (d + 1))) θ x = 0 := by
      funext i; simp [noiseField]
    refine ⟨fun _ _ i => hx i, ?_⟩
    intro φ _ t _
    simp only [hB, hCov, smul_zero, sub_zero]
    have hgen : sphGenerator (Measure.dirac (0 : HeadParam (d + 1)))
        (fun _ => 0) (noiseField β 0 0 (Measure.dirac (0 : HeadParam (d + 1)))) φ
        (fun _ : Idx (n + 1) => (basePoint d : EucSpace (d + 1))) = 0 := by
      simp [sphGenerator, hnoise]
    simp only [hgen, integral_zero]
    exact hasDerivWithinAt_const _ _ _

end Transformer.Homogenized
