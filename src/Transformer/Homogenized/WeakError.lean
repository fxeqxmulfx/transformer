/-
# Homogenized Transformers — weak approximation and false improved rates

Source: arXiv:2604.01978v1, §2.3.2, `thm:weak_error_clean`,
its improved-rate paragraph, and §6.1, `lem:stability_generator`.

The source compares its actual normalized random chain to a
homogenized SDE at a coarser rate `O(η)`. That theorem remains
open below, with the source's initial and probability conditions.

The stronger modified-equation estimate is false even at grid
times. A deterministic value matrix `I+J` has the same projected
drift as `J`, but a different second-order normalized discrete
step. The exact phase discrepancy produces an error of order
`η` at physical time one, against the proposed `O(η²)` bound.

The supporting §6.1 module also proves the negation of the
source's one-step estimate. These refutations preserve the
original dynamics and full source assumptions; the false
claims are replaced by theorems naming their counterexamples.
-/

import Transformer.Homogenized.Section6_OneStepRefutation

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Theorem (thm:weak_error_clean).**  Under `ass:high_order_short`, for any
`φ ∈ C⁴((𝕊^{d-1})^n)` there is `C ≥ 1`, depending on `‖φ‖_{C⁴}` but not on
`η, α, L`, with

  `sup_{t ∈ [0,t_L]} |𝔼φ(X(t)) - 𝔼φ(X^η(t))| ≤ C e^{C t_L} η (t_L + 1) max(1,α)`,

where `X` solves `eq: first.sde`, `X^η` interpolates the chain
`eq:update_tokens`, `t_L = ηL` and `α = η ς²/H`.

**What the source says and what is carried here.**  `X` and `X^η` start from
the same deterministic configuration; the manuscript's `X(0) = X^0` is written
out as the hypothesis `X 0 ω = x₀`, because without it the two expectations at
`t = 0` need not agree and the bound is false.  The solution of
`eq: first.sde` is taken in the sense of `IsFirstSde`, i.e. through the
generator: see `Transformer.Homogenized.Generator`.

The number of heads is positive.  The source writes `Σ_{h=1}^H`, so `H ≥ 1` is
implicit there, and the hypothesis is needed: at `H = 0` the update of
`eq:update_tokens` is the identity and the chain never moves, while the
limiting dynamics has drift `b` and does, so the bound fails for every `φ` that
separates `X(t_L)` from `X⁰`.

Not proved here.

Source: arXiv:2604.01978v1, `thm:weak_error_clean`. -/
theorem weak_error_clean {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : HasHighOrderLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω') [IsProbabilityMeasure P']
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsFirstSde β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ t ∈ Set.Icc (0 : ℝ) (η * L),
        |(∫ ω', φ (X t ω') ∂P') - ∫ ω, φ (interpChain η (Xd ω) t) ∂P| ≤
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max 1 (alphaOf η s H) := by
  sorry

/-- The hypotheses of `weak_error_clean` are satisfiable: the degenerate weight
law `ρ* = δ_0` satisfies `ass:high_order_short` by
`hasHighOrderLaw_dirac_zero`, and a constant `φ` is `C⁴`. -/
example (d n : ℕ) :
    0 < 1 ∧ HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, hasHighOrderLaw_dirac_zero d, contDiff_const⟩

/-- **The improved rate is false even at grid times.** The paragraph
following `thm:weak_error_clean` claims that the modified equation
`eq:SDE_ito_clean` has weak error
`C e^{C t_L} η (t_L+1) max(η,α)`. The original display takes the supremum
throughout `[0,t_L]`. The former formal statement restricted it to grid
times to avoid the interpolation obstruction proved separately in
`WeakErrorRefutation`.

The grid restriction is still insufficient. Take the actual head `(I+J,0)`,
one planar token, and one head. The high-order law holds with zero centered
fluctuations and variance proxy zero. The exact modified SDE is unit-speed
rotation, while each actual layer rotates by `arctan(η/(1+η))`.
At `η=1/m`, `L=m`, and grid time one, the coordinate error is at least
`cos(1)/(m+1)`. The claimed right side is `2 C e^C/m²`.

The negation below keeps every hypothesis and quantifier of the former
grid-time statement: the model and C⁴ observable are fixed before `C`,
and both actual processes start from the same deterministic unit token
on probability spaces. It therefore refutes both the grid-time variant
and the stronger source display. The distinct coarser theorem
`weak_error_clean` remains open.

Source: arXiv:2604.01978v1, §2.3.2, the paragraph following
`thm:weak_error_clean`, and §6.1, its modified-equation proof. -/
theorem not_weak_error_modified :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0)
      (ρ : Measure (HeadParam d)), HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
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
          C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H) := by
  intro h
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) Nat.one_pos 1 0 0
    (Measure.dirac radialRotHead) (radialRotHead_law 1).1 _ contDiff_radialRotObservable
  obtain ⟨m, hm, hgap⟩ := radialRot_grid_constant C hC
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hη : (0 : ℝ) < 1 / (m : ℝ) := by positivity
  have hsde : IsModifiedSde (1 / (m : ℝ)) 1 (alphaOf (1 / (m : ℝ)) 0 1) 0
      (Measure.dirac radialRotHead) (Measure.dirac ()) (fun t (_ : Unit) => rotFlow t) := by
    simpa only [alphaOf, zero_pow (by decide : 2 ≠ 0), mul_zero, zero_div] using
      isModifiedSde_radialRotHead (1 / (m : ℝ)) 1
  have key := hb _ 0 hη (radialRotHead_law 1).2 m
    (fun _ => EuclideanSpace.single 0 1) (fun _ => by simp [PiLp.norm_single])
    Unit (Measure.dirac ()) _ _ (isRandomChain_radialRotChain hη 1)
    Unit (Measure.dirac ()) (fun t _ => rotFlow t) hsde
    (fun _ => by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp) m le_rfl
  rw [interpChain_natCast_mul hη] at key
  have ht : (m : ℝ) * (1 / (m : ℝ)) = 1 := by field_simp
  have ht' : (1 / (m : ℝ)) * (m : ℝ) = 1 := by field_simp
  simp only [ht, ht', integral_dirac, rotFlow_apply, Matrix.cons_val_one,
    Matrix.cons_val_fin_one, mul_one, alphaOf, zero_pow (by decide : 2 ≠ 0),
    mul_zero, zero_div, max_eq_left hη.le] at key
  exact (not_lt_of_ge key) hgap

/-- The numerical, head-law, and observable premises of the refuted claim
are satisfiable. The full counterexample in the proof above also verifies
both actual processes and their common unit initial configuration. -/
example (d n : ℕ) :
    0 < 1 ∧ HasHighOrderLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, hasHighOrderLaw_dirac_zero d, contDiff_const⟩

end Homogenized
end Transformer
