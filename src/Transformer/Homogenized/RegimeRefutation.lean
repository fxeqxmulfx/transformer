/-
# Homogenized Transformers — the printed rates of `cor:ode1` and `cor:ode2` are false

`cor:ode1` and `cor:ode2` of arXiv:2604.01978v1 display

  `sup_{t∈[0,t_L]} |𝔼φ(X^η(t)) - φ(X(t))| ≤ C e^{C t_L} η (t_L+1) max(η,α)`

for the piecewise-constant interpolation `X^η(t) = X^{⌊t/η⌋}`.  Both are
false.  At the rotation head of `RotationHead` — `d = 2`, one token,
`ρ* = δ_{(J,0)}`, so `α = 0` and `ass:high_order_short` holds with
`σ_V = σ_A = 0` — take `η_m = 1/(m+1)`, `L_m = 1` (so `αηL = 0`, `η²L → 0`,
`η³L → 0`), `φ(X) = (x₁)₁` and `t = η/2`.  The chain has not moved,
`X^η(η/2) = X⁰ = e₀`, while both limiting flows — they coincide here, the
corrector `∇_b b` vanishing — have turned by the angle `η/2`, so the left-hand
side is `sin(η/2) ≥ η/π`.  The right-hand side is
`C e^{Cη} η (η+1) η ≤ 2 C e^C η²`, smaller for `η` small.

The `Θ(η)` gap is the interpolation lag, which no rate carrying a factor `η²`
can absorb.  `Regimes` states both corollaries with the leading `η` removed;
this file is the proof that the printed rate cannot be kept.

Source: arXiv:2604.01978v1, `cor:ode1`, `cor:ode2`.
-/

import Transformer.Homogenized.Regimes
import Transformer.Homogenized.RotationHead
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory Filter

namespace Transformer
namespace Homogenized

/-- For every `C ≥ 1` some `η = 1/(m+1)` has
`C e^{Cη} η (η+1) η < sin(η/2)`: the printed bound of `cor:ode1` at `t_L = η`,
`α = 0`, against the displacement of the rotation at `t = η/2`. -/
theorem rot_gap (C : ℝ) (hC : 1 ≤ C) : ∃ m : ℕ,
    C * Real.exp (C * (1 / ((m : ℝ) + 1) * 1)) * (1 / ((m : ℝ) + 1)) *
      (1 / ((m : ℝ) + 1) * 1 + 1) * (1 / ((m : ℝ) + 1)) < Real.sin (1 / ((m : ℝ) + 1) / 2) := by
  set K := 8 * C * Real.exp C with hK
  refine ⟨⌈K⌉₊, ?_⟩
  set η := 1 / ((⌈K⌉₊ : ℝ) + 1) with hη
  have hKpos : 0 < K := by positivity
  have hm : K < (⌈K⌉₊ : ℝ) + 1 := (Nat.le_ceil K).trans_lt (lt_add_one _)
  have hη0 : 0 < η := by positivity
  have hη1 : η ≤ 1 := by
    rw [hη, div_le_one (by positivity)]; linarith [(Nat.cast_nonneg ⌈K⌉₊ : (0 : ℝ) ≤ _)]
  have hηK : η * K < 1 := by
    rw [hη, div_mul_eq_mul_div, one_mul, div_lt_one (by positivity)]; exact hm
  have hexp : Real.exp (C * (η * 1)) ≤ Real.exp C :=
    Real.exp_le_exp.2 (by nlinarith)
  have hsin : η / 4 < Real.sin (η / 2) := by
    have h1 := Real.mul_le_sin (x := η / 2) (by positivity) (by linarith [Real.pi_gt_three])
    have h2 : η / 4 < 2 / π * (η / 2) := by
      rw [show 2 / π * (η / 2) = η / π by field_simp]
      exact div_lt_div_of_pos_left hη0 Real.pi_pos Real.pi_lt_four
    linarith
  have hC0 : 0 < C := by linarith
  calc C * Real.exp (C * (η * 1)) * η * (η * 1 + 1) * η
      ≤ C * Real.exp C * η * 2 * η := by gcongr; linarith
    _ = (η * K) * η / 4 := by rw [hK]; ring
    _ < 1 * η / 4 := by gcongr
    _ = η / 4 := by ring
    _ < _ := hsin

/-- The hypothesis of `rot_gap` is satisfiable. -/
example : (1 : ℝ) ≤ 1 := le_rfl

/-- The observable `φ(X) = (x₁)₁` is linear, hence `C⁴`. -/
theorem contDiff_coord : ContDiff ℝ 4 (fun x : Idx 1 → EucSpace 2 => x 0 1) :=
  ((EuclideanSpace.proj (1 : Fin 2) : EucSpace 2 →L[ℝ] ℝ).comp
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Idx 1 => EucSpace 2) 0)).contDiff

/-- At `t = η/2` the chain is still at `e₀`, and the rotation is at angle `η/2`:
the error in `φ(X) = (x₁)₁` is `sin(η/2)`. -/
theorem rot_error (η : ℝ) (hη : 0 < η) (hη1 : η ≤ 1) :
    |(∫ ω, (fun x : Idx 1 → EucSpace 2 => x 0 1)
        (interpChain η ((fun _ : Unit => rotChain η 0) ω) (η / 2)) ∂Measure.dirac ()) -
      (fun x : Idx 1 → EucSpace 2 => x 0 1) (rotFlow (η / 2))| = Real.sin (η / 2) := by
  have hfl : ⌊η / 2 / η⌋₊ = 0 := by
    rw [div_div_cancel_left' hη.ne', Nat.floor_eq_zero]; norm_num
  rw [integral_dirac]
  simp only [interpChain, hfl]
  rw [rotFlow_apply]
  have hs : 0 ≤ Real.sin (η / 2) :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by positivity) (by linarith [Real.pi_gt_three])
  simp [rotChain, abs_of_nonneg hs]

/-- The hypotheses of `rot_error` are satisfiable: `η = 1`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := ⟨one_pos, le_rfl⟩

/-- **`cor:ode1` is false as printed.**  The display
`sup_{t∈[0,t_L]} |𝔼φ(X^η(t)) - φ(X(t))| ≤ C e^{C t_L} η (t_L+1) max(η,α)`
fails at the rotation head: see the module docstring.  `ballistic_regime`
states the corollary with the leading `η` removed.

Source: arXiv:2604.01978v1, `cor:ode1`. -/
theorem not_ballistic_regime_printed :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
      ∀ (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ), IsVarianceProxy d n β ρ s → (∀ m, 0 < η m) →
      Tendsto (fun m => alphaOf (η m) s H * η m * L m) atTop (nhds 0) →
      Tendsto (fun m => η m ^ 2 * L m) atTop (nhds 0) →
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
  intro h
  have hη : ∀ m : ℕ, (0 : ℝ) < 1 / ((m : ℝ) + 1) := fun m => by positivity
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) one_pos 0 0 0 (Measure.dirac rotHead)
    (hasHighOrderLaw_dirac _) _ contDiff_coord (fun m => 1 / ((m : ℝ) + 1)) (fun _ => 1) 0
    (isVarianceProxy_rotHead 0) hη (by simp [alphaOf])
    (by simpa using ((tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).pow 2))
  obtain ⟨m, hm⟩ := rot_gap C hC
  have key := hb m _ (fun _ => by simp [PiLp.norm_single]) Unit (Measure.dirac ()) _ _
    (isRandomChain_rotChain _ 0) rotFlow (isBallisticFlow_rotFlow 0)
    (by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp) (1 / ((m : ℝ) + 1) / 2)
    ⟨by positivity, by push_cast; linarith [hη m]⟩
  rw [rot_error _ (hη m) (by rw [div_le_one (by positivity)]; simp)] at key
  simp only [alphaOf, Nat.cast_one, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    mul_zero, zero_div, max_eq_left (hη m).le] at key
  linarith

/-- **`cor:ode2` is false as printed.**  The same display against
`eq: deterministic.modified` fails at the rotation head, whose corrector
vanishes.  `modified_regime` states the corollary with the leading `η` removed.

Source: arXiv:2604.01978v1, `cor:ode2`. -/
theorem not_modified_regime_printed :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
      ∀ (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ), IsVarianceProxy d n β ρ s → (∀ m, 0 < η m) →
      Tendsto (fun m => alphaOf (η m) s H * η m * L m) atTop (nhds 0) →
      Tendsto (fun m => η m ^ 3 * L m) atTop (nhds 0) →
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
  intro h
  have hη : ∀ m : ℕ, (0 : ℝ) < 1 / ((m : ℝ) + 1) := fun m => by positivity
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) one_pos 0 0 0 (Measure.dirac rotHead)
    (hasHighOrderLaw_dirac _) _ contDiff_coord (fun m => 1 / ((m : ℝ) + 1)) (fun _ => 1) 0
    (isVarianceProxy_rotHead 0) hη (by simp [alphaOf])
    (by simpa using ((tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).pow 3))
  obtain ⟨m, hm⟩ := rot_gap C hC
  have key := hb m _ (fun _ => by simp [PiLp.norm_single]) Unit (Measure.dirac ()) _ _
    (isRandomChain_rotChain _ 0) rotFlow (isModifiedFlow_rotFlow _ 0)
    (by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp) (1 / ((m : ℝ) + 1) / 2)
    ⟨by positivity, by push_cast; linarith [hη m]⟩
  rw [rot_error _ (hη m) (by rw [div_le_one (by positivity)]; simp)] at key
  simp only [alphaOf, Nat.cast_one, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    mul_zero, zero_div, max_eq_left (hη m).le] at key
  linarith

end Homogenized
end Transformer
