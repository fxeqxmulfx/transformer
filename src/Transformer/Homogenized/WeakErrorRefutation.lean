/-
# Homogenized Transformers — the improved rate off the grid is false

The paragraph after `thm:weak_error_clean` of arXiv:2604.01978v1 states that
the chain is approximated by the modified equation `eq:SDE_ito_clean` at the
improved rate `C e^{C t_L} η (t_L+1) max(η,α)`, uniformly on `[0, t_L]` as the
theorem's display has it; `cor:SDE` states the same display in the diffusive
scaling.  Both are false for the piecewise-constant `X^η(t) = X^{⌊t/η⌋}`, for
the reason of `RegimeRefutation`: at the rotation head of `RotationHead` the
noise vanishes (`σ = 0`, so `√α/σ = 0`), the deterministic rotation solves
`eq:SDE_ito_clean`, and at `t = η/2` it has turned by `η/2` while the chain
has not moved — an error `sin(η/2) = Θ(η)` against a bound `Θ(η²)`.

`weak_error_modified` and `diffusive_regime` therefore state the improved rate
at the grid times `t = ℓη` only.

Source: arXiv:2604.01978v1, §2.3.2 (the paragraph after
`thm:weak_error_clean`), `cor:SDE`.
-/

import Transformer.Homogenized.RegimeRefutation

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory Filter

namespace Transformer
namespace Homogenized

/-- At variance proxy `σ = 0` the diffusion kernel `(√α/σ) G` vanishes.

Source: arXiv:2604.01978v1, `eq:G_alpha_def`. -/
theorem noiseField_zero {d n : ℕ} (β α : ℝ) (ρ : Measure (HeadParam d)) (θ : HeadParam d) :
    noiseField (n := n) β α 0 ρ θ = fun _ => 0 := by
  funext x i
  simp [noiseField]

/-- The rotation, as a curve in `(ℝ²)¹`, has velocity `b`. -/
theorem hasDerivAt_rotFlow_pi (β : ℝ) (t : ℝ) :
    HasDerivAt rotFlow (bField β (Measure.dirac rotHead) (rotFlow t)) t :=
  hasDerivAt_pi.2 fun i => hasDerivAt_rotFlow β t i

/-- The deterministic rotation, on a one-point probability space, solves
`eq:SDE_ito_clean` for `ρ* = δ_{(J,0)}` and `σ = 0`: the noise vanishes, the
corrector vanishes, and `t ↦ φ(x(t))` has derivative `Dφ(x(t))[b(x(t))]`.

Source: arXiv:2604.01978v1, `eq:SDE_ito_clean`, `def: ito.formula`. -/
theorem isModifiedSde_rotFlow (η β α : ℝ) :
    IsModifiedSde η β α 0 (Measure.dirac rotHead) (Measure.dirac ())
      (fun t (_ : Unit) => rotFlow t) := by
  refine ⟨fun t _ => norm_rotFlow t, fun φ hφ t _ => ?_⟩
  simp only [integral_dirac, sphGenerator, noiseField_zero, sphHess_zero, integral_zero,
    mul_zero, add_zero]
  have hB : (fun i => bField β (Measure.dirac rotHead) (rotFlow t) i -
      (η / 2) • covDerivB β (Measure.dirac rotHead) (rotFlow t) i) =
      bField β (Measure.dirac rotHead) (rotFlow t) := by
    funext i
    rw [covDerivB_rotHead β _ (norm_rotFlow t 0), smul_zero, sub_zero]
  rw [hB]
  have hd : Differentiable ℝ φ := hφ.differentiable (by simp)
  exact ((hd _).hasFDerivAt.comp_hasDerivAt t (hasDerivAt_rotFlow_pi β t)).hasDerivWithinAt

/-- **The improved rate is false off the grid.**  The rate
`C e^{C t_L} η (t_L+1) max(η,α)` against `eq:SDE_ito_clean`, uniformly in
`t ∈ [0, t_L]`, with `C` depending on `φ` only, fails at the rotation head:
see the module docstring.  `weak_error_modified` states it at grid times.

Source: arXiv:2604.01978v1, §2.3.2, the paragraph after `thm:weak_error_clean`. -/
theorem not_weak_error_modified_printed :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
      ∃ C : ℝ, 1 ≤ C ∧
        ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
        ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
        ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
          (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
          IsRandomChain η β ρ P Θ Xd x₀ →
        ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
          (X : ℝ → Ω' → (Idx n → EucSpace d)),
          IsModifiedSde η β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
        ∀ t ∈ Set.Icc (0 : ℝ) (η * L),
          |(∫ ω', φ (X t ω') ∂P') - ∫ ω, φ (interpChain η (Xd ω) t) ∂P| ≤
            C * Real.exp (C * (η * L)) * η * (η * L + 1) * max η (alphaOf η s H) := by
  intro h
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) one_pos 0 0 0 (Measure.dirac rotHead)
    (hasHighOrderLaw_dirac _) _ contDiff_coord
  obtain ⟨m, hm⟩ := rot_gap C hC
  have hη : (0 : ℝ) < 1 / ((m : ℝ) + 1) := by positivity
  have key := hb _ 0 hη (isVarianceProxy_rotHead 0) 1 _ (fun _ => by simp [PiLp.norm_single])
    Unit (Measure.dirac ()) _ _ (isRandomChain_rotChain _ 0) Unit (Measure.dirac ())
    (fun t _ => rotFlow t) (isModifiedSde_rotFlow _ 0 _)
    (fun _ => by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp)
    (1 / ((m : ℝ) + 1) / 2) ⟨by positivity, by push_cast; linarith⟩
  have e := rot_error _ hη (by rw [div_le_one (by positivity)]; simp)
  rw [abs_sub_comm] at key
  simp only [integral_dirac] at key e
  rw [e] at key
  simp only [alphaOf, Nat.cast_one, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    mul_zero, zero_div, max_eq_left hη.le] at key
  linarith

/-- **`cor:SDE` is false as printed.**  Its display, uniform in
`t ∈ [0, t_L]`, fails at the rotation head with `η_m = 1/(m+1)`, `L_m = 1`:
see the module docstring.  `diffusive_regime` states it at grid times.

Source: arXiv:2604.01978v1, `cor:SDE`. -/
theorem not_diffusive_regime_printed :
    ¬ ∀ {d n H : ℕ}, 0 < H → ∀ (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      HasHighOrderLaw d σV σA ρ →
      ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ 4 φ →
      ∀ (η : ℕ → ℝ) (L : ℕ → ℕ) (s : ℝ), IsVarianceProxy d n β ρ s → (∀ m, 0 < η m) →
      (∃ M : ℝ, ∀ m, alphaOf (η m) s H * η m * L m ≤ M) →
      Tendsto (fun m => η m ^ 3 * L m) atTop (nhds 0) →
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
  intro h
  have hη : ∀ m : ℕ, (0 : ℝ) < 1 / ((m : ℝ) + 1) := fun m => by positivity
  obtain ⟨C, hC, hb⟩ := h (d := 2) (n := 1) (H := 1) one_pos 0 0 0 (Measure.dirac rotHead)
    (hasHighOrderLaw_dirac _) _ contDiff_coord (fun m => 1 / ((m : ℝ) + 1)) (fun _ => 1) 0
    (isVarianceProxy_rotHead 0) hη ⟨0, fun m => by simp [alphaOf]⟩
    (by simpa using ((tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).pow 3))
  obtain ⟨m, hm⟩ := rot_gap C hC
  have key := hb m _ (fun _ => by simp [PiLp.norm_single]) Unit (Measure.dirac ()) _ _
    (isRandomChain_rotChain _ 0) Unit (Measure.dirac ()) (fun t _ => rotFlow t)
    (isModifiedSde_rotFlow _ 0 _)
    (fun _ => by funext i; rw [rotFlow_apply]; ext k; fin_cases k <;> simp)
    (1 / ((m : ℝ) + 1) / 2) ⟨by positivity, by push_cast; linarith [hη m]⟩
  have e := rot_error _ (hη m) (by rw [div_le_one (by positivity)]; simp)
  simp only [integral_dirac] at key e
  rw [e] at key
  simp only [alphaOf, Nat.cast_one, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    mul_zero, zero_div, max_eq_left (hη m).le] at key
  linarith

end Homogenized
end Transformer
