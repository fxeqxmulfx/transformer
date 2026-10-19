module

public import Transformer.Interpolation.Section1_CapContractionPhases
public import Transformer.Interpolation.Section1_BoundedPhaseSequences

/-!
# Physical realizations of fixed and cap-contraction phases

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §5, the norm
estimate in its proof. A fixed phase is realized at a physical duration
`H` by scaling its two output matrices by `1/H`; its score and gate
parameters are kept, so its complete cost is at most its own cost
times `1/H + 1`. A balanced ridge phase of intrinsic clock `τ` is
realized at duration `H` by the balanced phase of clock `τ/H`, at
`√(τ/H)` times its gate size; a clock at most `C₀(1 + L)` turns this
into an additive bound in `1/H` and `L`.

Both realizations have exactly the intrinsic endpoint law. One constant,
chosen before the clock and the duration, bounds the realizations of a
fixed valid list of phases and of every cap contraction list together.
That is the hypothesis of `exists_bounded_phase_sequence_controls`.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Scaling the output matrices of a fixed phase by `1/H` costs at most
its own cost times `1/H + 1`. Source auxiliary: arXiv:2411.04551v3,
§1 and §5, time rescaling of a fixed phase. -/
theorem parameterCost_rescalePhase_le (P : AutonomousPhase d) (H : ℝ) (hH : 0 < H) :
    parameterCost (rescalePhase H⁻¹ P).1 ≤ parameterCost P.1 * (1 / H + 1) := by
  have hinv : 0 ≤ H⁻¹ := inv_nonneg.mpr hH.le
  change ‖H⁻¹ • P.1.V‖ + ‖P.1.B‖ + ‖H⁻¹ • P.1.W‖ + ‖P.1.U‖ + ‖P.1.b‖ ≤ _
  rw [norm_smul, norm_smul, Real.norm_of_nonneg hinv, one_div]
  unfold parameterCost
  have hV := norm_nonneg P.1.V
  have hB := norm_nonneg P.1.B
  have hW := norm_nonneg P.1.W
  have hU := norm_nonneg P.1.U
  have hb := norm_nonneg P.1.b
  nlinarith [mul_nonneg hinv hB, mul_nonneg hinv hU, mul_nonneg hinv hb]

/-- Every fixed phase is realized at every positive physical duration,
with its exact endpoint law and an additive inverse-time cost.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, the fixed
preprocessing, matching and inverse phases. -/
theorem exists_rescaled_phase_realization (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (hB : P.1.B = 0) (H : ℝ) (hH : 0 < H) :
    ∃ R : AutonomousPhase d, IsAutonomousPhase R ∧ R.1.B = 0 ∧
      (∀ ρ : Perspective.ProbSphere d, R.2 ρ H = P.2 ρ 1) ∧
      parameterCost R.1 ≤ parameterCost P.1 * (1 / H + 1) := by
  refine ⟨rescalePhase H⁻¹ P, hP.rescale P H⁻¹ (inv_ne_zero hH.ne'),
    rescalePhase_B_zero P H⁻¹ hB, fun ρ => ?_, parameterCost_rescalePhase_le P H hH⟩
  change P.2 ρ (H⁻¹ * H) = P.2 ρ 1
  rw [inv_mul_cancel₀ hH.ne']

/-- A balanced ridge phase of clock `τ` is realized at every positive
physical duration `H` by the balanced phase of clock `τ/H`, at the
square-root cost of its gate. Source auxiliary: arXiv:2411.04551v3,
§1 and §5, prescribed-time ridge phases. -/
theorem exists_balanced_phase_realization (i : Fin d) (p : SSphere d) (u : EucSpace d)
    (b τ : ℝ) (hτ : 0 ≤ τ) (H : ℝ) (hH : 0 < H) :
    ∃ R : AutonomousPhase d, IsAutonomousPhase R ∧ R.1.B = 0 ∧
      (∀ ρ : Perspective.ProbSphere d,
        R.2 ρ H = (balancedRidgePhase i (p : EucSpace d) u b τ).2 ρ 1) ∧
      parameterCost R.1 ≤ sqrt (τ / H) * (1 + ‖u‖ + |b|) := by
  have hκ : 0 ≤ τ / H := div_nonneg hτ hH.le
  refine ⟨balancedRidgePhase i (p : EucSpace d) u b (τ / H),
    isAutonomousPhase_balancedRidge i _ u b _, balancedRidgePhase_B_zero i _ u b _,
    fun ρ => ?_, norm_balancedRidgeParams_unit_output_le i p u b (τ / H)⟩
  rw [balancedRidgePhase_law i _ u b _ hκ, balancedRidgePhase_law i _ u b τ hτ,
    div_mul_cancel₀ τ hH.ne', mul_one]

/-- With a clock at most `C₀(1 + L)`, the balanced realization costs at
most its gate size times `(C₀ + 1)(1/H + 1 + L)`. Source auxiliary:
arXiv:2411.04551v3, §5, the logarithmic concentration cost. -/
theorem exists_balanced_phase_realization_additive (i : Fin d) (p : SSphere d)
    (u : EucSpace d) (b τ C₀ L : ℝ) (hτ : 0 ≤ τ) (hC₀ : 0 ≤ C₀) (hL : 0 ≤ L)
    (hτC : τ ≤ C₀ * (1 + L)) (H : ℝ) (hH : 0 < H) :
    ∃ R : AutonomousPhase d, IsAutonomousPhase R ∧ R.1.B = 0 ∧
      (∀ ρ : Perspective.ProbSphere d,
        R.2 ρ H = (balancedRidgePhase i (p : EucSpace d) u b τ).2 ρ 1) ∧
      parameterCost R.1 ≤ (1 + ‖u‖ + |b|) * (C₀ + 1) * (1 / H + 1 + L) := by
  obtain ⟨R, hR, hRB, hRlaw, hRcost⟩ := exists_balanced_phase_realization i p u b τ hτ H hH
  refine ⟨R, hR, hRB, hRlaw, hRcost.trans ?_⟩
  have hs : sqrt (τ / H) ≤ (C₀ + 1) * (1 / H + 1 + L) :=
    (sqrt_le_sqrt (div_le_div_of_nonneg_right hτC hH.le)).trans
      (sqrt_clock_le_additive C₀ H L hC₀ hH hL)
  calc sqrt (τ / H) * (1 + ‖u‖ + |b|) ≤ (C₀ + 1) * (1 / H + 1 + L) * (1 + ‖u‖ + |b|) :=
        mul_le_mul_of_nonneg_right hs (by positivity)
    _ = _ := by ring

/-- One constant, chosen before the clock and the duration, bounds the
realizations of every phase of a fixed valid list and of every cap
contraction whose clock is at most `C₀(1 + L)`. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, the norm estimate of `thm: targets.atoms`. -/
theorem exists_fixed_and_cap_realization_constant (i : Fin d) (F : List (AutonomousPhase d))
    (hF : ∀ P ∈ F, IsAutonomousPhase P ∧ P.1.B = 0) (Y : Idx N → SSphere d) (c C₀ : ℝ)
    (hc : 0 ≤ c) (hC₀ : 0 ≤ C₀) :
    ∃ C₁ : ℝ, 0 ≤ C₁ ∧ ∀ τ L : ℝ, 0 ≤ τ → 0 ≤ L → τ ≤ C₀ * (1 + L) →
      ∀ (l : List (Idx N)) (P : AutonomousPhase d),
        (P ∈ F ∨ P ∈ capContractionPhases i Y c τ l) → ∀ H : ℝ, 0 < H →
          ∃ R : AutonomousPhase d, IsAutonomousPhase R ∧ R.1.B = 0 ∧
            (∀ ρ : Perspective.ProbSphere d, R.2 ρ H = P.2 ρ 1) ∧
            parameterCost R.1 ≤ C₁ * (1 / H + 1 + L) := by
  obtain ⟨Cfix, hCfix⟩ : ∃ C : ℝ, C = (F.map fun P => parameterCost P.1).sum := ⟨_, rfl⟩
  have hterms : ∀ x ∈ F.map (fun P => parameterCost P.1), 0 ≤ x := by
    intro x hx
    obtain ⟨P, _, rfl⟩ := List.mem_map.mp hx
    exact parameterCost_nonneg P.1
  have hCfix0 : 0 ≤ Cfix := hCfix ▸ List.sum_nonneg hterms
  have hD : 0 ≤ (2 + c) * (C₀ + 1) := by positivity
  refine ⟨Cfix + (2 + c) * (C₀ + 1), add_nonneg hCfix0 hD, ?_⟩
  intro τ L hτ hL hτC l P hP H hH
  have hX : 0 ≤ 1 / H + 1 + L := by positivity
  rcases hP with hP | hP
  · obtain ⟨R, hR, hRB, hRlaw, hRcost⟩ :=
      exists_rescaled_phase_realization P (hF P hP).1 (hF P hP).2 H hH
    refine ⟨R, hR, hRB, hRlaw, hRcost.trans ?_⟩
    have hle : parameterCost P.1 ≤ Cfix :=
      hCfix ▸ List.single_le_sum hterms _ (List.mem_map_of_mem hP)
    calc parameterCost P.1 * (1 / H + 1) ≤ Cfix * (1 / H + 1 + L) :=
          mul_le_mul hle (by linarith) (by positivity) hCfix0
      _ ≤ (Cfix + (2 + c) * (C₀ + 1)) * (1 / H + 1 + L) :=
          mul_le_mul_of_nonneg_right (by linarith) hX
  · obtain ⟨k, _, rfl⟩ := List.mem_map.mp hP
    obtain ⟨R, hR, hRB, hRlaw, hRcost⟩ := exists_balanced_phase_realization_additive i (Y k)
      (-(Y k : EucSpace d)) (-c) τ C₀ L hτ hC₀ hL hτC H hH
    refine ⟨R, hR, hRB, hRlaw, hRcost.trans ?_⟩
    rw [norm_neg, mem_sphere_zero_iff_norm.mp (Y k).2, abs_neg, abs_of_nonneg hc]
    calc (1 + 1 + c) * (C₀ + 1) * (1 / H + 1 + L) = (2 + c) * (C₀ + 1) * (1 / H + 1 + L) := by
          ring
      _ ≤ (Cfix + (2 + c) * (C₀ + 1)) * (1 / H + 1 + L) :=
          mul_le_mul_of_nonneg_right (by linarith) hX

/-- A unit duration instantiates the rescaling bound for an actual
ridge phase. -/
example : parameterCost (rescalePhase (1 : ℝ)⁻¹ (ridgePhase (0 : Fin 2) 0 0 1)).1 ≤
    parameterCost (ridgePhase (0 : Fin 2) 0 0 1).1 * (1 / 1 + 1) :=
  parameterCost_rescalePhase_le _ 1 one_pos

/-- An actual ridge phase is valid with zero scores, and a unit duration
is positive: the hypotheses of the fixed realization. -/
example : IsAutonomousPhase (ridgePhase (0 : Fin 2) 0 0 1) ∧
    (ridgePhase (0 : Fin 2) 0 0 1).1.B = 0 ∧ (0 : ℝ) < 1 :=
  ⟨isAutonomousPhase_ridge _ _ _ _, rfl, one_pos⟩

/-- A zero clock, zero clock constants and a unit duration meet the
balanced and additive hypotheses; the fixed list of one actual ridge
phase meets the hypothesis of the common constant. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 * (1 + 0) ∧ (0 : ℝ) < 1 ∧
    ∀ P ∈ [ridgePhase (0 : Fin 2) 0 0 1], IsAutonomousPhase P ∧ P.1.B = 0 := by
  refine ⟨le_rfl, le_rfl, by norm_num, one_pos, ?_⟩
  intro P hP
  rw [List.mem_singleton.mp hP]
  exact ⟨isAutonomousPhase_ridge _ _ _ _, rfl⟩

end Transformer.Interpolation
