module

public import Transformer.Interpolation.Section3_QuantitativeCenterClocks
public import Transformer.Interpolation.Section3_PhaseLawMaps

/-!
# Bounded physical realization of actual original phase sequences

ArXiv:2411.04551v3, §1 and §5, assembling the controls for
`thm: targets.atoms`. Each original endpoint map is assumed to have
an actual autonomous realization at every positive physical duration,
with a proved common additive inverse-time and accuracy cost.

Appending those actual realizations preserves their exact endpoint
maps. A half-horizon induction gives at most one additional constant
piece per phase and a finite explicit cost factor. The construction
uses the actual global weak joining theorem, with its existence,
uniqueness and constant final tail. No well-posedness of joined
controls is assumed as an input.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The source's complete sum norm on actual network parameters.
Source: arXiv:2411.04551v3, §1 and §5, `thm: targets.atoms`.
This function includes all four operators and the actual bias vector. -/
noncomputable def parameterCost (P : Params d) : ℝ :=
  ‖P.V‖ + ‖P.B‖ + ‖P.W‖ + ‖P.U‖ + ‖P.b‖

/-- Every actual parameter cost is nonnegative. Source auxiliary:
arXiv:2411.04551v3, §5, the complete norm estimate. -/
theorem parameterCost_nonneg (P : Params d) : 0 ≤ parameterCost P := by
  unfold parameterCost
  positivity

/-- A duration followed by its reciprocal recovers the actual phase
parameters exactly. Source: arXiv:2411.04551v3, §1 and §5,
physical realization of each original autonomous clock. -/
theorem rescalePhase_inverse_params (P : AutonomousPhase d) (H : ℝ) (hH : H ≠ 0) :
    (rescalePhase H⁻¹ (rescalePhase H P)).1 = P.1 := by
  simp only [rescalePhase, rescaleTimeParams, smul_smul, inv_mul_cancel₀ hH, one_smul]

/-- A finite list of original endpoint maps with actual bounded
physical realizations has bounded piecewise constant controls.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, composing the
fixed preprocessing and quantitative concentration stages. -/
theorem exists_bounded_phase_sequence_controls (i : Fin d) (L : List (AutonomousPhase d))
    (C A : ℝ) (hC : 0 ≤ C) (hA : 0 ≤ A)
    (hreal : ∀ P ∈ L, ∀ H : ℝ, 0 < H → ∃ R : AutonomousPhase d,
      IsAutonomousPhase R ∧ R.1.B = 0 ∧
        (∀ ρ : Perspective.ProbSphere d, R.2 ρ H = P.2 ρ 1) ∧
        parameterCost R.1 ≤ C * (1 / H + 1 + A))
    (T : ℝ) (hT : 0 < T) :
    ∃ (Q : AutonomousPhase d) (θ : TimeParams d),
      IsAutonomousPhase Q ∧ Q.1.B = 0 ∧
      (∀ s : ℝ, T ≤ s → θ s = Q.1) ∧
      PiecewiseConstant d θ T (L.length + 1) ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      (∀ s : ℝ, parameterCost (θ s) ≤ (2 : ℝ) ^ L.length * C * (1 / T + 1 + A)) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ μ ν : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ → cauchyPB d θ ν →
        μ 0 = ν 0 → μ = ν) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        μ T = phaseSequenceLaw L (μ 0) := by
  revert hreal
  induction L using List.reverseRecOn generalizing T with
  | nil =>
    intro _
    let Q := ridgePhase i 0 0 0
    have hQ : IsAutonomousPhase Q := isAutonomousPhase_ridge i 0 0 0
    have hzero (ρ : Perspective.ProbSphere d) (t : ℝ) : Q.2 ρ t = ρ :=
      ridgeLaw_eq_of_inactive_ae 0 0 0 ρ (ae_of_all _ fun _ => by simp) t
    refine ⟨Q, (fun _ => Q.1), hQ, rfl, (fun _ _ => rfl),
      piecewiseConstant_constant Q.1 T hT.le, (fun _ => rfl), ?_,
      (fun ρ => (hQ.exists_unique Q ρ).exists), hQ.unique Q, ?_⟩
    · intro s
      have hz : parameterCost Q.1 ≤ 0 := by
        simpa only [parameterCost, Q, ridgePhase, norm_zero, zero_add, add_zero, abs_zero] using
          norm_reluRidgeParams_sum_le i (0 : EucSpace d) 0 0
      exact hz.trans (by positivity)
    · intro μ hμ
      exact (congrFun (hQ.2.2 μ hμ) T).trans (hzero _ T)
  | append_singleton L P ih =>
    intro hreal
    let S := T / 2
    let H := T - S
    have hS : 0 < S := half_pos hT
    have hST : S < T := half_lt_self hT
    have hH : 0 < H := sub_pos.mpr hST
    have hprefix := fun R hR => hreal R (List.mem_append.mpr (Or.inl hR))
    obtain ⟨Q, θ, hQ, _, htail, hpc, hB, hnorm, hex, hu, hend⟩ := ih S hS hprefix
    obtain ⟨R, hR, hRB, hRend, hRnorm⟩ :=
      hreal P (List.mem_append.mpr (Or.inr (List.mem_singleton.mpr rfl))) H hH
    let P' := rescalePhase H R
    have hP' : IsAutonomousPhase P' := hR.rescale R H hH.ne'
    have hP'B : P'.1.B = 0 := rescalePhase_B_zero R H hRB
    obtain ⟨hfinal, hfinalB, htail', hpc', hB', hex', hu', hend'⟩ :=
      appendAutonomousPhase_spec S T hS hST θ Q hQ htail hex hu
        (L.length + 1) hpc hB (phaseSequenceLaw L) hend P' hP' hP'B
    have hparams : (rescalePhase ((T - S)⁻¹) P').1 = R.1 :=
      rescalePhase_inverse_params R H hH.ne'
    have hunit (ρ : Perspective.ProbSphere d) : P'.2 ρ 1 = P.2 ρ 1 := by
      change R.2 ρ (H * 1) = P.2 ρ 1
      rw [mul_one, hRend]
    have hscaleS : 1 / S + 1 + A ≤ 2 * (1 / T + 1 + A) := by
      have he : 1 / S = 2 * (1 / T) := by dsimp only [S]; ring
      rw [he]
      linarith
    have hscaleH : 1 / H + 1 + A ≤ 2 * (1 / T + 1 + A) := by
      have he : H = S := by dsimp only [H, S]; ring
      rw [he]
      exact hscaleS
    have hpow : (1 : ℝ) ≤ 2 ^ L.length := one_le_pow₀ (by norm_num)
    refine ⟨rescalePhase ((T - S)⁻¹) P', appendAutonomousPhase S T θ P',
      hfinal, hfinalB, htail', ?_, hB', ?_, hex', hu', ?_⟩
    · simpa only [List.length_append, List.length_singleton] using hpc'
    · intro s
      have hcost : parameterCost (appendAutonomousPhase S T θ P' s) ≤
          2 ^ L.length * C * (2 * (1 / T + 1 + A)) := by
        by_cases hs : s < S
        · rw [appendAutonomousPhase_of_lt S T θ P' s hs]
          exact (hnorm s).trans (mul_le_mul_of_nonneg_left hscaleS (by positivity))
        · have he : appendAutonomousPhase S T θ P' s = R.1 := by
            change (concatenate S θ (fun _ => (rescalePhase ((T - S)⁻¹) P').1)) s = _
            rw [concatenate_of_ge S θ _ s (not_lt.mp hs), hparams]
          rw [he]
          have hb := hRnorm.trans (mul_le_mul_of_nonneg_left hscaleH hC)
          exact hb.trans (by nlinarith [mul_nonneg hC (by positivity : 0 ≤ 2 * (1 / T + 1 + A))])
      simpa only [List.length_append, List.length_singleton, pow_succ, mul_assoc,
        mul_left_comm C (2 : ℝ)] using hcost
    · intro μ hμ
      rw [phaseSequenceLaw_append, phaseSequenceLaw_cons, phaseSequenceLaw_nil]
      exact (hend' μ hμ).trans (hunit _)

/-- A genuine projection phase has actual bounded realizations at
every positive duration, so the finite-sequence hypothesis has a
nonempty instance. Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
example : ∃ L : List (AutonomousPhase 2), L ≠ [] ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ ∀ P ∈ L, ∀ H : ℝ, 0 < H → ∃ R : AutonomousPhase 2,
      IsAutonomousPhase R ∧ R.1.B = 0 ∧
      (∀ ρ : Perspective.ProbSphere 2, R.2 ρ H = P.2 ρ 1) ∧
      parameterCost R.1 ≤ 2 * (1 / H + 1 + (0 : ℝ)) := by
  let w := basePoint 1
  let P := meanAxisPhase w 1
  refine ⟨[P], by simp, by norm_num, le_rfl, one_pos, ?_⟩
  intro Q hQ H hH
  have he : Q = P := List.mem_singleton.mp hQ
  subst Q
  let R := meanAxisPhase w (1 / H)
  refine ⟨R, isAutonomousPhase_meanAxis w _, rfl, ?_, ?_⟩
  · intro ρ
    change meanAxisLaw w (1 / H) ρ H = meanAxisLaw w 1 ρ 1
    rw [meanAxisLaw_scale, div_mul_cancel₀ _ hH.ne', meanAxisLaw_scale, one_mul]
  · have hn := norm_meanAxisParams_sum_le w (1 / H)
    have hpos : 0 ≤ (1 : ℝ) / H := div_nonneg zero_le_one hH.le
    change parameterCost (meanAxisParams (w : EucSpace 2) (1 / H)) ≤ _
    rw [abs_of_nonneg hpos] at hn
    exact hn.trans (by linarith)

end Transformer.Interpolation
