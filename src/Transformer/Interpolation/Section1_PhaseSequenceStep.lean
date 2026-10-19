module

public import Transformer.Interpolation.Section1_AutonomousPhases
public import Transformer.Interpolation.Section1_ControlPartitions

/-!
# Appending one actual autonomous phase at a prescribed horizon

ArXiv:2411.04551v3, §1 and §3.2, the composition of control phases.
An already proved original prefix ends at positive time S and continues
with its last constant field. Append the actual phase P until T>S,
scaling its outputs so that the new physical duration realizes unit
intrinsic time. The new control retains a proved constant final phase.

The explicit switching grid adds one piece, and the source B=0 constraint
is preserved. The generic original weak joining theorem supplies global
existence and uniqueness. Its prefix and tail identities prove the
exact new endpoint for every original weak solution. All hypotheses
concern the previously proved prefix or the specified original phase;
none presupposes well-posedness of the newly joined controls.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Append the original constant phase with its exact prescribed-time
output scaling. Source: arXiv:2411.04551v3, §1 and §3.2. -/
noncomputable def appendAutonomousPhase (S T : ℝ) (θ : TimeParams d)
    (P : AutonomousPhase d) : TimeParams d :=
  concatenate S θ (fun _ => (rescalePhase ((T - S)⁻¹) P).1)

/-- Before the switch the actual old parameters are retained exactly.
Source: arXiv:2411.04551v3, §1 and §3.2, control concatenation. -/
theorem appendAutonomousPhase_of_lt (S T : ℝ) (θ : TimeParams d) (P : AutonomousPhase d)
    (s : ℝ) (hs : s < S) : appendAutonomousPhase S T θ P s = θ s := by
  simp only [appendAutonomousPhase, concatenate, ite_eq_left hs]

/-- The new physical interval realizes exactly one original phase clock.
Source: arXiv:2411.04551v3, §3.2, prescribed-time rescaling. -/
theorem rescalePhase_unit_clock (P : AutonomousPhase d) (S T : ℝ) (hST : S < T)
    (ρ : Perspective.ProbSphere d) :
    (rescalePhase ((T - S)⁻¹) P).2 ρ (T - S) = P.2 ρ 1 := by
  change P.2 ρ ((T - S)⁻¹ * (T - S)) = P.2 ρ 1
  rw [inv_mul_cancel₀ (sub_pos.mpr hST).ne']

/-- After the new horizon the actual control retains its last constant
parameters. Source: arXiv:2411.04551v3, §3.2, composing phases. -/
theorem appendAutonomousPhase_tail (S T : ℝ) (θ : TimeParams d) (P : AutonomousPhase d)
    (hST : S ≤ T) (s : ℝ) (hs : T ≤ s) :
    appendAutonomousPhase S T θ P s = (rescalePhase ((T - S)⁻¹) P).1 :=
  concatenate_of_ge S θ _ s (hST.trans hs)

/-- The actual append adds one control piece.
Source: arXiv:2411.04551v3, §3.2, switch counting. -/
theorem piecewiseConstant_appendAutonomousPhase (S T : ℝ) (θ : TimeParams d)
    (P : AutonomousPhase d) (K : ℕ) (hST : S ≤ T) (hpc : PiecewiseConstant d θ S K) :
    PiecewiseConstant d (appendAutonomousPhase S T θ P) T (K + 1) :=
  piecewiseConstant_concatenate_constant θ _ S T K hST hpc

/-- The actual joined controls retain zero scores at every time.
Source: arXiv:2411.04551v3, §3, `eq: average.vf`. -/
theorem appendAutonomousPhase_B_zero (S T : ℝ) (θ : TimeParams d) (P : AutonomousPhase d)
    (hB : ∀ s : ℝ, (θ s).B = 0) (hPB : P.1.B = 0) (s : ℝ) :
    (appendAutonomousPhase S T θ P s).B = 0 := by
  by_cases hs : s < S
  · simpa only [appendAutonomousPhase, concatenate, ite_eq_left hs] using hB s
  · simpa only [appendAutonomousPhase, concatenate, ite_eq_right hs] using
      rescalePhase_B_zero P _ hPB

/-- Appending one original phase retains weak well-posedness, the final
constant tail and the exact composed endpoint for every original solution.
Source: arXiv:2411.04551v3, §1 and §3.2, control composition. -/
theorem appendAutonomousPhase_spec (S T : ℝ) (hS : 0 < S) (hST : S < T)
    (θ : TimeParams d) (Q : AutonomousPhase d) (hQ : IsAutonomousPhase Q)
    (htail : ∀ s : ℝ, S ≤ s → θ s = Q.1)
    (hex : ∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
      μ 0 = ρ ∧ cauchyPB d θ μ)
    (hu : ∀ μ ν : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ → cauchyPB d θ ν →
      μ 0 = ν 0 → μ = ν)
    (K : ℕ) (hpc : PiecewiseConstant d θ S K) (hB : ∀ s : ℝ, (θ s).B = 0)
    (Ψ : Perspective.ProbSphere d → Perspective.ProbSphere d)
    (hend : ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ → μ S = Ψ (μ 0))
    (P : AutonomousPhase d) (hP : IsAutonomousPhase P) (hPB : P.1.B = 0) :
    IsAutonomousPhase (rescalePhase ((T - S)⁻¹) P) ∧
      (rescalePhase ((T - S)⁻¹) P).1.B = 0 ∧
      (∀ s : ℝ, T ≤ s → appendAutonomousPhase S T θ P s = (rescalePhase ((T - S)⁻¹) P).1) ∧
      PiecewiseConstant d (appendAutonomousPhase S T θ P) T (K + 1) ∧
      (∀ s : ℝ, (appendAutonomousPhase S T θ P s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d (appendAutonomousPhase S T θ P) μ) ∧
      (∀ μ ν : ℝ → Perspective.ProbSphere d, cauchyPB d (appendAutonomousPhase S T θ P) μ →
        cauchyPB d (appendAutonomousPhase S T θ P) ν → μ 0 = ν 0 → μ = ν) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d (appendAutonomousPhase S T θ P) μ →
        μ T = P.2 (Ψ (μ 0)) 1 := by
  let R := rescalePhase ((T - S)⁻¹) P
  have hd : T - S ≠ 0 := (sub_pos.mpr hST).ne'
  have hR : IsAutonomousPhase R := hP.rescale P _ (inv_ne_zero hd)
  have hRB : R.1.B = 0 := rescalePhase_B_zero P _ hPB
  have hQex (ρ : Perspective.ProbSphere d) := (hQ.exists_unique Q ρ).exists
  have hRex (ρ : Perspective.ProbSphere d) := (hR.exists_unique R ρ).exists
  refine ⟨hR, hRB, appendAutonomousPhase_tail S T θ P hST.le,
    piecewiseConstant_appendAutonomousPhase S T θ P K hST.le hpc,
    appendAutonomousPhase_B_zero S T θ P hB hPB, ?_, ?_, ?_⟩
  · intro ρ
    exact exists_cauchyPB_concatenate S hS θ (fun _ => R.1) ρ (hex ρ) hRex
  · exact cauchyPB_concatenate_general_unique Q.1 S hS θ (fun _ => R.1) htail hQex hex hu hRex
      (hR.unique R)
  · intro μ hμ
    -- Reconstruct the proved old prefix from the actual original initial law.
    obtain ⟨η, hη0, hη⟩ := hex (μ 0)
    have hs := cauchyPB_concatenate_prefix_of_wellposed_tail Q.1 S hS θ (fun _ => R.1)
      htail hQex hu η μ hη hμ hη0.symm S le_rfl
    have hμS : μ S = Ψ (μ 0) := by rw [hs, hend η hη, hη0]
    -- The rescaled final law reaches exactly the selected unit-clock endpoint.
    have he := cauchyPB_concatenate_tail_of_unique S hS θ (fun _ => R.1) (hR.unique R)
      μ (R.2 (μ S)) hμ (hR.2.1 (μ S)) (hR.1 (μ S)) T hST.le
    rw [he, hμS]
    exact rescalePhase_unit_clock P S T hST (Ψ (μ 0))

/-- A nonzero ridge prefix and a rank-one attention phase realize
all induction inputs, with positive switch and final times, the actual
zero-score constraint and the true original prefix endpoint. -/
example : ∃ (Q P : AutonomousPhase 2) (θ : TimeParams 2)
    (Ψ : Perspective.ProbSphere 2 → Perspective.ProbSphere 2),
    IsAutonomousPhase Q ∧ IsAutonomousPhase P ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) < 3 ∧
    (∀ s : ℝ, 1 ≤ s → θ s = Q.1) ∧
    (∀ ρ : Perspective.ProbSphere 2, ∃ μ : ℝ → Perspective.ProbSphere 2,
      μ 0 = ρ ∧ cauchyPB 2 θ μ) ∧
    (∀ μ ν : ℝ → Perspective.ProbSphere 2, cauchyPB 2 θ μ → cauchyPB 2 θ ν →
      μ 0 = ν 0 → μ = ν) ∧
    PiecewiseConstant 2 θ 1 1 ∧ (∀ s : ℝ, (θ s).B = 0) ∧ P.1.B = 0 ∧
    ∀ μ : ℝ → Perspective.ProbSphere 2, cauchyPB 2 θ μ → μ 1 = Ψ (μ 0) := by
  let p : EucSpace 2 := basePoint 1
  let w : SSphere 2 := basePoint 1
  let Q : AutonomousPhase 2 := (reluRidgeParams (0 : Fin 2) p 0 1, ridgeLaw p 0 1)
  let P : AutonomousPhase 2 := (meanAxisParams (w : EucSpace 2) 1, meanAxisLaw w 1)
  have hQ : IsAutonomousPhase Q :=
    ⟨ridgeLaw_zero p 0 1, cauchyPB_ridgeLaw 0 p 0 1,
      fun μ hμ => cauchyPB_ridge_eq 0 p 0 1 (μ 0) μ hμ rfl⟩
  have hP : IsAutonomousPhase P :=
    ⟨meanAxisLaw_zero w 1, cauchyPB_meanAxisLaw w 1, cauchyPB_meanAxis_eq w 1⟩
  refine ⟨Q, P, (fun _ => Q.1), (fun ρ => Q.2 ρ 1), hQ, hP,
    zero_lt_one, by norm_num, (fun _ _ => rfl),
    (fun ρ => (hQ.exists_unique Q ρ).exists), hQ.unique Q,
    piecewiseConstant_constant Q.1 1 zero_le_one, (fun _ => rfl), rfl, ?_⟩
  intro μ hμ
  exact congrFun (hQ.2.2 μ hμ) 1

end Transformer.Interpolation
