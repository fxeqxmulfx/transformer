module

public import Transformer.Interpolation.Section2_LocalPhases

/-!
# Perceptron control curves for finite phase sequences

ArXiv:2411.04551v3, §2.2, `prop: compression`, whose controls have
`𝐕 ≡ 0`. The construction of `exists_phase_sequence_controls` joins
the phases one at a time at the half horizon and scales the outputs of
each appended phase; the scaling multiplies `𝐕` by the clock factor,
so phases with `𝐕 = 0` give a curve with `𝐕 ≡ 0`. The zero ridge of
the induction base has `𝐕 = 0` too.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Output scaling keeps a zero attention value matrix. Source auxiliary:
arXiv:2411.04551v3, §2.2, `𝐕 ≡ 0`. -/
theorem rescalePhase_V_zero (P : AutonomousPhase d) (κ : ℝ) (hV : P.1.V = 0) :
    (rescalePhase κ P).1.V = 0 := by
  change κ • P.1.V = 0
  rw [hV]
  exact smul_zero (A := ParamMatrix d) κ

/-- Appending a phase with zero attention to a curve with zero attention
keeps zero attention. Source auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem appendAutonomousPhase_V_zero (S T : ℝ) (θ : TimeParams d) (P : AutonomousPhase d)
    (hV : ∀ s : ℝ, (θ s).V = 0) (hPV : P.1.V = 0) (s : ℝ) :
    (appendAutonomousPhase S T θ P s).V = 0 := by
  by_cases hs : s < S
  · simpa only [appendAutonomousPhase, concatenate, ite_eq_left hs] using hV s
  · simpa only [appendAutonomousPhase, concatenate, ite_eq_right hs] using
      rescalePhase_V_zero P _ hPV

/-- Any finite sequence of proved perceptron phases has piecewise-constant
controls with `𝐕 ≡ 0` and `𝐁 ≡ 0`, global weak existence and uniqueness,
and its exact endpoint at the prescribed time. Source auxiliary:
arXiv:2411.04551v3, §2.2, `prop: compression`. -/
theorem exists_perceptron_phase_sequence_controls (i : Fin d) (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0 ∧ P.1.V = 0) (T : ℝ) (hT : 0 < T) :
    ∃ (Q : AutonomousPhase d) (θ : TimeParams d),
      IsAutonomousPhase Q ∧ Q.1.B = 0 ∧
      (∀ s : ℝ, T ≤ s → θ s = Q.1) ∧
      PiecewiseConstant d θ T (L.length + 1) ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      (∀ s : ℝ, (θ s).V = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ μ ν : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ → cauchyPB d θ ν →
        μ 0 = ν 0 → μ = ν) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        μ T = phaseSequenceLaw L (μ 0) := by
  revert hL
  induction L using List.reverseRecOn generalizing T with
  | nil =>
    intro _
    let Q : AutonomousPhase d := (reluRidgeParams i 0 0 0, ridgeLaw 0 0 0)
    have hQ : IsAutonomousPhase Q :=
      ⟨ridgeLaw_zero 0 0 0, cauchyPB_ridgeLaw i 0 0 0,
        fun μ hμ => cauchyPB_ridge_eq i 0 0 0 (μ 0) μ hμ rfl⟩
    have hzero (ρ : Perspective.ProbSphere d) (t : ℝ) : Q.2 ρ t = ρ :=
      ridgeLaw_eq_of_inactive_ae 0 0 0 ρ (ae_of_all _ fun _ => by simp) t
    refine ⟨Q, (fun _ => Q.1), hQ, rfl, (fun _ _ => rfl),
      piecewiseConstant_constant Q.1 T hT.le, (fun _ => rfl), (fun _ => rfl),
      (fun ρ => (hQ.exists_unique Q ρ).exists), hQ.unique Q, ?_⟩
    intro μ hμ
    exact (congrFun (hQ.2.2 μ hμ) T).trans (hzero _ T)
  | append_singleton L P ih =>
    intro hL
    have hprefix (R : AutonomousPhase d) (hR : R ∈ L) :
        IsAutonomousPhase R ∧ R.1.B = 0 ∧ R.1.V = 0 :=
      hL R (List.mem_append.mpr (Or.inl hR))
    have hP : IsAutonomousPhase P ∧ P.1.B = 0 ∧ P.1.V = 0 :=
      hL P (List.mem_append.mpr (Or.inr (List.mem_singleton.mpr rfl)))
    let S := T / 2
    have hS : 0 < S := half_pos hT
    have hST : S < T := half_lt_self hT
    obtain ⟨Q, θ, hQ, hQB, htail, hpc, hB, hV, hex, hu, hend⟩ := ih S hS hprefix
    obtain ⟨hR, hRB, htail', hpc', hB', hex', hu', hend'⟩ :=
      appendAutonomousPhase_spec S T hS hST
      θ Q hQ htail hex hu (L.length + 1) hpc hB (phaseSequenceLaw L) hend P hP.1 hP.2.1
    refine ⟨rescalePhase ((T - S)⁻¹) P, appendAutonomousPhase S T θ P,
      hR, hRB, htail', ?_, hB', appendAutonomousPhase_V_zero S T θ P hV hP.2.2, hex', hu', ?_⟩
    · simpa only [List.length_append, List.length_singleton] using hpc'
    · intro μ hμ
      rw [phaseSequenceLaw_append, phaseSequenceLaw_cons, phaseSequenceLaw_nil]
      exact hend' μ hμ

/-- One proved ridge phase, which has zero attention, is realized at unit
horizon with two pieces. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2) (μ : ℝ → Perspective.ProbSphere 2),
    PiecewiseConstant 2 θ 1 2 ∧ (∀ s : ℝ, (θ s).V = 0) ∧ μ 0 = ρ ∧ cauchyPB 2 θ μ := by
  have hL : ∀ P ∈ [ridgePhase (0 : Fin 2) (basePoint 1 : EucSpace 2) 0 1],
      IsAutonomousPhase P ∧ P.1.B = 0 ∧ P.1.V = 0 := by
    intro P hP
    rw [List.mem_singleton.mp hP]
    exact ⟨isAutonomousPhase_ridge _ _ _ _, rfl, rfl⟩
  obtain ⟨_, θ, -, -, -, hpc, -, hV, hex, -, -⟩ :=
    exists_perceptron_phase_sequence_controls (0 : Fin 2) _ hL 1 one_pos
  obtain ⟨μ, hμ0, hμ⟩ := hex ρ
  exact ⟨θ, μ, hpc, hV, hμ0, hμ⟩

end Transformer.Interpolation
