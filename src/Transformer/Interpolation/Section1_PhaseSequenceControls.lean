module

public import Transformer.Interpolation.Section1_PhaseSequenceStep
public import Mathlib.Data.List.Induction

/-!
# Actual control curves realizing finite autonomous phase sequences

ArXiv:2411.04551v3, §1 and §3.2, successive perceptron and
attention controls. The concrete phase constructors below pair the
original network parameters with their already proved actual weak laws.
Both have zero score matrix and global original law identification.

Every finite sequence of such proved phases is realized at any positive
prescribed horizon. A zero initial phase supplies the induction base.
Appending a phase scales its actual outputs to retain its unit intrinsic
clock, gives the exact endpoint for every original weak solution and
adds one constant piece. The resulting bound is list length plus one.
Global original weak existence and uniqueness and the actual final
constant tail are retained throughout the construction.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual single-neuron parameters and their proved spherical law.
Source: arXiv:2411.04551v3, §1 and §3.2, the perceptron phase. -/
noncomputable def ridgePhase (i : Fin d) (p u : EucSpace d) (b : ℝ) : AutonomousPhase d :=
  (reluRidgeParams i p u b, ridgeLaw p u b)

/-- Every original ridge phase has the stated original weak law.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2. -/
theorem isAutonomousPhase_ridge (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    IsAutonomousPhase (ridgePhase i p u b) :=
  ⟨ridgeLaw_zero p u b, cauchyPB_ridgeLaw i p u b,
    fun μ hμ => cauchyPB_ridge_eq i p u b (μ 0) μ hμ rfl⟩

/-- Original perceptron phase data have zero score matrix.
Source: arXiv:2411.04551v3, §3, `eq: average.vf`. -/
theorem ridgePhase_B_zero (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    (ridgePhase i p u b).1.B = 0 := rfl

/-- Actual rank-one attention parameters paired with their nonlinear law.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
noncomputable def meanAxisPhase (w : SSphere d) (κ : ℝ) : AutonomousPhase d :=
  (meanAxisParams (w : EucSpace d) κ, meanAxisLaw w κ)

/-- Every original rank-one phase has the stated nonlinear weak law.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem isAutonomousPhase_meanAxis (w : SSphere d) (κ : ℝ) :
    IsAutonomousPhase (meanAxisPhase w κ) :=
  ⟨meanAxisLaw_zero w κ, cauchyPB_meanAxisLaw w κ, cauchyPB_meanAxis_eq w κ⟩

/-- Original rank-one phase data have zero score matrix.
Source: arXiv:2411.04551v3, §3, `eq: average.vf`. -/
theorem meanAxisPhase_B_zero (w : SSphere d) (κ : ℝ) :
    (meanAxisPhase w κ).1.B = 0 := rfl

/-- Any finite sequence of proved original autonomous phases has actual
controls, global weak existence and uniqueness, and its exact endpoint
at the prescribed time. Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem exists_phase_sequence_controls (i : Fin d) (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) (T : ℝ) (hT : 0 < T) :
    ∃ (Q : AutonomousPhase d) (θ : TimeParams d),
      IsAutonomousPhase Q ∧ Q.1.B = 0 ∧
      (∀ s : ℝ, T ≤ s → θ s = Q.1) ∧
      PiecewiseConstant d θ T (L.length + 1) ∧ (∀ s : ℝ, (θ s).B = 0) ∧
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
      piecewiseConstant_constant Q.1 T hT.le, (fun _ => rfl),
      (fun ρ => (hQ.exists_unique Q ρ).exists), hQ.unique Q, ?_⟩
    intro μ hμ
    exact (congrFun (hQ.2.2 μ hμ) T).trans (hzero _ T)
  | append_singleton L P ih =>
    intro hL
    have hprefix (R : AutonomousPhase d) (hR : R ∈ L) :
        IsAutonomousPhase R ∧ R.1.B = 0 :=
      hL R (List.mem_append.mpr (Or.inl hR))
    have hP : IsAutonomousPhase P ∧ P.1.B = 0 :=
      hL P (List.mem_append.mpr (Or.inr (List.mem_singleton.mpr rfl)))
    let S := T / 2
    have hS : 0 < S := half_pos hT
    have hST : S < T := half_lt_self hT
    obtain ⟨Q, θ, hQ, hQB, htail, hpc, hB, hex, hu, hend⟩ :=
      ih S hS hprefix
    obtain ⟨hR, hRB, htail', hpc', hB', hex', hu', hend'⟩ :=
      appendAutonomousPhase_spec S T hS hST
      θ Q hQ htail hex hu (L.length + 1) hpc hB (phaseSequenceLaw L) hend P hP.1 hP.2
    refine ⟨rescalePhase ((T - S)⁻¹) P, appendAutonomousPhase S T θ P,
      hR, hRB, htail', ?_, hB', hex', hu', ?_⟩
    · simpa only [List.length_append, List.length_singleton] using hpc'
    · intro μ hμ
      rw [phaseSequenceLaw_append, phaseSequenceLaw_cons, phaseSequenceLaw_nil]
      exact hend' μ hμ

/-- Realize the finite actual solution-map composition with zero scores
and at most one extra constant piece. Every original weak endpoint is
identified. Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem exists_phase_sequence_realization (i : Fin d) (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) (T : ℝ) (hT : 0 < T) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T (L.length + 1) ∧
      (∀ s : ℝ, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        μ T = phaseSequenceLaw L (μ 0) := by
  obtain ⟨Q, θ, -, -, -, hpc, hB, hex, -, hend⟩ := exists_phase_sequence_controls i L hL T hT
  exact ⟨θ, hpc, hB, hex, hend⟩

/-- A genuine ridge followed by nonlinear rank-one attention is
realized by actual zero-score controls with three pieces, global original
weak existence and the exact endpoint for every original solution. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2)
    (μ : ℝ → Perspective.ProbSphere 2),
    PiecewiseConstant 2 θ 1 3 ∧ (∀ s : ℝ, (θ s).B = 0) ∧
    μ 0 = ρ ∧ cauchyPB 2 θ μ ∧
    ∀ ν : ℝ → Perspective.ProbSphere 2, cauchyPB 2 θ ν →
      ν 1 = meanAxisLaw (basePoint 1) 1
        (ridgeLaw (basePoint 1 : EucSpace 2) 0 1 (ν 0) 1) 1 := by
  let p : EucSpace 2 := basePoint 1
  let w : SSphere 2 := basePoint 1
  let L := [ridgePhase (0 : Fin 2) p 0 1, meanAxisPhase w 1]
  have hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0 := by
    intro P hP
    simp only [L, List.mem_cons, List.not_mem_nil, or_false] at hP
    rcases hP with rfl | rfl
    · exact ⟨isAutonomousPhase_ridge 0 p 0 1, ridgePhase_B_zero 0 p 0 1⟩
    · exact ⟨isAutonomousPhase_meanAxis w 1, meanAxisPhase_B_zero w 1⟩
  obtain ⟨θ, hpc, hB, hex, hend⟩ :=
    exists_phase_sequence_realization (0 : Fin 2) L hL 1 zero_lt_one
  obtain ⟨μ, hμ0, hμ⟩ := hex ρ
  exact ⟨θ, μ, hpc, hB, hμ0, hμ, hend⟩

end Transformer.Interpolation
