module

public import Transformer.Interpolation.Section1_GeneralPhaseUniqueness
public import Transformer.Interpolation.Section3_RidgeWellPosed

/-!
# Exact autonomous phases of the original weak equation

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`, and the
successive control phases of §3.2. Phase data consist of actual constant
network parameters and an explicitly specified probability law.
The predicate below relates these two arguments by the original weak
equation, initial condition and global law identification.

Its premises are proved for concrete attention and perceptron phases;
no well-posedness is assumed for their subsequent joined controls.
Nonzero clock rescaling preserves this relation. Time translation and
actual weak uniqueness give the group and inverse identities for every
such phase, and hence bijectivity of each finite composition.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual constant network parameters paired with their proposed law.
Source: arXiv:2411.04551v3, §1, the original Cauchy problem. -/
abbrev AutonomousPhase (d : ℕ) :=
  Params d × (Perspective.ProbSphere d → ℝ → Perspective.ProbSphere d)

/-- Predicate of parameter-and-law data: the stated law starts correctly,
solves the original weak equation and identifies every original solution.
Source: arXiv:2411.04551v3, §1, well-posed autonomous phases. -/
def IsAutonomousPhase (P : AutonomousPhase d) : Prop :=
  (∀ ρ : Perspective.ProbSphere d, P.2 ρ 0 = ρ) ∧
  (∀ ρ : Perspective.ProbSphere d, cauchyPB d (fun _ => P.1) (P.2 ρ)) ∧
  ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d (fun _ => P.1) μ → μ = P.2 (μ 0)

/-- Actual output-matrix scaling paired with the exact changed law clock.
Source: arXiv:2411.04551v3, §1 and §3.2, time rescaling. -/
noncomputable def rescalePhase (κ : ℝ) (P : AutonomousPhase d) : AutonomousPhase d :=
  (rescaleTimeParams κ (fun _ => P.1) 0, fun ρ t => P.2 ρ (κ * t))

/-- Valid phase data supply original global weak existence and uniqueness.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem IsAutonomousPhase.exists_unique (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (ρ : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d (fun _ => P.1) μ := by
  refine ⟨P.2 ρ, ⟨hP.1 ρ, hP.2.1 ρ⟩, fun μ hμ => ?_⟩
  rw [hP.2.2 μ hμ.2, hμ.1]

/-- The original solution is unique for fixed initial data.
Source: arXiv:2411.04551v3, §1, the autonomous Cauchy problem. -/
theorem IsAutonomousPhase.unique (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => P.1) μ)
    (hν : cauchyPB d (fun _ => P.1) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  rw [hP.2.2 μ hμ, hP.2.2 ν hν, h0]

/-- Nonzero clock rescaling preserves actual original phase validity.
Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem IsAutonomousPhase.rescale (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (κ : ℝ) (hκ : κ ≠ 0) : IsAutonomousPhase (rescalePhase κ P) := by
  have hθ : rescaleTimeParams κ (fun _ => P.1) = fun _ => (rescalePhase κ P).1 := rfl
  refine ⟨fun ρ => by simpa only [rescalePhase, mul_zero] using hP.1 ρ, ?_, ?_⟩
  · intro ρ
    have hh := cauchyPB_rescaleTimeParams κ hκ (fun _ => P.1) (P.2 ρ) (hP.2.1 ρ)
    rwa [hθ] at hh
  · intro μ hμ
    have hback := cauchyPB_unscaleTimeParams κ hκ (fun _ => P.1) μ (by rwa [hθ])
    have he := hP.2.2 _ hback
    funext t
    have hh := congrFun he (κ * t)
    simpa only [rescalePhase, inv_mul_cancel_left₀ hκ, mul_zero] using hh

/-- The source zero-score constraint survives exact output scaling.
Source: arXiv:2411.04551v3, §3, `eq: average.vf`. -/
theorem rescalePhase_B_zero (P : AutonomousPhase d) (κ : ℝ) (hB : P.1.B = 0) :
    (rescalePhase κ P).1.B = 0 := hB

/-- Actual autonomous law maps compose with their true elapsed times.
Source: arXiv:2411.04551v3, §1 and §3.2, solution maps. -/
theorem IsAutonomousPhase.law_add (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (ρ : Perspective.ProbSphere d) (s t : ℝ) : P.2 ρ (s + t) = P.2 (P.2 ρ s) t := by
  have hh : cauchyPB d (fun _ => P.1) (fun r => P.2 ρ (s + r)) :=
    cauchyPB_translateTimeParams s (fun _ => P.1) (P.2 ρ) (hP.2.1 ρ)
  have he := congrFun (hP.2.2 _ hh) t
  simpa only [add_zero] using he

/-- The negative clock inverts the actual original probability map.
Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem IsAutonomousPhase.law_inverse (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (ρ : Perspective.ProbSphere d) (t : ℝ) : P.2 (P.2 ρ t) (-t) = ρ := by
  rw [← hP.law_add P ρ t (-t), add_neg_cancel, hP.1]

/-- Every complete original phase preserves distinct probabilities.
Source: arXiv:2411.04551v3, §1 and §3.2, invertibility. -/
theorem IsAutonomousPhase.bijective_law (P : AutonomousPhase d) (hP : IsAutonomousPhase P)
    (t : ℝ) : Function.Bijective (fun ρ => P.2 ρ t) := by
  have hl : Function.LeftInverse (fun ρ => P.2 ρ (-t)) (fun ρ => P.2 ρ t) :=
    fun ρ => hP.law_inverse P ρ t
  refine ⟨hl.injective, fun ρ => ⟨P.2 ρ (-t), ?_⟩⟩
  simpa only [neg_neg] using hP.law_inverse P ρ (-t)

/-- Compose the specified actual phase laws at unit intrinsic clocks.
Source: arXiv:2411.04551v3, §3.2, successive control phases. -/
noncomputable def phaseSequenceLaw (L : List (AutonomousPhase d))
    (ρ : Perspective.ProbSphere d) : Perspective.ProbSphere d :=
  L.foldl (fun ν P => P.2 ν 1) ρ

/-- The empty phase sequence acts as the identity.
Source: arXiv:2411.04551v3, §3.2, the induction base. -/
theorem phaseSequenceLaw_nil (ρ : Perspective.ProbSphere d) : phaseSequenceLaw [] ρ = ρ := rfl

/-- The first original law acts before the rest of the sequence.
Source: arXiv:2411.04551v3, §3.2, composing solution maps. -/
theorem phaseSequenceLaw_cons (P : AutonomousPhase d) (L : List (AutonomousPhase d))
    (ρ : Perspective.ProbSphere d) : phaseSequenceLaw (P :: L) ρ = phaseSequenceLaw L (P.2 ρ 1) := rfl

/-- Appending phases composes their actual probability maps.
Source: arXiv:2411.04551v3, §3.2, composing controls. -/
theorem phaseSequenceLaw_append (L M : List (AutonomousPhase d)) (ρ : Perspective.ProbSphere d) :
    phaseSequenceLaw (L ++ M) ρ = phaseSequenceLaw M (phaseSequenceLaw L ρ) := List.foldl_append

/-- A finite sequence of proved original phases is bijective on laws.
Source: arXiv:2411.04551v3, §1 and §3.2, invertible solution maps. -/
theorem bijective_phaseSequenceLaw (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P) : Function.Bijective (phaseSequenceLaw L) := by
  induction L with
  | nil => exact Function.bijective_id
  | cons P L ih =>
    have hP := hL P List.mem_cons_self
    have ht := ih (fun Q hQ => hL Q (List.mem_cons_of_mem _ hQ))
    exact ht.comp (hP.bijective_law P 1)

/-- A nonzero actual projection neuron, an off-axis atom and a nonempty
phase list jointly realize phase validity, the zero-score constraint,
nonzero rescaling and all finite-composition premises. -/
example : ∃ (P : AutonomousPhase 2) (L : List (AutonomousPhase 2))
    (ρ : Perspective.ProbSphere 2) (μ : ℝ → Perspective.ProbSphere 2),
    IsAutonomousPhase P ∧ P.1.B = 0 ∧ (2 : ℝ) ≠ 0 ∧ L = [P] ∧
    (∀ Q ∈ L, IsAutonomousPhase Q) ∧ cauchyPB 2 (fun _ => P.1) μ ∧ μ 0 = ρ := by
  let p : EucSpace 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let ρ := Perspective.diracProb 2 z
  let P : AutonomousPhase 2 := (reluRidgeParams (0 : Fin 2) p 0 1, ridgeLaw p 0 1)
  have hP : IsAutonomousPhase P :=
    ⟨ridgeLaw_zero p 0 1, cauchyPB_ridgeLaw 0 p 0 1,
      fun μ hμ => cauchyPB_ridge_eq 0 p 0 1 (μ 0) μ hμ rfl⟩
  refine ⟨P, [P], ρ, P.2 ρ, hP, rfl, by norm_num, rfl, ?_, hP.2.1 ρ, hP.1 ρ⟩
  intro Q hQ
  rw [List.mem_singleton] at hQ
  subst Q
  exact hP

end Transformer.Interpolation
