module

public import Transformer.Interpolation.Section3_BallFlow
public import Transformer.Interpolation.Section1_ConcatenatedPrimitives
public import Transformer.Interpolation.Section3_TwoRidgeScaling

/-!
# Joining actual particle flows at an original neural control switch

ArXiv:2411.04551v3, §4, `eq: neural.ode.sphere`, and Appendix,
the successive phases of `lem: tubular.mass.movement`.

The second particle flow starts from the first flow's switching value.
Joining the original matrices and biases with the same elapsed clock
joins their actual velocities along these trajectories. The integrated
particle equations match at the switch and the vector primitive theorem
then gives the original integrated equation for the complete joined flow.

On a prescribed nonnegative interval, the joined map is Lipschitz and
bijective. The two fixed complements combine to the complement of their
union, as required for the source's chain of supported ball transports.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual second flow acts on the first flow's switching value.
Source: arXiv:2411.04551v3, Appendix, successive applications of
`lem: two.balls` in `lem: tubular.mass.movement`. -/
noncomputable def concatenateFlow (S : ℝ) (φ₁ φ₂ : ℝ → SSphere d → SSphere d)
    (t : ℝ) (x : SSphere d) : SSphere d :=
  concatenate S (fun s => φ₁ s x) (fun s => φ₂ s (φ₁ S x)) t

/-- Joining the actual controls and flow joins their original velocities
with the same clock. Source: arXiv:2411.04551v3, §4,
`eq: neural.ode.sphere`, and Appendix, `lem: tubular.mass.movement`. -/
theorem neuralVF_concatenate (S : ℝ) (W₁ U₁ W₂ U₂ : ℝ → ParamMatrix d)
    (b₁ b₂ : ℝ → EucSpace d) (φ₁ φ₂ : ℝ → SSphere d → SSphere d) (x : SSphere d) (t : ℝ) :
    neuralVF d (concatenate S W₁ W₂) (concatenate S U₁ U₂) (concatenate S b₁ b₂)
      t (concatenateFlow S φ₁ φ₂ t x) =
    concatenate S (fun s => neuralVF d W₁ U₁ b₁ s (φ₁ s x))
      (fun s => neuralVF d W₂ U₂ b₂ s (φ₂ s (φ₁ S x))) t := by
  by_cases ht : t < S <;>
    simp only [neuralVF, concatenateFlow, concatenate, ht, ite_true, ite_false]

/-- Before the switch, the joined point is exactly the first flow's
point. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the first phase. -/
theorem concatenateFlow_of_lt (S : ℝ) (φ₁ φ₂ : ℝ → SSphere d → SSphere d)
    (t : ℝ) (ht : t < S) (x : SSphere d) : concatenateFlow S φ₁ φ₂ t x = φ₁ t x :=
  concatenate_of_lt S _ _ t ht

/-- After the switch the second flow uses elapsed time and the first
endpoint. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the appended phase. -/
theorem concatenateFlow_of_ge (S : ℝ) (φ₁ φ₂ : ℝ → SSphere d → SSphere d)
    (t : ℝ) (ht : S ≤ t) (x : SSphere d) :
    concatenateFlow S φ₁ φ₂ t x = φ₂ (t - S) (φ₁ S x) :=
  concatenate_of_ge S _ _ t ht

/-- The joined particle curve solves the actual original integrated
neural equation, including at and across the switch. Source:
arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem isNeuralFlow_concatenate (S : ℝ) (hS : 0 < S)
    (W₁ U₁ W₂ U₂ : ℝ → ParamMatrix d) (b₁ b₂ : ℝ → EucSpace d)
    (φ₁ φ₂ : ℝ → SSphere d → SSphere d)
    (h₁ : IsNeuralFlow d W₁ U₁ b₁ φ₁) (h₂ : IsNeuralFlow d W₂ U₂ b₂ φ₂) :
    IsNeuralFlow d (concatenate S W₁ W₂) (concatenate S U₁ U₂) (concatenate S b₁ b₂)
      (concatenateFlow S φ₁ φ₂) := by
  intro x t
  let F₁ (s : ℝ) : EucSpace d := φ₁ s x
  let F₂ (s : ℝ) : EucSpace d := φ₂ s (φ₁ S x)
  let g₁ (s : ℝ) := neuralVF d W₁ U₁ b₁ s (φ₁ s x)
  let g₂ (s : ℝ) := neuralVF d W₂ U₂ b₂ s (φ₂ s (φ₁ S x))
  have hF₁ (s : ℝ) : F₁ s = F₁ 0 + ∫ r in (0 : ℝ)..s, g₁ r := by
    change (φ₁ s x : EucSpace d) = (φ₁ 0 x : EucSpace d) + _
    rw [IsNeuralFlow.zero W₁ U₁ b₁ φ₁ h₁ x]
    exact (h₁ x s).2
  have hF₂ (s : ℝ) : F₂ s = F₂ 0 + ∫ r in (0 : ℝ)..s, g₂ r := by
    change (φ₂ s (φ₁ S x) : EucSpace d) = (φ₂ 0 (φ₁ S x) : EucSpace d) + _
    rw [IsNeuralFlow.zero W₂ U₂ b₂ φ₂ h₂ (φ₁ S x)]
    exact (h₂ (φ₁ S x) s).2
  have hmatch : F₂ 0 = F₁ S := congrArg Subtype.val (IsNeuralFlow.zero W₂ U₂ b₂ φ₂ h₂ (φ₁ S x))
  have he := concatenate_primitives S hS F₁ F₂ g₁ g₂
    (IsNeuralFlow.intervalIntegrable W₁ U₁ b₁ φ₁ h₁ x)
    (IsNeuralFlow.intervalIntegrable W₂ U₂ b₂ φ₂ h₂ (φ₁ S x)) hF₁ hF₂ hmatch t
  simp_rw [neuralVF_concatenate]
  refine ⟨he.1, ?_⟩
  have hval (s : ℝ) : (concatenateFlow S φ₁ φ₂ s x : EucSpace d) = concatenate S F₁ F₂ s := by
    by_cases hs : s < S <;> simp only [concatenateFlow, concatenate, hs, ite_true, ite_false, F₁, F₂]
  rw [hval, he.2, concatenate_zero S hS]
  change F₁ 0 + _ = x + _
  rw [show F₁ 0 = (x : EucSpace d) from congrArg Subtype.val (IsNeuralFlow.zero W₁ U₁ b₁ φ₁ h₁ x)]

/-- Supported original phases retain Lipschitz, bijective slices and fix
the complement of the union. Source: arXiv:2411.04551v3, Appendix,
the moreover clause of `lem: tubular.mass.movement`. -/
theorem isTransportOutside_concatenate (S T : ℝ) (hS : 0 ≤ S)
    (φ₁ φ₂ : ℝ → SSphere d → SSphere d) (A B : Set (SSphere d))
    (h₁ : IsTransportOutside d φ₁ A S) (h₂ : IsTransportOutside d φ₂ B (T - S)) :
    IsTransportOutside d (concatenateFlow S φ₁ φ₂) (A ∪ B) T := by
  intro t ht
  by_cases hs : t < S
  · obtain ⟨⟨L, hL⟩, hb, hfix⟩ := h₁ t ⟨ht.1, hs.le⟩
    have he : concatenateFlow S φ₁ φ₂ t = φ₁ t := by
      funext x
      exact concatenate_of_lt S _ _ t hs
    rw [he]
    exact ⟨⟨L, hL⟩, hb, fun x hx => hfix x (fun ha => hx (Or.inl ha))⟩
  · have hSt : S ≤ t := le_of_not_gt hs
    obtain ⟨⟨L₁, hL₁⟩, hb₁, hfix₁⟩ := h₁ S ⟨hS, le_rfl⟩
    obtain ⟨⟨L₂, hL₂⟩, hb₂, hfix₂⟩ := h₂ (t - S) ⟨sub_nonneg.mpr hSt, sub_le_sub_right ht.2 S⟩
    have he : concatenateFlow S φ₁ φ₂ t = φ₂ (t - S) ∘ φ₁ S := by
      funext x
      exact concatenate_of_ge S _ _ t hSt
    rw [he]
    refine ⟨⟨L₂ * L₁, hL₂.comp hL₁⟩, hb₂.comp hb₁, ?_⟩
    intro x hx
    change φ₂ (t - S) (φ₁ S x) = x
    rw [hfix₁ x (fun ha => hx (Or.inl ha)), hfix₂ x (fun hb => hx (Or.inr hb))]

/-- Complete original phases retain Lipschitz and bijective slices at
every real time after joining. Auxiliary for arXiv:2411.04551v3,
Appendix, `lem: tubular.mass.movement`, and §4, `eq: neural.ode.sphere`. -/
theorem concatenateFlow_lipschitz_bijective (S : ℝ) (φ₁ φ₂ : ℝ → SSphere d → SSphere d)
    (h₁ : ∀ t, (∃ L : NNReal, LipschitzWith L (φ₁ t)) ∧ Function.Bijective (φ₁ t))
    (h₂ : ∀ t, (∃ L : NNReal, LipschitzWith L (φ₂ t)) ∧ Function.Bijective (φ₂ t))
    (t : ℝ) :
    (∃ L : NNReal, LipschitzWith L (concatenateFlow S φ₁ φ₂ t)) ∧
      Function.Bijective (concatenateFlow S φ₁ φ₂ t) := by
  by_cases ht : t < S
  · have he : concatenateFlow S φ₁ φ₂ t = φ₁ t := by
      funext x
      exact concatenateFlow_of_lt S φ₁ φ₂ t ht x
    rw [he]
    exact h₁ t
  · obtain ⟨⟨L₁, hL₁⟩, hb₁⟩ := h₁ S
    obtain ⟨⟨L₂, hL₂⟩, hb₂⟩ := h₂ (t - S)
    have he : concatenateFlow S φ₁ φ₂ t = φ₂ (t - S) ∘ φ₁ S := by
      funext x
      exact concatenateFlow_of_ge S φ₁ φ₂ t (le_of_not_gt ht) x
    rw [he]
    exact ⟨⟨L₂ * L₁, hL₂.comp hL₁⟩, hb₂.comp hb₁⟩

/-- Two nonzero original projection phases, the second at twice the
speed, supply actual integrated flows and transport clauses together.
Positive switching and elapsed horizons jointly realize all hypotheses. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (W₁ U₁ W₂ U₂ : ℝ → ParamMatrix 2)
    (b₁ b₂ : ℝ → EucSpace 2) (φ₁ φ₂ : ℝ → SSphere 2 → SSphere 2),
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    IsNeuralFlow 2 W₁ U₁ b₁ φ₁ ∧ IsNeuralFlow 2 W₂ U₂ b₂ φ₂ ∧
    IsTransportOutside 2 φ₁ univ 1 ∧ IsTransportOutside 2 φ₂ univ (2 - 1) ∧
    (∀ t, (∃ L : NNReal, LipschitzWith L (φ₁ t)) ∧ Function.Bijective (φ₁ t)) ∧
    (∀ t, (∃ L : NNReal, LipschitzWith L (φ₂ t)) ∧ Function.Bijective (φ₂ t)) ∧
    (ρ : Measure (SSphere 2)) univ = 1 := by
  let p : EucSpace 2 := basePoint 1
  let θ₁ := twoRidgeParams (0 : Fin 2) 1 p 0 1 0 0 0
  let θ₂ := twoRidgeParams (0 : Fin 2) 1 ((2 : ℝ) • p) 0 1 0 0 0
  refine ⟨(fun _ => θ₁.W), (fun _ => θ₁.U), (fun _ => θ₂.W), (fun _ => θ₂.U),
    (fun _ => θ₁.b), (fun _ => θ₂.b), (fun t x => twoRidgeFlow p 0 1 0 0 0 x t),
    (fun t x => twoRidgeFlow ((2 : ℝ) • p) 0 1 0 0 0 x t), zero_lt_one, zero_le_one,
    isNeuralFlow_twoRidgeFlow 0 1 (by norm_num) _ _ _ _ _ _,
    isNeuralFlow_twoRidgeFlow 0 1 (by norm_num) _ _ _ _ _ _, ?_, ?_, ?_, ?_,
    IsProbabilityMeasure.measure_univ⟩
  · exact twoRidgeFlow_isTransportOutside _ _ _ _ _ _ univ (fun x hx => (hx (mem_univ x)).elim) 1
  · exact twoRidgeFlow_isTransportOutside _ _ _ _ _ _ univ (fun x hx => (hx (mem_univ x)).elim) (2 - 1)
  · exact fun t => ⟨exists_lipschitzWith_twoRidgeFlow _ _ _ _ _ _ t,
      bijective_twoRidgeFlow _ _ _ _ _ _ t⟩
  · exact fun t => ⟨exists_lipschitzWith_twoRidgeFlow _ _ _ _ _ _ t,
      bijective_twoRidgeFlow _ _ _ _ _ _ t⟩

end Transformer.Interpolation
