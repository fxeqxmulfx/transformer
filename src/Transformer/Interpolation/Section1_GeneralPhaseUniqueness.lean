/-
# Original weak uniqueness for general joined phases

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and the control
compositions in §3.2, `prop: separation`. A proved prefix can be
followed by any globally well-posed original branch. Its previous
constant tail needs global weak existence so that a new prefix can
be extended back to the old global controls.

The later branch is identified by gluing a translated actual branch
before the switch to the original joined law after it. Translating back
produces a global solution of the later original equation, to which its
proved uniqueness applies. This argument works for nonlinear attention
as well as perceptron fields; it assumes no characteristic representation.
-/

import Transformer.Interpolation.Section1_PhaseConcatenation
import Transformer.Interpolation.Section1_TimeTranslation
import Transformer.Interpolation.Section3_MeanAxisTransport

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Global weak existence of the old constant tail and proved prefix
uniqueness identify every new original prefix. Source:
arXiv:2411.04551v3, §1 and §3.2, composing control phases. -/
theorem cauchyPB_concatenate_prefix_of_wellposed_tail (θc : Params d)
    (S : ℝ) (hS : 0 < S) (θ₁ θ₂ : TimeParams d)
    (htail : ∀ s : ℝ, S ≤ s → θ₁ s = θc)
    (hextail : ∀ ρ : Perspective.ProbSphere d, ∃ η : ℝ → Perspective.ProbSphere d,
      η 0 = ρ ∧ cauchyPB d (fun _ => θc) η)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ₁ α → cauchyPB d θ₁ β →
      α 0 = β 0 → α = β)
    (μ₁ μ : ℝ → Perspective.ProbSphere d) (hμ₁ : cauchyPB d θ₁ μ₁)
    (hμ : cauchyPB d (concatenate S θ₁ θ₂) μ) (h0 : μ 0 = μ₁ 0) (t : ℝ) (ht : t ≤ S) :
    μ t = μ₁ t := by
  obtain ⟨η, hη0, hη⟩ := hextail (μ S)
  let M := concatenate S μ η
  have hM := cauchyPB_concatenate S hS (concatenate S θ₁ θ₂) (fun _ => θc) μ η hμ hη hη0
  have hθ : concatenate S (concatenate S θ₁ θ₂) (fun _ => θc) = θ₁ := by
    funext s
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs]
    · simp only [concatenate, ite_eq_right hs]
      exact (htail s (le_of_not_gt hs)).symm
  rw [hθ] at hM
  have he : M = μ₁ := hunique M μ₁ hM hμ₁ ((concatenate_zero S hS μ η).trans h0)
  have htime : M t = μ t := concatenate_of_le S μ η t ht hη0
  exact htime.symm.trans (congrFun he t)

/-- Every later original law equals the actual translated branch from
the switch law, using only that branch's proved global uniqueness.
Source: arXiv:2411.04551v3, §1 and §3.2, control composition. -/
theorem cauchyPB_concatenate_tail_of_unique (S : ℝ) (hS : 0 < S) (θ₁ θ₂ : TimeParams d)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ₂ α → cauchyPB d θ₂ β →
      α 0 = β 0 → α = β)
    (μ η : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (concatenate S θ₁ θ₂) μ)
    (hη : cauchyPB d θ₂ η) (hη0 : η 0 = μ S) (t : ℝ) (ht : S ≤ t) : μ t = η (t - S) := by
  let α (s : ℝ) := η (-S + s)
  let β (s : ℝ) := μ (S + s)
  let M := concatenate S α β
  have hα : cauchyPB d (translateTimeParams (-S) θ₂) α :=
    cauchyPB_translateTimeParams (-S) θ₂ η hη
  have hβ : cauchyPB d (translateTimeParams S (concatenate S θ₁ θ₂)) β :=
    cauchyPB_translateTimeParams S (concatenate S θ₁ θ₂) μ hμ
  have hmatch : β 0 = α S := by
    dsimp only [α, β]
    simpa only [add_zero, neg_add_cancel] using hη0.symm
  have hM := cauchyPB_concatenate S hS _ _ α β hα hβ hmatch
  have hθ : concatenate S (translateTimeParams (-S) θ₂)
      (translateTimeParams S (concatenate S θ₁ θ₂)) = translateTimeParams (-S) θ₂ := by
    funext s
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs]
    · rw [concatenate_of_ge S _ _ s (le_of_not_gt hs)]
      change (concatenate S θ₁ θ₂) (S + (s - S)) = θ₂ (-S + s)
      rw [show S + (s - S) = s by ring, concatenate_of_ge S _ _ s (le_of_not_gt hs)]
      congr 1
      ring
  rw [hθ] at hM
  have hback : cauchyPB d θ₂ (fun s => M (S + s)) := by
    simpa only [neg_neg] using cauchyPB_untranslateTimeParams (-S) θ₂ M hM
  have hMμ (s : ℝ) (hs : S ≤ s) : M s = μ s := by
    change concatenate S α β s = μ s
    rw [concatenate_of_ge S α β s hs]
    change μ (S + (s - S)) = μ s
    congr 1
    ring
  have h0 : (fun s => M (S + s)) 0 = η 0 := by
    change M (S + 0) = η 0
    rw [add_zero, hMμ S le_rfl, hη0]
  have he := hunique _ η hback hη h0
  have hh := congrFun he (t - S)
  change M (S + (t - S)) = η (t - S) at hh
  rw [show S + (t - S) = t by ring, hMμ t ht] at hh
  exact hh

/-- Proved original branch well-posedness gives global weak uniqueness
for their actual joining. Source: arXiv:2411.04551v3, §1 and §3.2.
The old constant tail has only an existence premise. -/
theorem cauchyPB_concatenate_general_unique (θc : Params d) (S : ℝ) (hS : 0 < S)
    (θ₁ θ₂ : TimeParams d) (htail : ∀ s : ℝ, S ≤ s → θ₁ s = θc)
    (hextail : ∀ ρ : Perspective.ProbSphere d, ∃ η : ℝ → Perspective.ProbSphere d,
      η 0 = ρ ∧ cauchyPB d (fun _ => θc) η)
    (hex₁ : ∀ ρ : Perspective.ProbSphere d, ∃ η : ℝ → Perspective.ProbSphere d,
      η 0 = ρ ∧ cauchyPB d θ₁ η)
    (hu₁ : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ₁ α → cauchyPB d θ₁ β →
      α 0 = β 0 → α = β)
    (hex₂ : ∀ ρ : Perspective.ProbSphere d, ∃ η : ℝ → Perspective.ProbSphere d,
      η 0 = ρ ∧ cauchyPB d θ₂ η)
    (hu₂ : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ₂ α → cauchyPB d θ₂ β →
      α 0 = β 0 → α = β)
    (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S θ₁ θ₂) μ) (hν : cauchyPB d (concatenate S θ₁ θ₂) ν)
    (h0 : μ 0 = ν 0) : μ = ν := by
  obtain ⟨η, hη0, hη⟩ := hex₁ (μ 0)
  have hpreμ (t : ℝ) (ht : t ≤ S) : μ t = η t :=
    cauchyPB_concatenate_prefix_of_wellposed_tail θc S hS θ₁ θ₂ htail hextail hu₁ η μ
      hη hμ hη0.symm t ht
  have hpreν (t : ℝ) (ht : t ≤ S) : ν t = η t :=
    cauchyPB_concatenate_prefix_of_wellposed_tail θc S hS θ₁ θ₂ htail hextail hu₁ η ν
      hη hν (h0.symm.trans hη0.symm) t ht
  have hmatch : μ S = ν S := (hpreμ S le_rfl).trans (hpreν S le_rfl).symm
  obtain ⟨ξ, hξ0, hξ⟩ := hex₂ (μ S)
  funext t
  by_cases ht : t ≤ S
  · exact (hpreμ t ht).trans (hpreν t ht).symm
  · have hst : S ≤ t := (lt_of_not_ge ht).le
    rw [cauchyPB_concatenate_tail_of_unique S hS θ₁ θ₂ hu₂ μ ξ hμ hξ hξ0 t hst,
      cauchyPB_concatenate_tail_of_unique S hS θ₁ θ₂ hu₂ ν ξ hν hξ (hξ0.trans hmatch) t hst]

/-- A nonzero perceptron prefix and an actual nonlinear rank-one tail
supply all branch-existence, uniqueness, matching-data and original
weak-equation hypotheses for the same joined control. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (θc : Params 2) (θ₂ : TimeParams 2)
    (μ₁ μ η : ℝ → Perspective.ProbSphere 2),
    (∀ σ : Perspective.ProbSphere 2, ∃ ξ : ℝ → Perspective.ProbSphere 2,
      ξ 0 = σ ∧ cauchyPB 2 (fun _ => θc) ξ) ∧
    (∀ α β : ℝ → Perspective.ProbSphere 2, cauchyPB 2 (fun _ => θc) α →
      cauchyPB 2 (fun _ => θc) β → α 0 = β 0 → α = β) ∧
    (∀ σ : Perspective.ProbSphere 2, ∃ ξ : ℝ → Perspective.ProbSphere 2,
      ξ 0 = σ ∧ cauchyPB 2 θ₂ ξ) ∧
    (∀ α β : ℝ → Perspective.ProbSphere 2, cauchyPB 2 θ₂ α →
      cauchyPB 2 θ₂ β → α 0 = β 0 → α = β) ∧
    cauchyPB 2 (fun _ => θc) μ₁ ∧
    cauchyPB 2 (concatenate 1 (fun _ => θc) θ₂) μ ∧ μ 0 = μ₁ 0 ∧
    cauchyPB 2 θ₂ η ∧ η 0 = μ 1 ∧ (0 : ℝ) < 1 := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let w : SSphere 2 := basePoint 1
  let θc := twoRidgeParams (0 : Fin 2) 1 p 0 1 q 0 1
  let θ₂ : TimeParams 2 := fun _ => meanAxisParams (w : EucSpace 2) 1
  have hex₁ (σ : Perspective.ProbSphere 2) : ∃ ξ : ℝ → Perspective.ProbSphere 2,
      ξ 0 = σ ∧ cauchyPB 2 (fun _ => θc) ξ :=
    ⟨twoRidgeLaw p 0 1 q 0 1 σ, twoRidgeLaw_zero _ _ _ _ _ _ _,
      cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _⟩
  have hex₂ := exists_cauchyPB_meanAxis w 1
  obtain ⟨μ, hμ0, hμ⟩ := exists_cauchyPB_concatenate 1 zero_lt_one
    (fun _ => θc) θ₂ ρ (hex₁ ρ) hex₂
  let μ₁ := twoRidgeLaw p 0 1 q 0 1 ρ
  let η := meanAxisLaw w 1 (μ 1)
  exact ⟨θc, θ₂, μ₁, μ, η, hex₁,
    cauchyPB_twoRidge_unique 0 1 (by norm_num) p 0 1 q 0 1,
    hex₂, cauchyPB_meanAxis_unique w 1,
    cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _, hμ,
    hμ0.trans (twoRidgeLaw_zero _ _ _ _ _ _ _).symm,
    cauchyPB_meanAxisLaw w 1 (μ 1), meanAxisLaw_zero w 1 (μ 1), zero_lt_one⟩

end Transformer.Interpolation
