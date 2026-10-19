module

public import Transformer.Interpolation.Section3_MeanAxisTransport
public import Transformer.Interpolation.Section1_WeakConcatenation

/-!
# Original rank-one weak transport across control switches

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2, Steps 1 and 3
of `lem: induction.barycenter`. Every original weak law follows the
proved nonlinear mean-attention flow during an initial constant phase,
regardless of the later controls. Extend the original prefix at the
switch by a constructed genuine mean-attention weak law, then apply
actual global uniqueness to the resulting constant-control equation.

For a last constant phase, reverse the original controls and curve from
the observation endpoint. The prefix theorem and the actual inverse
probability map give the exact elapsed-time law after the last switch.
Both phase boundaries are included; no uniqueness is assumed for the
full switched equation, and the earlier or later controls are arbitrary.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original weak law follows the mean-attention flow through its
first constant phase, including the switch. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem cauchyPB_meanAxis_prefix (w : SSphere d) (κ S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, s < S → θ s = meanAxisParams (w : EucSpace d) κ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : t ≤ S) :
    μ t = meanAxisLaw w κ (μ 0) t := by
  let θc : TimeParams d := fun _ => meanAxisParams (w : EucSpace d) κ
  let ν := meanAxisLaw w κ (μ S)
  let M := concatenate S μ ν
  have hmatch : ν 0 = μ S := meanAxisLaw_zero w κ (μ S)
  have hM := cauchyPB_concatenate S hS θ θc μ ν hμ (cauchyPB_meanAxisLaw w κ (μ S)) hmatch
  have hconst : concatenate S θ θc = θc := by
    funext s
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs]
      exact hθ s hs
    · simp only [concatenate, ite_eq_right hs, θc]
  rw [hconst] at hM
  have he := congrFun (cauchyPB_meanAxis_eq w κ M hM) t
  have hzero : M 0 = μ 0 := concatenate_zero S hS μ ν
  have htime : M t = μ t := concatenate_of_le S μ ν t ht hmatch
  rwa [hzero, htime] at he

/-- Equal initial data force equality through the original initial
mean-attention phase. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem cauchyPB_meanAxis_prefix_unique (w : SSphere d) (κ S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, s < S → θ s = meanAxisParams (w : EucSpace d) κ)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (h0 : μ 0 = ν 0) (t : ℝ) (ht : t ≤ S) : μ t = ν t := by
  rw [cauchyPB_meanAxis_prefix w κ S hS θ hθ μ hμ t ht,
    cauchyPB_meanAxis_prefix w κ S hS θ hθ ν hν t ht, h0]

/-- After the last switch, every original weak law follows the actual
mean-attention law at elapsed time. Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem cauchyPB_meanAxis_tail (w : SSphere d) (κ S : ℝ) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, S < s → θ s = meanAxisParams (w : EucSpace d) κ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : 0 ≤ t) :
    μ (S + t) = meanAxisLaw w κ (μ S) t := by
  by_cases he0 : t = 0
  · subst t
    rw [add_zero, meanAxisLaw_zero]
  have hpos : 0 < t := lt_of_le_of_ne ht (Ne.symm he0)
  have hrev (r : ℝ) (hr : r < t) :
      reverseTimeParams (S + t) θ r = meanAxisParams (w : EucSpace d) (-κ) := by
    dsimp only [reverseTimeParams, rescaleTimeParams, translateTimeParams]
    have hc : S + t + (-1 * r) = S + t - r := by ring
    rw [hc, hθ (S + t - r) (by linarith)]
    simp only [meanAxisParams, smul_smul, neg_one_mul]
    congr 1
    ext x
    simp only [smul_apply, zero_apply, smul_zero]
  have hμrev := cauchyPB_reverseTimeParams (S + t) θ μ hμ
  have he := cauchyPB_meanAxis_prefix w (-κ) t hpos (reverseTimeParams (S + t) θ) hrev
    (fun r => μ (S + t - r)) hμrev t le_rfl
  simp only [add_sub_cancel_right, sub_zero] at he
  have hh := congrArg (fun ρ => meanAxisLaw w κ ρ t) he
  have hi (ρ : Perspective.ProbSphere d) : meanAxisLaw w κ (meanAxisLaw w (-κ) ρ t) t = ρ := by
    simpa only [neg_neg] using meanAxisLaw_inverse w (-κ) ρ t
  rw [hi] at hh
  exact hh.symm

/-- The final-phase formula retains the original absolute observation
time. Auxiliary for arXiv:2411.04551v3, §3.2, Step 3. -/
theorem cauchyPB_meanAxis_tail_after (w : SSphere d) (κ S : ℝ) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, S < s → θ s = meanAxisParams (w : EucSpace d) κ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : S ≤ t) :
    μ t = meanAxisLaw w κ (μ S) (t - S) := by
  have he := cauchyPB_meanAxis_tail w κ S θ hθ μ hμ (t - S) (sub_nonneg.mpr ht)
  simpa only [add_sub_cancel] using he

/-- Appending the actual rank-one controls gives their true nonlinear
last-phase law. Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem cauchyPB_meanAxis_concatenate_tail (w : SSphere d) (κ S : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S θ (fun _ => meanAxisParams (w : EucSpace d) κ)) μ)
    (t : ℝ) (ht : S ≤ t) : μ t = meanAxisLaw w κ (μ S) (t - S) := by
  apply cauchyPB_meanAxis_tail_after w κ S _ ?_ μ hμ t ht
  intro s hs
  simp only [concatenate, ite_eq_right (not_lt_of_ge hs.le)]

/-- The first-phase formula keeps the prescribed original initial law.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem cauchyPB_meanAxis_prefix_initial (w : SSphere d) (κ S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, s < S → θ s = meanAxisParams (w : EucSpace d) κ)
    (ρ : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (h0 : μ 0 = ρ) (t : ℝ) (ht : t ≤ S) :
    μ t = meanAxisLaw w κ ρ t := by
  rw [cauchyPB_meanAxis_prefix w κ S hS θ hθ μ hμ t ht, h0]

/-- The last-phase formula keeps the prescribed law at the actual switch.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 3. -/
theorem cauchyPB_meanAxis_tail_initial (w : SSphere d) (κ S : ℝ) (θ : TimeParams d)
    (hθ : ∀ s : ℝ, S < s → θ s = meanAxisParams (w : EucSpace d) κ)
    (ρ : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (h0 : μ S = ρ) (t : ℝ) (ht : S ≤ t) :
    μ t = meanAxisLaw w κ ρ (t - S) := by
  rw [cauchyPB_meanAxis_tail_after w κ S θ hθ μ hμ t ht, h0]

/-- Two different nonzero value factors and their actual joined weak
laws realize both constant-control phase hypotheses. The laws match at
the genuine switch, and the same initial data realize prefix uniqueness. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (w : SSphere 2) (θ : TimeParams 2)
    (μ ν : ℝ → Perspective.ProbSphere 2) (σ : Perspective.ProbSphere 2),
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 ∧
    (∀ s : ℝ, s < 1 → θ s = meanAxisParams (w : EucSpace 2) 1) ∧
    (∀ s : ℝ, 1 < s → θ s = meanAxisParams (w : EucSpace 2) 2) ∧
    cauchyPB 2 θ μ ∧ cauchyPB 2 θ ν ∧ μ 0 = ρ ∧ μ 0 = ν 0 ∧ μ 1 = σ := by
  let w : SSphere 2 := basePoint 1
  let θ₁ : TimeParams 2 := fun _ => meanAxisParams (w : EucSpace 2) 1
  let θ₂ : TimeParams 2 := fun _ => meanAxisParams (w : EucSpace 2) 2
  let μ₁ := meanAxisLaw w 1 ρ
  let μ₂ := meanAxisLaw w 2 (μ₁ 1)
  let θ := concatenate 1 θ₁ θ₂
  let μ := concatenate 1 μ₁ μ₂
  have hmatch : μ₂ 0 = μ₁ 1 := meanAxisLaw_zero w 2 _
  have hμ : cauchyPB 2 θ μ := cauchyPB_concatenate 1 zero_lt_one θ₁ θ₂ μ₁ μ₂
    (cauchyPB_meanAxisLaw w 1 ρ) (cauchyPB_meanAxisLaw w 2 _) hmatch
  refine ⟨w, θ, μ, μ, μ₁ 1, zero_lt_one, by norm_num, ?_, ?_, hμ, hμ, ?_, rfl, ?_⟩
  · intro s hs
    simp only [θ, concatenate, ite_eq_left hs, θ₁]
  · intro s hs
    simp only [θ, concatenate, ite_eq_right (not_lt_of_ge hs.le), θ₂]
  · change concatenate 1 μ₁ μ₂ 0 = ρ
    rw [concatenate_zero 1 zero_lt_one]
    exact meanAxisLaw_zero w 1 ρ
  · change concatenate 1 μ₁ μ₂ 1 = μ₁ 1
    rw [concatenate_switch, hmatch]

end Transformer.Interpolation
