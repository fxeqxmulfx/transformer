/-
# Original target-support bounds imply the genuine Wasserstein accuracy

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §5,
`lem: monge`. Couple a probability law to a target atom by the
constant map. If its actual support lies in a chordal ball of radius
r, the genuine quadratic transport cost is at most r squared, hence
the original W2 distance is at most r.

Apply this to the constructed original quadrant-to-target controls.
Every original weak endpoint has the same target support bound, so
every such endpoint has the requested W2 accuracy. The controls also
construct a complete family of global original weak solutions.

The piece count is at most thirteen N plus one, and therefore at most
five dN for nonempty data in dimension at least three. These are the
existence and switch conclusions for positive-quadrant data; no
quantitative parameter-norm bound is asserted in this auxiliary.
-/

import Transformer.Interpolation.Section4_QuadrantLawMatching
import Transformer.Interpolation.Monge

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The actual support radius bounds genuine W2 distance to the
target atom, with sharp constant one. Source: arXiv:2411.04551v3,
§1, `thm: targets.atoms`, and §5, `lem: monge`. -/
theorem W2_le_radius_of_support (ν : Perspective.ProbSphere d) (p : SSphere d)
    (r : ℝ) (hr : 0 ≤ r)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p r) :
    W2 d (ν : Measure (SSphere d)) (Measure.dirac p) ≤ r := by
  have hmap : Measure.map (fun _ : SSphere d => p) (ν : Measure (SSphere d)) = Measure.dirac p := by
    rw [Measure.map_const, IsProbabilityMeasure.measure_univ, one_smul]
  have hcost := monge d ν id (fun _ => p) measurable_id measurable_const
  rw [Measure.map_id, hmap] at hcost
  apply hcost.trans
  apply Real.sqrt_le_iff.mpr
  refine ⟨hr, ?_⟩
  have hi : Integrable (fun x : SSphere d => ‖(x : EucSpace d) - (p : EucSpace d)‖ ^ 2)
      (ν : Measure (SSphere d)) :=
    ((continuous_subtype_val.sub continuous_const).norm.pow 2).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hb : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)),
      ‖(x : EucSpace d) - (p : EucSpace d)‖ ^ 2 ≤ r ^ 2 := by
    filter_upwards [Measure.support_mem_ae (μ := (ν : Measure (SSphere d)))] with x hx
    have hn : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r := by
      simpa only [Metric.mem_closedBall, Subtype.dist_eq, dist_eq_norm] using hν hx
    nlinarith [norm_nonneg ((x : EucSpace d) - (p : EucSpace d))]
  have hm := integral_mono_ae hi (integrable_const (r ^ 2)) hb
  rwa [integral_const, probReal_univ, one_smul] at hm

/-- All original weak endpoints have the prescribed target accuracy
under the same common quadrant controls. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §5. -/
theorem exists_quadrant_target_accuracy_controls (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) (Y : Idx N → SSphere d)
    (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 13 * N + 1 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ ν : ℝ → Perspective.ProbSphere d,
        ν 0 = ρ ∧ cauchyPB d θ ν) ∧
      ∀ k, ∀ ν : ℝ → Perspective.ProbSphere d, ν 0 = μ k → cauchyPB d θ ν →
        W2 d (ν T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  obtain ⟨θ, K, hK, hpc, hB, hex, hfinal⟩ := exists_quadrant_target_controls hd hN i μ hμ hinj Y hYinj hY ε T hε hT
  refine ⟨θ, K, hK, hpc, hB, hex, ?_⟩
  intro k ν h0 hν
  exact W2_le_radius_of_support (ν T) (Y k) ε hε.le (hfinal k ν h0 hν)

/-- Constructed original weak solutions realize simultaneous target
accuracy for every positive tolerance and horizon. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, quadrant input and targets. -/
theorem exists_quadrant_target_accuracy_solutions (hd : 3 ≤ d) (hN : 1 ≤ N) (i : Fin d)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) (Y : Idx N → SSphere d)
    (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (ν : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 13 * N + 1 ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, ν k 0 = μ k ∧ cauchyPB d θ (ν k)) ∧
      ∀ k, W2 d (ν k T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  obtain ⟨θ, K, hK, hpc, hB, hex, hfinal⟩ :=
    exists_quadrant_target_accuracy_controls hd hN i μ hμ hinj Y hYinj hY ε T hε hT
  choose ν h0 hν using fun k => hex (μ k)
  exact ⟨θ, K, ν, hK, hpc, hB, (fun k => ⟨h0 k, hν k⟩), fun k => hfinal k (ν k) (h0 k) (hν k)⟩

/-- One uniform five-dN piece bound suffices for genuine quadrant
target accuracy. Source auxiliary: arXiv:2411.04551v3, §1,
the existence and switch count of `thm: targets.atoms` for these data. -/
theorem exists_quadrant_target_accuracy_linear_pieces (hd : 3 ≤ d) (hN : 1 ≤ N)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective μ) (Y : Idx N → SSphere d)
    (hYinj : Function.Injective Y) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (ν : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 5 * d * N ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, ν k 0 = μ k ∧ cauchyPB d θ (ν k)) ∧
      ∀ k, W2 d (ν k T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨θ, K, ν, hK, hpc, hB, hν, hfinal⟩ :=
    exists_quadrant_target_accuracy_solutions hd hN i μ hμ hinj Y hYinj hY ε T hε hT
  refine ⟨θ, K, ν, hK.trans ?_, hpc, hB, hν, hfinal⟩
  nlinarith

/-- Distinct genuine original quadrant laws with exactly equal
initial means satisfy the input hypotheses. Thus the ridge refinement
case is inhabited as well as the simpler non-collinear case. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 3) (Y : Idx 2 → SSphere 3),
    3 ≤ 3 ∧ 1 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Function.Injective μ ∧
      (∀ k, (μ k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
      Function.Injective Y ∧ (∀ k, Y k ∈ positiveQuadrant 3) ∧
      barycenter 3 (μ 0) = barycenter 3 (μ 1) := by
  obtain ⟨ρ, σ, hne, hρ, hσ, hmean, _⟩ := perturbation_equal_barycenter_counterexample
  let μ : Idx 2 → Perspective.ProbSphere 3 := ![ρ, σ]
  let Y (k : Idx 2) := quadrantThreePoint k.castSucc
  refine ⟨μ, Y, le_rfl, by decide, one_pos, one_pos, ?_, ?_, ?_, ?_, hmean⟩
  · intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact False.elim (hne he)
    · exact False.elim (hne he.symm)
    · rfl
  · intro k
    fin_cases k
    · exact hρ
    · exact hσ
  · intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  · intro k
    exact quadrantThreePoint_mem k.castSucc

/-- A genuine target Dirac realizes a nonnegative positive support
radius, so the support-to-W2 hypotheses are jointly satisfiable. -/
example : (0 : ℝ) ≤ 1 ∧
    (Perspective.diracProb 3 (quadrantThreePoint 0) : Measure (SSphere 3)).support ⊆
      Metric.closedBall (quadrantThreePoint 0) 1 := by
  refine ⟨zero_le_one, ?_⟩
  intro x hx
  rw [eq_of_mem_support_dirac hx]
  simp

end Transformer.Interpolation
