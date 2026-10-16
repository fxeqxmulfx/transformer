/-
# Selective contraction by three actual control phases

ArXiv:2411.04551v3, §3.2, Steps 1–3 of `lem: induction.barycenter`.
One transverse attention axis isolates any finite family with nonzero
axial means from the selected zero-mean inputs. The latter have a common
positive height floor at the target pole. The inactive gate fixes all
isolated laws, and reversing the actual attention restores them exactly.

The contraction time is chosen uniformly before all selected laws.
The middle radius includes the proved reverse-attention expansion factor.
Three equal physical slots give the prescribed positive horizon and an
absolute three-piece control count. The unselected family may be empty;
no nonemptiness or stronger weak-solution assumption is introduced.
-/

import Transformer.Interpolation.Section3_SelectiveClocks
import Transformer.Interpolation.Section3_MeanAxisAttraction

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A finite family is restored exactly while every selected original
law contracts to a prescribed radius at the prescribed time. Source:
arXiv:2411.04551v3, §3.2, Steps 1–3. One common axis replaces the source's
basis cycle; the finite avoidance lemma proves its availability. -/
theorem exists_selective_three_phase_control {ι : Type*} [Fintype ι]
    (i j : Fin d) (hij : i ≠ j)
    (p w : SSphere d)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support)
    (A : Set (Perspective.ProbSphere d))
    (a : ℝ) (ha : 0 < a)
    (hA : ∀ ν ∈ A, meanAxisMoment (w : EucSpace d) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere d)).support,
        a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d))
    (T r : ℝ) (hT : 0 < T) (hr : 0 < r) :
    ∃ θ : TimeParams d,
      PiecewiseConstant d θ T 3 ∧
      (∀ ρ : Perspective.ProbSphere d,
        ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ ν ∈ A, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ν → cauchyPB d θ μ →
        (μ T : Measure (SSphere d)).support ⊆ Metric.closedBall p r) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = η k → cauchyPB d θ μ → μ T = η k := by
  -- The half-height gate leaves a strict activation margin for selected inputs.
  let c := a / 2
  have hc : 0 < c := by
    dsimp only [c]
    positivity
  have hca : c < a := by
    dsimp only [c]
    linarith
  -- The finite intersection remains valid when there are no unselected inputs.
  have he : ∀ᶠ t : ℝ in Filter.atTop, ∀ k,
      (meanAxisLaw w 1 (η k) t : Measure (SSphere d)).support ⊆
        Metric.ball w (c / 2) ∪ Metric.ball (antipode d w) (c / 2) :=
    Filter.eventually_all.mpr (fun k => eventually_meanAxisLaw_two_poles w 1 (η k)
      zero_lt_one (hη k) (hmiss k).1 (hmiss k).2 (c / 2) (by positivity))
  obtain ⟨t₁, hpre, ht₁⟩ := (he.and (Filter.eventually_gt_atTop (0 : ℝ))).exists
  -- Reversal expands radii by at most exp t₁, independently of the middle clock.
  let δ := r / (2 * exp t₁)
  have hδ : 0 < δ := by
    dsimp only [δ]
    positivity
  have hrad : δ * exp t₁ ≤ r := by
    have hh : δ * (2 * exp t₁) = r := div_mul_cancel₀ r (ne_of_gt (by positivity))
    have hp : 0 < δ * exp t₁ := mul_pos hδ (exp_pos _)
    nlinarith
  -- This single contraction clock works for every law with the same height floor.
  obtain ⟨t₂, -, hcenter⟩ := exists_time_ridgeLaw_center_radius p a c δ hc.le hca hδ
  let θ := selectiveControls i j p w c t₁ t₂ T
  refine ⟨θ, piecewiseConstant_selectiveControls i j p w c t₁ t₂ T hT,
    exists_cauchyPB_selectiveControls i j hij p w c t₁ t₂ T hT, ?_, ?_⟩
  · intro ν hν μ h0 hμ
    have hs := selectiveControls_selected_radius i j hij p w c t₁ t₂ T δ hT ht₁.le hwp ν
      (hA ν hν).1 (hcenter ν (hA ν hν).2) μ hμ h0
    intro x hx
    exact (hs hx).trans hrad
  · intro k μ h0 hμ
    -- The actual gate fixes the isolated law, so the last phase is its true inverse.
    have hfixed := ridgeLaw_inactive_near_orthogonal_poles p w (p : EucSpace d) c hc hwp
      (meanAxisLaw w 1 (η k) t₁) (hpre k) t₂
    exact selectiveControls_restore_fixed i j hij p w c t₁ t₂ T hT
      (η k) hfixed μ hμ h0

/-- A genuine off-axis unselected atom and a selected transverse pole
jointly satisfy every isolation, support, positive-margin and clock
hypothesis. Both the selected set and the finite unselected family are
nonempty, so neither the contraction nor the restoration is vacuous. -/
example : ∃ (i j : Fin 2) (p w : SSphere 2)
    (η : Fin 1 → Perspective.ProbSphere 2)
    (A : Set (Perspective.ProbSphere 2)) (a : ℝ),
    i ≠ j ∧ inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 ∧
    (∀ k, meanAxisMoment (w : EucSpace 2) (η k) ≠ 0) ∧
    (∀ k, w ∉ (η k : Measure (SSphere 2)).support ∧
      antipode 2 w ∉ (η k : Measure (SSphere 2)).support) ∧
    A.Nonempty ∧ 0 < a ∧
    (∀ ν ∈ A, meanAxisMoment (w : EucSpace 2) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere 2)).support,
        a ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2)) ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 := by
  let p : SSphere 2 := basePoint 1
  let w : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let z : SSphere 2 := ⟨EuclideanSpace.single 0 (3 / 5 : ℝ) +
      EuclideanSpace.single 1 (4 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]
    norm_num⟩
  let ν := Perspective.diracProb 2 p
  let η : Fin 1 → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 z
  let A : Set (Perspective.ProbSphere 2) := {ν}
  have hwp : inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 := by
    simp [w, p, basePoint, EuclideanSpace.inner_single_left]
  have hmean (k : Fin 1) : meanAxisMoment (w : EucSpace 2) (η k) ≠ 0 := by
    norm_num [η, meanAxisMoment, barycenter, Perspective.diracProb,
      w, z, inner_add_right, EuclideanSpace.inner_single_left]
  have hmiss (y : SSphere 2) (hy : (y : EucSpace 2) 0 = 0) :
      y ∉ (Perspective.diracProb 2 z : Measure (SSphere 2)).support := by
    intro hh
    have he : y = z := eq_of_mem_support_dirac hh
    have hh₀ := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) he
    norm_num [hy, z] at hh₀
  have hw₀ : (w : EucSpace 2) 0 = 0 := by simp [w]
  have haw₀ : (antipode 2 w : EucSpace 2) 0 = 0 := by simp [antipode, hw₀]
  have hselected : ∀ ρ ∈ A,
      meanAxisMoment (w : EucSpace 2) ρ = 0 ∧
      ∀ x ∈ (ρ : Measure (SSphere 2)).support,
        (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
    intro ρ hρ
    have he : ρ = ν := hρ
    subst ρ
    constructor
    · simpa [ν, meanAxisMoment, barycenter, Perspective.diracProb] using hwp
    · intro x hx
      rw [eq_of_mem_support_dirac hx, real_inner_self_eq_norm_sq,
        mem_sphere_zero_iff_norm.mp p.2]
      norm_num
  refine ⟨0, 1, p, w, η, A, 1 / 2, by norm_num, hwp, hmean,
    fun _ => ⟨hmiss w hw₀, hmiss (antipode 2 w) haw₀⟩,
    ⟨ν, rfl⟩, by norm_num, hselected, zero_lt_one, by norm_num⟩

end Transformer.Interpolation
