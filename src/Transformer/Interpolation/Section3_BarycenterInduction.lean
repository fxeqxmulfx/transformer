module

public import Transformer.Interpolation.Section3_BarycenterContraction
public import Transformer.Interpolation.Section1_ControlPartitions

/-!
# The original barycenter induction with a uniform control bound

ArXiv:2411.04551v3, §3.2, `lem: induction.barycenter`.
In ambient dimension at least two, one common transverse axis and one
gated perceptron give three actual control phases. The proof applies to
every original weak solution and gives global weak existence for all
initial data. The original geodesic hulls obey the angular target radius.

The zero-dimensional sphere cannot support a probability. The strict
quadrant in ambient dimension one supports only the positive Dirac law,
so a constant zero control proves that case. Thus the absolute constant
three is chosen before dimension, family size and all data.

The source inconsistently alternates between target indices N and j.
As in `induction_barycenter`, j denotes that same selected law throughout;
the collinearity, both final hulls, the target pole and the restoration
clause refer to this one index. No original data hypothesis is removed.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

open Perspective

/-- The source's original shrinking conclusion has uniform constant
three in its O(d · N) control bound. Source: arXiv:2411.04551v3, §3.2,
`lem: induction.barycenter`, with the inconsistent N/j target indices
consistently read as j. Every original weak solution is covered. -/
theorem exists_barycenter_induction_controls :
    ∀ (d N : ℕ) (T ε : ℝ) (j : Idx N)
      (μ₀ : Idx N → Perspective.ProbSphere d)
      (ν₀ : Perspective.ProbSphere d),
      0 < T → 0 < ε →
      (∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) →
      (∀ i i' : Idx N, i ≠ i' →
        ∀ γ : ℝ, barycenter d (μ₀ i) ≠ γ • barycenter d (μ₀ i')) →
      (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d →
      (∃ γ : ℝ, barycenter d ν₀ = γ • barycenter d (μ₀ j)) →
      ∃ (θ : TimeParams d) (K : ℕ),
        K ≤ 3 * (d * N) ∧ PiecewiseConstant d θ T K ∧
        (∀ i : Idx N, ∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
        (∃ ν : ℝ → Perspective.ProbSphere d,
          ν 0 = ν₀ ∧ cauchyPB d θ ν) ∧
        (∀ μ ν : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ j → cauchyPB d θ μ →
          ν 0 = ν₀ → cauchyPB d θ ν →
          ∀ x ∈ convG d (ν T : Measure (SSphere d)).support ∪
              convG d (μ T : Measure (SSphere d)).support,
            Real.arccos (inner (𝕜 := ℝ) (x : EucSpace d)
              (‖barycenter d (μ₀ j)‖⁻¹ • barycenter d (μ₀ j))) < ε) ∧
        ∀ i : Idx N, i ≠ j → ∀ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ → μ T = μ₀ i := by
  intro d N T ε j μ₀ ν₀ hT hε hμQ hsep hνQ hcol
  have hN : 1 ≤ N := by
    have hh := j.isLt
    omega
  -- A single transverse attention axis is available in dimension at least two.
  by_cases hd : 2 ≤ d
  · obtain ⟨θ, hθ, hex, htarget, hrest⟩ :=
      exists_barycenter_contraction_controls hd T ε j μ₀ ν₀
      hT hε hμQ (fun i hi => hsep i j hi) hνQ hcol
    have hdN : 1 ≤ d * N := Nat.mul_pos (by omega) (by omega)
    refine ⟨θ, 3, ?_, hθ, fun i => hex (μ₀ i), hex ν₀, htarget, hrest⟩
    simpa only [mul_one] using Nat.mul_le_mul_left 3 hdN
  · have hd' : d ≤ 1 := by omega
    interval_cases d
    -- The zero-dimensional ambient sphere has no probability laws.
    · obtain ⟨x, -⟩ := nonempty_of_measure_ne_zero (μ := (ν₀ : Measure (SSphere 0)))
        (s := Set.univ) (by rw [measure_univ]; exact one_ne_zero)
      have hx0 : (x : EucSpace 0) = 0 := by
        ext i
        exact Fin.elim0 i
      have hn := mem_sphere_zero_iff_norm.mp x.2
      rw [hx0, norm_zero] at hn
      exact (zero_ne_one hn).elim
    -- In dimension one every original quadrant law is the same positive atom.
    · let w : SSphere 1 := basePoint 0
      let θ : TimeParams 1 := fun _ => meanAxisParams (w : EucSpace 1) 0
      have hex (ρ : Perspective.ProbSphere 1) :
          ∃ μ : ℝ → Perspective.ProbSphere 1,
          μ 0 = ρ ∧ cauchyPB 1 θ μ := exists_cauchyPB_meanAxis w 0 ρ
      have hstationary (μ : ℝ → Perspective.ProbSphere 1)
          (hμ : cauchyPB 1 θ μ) : μ T = μ 0 := by
        rw [cauchyPB_meanAxis_eq w 0 μ hμ, meanAxisLaw_of_zero_factor]
      have hμj : μ₀ j = Perspective.diracProb 1 w :=
        eq_dirac_of_positiveQuadrant_one (μ₀ j) (hμQ j)
      have hν : ν₀ = Perspective.diracProb 1 w :=
        eq_dirac_of_positiveQuadrant_one ν₀ hνQ
      have hmean :
          ‖barycenter 1 (μ₀ j)‖⁻¹ • barycenter 1 (μ₀ j) = (w : EucSpace 1) := by
        rw [hμj]
        simp [barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp w.2]
      have hhull (ρ : Perspective.ProbSphere 1)
          (hρ : ρ = Perspective.diracProb 1 w) :
          ∀ x ∈ convG 1 (ρ : Measure (SSphere 1)).support,
            arccos (inner ℝ (x : EucSpace 1) (w : EucSpace 1)) < ε := by
        intro x hx
        have hs : (ρ : Measure (SSphere 1)).support ⊆ {w} := by
          rw [hρ]
          intro y hy
          exact eq_of_mem_support_dirac hy
        rw [arccos_on_convG_singleton w x (convG_mono hs hx)]
        exact hε
      refine ⟨θ, 1, by omega,
        piecewiseConstant_constant _ T hT.le,
        fun i => hex (μ₀ i), hex ν₀, ?_, ?_⟩
      · intro μ ν hμ0 hμ hν0 hν' x hx
        rw [hmean]
        rcases hx with hx | hx
        · have hνT : ν T = Perspective.diracProb 1 w :=
            (hstationary ν hν').trans (hν0.trans hν)
          exact hhull (ν T) hνT x hx
        · have hμT : μ T = Perspective.diracProb 1 w :=
            (hstationary μ hμ).trans (hμ0.trans hμj)
          exact hhull (μ T) hμT x hx
      · intro i _ μ h0 hμ
        exact (hstationary μ hμ).trans h0

/-- The hypotheses of `exists_barycenter_induction_controls` are
satisfiable on a one-element
family: `N = 1` leaves the non-collinearity of *distinct* barycenters nothing to
check, `ν₀ = μ₀ 0 = δ_{basePoint 0}` is supported in `ℚ_1^0`, and the two
barycenters are collinear with `γ = 1`. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 1,
        (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i : Measure (SSphere 1))).support
          ⊆ positiveQuadrant 1) ∧
      (∀ i i' : Idx 1, i ≠ i' →
        ∀ γ : ℝ, barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i)
          ≠ γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i')) ∧
      ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support
        ⊆ positiveQuadrant 1 ∧
      (∃ γ : ℝ, barycenter 1 (diracProb 1 (basePoint 0))
        = γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) 0)) := by
  have hQ : ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support
      ⊆ positiveQuadrant 1 := fun x hx => by
    have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
    rw [eq_of_mem_support_dirac hx']
    exact basePoint_mem_positiveQuadrant
  exact ⟨one_pos, one_pos, fun _ => hQ,
    fun i i' hne => absurd (Subsingleton.elim i i') hne,
    hQ,
    1, (one_smul ℝ _).symm⟩

end Transformer.Interpolation
