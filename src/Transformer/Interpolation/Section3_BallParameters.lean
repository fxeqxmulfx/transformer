module

public import Transformer.Interpolation.Section3_CapAttractor
public import Transformer.Interpolation.Section3_MassConcentration
public import Transformer.Interpolation.Section3_TwoRidgeScaling
public import Transformer.Interpolation.Section3_NullRepeller

/-!
# Explicit original two-neuron data for each ball-chain phase

ArXiv:2411.04551v3, Appendix, `lem: two.balls` and its use in
`lem: tubular.mass.movement`.

The proper-ball attractor and the whole-sphere projection are actual
two-neuron fields. For the specified initial probability, attraction
gives the required mass in any open target inside the active ball.
Scaling the two output vectors places that endpoint at the prescribed
nonzero time and keeps the complement fixed for every real time.

The six returned vectors and scalars are the true activation and output
data, so successive phases can use their established global weak
existence and uniqueness. Their choice may depend on the initial law,
as required by the corrected quantifier order of the ball lemmas.
-/

@[expose] public section

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual attraction supplies an output rate realizing the mass bound
at the prescribed time, with the same fixed complement. Auxiliary for
arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_twoRidge_attraction_mass_rate (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) (S A : Set (SSphere d)) (hS : MeasurableSet S)
    (hA : IsOpen A) (hAS : A ⊆ S) (ω : SSphere d) (hω : ω ∈ A)
    (ν : Perspective.ProbSphere d)
    (hactive : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      x ∈ S → Tendsto (twoRidgeFlow p u b q v c x) atTop (nhds ω))
    (hfix : ∀ x ∉ S, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) :
    ∃ κ : ℝ, (∀ x ∉ S, ∀ t : ℝ, twoRidgeFlow (κ • p) u b (κ • q) v c x t = x) ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) S ≤
        (twoRidgeLaw (κ • p) u b (κ • q) v c ν T : Measure (SSphere d)) A := by
  let Φ (t : ℝ) (x : SSphere d) := twoRidgeFlow p u b q v c x t
  let η := min ε 1
  have hη : 0 < η := lt_min hε zero_lt_one
  have hm := eventually_map_mass_ge ν Φ (fun t => measurable_twoRidgeFlow_initial _ _ _ _ _ _ t)
    S A hS hA hAS ω hω hactive hfix η hη (min_le_right _ _)
  obtain ⟨τ, hτ⟩ := hm.exists
  refine ⟨τ / T, ?_, ?_⟩
  · intro x hx t
    rw [twoRidgeFlow_smul_output]
    exact hfix x hx _
  · rw [twoRidgeLaw_smul_output, div_mul_cancel₀ τ hT, coe_twoRidgeLaw]
    exact (mul_le_mul_of_nonneg_right
      (ENNReal.ofReal_le_ofReal (by have hh := min_le_left ε 1; change min ε 1 ≤ ε at hh; linarith))
      (by positivity)).trans hτ

/-- A proper ball supplies explicit constant two-neuron data for its
next mass-transport phase. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, via `lem: two.balls`. -/
theorem exists_twoRidge_proper_ball_mass_parameters (z : SSphere d) (R : ℝ)
    (hR : 0 < R) (hR2 : R ≤ 2) (A : Set (SSphere d)) (hA : IsOpen A)
    (hAS : A ⊆ Metric.ball z R) (hne : A.Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ),
      (∀ x ∉ Metric.ball z R, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x) ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z R) ≤
        (twoRidgeLaw p u b q v c ν T : Measure (SSphere d)) A := by
  obtain ⟨ω, hω⟩ := hne
  obtain ⟨w, h, ℓ, hattract, hfixed⟩ := exists_twoRidge_ball_attractor z ω R hR hR2 (hAS hω)
  let c := 1 - R ^ 2 / 2
  let q := capTargetOutput z w ω c h ℓ
  obtain ⟨κ, hfix, hmass⟩ := exists_twoRidge_attraction_mass_rate
    (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h)
    (Metric.ball z R) A Metric.isOpen_ball.measurableSet hA hAS ω hω ν
    (Eventually.of_forall hattract) hfixed ε T hε hT
  exact ⟨κ • (z : EucSpace d), -(z : EucSpace d), -c, κ • q, -(w : EucSpace d), -h, hfix, hmass⟩

/-- A null repelling pole supplies actual constant data when the active
ball is the whole sphere. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem exists_twoRidge_whole_mass_parameters (hd : 2 ≤ d)
    (A : Set (SSphere d)) (hA : IsOpen A) (hne : A.Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ),
      (∀ x ∉ (univ : Set (SSphere d)), ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x) ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) univ ≤
        (twoRidgeLaw p u b q v c ν T : Measure (SSphere d)) A := by
  obtain ⟨p, hpA, hnull⟩ := exists_target_null_antipode hd ν A hA hne
  have hactive := ae_tendsto_twoRidge_projection ν p hnull
  obtain ⟨κ, hfix, hmass⟩ := exists_twoRidge_attraction_mass_rate
    (p : EucSpace d) 0 1 0 0 0 univ A MeasurableSet.univ hA (subset_univ _) p hpA ν
    (hactive.mono fun _ hx _ => hx) (fun x hx => (hx (mem_univ x)).elim) ε T hε hT
  exact ⟨κ • (p : EucSpace d), 0, 1, κ • (0 : EucSpace d), 0, 0, hfix, hmass⟩

/-- Every nonempty open target inside an arbitrary spherical ball admits
explicit original two-neuron mass-transport data. Auxiliary for
arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem exists_twoRidge_ball_mass_parameters (hd : 2 ≤ d)
    (z : SSphere d) (R : ℝ) (A : Set (SSphere d)) (hA : IsOpen A)
    (hAS : A ⊆ Metric.ball z R) (hne : A.Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ),
      (∀ x ∉ Metric.ball z R, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x) ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z R) ≤
        (twoRidgeLaw p u b q v c ν T : Measure (SSphere d)) A := by
  by_cases hR2 : R ≤ 2
  · obtain ⟨ω, hω⟩ := hne
    have hwr : dist ω z < R := hAS hω
    exact exists_twoRidge_proper_ball_mass_parameters z R
      ((dist_nonneg : 0 ≤ dist ω z).trans_lt hwr) hR2 A hA hAS ⟨ω, hω⟩ ε T hε hT ν
  · have hs := spherical_ball_eq_univ_of_two_lt z R (lt_of_not_ge hR2)
    simpa only [hs] using exists_twoRidge_whole_mass_parameters hd A hA hne ε T hε hT ν

/-- Consecutive overlapping balls give the exact next-phase mass
estimate with explicit neuron data. Source: arXiv:2411.04551v3,
Appendix, `lem: two.balls`, as used by `lem: tubular.mass.movement`. -/
theorem exists_twoRidge_overlap_mass_parameters (hd : 2 ≤ d)
    (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hmeet : (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁).Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ),
      (∀ x ∉ Metric.ball z₀ R₀, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x) ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z₀ R₀) ≤
        (twoRidgeLaw p u b q v c ν T : Measure (SSphere d)) (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁) :=
  exists_twoRidge_ball_mass_parameters hd z₀ R₀ (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁)
    (Metric.isOpen_ball.inter Metric.isOpen_ball) inter_subset_left hmeet ε T hε hT ν

/-- An actual open spherical ball and an atomic initial law jointly
satisfy all proper-ball, whole-sphere target, accuracy and time premises. -/
example : ∃ (z : SSphere 2) (A : Set (SSphere 2)) (ν : Perspective.ProbSphere 2),
    2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 ∧ IsOpen A ∧
    A ⊆ Metric.ball z 1 ∧ A.Nonempty ∧
    (Metric.ball z 1 ∩ Metric.ball z 1).Nonempty ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    (ν : Measure (SSphere 2)) univ = 1 := by
  let z : SSphere 2 := basePoint 1
  refine ⟨z, Metric.ball z 1, Perspective.diracProb 2 z, le_rfl, zero_lt_one,
    by norm_num, Metric.isOpen_ball, Subset.rfl, ⟨z, by simp⟩, ⟨z, by simp, by simp⟩, zero_lt_one,
    by norm_num, IsProbabilityMeasure.measure_univ⟩

/-- The true projection flow and a Dirac law at its attracting pole
satisfy the attraction and fixed-complement hypotheses simultaneously. -/
example (p : SSphere 2) :
    MeasurableSet (univ : Set (SSphere 2)) ∧ IsOpen (Metric.ball p 1) ∧
    Metric.ball p 1 ⊆ univ ∧ p ∈ Metric.ball p 1 ∧
    (∀ᵐ x ∂(Perspective.diracProb 2 p : Measure (SSphere 2)),
      x ∈ univ → Tendsto (twoRidgeFlow (p : EucSpace 2) 0 1 0 0 0 x) atTop (nhds p)) ∧
    (∀ x ∉ (univ : Set (SSphere 2)), ∀ t : ℝ, twoRidgeFlow (p : EucSpace 2) 0 1 0 0 0 x t = x) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 := by
  refine ⟨MeasurableSet.univ, Metric.isOpen_ball, subset_univ _, by simp,
    ?_, fun x hx => (hx (mem_univ x)).elim, zero_lt_one, by norm_num⟩
  apply (ae_tendsto_twoRidge_projection (Perspective.diracProb 2 p) p ?_).mono
  · intro x hx _
    exact hx
  · simp [Perspective.diracProb, (antipode_ne 2 p).symm]

end Transformer.Interpolation
