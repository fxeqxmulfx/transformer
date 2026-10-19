module

public import Transformer.Interpolation.Section3_CapAttractor
public import Transformer.Interpolation.Section3_MassConcentration
public import Transformer.Interpolation.Section3_TwoRidgeScaling
public import Transformer.Interpolation.Section3_NullRepeller

/-!
# Constant original controls for spherical ball concentration

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
`eq: claim.statement.lemmaA1` and the quadratic Wasserstein conclusion.

Every proper ball uses the calibrated two-neuron attractor, while a ball
covering the sphere uses a projection with a null repelling pole. For the
given initial probability, dominated convergence supplies one late time
giving the mass and quadratic cost bounds simultaneously. Output scaling
realizes that same endpoint at the requested nonzero terminal time.

The constructed constant matrices have the original width `d`. All weak
solutions are identified with their actual flow pushforwards. The source's
fixed-complement, Lipschitz and bijective flow clauses hold as stated.
-/

@[expose] public section

open Real Set MeasureTheory Filter
open scoped Topology ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A proper ball admits actual constant controls concentrating its mass
in any open inner target region, with the exact collapsed measure as target.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem proper_ball_control (i j : Fin d) (hij : i ≠ j) (z : SSphere d) (R : ℝ)
    (hR : 0 < R) (hR2 : R ≤ 2) (A : Set (SSphere d)) (hA : IsOpen A)
    (hAS : A ⊆ Metric.ball z R) (ω : SSphere d) (hω : ω ∈ A)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0)
    (ν : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (a : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => a) φ ∧
      IsTransportOutside d φ (Metric.ball z R) T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (constNeural d W U a) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d (constNeural d W U a) μ →
        (μ T : Measure (SSphere d)) = Measure.map (φ T) (ν : Measure (SSphere d)) ∧
        ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z R) ≤ (μ T : Measure (SSphere d)) A ∧
        W2 d (μ T) (collapseInto d ν (Metric.ball z R) ω) ≤ ε := by
  obtain ⟨w, h, ℓ, hattract, hfixed⟩ := exists_twoRidge_ball_attractor z ω R hR hR2 (hAS hω)
  let c := 1 - R ^ 2 / 2
  let q := capTargetOutput z w ω c h ℓ
  let Φ (t : ℝ) (x : SSphere d) :=
    twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x t
  have hΦ (t : ℝ) : Measurable (Φ t) := measurable_twoRidgeFlow_initial _ _ _ _ _ _ t
  let η := min ε 1
  have hη : 0 < η := lt_min hε zero_lt_one
  have hη1 : η ≤ 1 := min_le_right _ _
  have hηε : η ≤ ε := min_le_left _ _
  obtain ⟨τ, _, hm, hw⟩ := exists_time_mass_and_W2 ν Φ hΦ (Metric.ball z R) A
    Metric.isOpen_ball.measurableSet hA hAS ω hω (Eventually.of_forall hattract) hfixed η hη hη1
  obtain ⟨W, U, a, φ, hflow, hout, hex, hterm, hmap⟩ := exists_twoRidge_neural_realization i j hij
    (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) (Metric.ball z R) hfixed τ T hT ν
  refine ⟨W, U, a, φ, hflow, hout, hex, fun μ h0 hμ => ?_⟩
  have ht := hterm μ h0 hμ
  refine ⟨?_, ?_, ?_⟩
  · rw [ht, coe_twoRidgeLaw, hmap]
  · rw [ht, coe_twoRidgeLaw]
    exact (mul_le_mul_of_nonneg_right
      (ENNReal.ofReal_le_ofReal (by linarith : 1 - ε ≤ 1 - η)) (by positivity)).trans hm
  · rw [ht, coe_twoRidgeLaw]
    exact hw.trans hηε

/-- Whole-sphere data admit the same original control conclusions by
choosing a nearby attractor whose repelling pole is null for this law.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem whole_sphere_control (hd : 2 ≤ d) (i j : Fin d) (hij : i ≠ j)
    (A : Set (SSphere d)) (hA : IsOpen A) (ω : SSphere d) (hω : ω ∈ A)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (a : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => a) φ ∧
      IsTransportOutside d φ univ T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (constNeural d W U a) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d (constNeural d W U a) μ →
        (μ T : Measure (SSphere d)) = Measure.map (φ T) (ν : Measure (SSphere d)) ∧
        ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) univ ≤ (μ T : Measure (SSphere d)) A ∧
        W2 d (μ T) (collapseInto d ν univ ω) ≤ ε := by
  obtain ⟨p, hpA, hpω, hnull⟩ :=
    exists_target_null_antipode_near hd ν A hA ω hω (ε / 2) (by positivity)
  let Φ (t : ℝ) (x : SSphere d) := twoRidgeFlow (p : EucSpace d) 0 1 0 0 0 x t
  have hΦ (t : ℝ) : Measurable (Φ t) := measurable_twoRidgeFlow_initial _ _ _ _ _ _ t
  have hl : ∀ᵐ x ∂(ν : Measure (SSphere d)),
      Tendsto (fun t => Φ t x) atTop (nhds p) :=
    ae_tendsto_twoRidge_projection ν p hnull
  have hfix : ∀ x ∉ (univ : Set (SSphere d)), ∀ t : ℝ, Φ t x = x :=
    fun x hx => (hx (mem_univ x)).elim
  let η := min ε 1
  have hη : 0 < η := lt_min hε zero_lt_one
  have hηε : η ≤ ε := min_le_left _ _
  have hm := eventually_map_mass_ge ν Φ hΦ univ A
    MeasurableSet.univ hA (subset_univ _) p hpA
    (hl.mono fun _ hx _ => hx) hfix η hη (min_le_right _ _)
  have hw := eventually_W2_map_dirac_lt_of_ae_tendsto ν Φ hΦ p ω hl ε (by linarith)
  obtain ⟨τ, hmτ, hwτ, _⟩ := (hm.and (hw.and (eventually_gt_atTop (0 : ℝ)))).exists
  obtain ⟨W, U, a, φ, hflow, hout, hex, hterm, hmap⟩ := exists_twoRidge_neural_realization i j hij
    (p : EucSpace d) 0 1 0 0 0 univ hfix τ T hT ν
  have hcollapse : collapseInto d ν univ ω = Measure.dirac ω := by simp [collapseInto]
  refine ⟨W, U, a, φ, hflow, hout, hex, fun μ h0 hμ => ?_⟩
  have ht := hterm μ h0 hμ
  refine ⟨?_, ?_, ?_⟩
  · rw [ht, coe_twoRidgeLaw, hmap]
  · rw [ht, coe_twoRidgeLaw]
    exact (mul_le_mul_of_nonneg_right
      (ENNReal.ofReal_le_ofReal (by linarith : 1 - ε ≤ 1 - η)) (by positivity)).trans hmτ
  · rw [ht, coe_twoRidgeLaw, hcollapse]
    exact hwτ.le

/-- An arbitrary ball and open target region satisfy all the source's
constant-control, weak-law, flow, mass and Wasserstein conclusions jointly.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem ball_control (hd : 2 ≤ d) (i j : Fin d) (hij : i ≠ j) (z : SSphere d) (R : ℝ)
    (A : Set (SSphere d)) (hA : IsOpen A) (hAS : A ⊆ Metric.ball z R)
    (ω : SSphere d) (hω : ω ∈ A)
    (ε T : ℝ) (hε : 0 < ε) (hT : T ≠ 0)
    (ν : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (a : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => a) φ ∧
      IsTransportOutside d φ (Metric.ball z R) T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (constNeural d W U a) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d (constNeural d W U a) μ →
        (μ T : Measure (SSphere d)) = Measure.map (φ T) (ν : Measure (SSphere d)) ∧
        ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z R) ≤ (μ T : Measure (SSphere d)) A ∧
        W2 d (μ T) (collapseInto d ν (Metric.ball z R) ω) ≤ ε := by
  by_cases hR2 : R ≤ 2
  · have hwr : dist ω z < R := hAS hω
    exact proper_ball_control i j hij z R ((dist_nonneg : 0 ≤ dist ω z).trans_lt hwr) hR2
      A hA hAS ω hω ε T hε hT ν
  · have hs := spherical_ball_eq_univ_of_two_lt z R (lt_of_not_ge hR2)
    simpa only [hs] using whole_sphere_control hd i j hij A hA ω hω ε T hε hT ν

/-- Two actual hidden coordinates, a genuine open ball and target,
positive accuracy, nonzero time and a two-atom probability jointly
realize all geometric, dimension and initial-law hypotheses. -/
example : ∃ (z ω : SSphere 2) (A : Set (SSphere 2)) (ν : Perspective.ProbSphere 2),
    2 ≤ 2 ∧ (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 ∧
    IsOpen A ∧ A ⊆ Metric.ball z 1 ∧ ω ∈ A ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧
    (ν : Measure (SSphere 2)) univ = 1 := by
  let z : SSphere 2 := basePoint 1
  refine ⟨z, z, Metric.ball z 1, halfDirac 2 z (antipode 2 z),
    le_rfl, by norm_num, zero_lt_one, by norm_num, Metric.isOpen_ball,
    Subset.rfl, ?_, zero_lt_one, by norm_num, by exact IsProbabilityMeasure.measure_univ⟩
  simp

end Transformer.Interpolation
