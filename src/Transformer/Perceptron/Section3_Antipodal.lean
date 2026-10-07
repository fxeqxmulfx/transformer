/-
# Antipodal two-atom SOPD critical points

The second part of `thm: bound`, §3.2 of arXiv:2601.21366v2, excludes
a single cluster at every positive temperature when the perceptron weights
are small. An antipodal pair with zero perceptron weights supplies the
counterexample at small temperature. This file verifies all its structural
hypotheses: stationarity, the second-order condition, and the required
presentation by distinct angles of `[0,2π)`.

SOPD follows from an actual energy minimum along every Monge perturbation.
The perturbation retains the two equal masses, and the antipodal pair
minimizes their interaction energy. Continuity at the start of the curve
then gives a nonnegative second derivative whenever it exists. This is
not a claim that the pair minimizes energy over all probability measures.
`Section3_TwoAtomEnergy` also proves existence of these second variations.
-/

import Transformer.Perceptron.Section3_TwoAtomEnergy
import Transformer.Perceptron.GeodesicCurve
import Transformer.Perceptron.Minimizer

open scoped BigOperators ENNReal
open Real MeasureTheory Filter

namespace Transformer.Perceptron

variable {d : ℕ}

/-- Coincident atoms merge into a single Dirac mass;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`.
This constructor permits coincidence for intermediate pushforwards;
`IsAtomicOnCircle` separately requires distinct initial atoms. -/
theorem twoAtomProb_self (x : SSphere d) :
    twoAtomProb x x = Perspective.diracProb d x := by
  apply ProbabilityMeasure.toMeasure_injective
  rw [coe_twoAtomProb, ← add_smul]
  change ((1 / 2 : ℝ≥0∞) + 1 / 2) • Measure.dirac x = Measure.dirac x
  simp only [one_div, ENNReal.inv_two_add_inv_two, one_smul]

/-- An antipodal pair is stationary for pure attention;
arXiv:2601.21366v2, §2.3, `eq: steady.state` and §3.2, `thm: bound`.
At both support points every interacting vector is radial. -/
theorem isStationary_twoAtomProb_antipodal (β : ℝ) (x : SSphere d) :
    IsStationary β 0 0 0 (twoAtomProb x (Perspective.antipode d x)) := by
  intro z hz
  rcases mem_support_twoAtomProb hz with rfl | rfl
  · rw [energyGrad, drift_zero, add_zero, integral_twoAtomProb]
    simp [proj, Perspective.antipode, inner_neg_right]
  · rw [energyGrad, drift_zero, add_zero, integral_twoAtomProb]
    simp [proj, Perspective.antipode, inner_neg_left]

/-- The antipodal pair has minimum energy along every Monge perturbation
issued from it; arXiv:2601.21366v2, §2.2 and §3.2, proof of `thm: bound`.
Only preservation of the fixed masses is used in the comparison. -/
theorem energy_twoAtomProb_antipodal_geodesic_min (β : ℝ) (hβ : 0 < β)
    (x : SSphere d) (ξ : SSphere d → EucSpace d)
    (ν : ℝ → Perspective.ProbSphere d)
    (hν : IsGeodesicFrom ξ (twoAtomProb x (Perspective.antipode d x)) ν) :
    ∀ t : ℝ, energy β 0 0 0 (ν 0) ≤ energy β 0 0 0 (ν t) := by
  intro t
  obtain ⟨T, hTm, -, hmap⟩ := hν t
  have ht : ν t = twoAtomProb (T x) (T (Perspective.antipode d x)) :=
    ProbabilityMeasure.toMeasure_injective
      (hmap.trans (map_twoAtomProb x (Perspective.antipode d x) T hTm))
  rw [geodesic_zero hν, ht]
  exact energy_twoAtomProb_antipodal_le β hβ x _ _

/-- These hypotheses hold for `β = 1` and the constant geodesic with zero
velocity; arXiv:2601.21366v2, §2.2, `eq:2order`. -/
example : (0 : ℝ) < 1 ∧
    IsGeodesicFrom (fun _ : SSphere 1 => (0 : EucSpace 1))
      (twoAtomProb (basePoint 0) (Perspective.antipode 1 (basePoint 0)))
      (fun _ => twoAtomProb (basePoint 0) (Perspective.antipode 1 (basePoint 0))) := by
  refine ⟨one_pos, fun _ => ⟨id, measurable_id, ?_, Measure.map_id.symm⟩⟩
  intro x
  rw [smul_zero, sphereExp_zero]
  rfl

/-- Every antipodal pair with two fixed equal masses is SOPD for pure
attention at positive temperature; arXiv:2601.21366v2, §2.2, `eq:2order`,
and §3.2, the atomic Hessian in the proof of `thm: bound`.
The proof uses the minimum among fixed-mass pairs, so no unproved
atomicity or strictness result is used. -/
theorem isSOPD_twoAtomProb_antipodal (β : ℝ) (hβ : 0 < β) (x : SSphere d) :
    IsSOPD β 0 0 0 0 (twoAtomProb x (Perspective.antipode d x)) := by
  refine ⟨isStationary_twoAtomProb_antipodal β x, fun ξ _ ν hν H hH => ?_⟩
  have h0 := geodesic_zero hν
  apply nonneg_of_hasDerivAt_deriv_of_isMin (J := fun t => energy β 0 0 0 (ν t))
    (energy_twoAtomProb_antipodal_geodesic_min β hβ x ξ ν hν) _ hH
  show Tendsto (fun t => energy β 0 0 0 (ν t)) (nhds 0)
    (nhds (energy β 0 0 0 (ν 0)))
  rw [h0]
  exact ((continuous_energy β (φ := (0 : ℝ → ℝ)) continuous_const 0 0).tendsto _).comp
    (tendsto_geodesic hν)

/-- The positive-temperature hypothesis is satisfied at `β = 1`;
arXiv:2601.21366v2, §3.2, `thm: bound`. -/
example : (0 : ℝ) < 1 := one_pos

/-- Zero output weights leave the same antipodal SOPD critical point for
any activation and primitive; arXiv:2601.21366v2, §2.3,
`eq: primitive.field` and §3.2, `thm: bound`.
In particular, the counterexample may use the identity activation, whose
Lipschitz constant is exactly one, with its quadratic primitive. -/
theorem isSOPD_twoAtomProb_antipodal_zero_weights (β : ℝ) (hβ : 0 < β)
    (φ σ : ℝ → ℝ) (x : SSphere d) :
    IsSOPD β φ σ 0 0 (twoAtomProb x (Perspective.antipode d x)) := by
  have hbase := isSOPD_twoAtomProb_antipodal β hβ x
  have hE : energy (d := d) β φ 0 0 = energy β (0 : ℝ → ℝ) 0 0 := by
    funext μ
    simp [energy, potential]
  have hS : IsStationary β σ 0 0 (twoAtomProb x (Perspective.antipode d x)) := by
    intro z hz
    simpa only [energyGrad, drift_zero] using hbase.1 z hz
  refine ⟨hS, fun ξ hξ ν hν H hH => ?_⟩
  rw [hE] at hH
  exact hbase.2 ξ hξ ν hν H hH

/-- The temperature hypothesis for this zero-weight family holds at
`β = 1`; arXiv:2601.21366v2, §3.2, `thm: bound`. -/
example : (0 : ℝ) < 1 := one_pos

/-- The angles `0` and `π` for the two-atom counterexample;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
noncomputable def antipodalCircleAngles (i : Idx 2) : ℝ := if i = 0 then 0 else π

/-- The equal-mass antipodal measure in the standard frame of `ℝ²`;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
noncomputable def antipodalCircleProb : Perspective.ProbSphere 2 :=
  twoAtomProb (basePoint 1) (Perspective.antipode 2 (basePoint 1))

/-- The two angles parametrize the stated antipodal pair;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
theorem circlePoint_antipodalCircleAngles (i : Idx 2) :
    circlePoint (antipodalCircleAngles i) =
      if i = 0 then basePoint 1 else Perspective.antipode 2 (basePoint 1) := by
  fin_cases i <;> apply Subtype.ext <;>
    simp [antipodalCircleAngles, circlePoint, greatCircle, Perspective.antipode]

/-- The angles lie in the source's chosen interval;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
theorem antipodalCircleAngles_mem (i : Idx 2) :
    antipodalCircleAngles i ∈ Set.Ico 0 (2 * π) := by
  fin_cases i
  · change 0 ≤ (0 : ℝ) ∧ 0 < 2 * π
    exact ⟨le_rfl, by positivity⟩
  · change 0 ≤ π ∧ π < 2 * π
    exact ⟨Real.pi_pos.le, by linarith [Real.pi_pos]⟩

/-- The angles are pairwise distinct, as required by the source;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
theorem antipodalCircleAngles_injective : Function.Injective antipodalCircleAngles := by
  intro i j hij
  have hp := Real.pi_pos
  fin_cases i <;> fin_cases j <;> simp_all [antipodalCircleAngles]

/-- The antipodal measure has the source's atomic presentation, with
`N = 2`, strictly positive masses summing to one and distinct angles;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
theorem isAtomicOnCircle_antipodalCircleProb :
    IsAtomicOnCircle 2 (fun _ => 1 / 2) antipodalCircleAngles antipodalCircleProb := by
  have hhalf : ENNReal.ofReal (1 / 2 : ℝ) = (1 / 2 : ℝ≥0∞) := by
    rw [ENNReal.ofReal_div_of_pos (by norm_num)]
    norm_num
  refine ⟨le_rfl, fun _ => by norm_num, by norm_num,
    antipodalCircleAngles_mem, antipodalCircleAngles_injective, ?_⟩
  rw [antipodalCircleProb, coe_twoAtomProb, Fin.sum_univ_two]
  simp only [circlePoint_antipodalCircleAngles]
  norm_num [hhalf]

/-- The concrete antipodal measure is SOPD at every positive temperature;
arXiv:2601.21366v2, §2.2, `eq:2order` and §3.2, `thm: bound`. -/
theorem isSOPD_antipodalCircleProb (β : ℝ) (hβ : 0 < β) :
    IsSOPD β 0 0 0 0 antipodalCircleProb :=
  isSOPD_twoAtomProb_antipodal β hβ (basePoint 1)

/-- This SOPD family includes `β = 1`;
arXiv:2601.21366v2, §3.2, `thm: bound`. -/
example : (0 : ℝ) < 1 := one_pos

/-- The energy at the concrete antipodal measure;
arXiv:2601.21366v2, §3.2, the atomic energy in the proof of `thm: bound`.
The cross interaction is `exp(-β)` and each self interaction is `exp β`. -/
theorem energy_antipodalCircleProb (β : ℝ) :
    energy β 0 0 0 antipodalCircleProb =
      (2 * β)⁻¹ * ((Real.exp β + Real.exp (-β)) / 2) := by
  rw [antipodalCircleProb, energy_twoAtomProb_zero]
  simp [Perspective.antipode, inner_neg_right]

end Transformer.Perceptron
