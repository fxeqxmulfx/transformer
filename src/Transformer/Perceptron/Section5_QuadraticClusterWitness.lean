module

public import Transformer.Perceptron.Section5_SymmetricForce

/-!
# A complete stationary cluster witness at the large-temperature threshold

ArXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound` and
`eq: pairwise.distance`, and §5.4, `thm: bound`.
The actual equal-mass quadratic pair at `β=4`, angles `1/8,-1/8`,
has source representatives `1/8,2π-1/8` and cluster width `1/4`.
Its single-neuron weight is `-exp(4 cos(1/4))`.

This witness simultaneously satisfies stationarity, the primitive
identity, the one-Lipschitz activation with zero value at zero,
the positive distinct-atom probability presentation, the threshold,
and the original circular-distance condition with two atoms.
It supports the first-order strengthening proved here; the source
assumes SOPD, which is not asserted for this close symmetric pair.
The actual measure is the two-atom probability already constructed.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer.Perceptron

/-- The two distinct source representatives in `[0,2π)`;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
noncomputable def quadraticClusterAngles (i : Idx 2) : ℝ :=
  if i = 0 then 1 / 8 else 2 * π - 1 / 8

/-- The actual symmetric two-atom probability, not a formal list;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
noncomputable def quadraticClusterProb : Perspective.ProbSphere 2 :=
  twoAtomProb (circlePoint (1 / 8)) (circlePoint (-(1 / 8 : ℝ)))

/-- The quadratic pin's actual weight for this configuration;
arXiv:2601.21366v2, §2.3, `eq: primitive.field`, with the stationary
weight computed in `isStationary_twoAtom_quadratic_pin`. -/
noncomputable def quadraticClusterWeights : Idx 2 → ℝ :=
  Pi.single 0 (-Real.exp (4 * Real.cos (1 / 4)))

/-- The single active neuron at the actual first unit axis;
arXiv:2601.21366v2, §2.3, `eq: primitive.field`. -/
noncomputable def quadraticClusterNeuron : Idx 2 → EucSpace 2 :=
  Pi.single 0 ((basePoint 1 : SSphere 2) : EucSpace 2)

/-- **The actual probability has the full positive distinct-atom presentation.**
Source: arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`.
The second source angle differs by one turn from the negative angle
used in the symmetric probability constructor. -/
theorem isAtomicOnCircle_quadraticCluster :
    IsAtomicOnCircle 2 (fun _ => 1 / 2) quadraticClusterAngles quadraticClusterProb := by
  refine {
    two_le := by norm_num
    mass_pos := fun i => by norm_num
    mass_sum := by norm_num
    angle_mem := ?_
    angle_inj := ?_
    eq_sum := ?_ }
  · intro i
    fin_cases i
    · change (1 / 8 : ℝ) ∈ Set.Ico 0 (2 * π)
      exact ⟨by norm_num, by nlinarith [Real.pi_gt_three]⟩
    · change (2 * π - 1 / 8 : ℝ) ∈ Set.Ico 0 (2 * π)
      exact ⟨by nlinarith [Real.pi_gt_three], by linarith⟩
  · intro i j hij
    fin_cases i <;> fin_cases j
    · rfl
    · change (1 / 8 : ℝ) = 2 * π - 1 / 8 at hij
      nlinarith [Real.pi_gt_three]
    · change (2 * π - 1 / 8 : ℝ) = 1 / 8 at hij
      nlinarith [Real.pi_gt_three]
    · rfl
  · rw [quadraticClusterProb, coe_twoAtomProb, Fin.sum_univ_two]
    change (1 / 2 : ℝ≥0∞) • Measure.dirac (circlePoint (1 / 8)) +
      (1 / 2 : ℝ≥0∞) • Measure.dirac (circlePoint (-(1 / 8 : ℝ))) =
      ENNReal.ofReal (1 / 2) • Measure.dirac (circlePoint (1 / 8)) +
      ENNReal.ofReal (1 / 2) • Measure.dirac (circlePoint (2 * π - 1 / 8))
    rw [show (2 * π - 1 / 8 : ℝ) = -(1 / 8 : ℝ) + 2 * π * (1 : ℤ) by ring,
      circlePoint_int_mul_period]
    simp only [ENNReal.ofReal_div_of_pos (show (0 : ℝ) < 2 by norm_num)]
    norm_num

/-- **The actual quadratic cluster is stationary at `β=4`.**
Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`, and §5.4,
the stationary hypotheses of the strengthened mass bound. -/
theorem isStationary_quadraticCluster :
    IsStationary 4 id quadraticClusterWeights quadraticClusterNeuron quadraticClusterProb := by
  simpa only [quadraticClusterWeights, quadraticClusterNeuron, quadraticClusterProb,
    show (2 * (1 / 8) : ℝ) = 1 / 4 by norm_num] using
    isStationary_twoAtom_quadratic_pin 4 (by norm_num) (1 / 8)

/-- The actual source scale at `β=4` is exactly `1/4`.
Source: arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`. -/
theorem quadraticCluster_scale : 1 / (2 * Real.sqrt 4) = (1 / 4 : ℝ) := by
  have hs : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  rw [hs]
  norm_num

/-- **Both atoms satisfy the original circular-distance condition.**
Source: arXiv:2601.21366v2, §3.2, `eq: pairwise.distance`,
and §5.4, `eq:dist.cond`. The pair crosses angle zero; the
nonzero integer turns are necessary for its source representatives. -/
theorem quadraticCluster_distance : ∀ i j : Idx 2, ∃ k : ℤ,
    |quadraticClusterAngles i - quadraticClusterAngles j + 2 * π * (k : ℝ)| ≤
      1 / (2 * Real.sqrt 4) := by
  intro i j
  rw [quadraticCluster_scale]
  fin_cases i <;> fin_cases j
  · exact ⟨0, by norm_num [quadraticClusterAngles]⟩
  · refine ⟨1, ?_⟩
    change |(1 / 8 : ℝ) - (2 * π - 1 / 8) + 2 * π * (1 : ℤ)| ≤ _
    rw [show (1 / 8 : ℝ) - (2 * π - 1 / 8) + 2 * π * (1 : ℤ) = 1 / 4 by ring]
    norm_num
  · refine ⟨-1, ?_⟩
    change |(2 * π - 1 / 8 : ℝ) - 1 / 8 + 2 * π * (-1 : ℤ)| ≤ _
    rw [show (2 * π - 1 / 8 : ℝ) - 1 / 8 + 2 * π * (-1 : ℤ) = -(1 / 4 : ℝ) by ring]
    norm_num
  · exact ⟨0, by norm_num [quadraticClusterAngles]⟩

/-- **The strengthened mass-bound hypotheses have a complete joint witness.**
Source: arXiv:2601.21366v2, §3.2, `thm: bound`, and §5.4,
`eq:dist.cond`. This includes a nonempty two-atom cluster at the
actual threshold, with the actual stationary probability measure. -/
theorem quadraticCluster_stationary_witness : (4 : ℝ) ≤ 4 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id r) r) ∧
    LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
    IsAtomicOnCircle 2 (fun _ => 1 / 2) quadraticClusterAngles quadraticClusterProb ∧
    IsStationary 4 id quadraticClusterWeights quadraticClusterNeuron quadraticClusterProb ∧
    2 ≤ (Finset.univ : Finset (Idx 2)).card ∧
    (∀ i ∈ (Finset.univ : Finset (Idx 2)), ∀ j ∈ Finset.univ, ∃ k : ℤ,
      |quadraticClusterAngles i - quadraticClusterAngles j + 2 * π * (k : ℝ)| ≤
        1 / (2 * Real.sqrt 4)) := by
  exact ⟨le_rfl, fun r => by simpa using hasDerivAt_pow 2 r, LipschitzWith.id, rfl,
    isAtomicOnCircle_quadraticCluster, isStationary_quadraticCluster, by norm_num,
    fun i hi j hj => quadraticCluster_distance i j⟩

/-- The actual interval representatives are `1/8,-1/8`, with the
same sphere points and total width `1/4`. This also verifies the
geometry used to witness the interval version of the mass bound.
Source: arXiv:2601.21366v2, §5.4, `eq:dist.cond`. -/
theorem quadraticCluster_interval :
    let α := fun i : Idx 2 => if i = 0 then (1 / 8 : ℝ) else -(1 / 8 : ℝ)
    (∀ i, circlePoint (α i) = circlePoint (quadraticClusterAngles i)) ∧
    (-(1 / 8 : ℝ) < 1 / 8) ∧
    (1 / 8 : ℝ) - -(1 / 8 : ℝ) ≤ 1 / (2 * Real.sqrt 4) ∧
    ∀ i : Idx 2, α i ∈ Set.Icc (-(1 / 8 : ℝ)) (1 / 8 : ℝ) := by
  intro α
  rw [quadraticCluster_scale]
  refine ⟨?_, by norm_num, by norm_num, ?_⟩
  · intro i
    fin_cases i
    · rfl
    · change circlePoint (-(1 / 8 : ℝ)) = circlePoint (2 * π - 1 / 8)
      rw [show (2 * π - 1 / 8 : ℝ) = -(1 / 8 : ℝ) + 2 * π * (1 : ℤ) by ring,
        circlePoint_int_mul_period]
  · intro i
    fin_cases i <;> norm_num [α]

end Transformer.Perceptron
