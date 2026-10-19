module

public import Transformer.Perceptron.Section5_KernelClusterBounds
public import Transformer.Perceptron.Section5_InteractionIntegral
public import Transformer.Perceptron.Section3_ClusterCounterexample

/-!
# Interaction-force increments on an actual atomic cluster

ArXiv:2601.21366v2, §5.4, `eq:T-in-bound` and `eq:T-out-bound`.
The source's negative intra-cluster curvature and global positive
curvature control the derivative of the actual interaction force.
Every real representative between the two cluster endpoints is
within the original cluster width of each cluster atom.

Integration against the source's atomic presentation gives the
original masses, including their unit-mass normalization. The
resulting derivative bound is `exp β (3/5-(11/10)M)`, where `M`
is the actual cluster mass. No Hessian or second derivative of
the activation is assumed in this interaction-only estimate.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **Integration uses the masses in the actual atomic presentation.**
Source: arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`, and
§5.4, the finite sums in the proof of `thm: bound`. No regularity
of the integrand is required for this finite Dirac measure. -/
theorem integral_of_isAtomicOnCircle {N : ℕ} {m θ : Idx N → ℝ}
    {μ : Perspective.ProbSphere 2} (hatom : IsAtomicOnCircle N m θ μ)
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (f : SSphere 2 → E) :
    ∫ z, f z ∂(μ : Measure (SSphere 2)) = ∑ i : Idx N, m i • f (circlePoint (θ i)) := by
  rw [hatom.eq_sum, integral_finsetSum_measure (fun i hi =>
    (integrable_dirac (f := f) (a := circlePoint (θ i)) (by finiteness)).smul_measure
      ENNReal.ofReal_ne_top)]
  simp only [integral_smul_measure, integral_dirac]
  apply Finset.sum_congr rfl
  intro i hi
  rw [ENNReal.toReal_ofReal (hatom.mass_pos i).le]

/-- **Every positive-mass atom belongs to the actual topological support.**
Source: arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`, and
§5.4, applying stationarity to each atom in `thm: bound`. -/
theorem circlePoint_mem_support_of_isAtomicOnCircle {N : ℕ} {m θ : Idx N → ℝ}
    {μ : Perspective.ProbSphere 2} (hatom : IsAtomicOnCircle N m θ μ) (i : Idx N) :
    circlePoint (θ i) ∈ (μ : Measure (SSphere 2)).support := by
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : circlePoint (θ i) ∈ U := mem_of_mem_nhds hU
  rw [hatom.eq_sum, Measure.finsetSum_apply]
  have hpos : 0 < (ENNReal.ofReal (m i) • Measure.dirac (circlePoint (θ i))) U := by
    rw [Measure.smul_apply, smul_eq_mul, Measure.dirac_apply_of_mem hx, mul_one]
    exact ENNReal.ofReal_pos.mpr (hatom.mass_pos i)
  have hle := Finset.single_le_sum
    (f := fun j : Idx N => (ENNReal.ofReal (m j) • Measure.dirac (circlePoint (θ j))) U)
    (fun j hj => bot_le) (Finset.mem_univ i)
  exact hpos.trans_le hle

/-- The atomic presentation and support inputs are jointly realized
by the actual antipodal probability with two positive masses. -/
example : IsAtomicOnCircle 2 (fun _ => 1 / 2) antipodalCircleAngles antipodalCircleProb ∧
    circlePoint (antipodalCircleAngles 0) ∈
      (antipodalCircleProb : Measure (SSphere 2)).support :=
  ⟨isAtomicOnCircle_antipodalCircleProb,
    circlePoint_mem_support_of_isAtomicOnCircle isAtomicOnCircle_antipodalCircleProb 0⟩

/-- **The cluster and global curvature bounds aggregate with the actual masses.**
Source: arXiv:2601.21366v2, §5.4, the splitting into
`eq:T-in-bound` and `eq:T-out-bound`. This algebraic step uses
unit total mass and allows zero weights; no geometric input is hidden. -/
theorem weighted_cluster_curvature_upper {N : ℕ} (m k : Idx N → ℝ)
    (hm : ∀ i, 0 ≤ m i) (hmass : ∑ i, m i = 1)
    (S : Finset (Idx N)) (a b : ℝ) (hglobal : ∀ i, k i ≤ b)
    (hcluster : ∀ i ∈ S, k i ≤ -a) :
    (∑ i, m i * k i) ≤ b - (a + b) * (∑ i ∈ S, m i) := by
  classical
  have hfilter : Finset.univ.filter (fun i : Idx N => i ∈ S) = S := by ext i; simp
  have hselected : (∑ i : Idx N, if i ∈ S then m i * (a + b) else 0) =
      (∑ i ∈ S, m i) * (a + b) := by
    rw [← Finset.sum_filter, hfilter, Finset.sum_mul]
  have hsum : (∑ i : Idx N, (m i * k i + (if i ∈ S then m i * (a + b) else 0))) ≤
      ∑ i : Idx N, m i * b := by
    apply Finset.sum_le_sum
    intro i hi
    by_cases hmem : i ∈ S
    · simp only [ite_eq_left hmem]
      nlinarith [mul_le_mul_of_nonneg_left (hcluster i hmem) (hm i)]
    · simp only [ite_eq_right hmem, add_zero]
      exact mul_le_mul_of_nonneg_left (hglobal i) (hm i)
  rw [Finset.sum_add_distrib, hselected, ← Finset.sum_mul, hmass, one_mul] at hsum
  linarith

/-- All weighted-curvature inputs are jointly satisfied by two equal
positive masses and the sharp negative self-curvature coefficients.
The algebra itself permits the full cluster or any smaller subset. -/
example : (∀ i : Idx 2, (0 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Idx 2, (fun _ : Idx 2 => (1 / 2 : ℝ)) i) = 1 ∧
    (∀ i : Idx 2, (fun _ : Idx 2 => (-1 / 2 : ℝ)) i ≤ (3 / 5 : ℝ)) ∧
    (∀ i ∈ (Finset.univ : Finset (Idx 2)),
      (fun _ : Idx 2 => (-1 / 2 : ℝ)) i ≤ -(1 / 2 : ℝ)) := by norm_num

/-- **The actual interaction-force derivative is bounded by cluster mass.**
Source: arXiv:2601.21366v2, §5.4, `eq:T-in-bound`,
`eq:T-out-bound`, and `eq:dist.cond`. The representatives describe
the actual same sphere atoms. Every point of the short interval
has the uniform negative curvature bound against each cluster atom. -/
theorem deriv_atomic_interaction_force_le {N : ℕ} {m θ : Idx N → ℝ}
    {μ : Perspective.ProbSphere 2} (hatom : IsAtomicOnCircle N m θ μ)
    (β : ℝ) (hβ : 4 ≤ β) (S : Finset (Idx N)) (α : Idx N → ℝ)
    (hrep : ∀ i, circlePoint (α i) = circlePoint (θ i)) (L U : ℝ)
    (hwidth : U - L ≤ 1 / (2 * Real.sqrt β))
    (hcluster : ∀ i ∈ S, α i ∈ Set.Icc L U) (r : ℝ) (hr : r ∈ Set.Icc L U) :
    deriv (fun t => ∫ z, circleInteractionForce β t z ∂(μ : Measure (SSphere 2))) r ≤
      Real.exp β * ((3 / 5 : ℝ) - (11 / 10 : ℝ) * (∑ i ∈ S, m i)) := by
  have hβ0 : 0 < β := by linarith
  rw [(hasDerivAt_integral_circleInteractionForce β hβ0 μ r).deriv,
    integral_of_isAtomicOnCircle hatom]
  simp only [smul_eq_mul]
  have hnegative (i : Idx N) (hi : i ∈ S) :
      circleInteractionCurvature β r (circlePoint (θ i)) ≤ -(1 / 2 : ℝ) * Real.exp β := by
    rw [← hrep i]
    apply circleInteractionCurvature_le_cluster β (by linarith) r (α i)
    exact abs_le.mpr ⟨by linarith [(hcluster i hi).1, hr.2],
      by linarith [(hcluster i hi).2, hr.1]⟩
  have h := weighted_cluster_curvature_upper m
    (fun i => circleInteractionCurvature β r (circlePoint (θ i)))
    (fun i => (hatom.mass_pos i).le) hatom.mass_sum S
    ((1 / 2 : ℝ) * Real.exp β) ((3 / 5 : ℝ) * Real.exp β)
    (fun i => circleInteractionCurvature_le_global β hβ r (circlePoint (θ i)))
    (by simpa only [neg_mul] using hnegative)
  convert h using 1
  ring

/-- All derivative-bound inputs are jointly realized by the actual
antipodal probability at `β=4` and the singleton cluster at angle zero.
The interval degenerates to that atom; the subsequent mass theorem
will additionally require two distinct cluster endpoints. -/
example : IsAtomicOnCircle 2 (fun _ => 1 / 2) antipodalCircleAngles antipodalCircleProb ∧
    (4 : ℝ) ≤ 4 ∧ (∀ i : Idx 2, circlePoint (antipodalCircleAngles i) =
      circlePoint (antipodalCircleAngles i)) ∧
    (0 : ℝ) - 0 ≤ 1 / (2 * Real.sqrt 4) ∧
    (∀ i ∈ ({0} : Finset (Idx 2)), antipodalCircleAngles i ∈ Set.Icc 0 0) ∧
    (0 : ℝ) ∈ Set.Icc 0 0 := by
  refine ⟨isAtomicOnCircle_antipodalCircleProb, le_rfl, fun i => rfl,
    by norm_num, ?_, by simp⟩
  intro i hi
  have hi0 : i = 0 := Finset.mem_singleton.mp hi
  subst i
  simp [antipodalCircleAngles]

end Transformer.Perceptron
