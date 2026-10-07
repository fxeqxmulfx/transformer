/-
# Generic ReLU parameters for a fixed measure

For the quantifiers of arXiv:2601.21366v2, §3.1, `thm: any.d` (iii),
an elementary parameter argument gives a stronger conclusion: the active
region is null for a stationary measure in a dense set of parameters.

At interior stationary parameters, varying an output weight kills each
tangential coefficient. If an input is active at a support point, its
projection is therefore zero. Moving that input along a nonzero tangent
direction preserves its positive activation, but makes its projection
nonzero. This contradicts interior stationarity in dimension at least two.
-/

import Transformer.Perceptron.Section3_StationaryInterior

open Real MeasureTheory Filter Topology

namespace Transformer.Perceptron

variable {d : ℕ}

/-- An interior stationary ReLU parameter has no active support point.
The input perturbation is tangential and keeps the activation fixed;
arXiv:2601.21366v2, §3.1, parameter variations of `thm: any.d` (iii). -/
theorem relu_inner_nonpos_of_mem_interior (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (p : Params d)
    (hp : p ∈ interior (stationaryParams (fun s => max s 0) μ)) (x : SSphere d)
    (hx : x ∈ (μ : Measure (SSphere d)).support) (j : Idx d) :
    inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d) ≤ 0 := by
  by_contra hn
  have hpos : 0 < inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d) := lt_of_not_ge hn
  have hcoef := neuron_coeff_eq_zero_of_mem_interior (fun s => max s 0) μ p hp x hx j
  rw [max_eq_left hpos.le] at hcoef
  have hproj : proj d (x : EucSpace d) (p.2.2 j) = 0 :=
    smul_right_injective (EucSpace d) hpos.ne' (by simpa only [smul_zero] using hcoef)
  have hnorm : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hx0 : (x : EucSpace d) ≠ 0 := norm_ne_zero_iff.mp (by rw [hnorm]; norm_num)
  obtain ⟨v, hv, hxv⟩ := exists_unit_inner_eq_zero hd hx0
  have hv0 : v ≠ 0 := norm_ne_zero_iff.mp (by rw [hv]; norm_num)
  have hvx : inner (𝕜 := ℝ) v (x : EucSpace d) = 0 := by
    rw [real_inner_comm]
    exact hxv
  let path : ℝ → Params d := fun t => (p.1, p.2.1, p.2.2 + Pi.single j (t • v))
  have ha : Continuous (fun t : ℝ => p.2.2 + Pi.single j (t • v)) := by
    apply continuous_pi
    intro k
    simp only [Pi.add_apply, Pi.single_apply]
    split_ifs <;> fun_prop
  have hc : Continuous path := continuous_const.prodMk (continuous_const.prodMk ha)
  have hzero : path 0 = p := by simp [path]
  have hnear : ∀ᶠ t in 𝓝 (0 : ℝ), path t ∈
      interior (stationaryParams (fun s => max s 0) μ) :=
    (hc.tendsto 0).eventually (isOpen_interior.mem_nhds (hzero.symm ▸ hp))
  obtain ⟨δ, hδ, hD⟩ := Metric.eventually_nhds_iff.mp hnear
  have hhalf := hD (y := δ / 2) (by
    rw [Real.dist_eq, sub_zero, abs_of_pos (half_pos hδ)]
    linarith)
  have haj : (path (δ / 2)).2.2 j = p.2.2 j + (δ / 2) • v := by simp [path]
  have hinner : inner (𝕜 := ℝ) ((path (δ / 2)).2.2 j) (x : EucSpace d) =
      inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d) := by
    rw [haj, inner_add_left, real_inner_smul_left, hvx, mul_zero, add_zero]
  have hnew := neuron_coeff_eq_zero_of_mem_interior (fun s => max s 0)
    μ (path (δ / 2)) hhalf x hx j
  rw [hinner, max_eq_left hpos.le] at hnew
  have hnewproj : proj d (x : EucSpace d) ((path (δ / 2)).2.2 j) = 0 :=
    smul_right_injective (EucSpace d) hpos.ne' (by simpa only [smul_zero] using hnew)
  have hpv : proj d (x : EucSpace d) v = v := by
    rw [proj, hxv, zero_smul, sub_zero]
  rw [haj, proj_add_smul, hproj, hpv, zero_add] at hnewproj
  exact hv0 (smul_right_injective (EucSpace d) (half_pos hδ).ne'
    (by simpa only [smul_zero] using hnewproj))

/-- The active region is null at interior stationary parameters: it is
disjoint from the support, whose complement is null;
arXiv:2601.21366v2, §3.1, active regions of `thm: any.d` (iii). -/
theorem measure_active_eq_zero_of_mem_interior (hd : 2 ≤ d)
    (μ : Perspective.ProbSphere d) (p : Params d)
    (hp : p ∈ interior (stationaryParams (fun s => max s 0) μ)) :
    (μ : Measure (SSphere d)) {x : SSphere d | ∃ j : Idx d,
      0 < inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d)} = 0 := by
  apply measure_mono_null _ Measure.measure_compl_support
  intro x hx hs
  obtain ⟨j, hj⟩ := hx
  exact (not_lt_of_ge (relu_inner_nonpos_of_mem_interior hd μ p hp x hs j)) hj

/-- The dense parameter set for the fixed measure: positive temperature and
either nonstationarity or interior stationarity;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
def reluGenericParams (μ : Perspective.ProbSphere d) : Set (Params d) :=
  {p | 0 < p.1 ∧ (p ∉ stationaryParams (fun s => max s 0) μ ∨
    p ∈ interior (stationaryParams (fun s => max s 0) μ))}

/-- The chosen set is dense at every positive temperature;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
theorem positive_subset_closure_reluGenericParams (μ : Perspective.ProbSphere d) :
    {p : Params d | 0 < p.1} ⊆ closure (reluGenericParams μ) := by
  intro p hp
  exact mem_closure_nonstationary_or_interior (fun s => max s 0) μ p hp

/-- On a dense set of parameters, stationarity of the fixed measure forces
zero mass in the active region. This preserves the source's order of
quantifiers and strengthens its countable atomicity conclusion;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
theorem exists_dense_active_null_params (hd : 2 ≤ d) (μ : Perspective.ProbSphere d) :
    ∃ U : Set (Params d), U ⊆ {p | 0 < p.1} ∧ {p | 0 < p.1} ⊆ closure U ∧
      ∀ p ∈ U, IsStationary p.1 (fun s => max s 0) p.2.1 p.2.2 μ →
        (μ : Measure (SSphere d)) {x : SSphere d | ∃ j : Idx d,
          0 < inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d)} = 0 := by
  refine ⟨reluGenericParams μ, fun _ hp => hp.1,
    positive_subset_closure_reluGenericParams μ, ?_⟩
  intro p hp hs
  rcases hp.2 with hn | hint
  · exact (hn hs).elim
  · exact measure_active_eq_zero_of_mem_interior hd μ p hint

/-- The dense set really includes stationary parameters for a Dirac measure
with inactive inputs, for arbitrary output weights. This supplies stationary
witnesses for the fixed-measure generic conclusion;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
theorem dirac_mem_reluGenericParams_and_stationary (β : ℝ) (hβ : 0 < β)
    (ω : Idx d → ℝ) (x : SSphere d) :
    (β, ω, fun _ : Idx d => -(x : EucSpace d)) ∈
        reluGenericParams (Perspective.diracProb d x) ∧
      IsStationary β (fun s => max s 0) ω (fun _ => -(x : EucSpace d))
        (Perspective.diracProb d x) := by
  have hi := mem_interior_stationaryParams_dirac_relu β ω x
  exact ⟨⟨hβ, Or.inr hi⟩, interior_subset hi⟩

/-- The dimension, interior and support hypotheses above are satisfied by a
Dirac atom in dimension two, with all output weights equal to one;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
example : (2 : ℕ) ≤ 2 ∧
    (1, (fun _ : Idx 2 => (1 : ℝ)),
        fun _ : Idx 2 => -((basePoint 1 : SSphere 2) : EucSpace 2)) ∈
      interior (stationaryParams (fun s => max s 0)
        (Perspective.diracProb 2 (basePoint 1))) ∧
    (basePoint 1 : SSphere 2) ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support := by
  refine ⟨le_rfl, mem_interior_stationaryParams_dirac_relu 1 (fun _ => 1) _, ?_⟩
  obtain ⟨y, hy⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)))
  have hyx : y = basePoint 1 := Interpolation.eq_of_mem_support_dirac hy
  exact hyx ▸ hy

/-- The stationary generic witness also has positive temperature;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
example : (0 : ℝ) < 1 ∧
    (1, (fun _ : Idx 2 => (1 : ℝ)),
        fun _ : Idx 2 => -((basePoint 1 : SSphere 2) : EucSpace 2)) ∈
      reluGenericParams (Perspective.diracProb 2 (basePoint 1)) ∧
    IsStationary 1 (fun s => max s 0) (fun _ : Idx 2 => 1)
      (fun _ => -((basePoint 1 : SSphere 2) : EucSpace 2))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨one_pos, dirac_mem_reluGenericParams_and_stationary 1 one_pos (fun _ => 1) _⟩

end Transformer.Perceptron
