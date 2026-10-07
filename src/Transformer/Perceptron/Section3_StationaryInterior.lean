/-
# Interior of the stationary parameter set

For the fixed measure in arXiv:2601.21366v2, §3.1, `thm: any.d` (iii),
stationarity on a parameter neighborhood forces every neuron's tangential
coefficient to vanish on the support. Vary one output weight and subtract
the stationarity equations; the attention term stays fixed.

The union of the nonstationary parameters and the interior of the stationary
parameters is dense. A Dirac measure with strictly inactive inputs supplies
an actual open region of stationary parameters, including nonzero weights.
-/

import Transformer.Perceptron.Section3_Genericity

open Real MeasureTheory Filter Topology
open scoped BigOperators

namespace Transformer.Perceptron

variable {d : ℕ}

/-- Parameters for which the fixed measure is stationary;
arXiv:2601.21366v2, §2.3, `eq: steady.state`, used in §3.1. -/
def stationaryParams (σ : ℝ → ℝ) (μ : Perspective.ProbSphere d) : Set (Params d) :=
  {p | IsStationary p.1 σ p.2.1 p.2.2 μ}

/-- Linearity of the tangent projection in a parameter perturbation;
arXiv:2601.21366v2, §3.1, proof of `thm: any.d` (iii). -/
theorem proj_add_smul (x y v : EucSpace d) (t : ℝ) :
    proj d x (y + t • v) = proj d x y + t • proj d x v := by
  unfold proj
  rw [inner_add_right, inner_smul_right, add_smul, mul_smul, smul_sub]
  abel

/-- Changing one output weight adds its scalar multiple of the neuron's
tangential coefficient. The attention field does not depend on the weights;
arXiv:2601.21366v2, §3.1, the field in the proof of `thm: any.d` (iii). -/
theorem energyGrad_add_weight (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d)
    (j : Idx d) (t : ℝ) :
    energyGrad β σ (ω + Pi.single j t) a μ x = energyGrad β σ ω a μ x +
      t • (σ (inner (𝕜 := ℝ) (a j) x) • proj d x (a j)) := by
  have hs : (∑ k : Idx d, ((ω + (Pi.single j t : Idx d → ℝ)) k *
      σ (inner (𝕜 := ℝ) (a k) x)) • a k) =
      (∑ k : Idx d, (ω k * σ (inner (𝕜 := ℝ) (a k) x)) • a k) +
        (t * σ (inner (𝕜 := ℝ) (a j) x)) • a j := by
    simp only [Pi.add_apply, add_mul, add_smul, Finset.sum_add_distrib,
      Pi.single_apply, ite_mul, zero_mul, ite_smul, zero_smul]
    rw [Fintype.sum_ite_eq']
  unfold energyGrad drift
  rw [hs, proj_add_smul, mul_smul]
  abel

/-- At an interior stationary parameter, every neuron's coefficient vanishes
at every support point. A small nonzero change of its output weight remains
stationary, and subtracting the two field equations proves the assertion;
arXiv:2601.21366v2, §3.1, the parameter variations of `thm: any.d` (iii). -/
theorem neuron_coeff_eq_zero_of_mem_interior (σ : ℝ → ℝ)
    (μ : Perspective.ProbSphere d) (p : Params d)
    (hp : p ∈ interior (stationaryParams σ μ)) (x : SSphere d)
    (hx : x ∈ (μ : Measure (SSphere d)).support) (j : Idx d) :
    σ (inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d)) •
      proj d (x : EucSpace d) (p.2.2 j) = 0 := by
  let path : ℝ → Params d := fun t => (p.1, p.2.1 + Pi.single j t, p.2.2)
  have hw : Continuous (fun t : ℝ => p.2.1 + Pi.single j t) := by
    apply continuous_pi
    intro k
    simp only [Pi.add_apply, Pi.single_apply]
    split_ifs <;> fun_prop
  have hc : Continuous path := continuous_const.prodMk (hw.prodMk continuous_const)
  have hzero : path 0 = p := by simp [path]
  have hnear : ∀ᶠ t in 𝓝 (0 : ℝ), path t ∈ stationaryParams σ μ := by
    exact (hc.tendsto 0).eventually (hzero.symm ▸ mem_interior_iff_mem_nhds.mp hp)
  obtain ⟨δ, hδ, hD⟩ := Metric.eventually_nhds_iff.mp hnear
  have hhalf : path (δ / 2) ∈ stationaryParams σ μ := hD (by
    rw [Real.dist_eq, sub_zero, abs_of_pos (half_pos hδ)]
    linarith)
  have hfield := hhalf x hx
  have hold := (interior_subset hp : p ∈ stationaryParams σ μ) x hx
  dsimp only [path] at hfield
  rw [energyGrad_add_weight p.1 σ p.2.1 p.2.2 μ x j (δ / 2), hold, zero_add] at hfield
  exact smul_right_injective (EucSpace d) (half_pos hδ).ne'
    (by simpa only [smul_zero] using hfield)

/-- Nonstationarity or interior stationarity is dense among positive-temperature
parameters. In any open neighborhood, either a nonstationary parameter exists,
or that neighborhood itself witnesses interior stationarity;
arXiv:2601.21366v2, §3.1, fixed-measure quantifiers of `thm: any.d` (iii). -/
theorem mem_closure_nonstationary_or_interior (σ : ℝ → ℝ)
    (μ : Perspective.ProbSphere d) (p : Params d) (hβ : 0 < p.1) :
    p ∈ closure {q : Params d | 0 < q.1 ∧
      (q ∉ stationaryParams σ μ ∨ q ∈ interior (stationaryParams σ μ))} := by
  apply _root_.mem_closure_iff.mpr
  intro O hO hpO
  let V : Set (Params d) := O ∩ {q | 0 < q.1}
  have hV : IsOpen V := hO.inter (isOpen_lt continuous_const continuous_fst)
  have hpV : p ∈ V := ⟨hpO, hβ⟩
  by_cases hn : ∃ q ∈ V, q ∉ stationaryParams σ μ
  · obtain ⟨q, hq, hqS⟩ := hn
    exact ⟨q, hq.1, hq.2, Or.inl hqS⟩
  · have hVS : V ⊆ stationaryParams σ μ := by
      intro q hq
      by_contra hqS
      exact hn ⟨q, hq, hqS⟩
    exact ⟨p, hpO, hβ, Or.inr (mem_interior.mpr ⟨V, hVS, hV, hpV⟩)⟩

/-- Inputs strictly opposite a Dirac atom lie in the interior stationary set
for ReLU, for arbitrary output weights. Strict negativity persists on an open
parameter neighborhood, where every activation at the atom is zero;
arXiv:2601.21366v2, §3.1, inactive regions in `thm: any.d` (iii). -/
theorem mem_interior_stationaryParams_dirac_relu (β : ℝ) (ω : Idx d → ℝ)
    (x : SSphere d) :
    (β, ω, fun _ : Idx d => -(x : EucSpace d)) ∈
      interior (stationaryParams (fun s => max s 0) (Perspective.diracProb d x)) := by
  let O : Set (Params d) := {p | ∀ j, inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d) < 0}
  have hO : IsOpen O := by
    dsimp [O]
    rw [Set.ofPred_forall]
    apply isOpen_iInter_of_finite
    intro j
    apply isOpen_lt _ continuous_const
    fun_prop
  refine mem_interior.mpr ⟨O, ?_, hO, ?_⟩
  · intro p hp
    apply isStationary_diracProb_of_radial p.1 (fun s => max s 0) p.2.1 p.2.2 x 0
    have hz (j : Idx d) : max (inner (𝕜 := ℝ) (p.2.2 j) (x : EucSpace d)) 0 = 0 :=
      max_eq_right (hp j).le
    simp only [hz, mul_zero, zero_smul, Finset.sum_const_zero]
  · intro j
    have hn : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
    rw [inner_neg_left, real_inner_self_eq_norm_sq, hn]
    norm_num

/-- The coefficient and density hypotheses hold at positive temperature with
nonzero weights: a two-dimensional Dirac atom and all inputs opposite it;
arXiv:2601.21366v2, §3.1, `thm: any.d` (iii). -/
example : (0 : ℝ) < 1 ∧
    (1, (fun _ : Idx 2 => (1 : ℝ)),
        fun _ : Idx 2 => -((basePoint 1 : SSphere 2) : EucSpace 2)) ∈
      interior (stationaryParams (fun s => max s 0)
        (Perspective.diracProb 2 (basePoint 1))) ∧
    (basePoint 1 : SSphere 2) ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support := by
  refine ⟨one_pos, mem_interior_stationaryParams_dirac_relu 1 (fun _ => 1) _, ?_⟩
  obtain ⟨y, hy⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero
    (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)))
  have hyx : y = basePoint 1 := Interpolation.eq_of_mem_support_dirac hy
  exact hyx ▸ hy

end Transformer.Perceptron
