module

public import Transformer.Interpolation.Section5_ClusterRotations
public import Transformer.Perspective.UniformAtomless

/-!
# The actual cluster law is atomless, and independent clusters differ

ArXiv:2411.04551v3, §5, `prop:generic-discrete`.

The actual cluster map is almost-everywhere measurable and its law is
rotation invariant. That probability law has no atoms in dimension at
least two, by transitivity of the orthogonal action and the infinitude
of the sphere. Mapping independent input tuples coordinatewise gives
the product of their cluster laws. Its diagonal has mass zero.

Almost-everywhere measurability suffices to push forward the product law.
The complement of the null collision event still has probability one.

The conclusions are proved from the explicit empirical clustering input;
the source's almost-sure clustering citation is not used as an axiom.
All hypotheses are realized by actual one-token flows and the constructed
uniform spherical law, including the full empirical-limit condition.
-/

@[expose] public section

open MeasureTheory Filter Topology

namespace Transformer.Interpolation

variable {d n : ℕ}

/-- The actual cluster law is a probability measure.
Source: arXiv:2411.04551v3, §5, the pushforward law in
`prop:generic-discrete`; measurability follows from the proved flow. -/
theorem isProbabilityMeasure_clusterLaw (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X)))) :
    IsProbabilityMeasure (ref.map f) := by
  have := isProbabilityMeasure_of_uniformTuple ref href
  exact (Measure.isProbabilityMeasure_map_iff
    (aemeasurable_of_empirical_saFlow_cluster β hβ hn ref f hf)).mpr inferInstance

example : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    ∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0))) :=
  ⟨le_rfl, one_pos, uniformTuple_uniformLaw (by norm_num) 1,
    empirical_saFlow_one_clusters 0 _⟩

/-- **The actual cluster law has no atoms.** The stronger dimension bound
`d ≥ 2` suffices for the source's `d ≥ 3` conclusion.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`, its atom argument. -/
theorem clusterLaw_singleton_zero (hd : 2 ≤ d) (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X))))
    (z : SSphere d) : (ref.map f) {z} = 0 := by
  have := isProbabilityMeasure_clusterLaw β hβ hn ref href f hf
  exact Perspective.measure_singleton_eq_zero_of_invariant d hd (ref.map f)
    (map_clusterLaw_rotation β hβ hn ref href f hf) z

example : 2 ≤ 3 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    ∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0))) :=
  ⟨by norm_num, le_rfl, one_pos, uniformTuple_uniformLaw (by norm_num) 1,
    empirical_saFlow_one_clusters 0 _⟩

/-- **Independent empirical inputs have equal actual limit clusters with
probability zero.** Coordinatewise pushforward preserves the product law,
whose diagonal is null by the atomless spherical-law argument.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem measure_independent_clusters_eq_zero (N : ℕ) (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X))))
    (i j : Idx N) (hij : i ≠ j) :
    (Measure.pi fun _ : Idx N => ref)
      {X : Idx N → SphereTuple d n | f (X i) = f (X j)} = 0 := by
  have := isProbabilityMeasure_of_uniformTuple ref href
  have := isProbabilityMeasure_clusterLaw β hβ hn ref href f hf
  have hmf := aemeasurable_of_empirical_saFlow_cluster β hβ hn ref f hf
  let ν := ref.map f
  let g (X : Idx N → SphereTuple d n) : SphereTuple d N := fun k => f (X k)
  have hmg : AEMeasurable g (Measure.pi fun _ : Idx N => ref) :=
    AEMeasurable.of_eval fun k => hmf.comp_quasiMeasurePreserving
      (Measure.quasiMeasurePreserving_eval (fun _ : Idx N => ref) k)
  have hmap : (Measure.pi fun _ : Idx N => ref).map g =
      Measure.pi fun _ : Idx N => ν := Measure.pi_map_pi (fun _ => hmf)
  have hν : Perspective.UniformTuple d N (Measure.pi fun _ : Idx N => ν) :=
    ⟨ν, inferInstance, map_clusterLaw_rotation β hβ hn ref href f hf, rfl⟩
  have hs : MeasurableSet {Y : SphereTuple d N | Y i = Y j} :=
    measurableSet_eq_fun (measurable_pi_apply i) (measurable_pi_apply j)
  change (Measure.pi fun _ : Idx N => ref) (g ⁻¹' {Y : SphereTuple d N | Y i = Y j}) = 0
  rw [← Measure.map_apply_of_aemeasurable hmg hs, hmap]
  exact Perspective.measure_coords_eq_eq_zero d N hd _ hν hij

/-- Every hypothesis, including distinct sample indices and the full
actual clustering input, has a concrete uniform one-token model. -/
example : 2 ≤ 3 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    (∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0)))) ∧ (0 : Idx 2) ≠ 1 :=
  ⟨by norm_num, le_rfl, one_pos, uniformTuple_uniformLaw (by norm_num) 1,
    empirical_saFlow_one_clusters 0 _, by decide⟩

/-- **Independent actual clusters are distinct with probability one.**
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`.
The empirical clustering conclusion is an explicit input. -/
theorem measure_independent_clusters_ne_one (N : ℕ) (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (href : Perspective.UniformTuple d n ref)
    (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X))))
    (i j : Idx N) (hij : i ≠ j) :
    (Measure.pi fun _ : Idx N => ref)
      {X : Idx N → SphereTuple d n | f (X i) ≠ f (X j)} = 1 := by
  have := isProbabilityMeasure_of_uniformTuple ref href
  have hzero := measure_independent_clusters_eq_zero N hd β hβ hn ref href f hf i j hij
  have hcompl : (Measure.pi fun _ : Idx N => ref)
      {X : Idx N → SphereTuple d n | f (X i) ≠ f (X j)}ᶜ = 0 := by
    simpa only [Set.compl_ofPred, not_not] using hzero
  exact (measure_of_measure_compl_eq_zero hcompl).trans measure_univ

example : 2 ≤ 3 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧ Perspective.UniformTuple 3 1
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3) ∧
    (∀ᵐ X ∂(Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0)))) ∧ (0 : Idx 2) ≠ 1 :=
  ⟨by norm_num, le_rfl, one_pos, uniformTuple_uniformLaw (by norm_num) 1,
    empirical_saFlow_one_clusters 0 _, by decide⟩

/-- The distinct-cluster conclusion itself holds for two actual uniform
one-token inputs in dimension three, with no clustering assumption left.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
example :
    (Measure.pi fun _ : Idx 2 => Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3)
      {X : Idx 2 → SphereTuple 3 1 | X 0 0 ≠ X 1 0} = 1 := by
  exact measure_independent_clusters_ne_one 2 (by norm_num) 0 le_rfl one_pos
    (Measure.pi fun _ : Idx 1 => MeanField.uniformLaw 3)
    (uniformTuple_uniformLaw (by norm_num) 1) (fun X => X 0)
    (empirical_saFlow_one_clusters 0 _) 0 1 (by decide)

end Transformer.Interpolation
