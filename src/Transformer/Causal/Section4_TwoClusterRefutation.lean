module

public import Transformer.Causal.Section4_AnisotropicBarrier
public import Transformer.Causal.UniformAtoms
public import Transformer.Metastability.Section4_UniformLaw
public import Transformer.Perspective.Section3_SmallBeta

/-!
# A positive-measure counterexample to the two-cluster conjecture

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`.
The conjecture asserts almost-everywhere convergence to the two
leading eigenvectors for arbitrary query and key matrices whenever
the value matrix has distinct positive real eigenvalues.

This is false for `V=diag(2,1)`, `Q=I`, `K=diag(0,4)`, `β=1`.
The original coordinate region proved invariant in the preceding
module is nonempty and open in the original two-token sphere.
The actual independent uniform initialization gives it positive mass.
Every point of this region has a true global causal trajectory whose
follower remains above second coordinate one quarter for all future
time, so its limit cannot be either leading eigenvector.

The counterexample exhibits all original spectral and uniform-law
hypotheses, including both distinct positive eigenvalues.
The leader has first coordinate above `99/100` and second-coordinate
absolute value below one hundredth. The follower has negative first
coordinate and second coordinate between one quarter and one half.
The original independent uniform initialization gives strictly
positive probability to this behavior.
-/

@[expose] public section

open Real Set MeasureTheory Filter Topology

namespace Transformer.Causal

open Perceptron

/-- The actual law of two independent uniform unit-circle points.
Source: arXiv:2411.04990v2, §4, the initialization in `thm1.5`. -/
noncomputable def anisotropicUniformLaw : Measure (SphereTuple 2 2) :=
  Measure.pi (fun _ : Fin 2 => MeanField.uniformLaw 2)

/-- The actual law satisfies exactly the original uniform-initialization
predicate. Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`. -/
theorem uniformTuple_anisotropicUniformLaw : Perspective.UniformTuple 2 2 anisotropicUniformLaw := by
  have h := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨MeanField.uniformLaw 2, h.1, h.2, rfl⟩

/-- The actual original product law is a probability measure.
Source: arXiv:2411.04990v2, §4, the probability in `thm1.5`. -/
theorem isProbabilityMeasure_anisotropicUniformLaw : IsProbabilityMeasure anisotropicUniformLaw := by
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) :=
    MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
  unfold anisotropicUniformLaw
  infer_instance

/-- The actual invariant coordinate region is open in the original
product sphere. Source: arXiv:2411.04990v2, §4, the state space of `thm1.5`. -/
theorem isOpen_anisotropicInitialRegion : IsOpen {X : SphereTuple 2 2 | anisotropicInitialRegion X} := by
  have hc (k i : Fin 2) : Continuous (fun X : SphereTuple 2 2 => (X k : EucSpace 2) i) :=
    (EuclideanSpace.proj i : EucSpace 2 →L[ℝ] ℝ).continuous.comp
      (continuous_subtype_val.comp (continuous_apply k))
  have h := (isOpen_lt (hc 0 1).abs (continuous_const : Continuous (fun _ : SphereTuple 2 2 => (1 / 100 : ℝ)))).inter
    ((isOpen_lt (continuous_const : Continuous (fun _ : SphereTuple 2 2 => (99 / 100 : ℝ))) (hc 0 0)).inter
      (((isOpen_lt (continuous_const : Continuous (fun _ : SphereTuple 2 2 => (1 / 4 : ℝ))) (hc 1 1)).inter
        (isOpen_lt (hc 1 1) (continuous_const : Continuous (fun _ : SphereTuple 2 2 => (1 / 2 : ℝ))))).inter
        (isOpen_lt (hc 1 0) (continuous_const : Continuous (fun _ : SphereTuple 2 2 => (0 : ℝ))))))
  simpa only [anisotropicInitialRegion, Set.ofPred_and, Set.mem_Ioo] using h

/-- The original nonempty invariant region has strictly positive mass
under the actual uniform product law. Source: arXiv:2411.04990v2,
§4, the almost-everywhere quantifier in Conjecture `thm1.5`.
Positivity follows from the genuine uniform spherical measure and its product. -/
theorem anisotropicInitialRegion_measure_pos :
    0 < anisotropicUniformLaw {X : SphereTuple 2 2 | anisotropicInitialRegion X} := by
  have : IsProbabilityMeasure (MeanField.uniformLaw 2) :=
    MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
  have : (MeanField.uniformLaw 2).IsOpenPosMeasure := by
    constructor
    intro U hU hne
    rw [MeanField.uniformLaw_apply]
    exact (uniformSphere_pos_of_isOpen 2 hU hne).ne'
  have : anisotropicUniformLaw.IsOpenPosMeasure := by
    unfold anisotropicUniformLaw
    infer_instance
  exact isOpen_anisotropicInitialRegion.measure_pos anisotropicUniformLaw
    ⟨(fun i => circlePoint ((![0, π - Real.arcsin (3 / 8)] : Fin 2 → ℝ) i)),
      anisotropicInitialRegion_witness⟩

/-- **Counterexample to Conjecture `thm1.5`, its actual almost-everywhere
conclusion.** The original uniform product initialization does not give
almost-sure convergence of every token to the leading unit eigenvectors
for these actual matrices. Source: arXiv:2411.04990v2, §4.
The witness trajectories solve the original masked equation for all real times. -/
theorem not_ae_two_cluster_anisotropic :
    ¬ ∀ᵐ X₀ ∂anisotropicUniformLaw, ∀ X : ℝ → SphereTuple 2 2, X 0 = X₀ →
      CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap X →
        ∀ k : Idx 2,
          Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (basePoint 1 : EucSpace 2)) ∨
          Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (-(basePoint 1 : EucSpace 2))) := by
  intro hae
  have hnull := ae_iff.mp hae
  have hregionNull : anisotropicUniformLaw {X₀ | anisotropicInitialRegion X₀} = 0 := by
    apply measure_mono_null _ hnull
    intro X₀ hinit hgood
    obtain ⟨X, h0, hX, hi⟩ := exists_anisotropic_CSA_in_region X₀ hinit
    rcases hgood X h0 hX (1 : Fin 2) with hplus | hminus
    · exact not_tendsto_axis_of_anisotropicRegion X hi (basePoint 1 : EucSpace 2)
        (by simp [basePoint]) hplus
    · exact not_tendsto_axis_of_anisotropicRegion X hi (-(basePoint 1 : EucSpace 2))
        (by simp [basePoint]) hminus
  exact anisotropicInitialRegion_measure_pos.ne' hregionNull

/-- **The full original hypotheses and failure of Conjecture `thm1.5`
are exhibited together.** Temperature one, an actual Euclidean eigenbasis
with distinct positive eigenvalues two and one, its leading unit vector,
and the genuine original uniform initialization satisfy the source's
assumptions. Its stated almost-everywhere two-cluster conclusion fails.
Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`. -/
theorem two_cluster_counterexample :
    (0 : ℝ) ≤ 1 ∧
    (∃ (e : Idx 2 → EucSpace 2) (μ : Idx 2 → ℝ), LinearIndependent ℝ e ∧
      (∀ i, anisotropicValueMap (e i) = μ i • e i) ∧ Function.Injective μ ∧
      (∀ i, 0 < μ i) ∧ ∀ i, μ i ≤ 2) ∧
    anisotropicValueMap (basePoint 1 : EucSpace 2) = (2 : ℝ) • (basePoint 1 : EucSpace 2) ∧
    Perspective.UniformTuple 2 2 anisotropicUniformLaw ∧
    ¬ ∀ᵐ X₀ ∂anisotropicUniformLaw, ∀ X : ℝ → SphereTuple 2 2, X 0 = X₀ →
      CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap X →
        ∀ k : Idx 2,
          Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (basePoint 1 : EucSpace 2)) ∨
          Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (-(basePoint 1 : EucSpace 2))) :=
  ⟨zero_le_one, anisotropicValueMap_eigenbasis, anisotropicValueMap_leading,
    uniformTuple_anisotropicUniformLaw, not_ae_two_cluster_anisotropic⟩

/-- The proposed two-cluster assertion fails even with both dimensions
fixed at two, under the literal original spectral hypotheses and uniform law.
Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`. -/
theorem not_forall_two_cluster :
    ¬ ∀ (β : ℝ) (Q K V : ParamMatrix 2) (lam : ℝ) (ξ : SSphere 2), 0 ≤ β →
      (∃ (e : Idx 2 → EucSpace 2) (μ : Idx 2 → ℝ), LinearIndependent ℝ e ∧
        (∀ i, V (e i) = μ i • e i) ∧ Function.Injective μ ∧ (∀ i, 0 < μ i) ∧ ∀ i, μ i ≤ lam) →
      V (ξ : EucSpace 2) = lam • (ξ : EucSpace 2) →
      ∀ σ : Measure (SphereTuple 2 2), Perspective.UniformTuple 2 2 σ →
        ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple 2 2, X 0 = X₀ → CSA 2 2 β Q K V X →
          ∀ k : Idx 2,
            Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (ξ : EucSpace 2)) ∨
            Tendsto (fun t => (X t k : EucSpace 2)) atTop (nhds (-(ξ : EucSpace 2))) := by
  intro h
  exact not_ae_two_cluster_anisotropic (h 1 (ContinuousLinearMap.id ℝ (EucSpace 2))
    anisotropicKeyMap anisotropicValueMap 2 (basePoint 1) zero_le_one anisotropicValueMap_eigenbasis
    anisotropicValueMap_leading anisotropicUniformLaw uniformTuple_anisotropicUniformLaw)

end Transformer.Causal
