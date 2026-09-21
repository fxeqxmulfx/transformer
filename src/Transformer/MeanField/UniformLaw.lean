/-
# Mean-Field Dynamics — the uniform law on the sphere, and densities against it

`Causal.uniformSphere` is built on the subspace σ-algebra of `𝕊^{d-1}`, while
`𝒫(𝕊^{d-1})` (`Perspective.ProbSphere`) carries the Borel one.  The two are
equal but not by definition; `uniformLaw` is the uniform law read over the
latter, and it is what `thm: mfclust` takes densities against.

`exists_density_mean_pos` shows that the hypotheses of `thm: mfclust` — an
`L²` density with `R₀ > 0` — can be met against any probability measure.

Source: arXiv:2512.01868v4, §4.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Causal.CapMass
import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- The σ-algebra `Perspective.sphereMeasurableSpace` of `𝒫(𝕊^{d-1})` — the
Borel one — is the subspace σ-algebra that `Causal.uniformSphere` is built on;
the two are equal, though not by definition. -/
theorem sphereMeasurableSpace_eq_subtype :
    Perspective.sphereMeasurableSpace d = Subtype.instMeasurableSpace :=
  (Subtype.borelSpace (Metric.sphere (0 : EucSpace d) 1)).measurable_eq.symm

/-- **The uniform law on `𝕊^{d-1}`** over the Borel σ-algebra of
`𝒫(𝕊^{d-1})`: `Causal.uniformSphere` carried across
`sphereMeasurableSpace_eq_subtype`.

Source: arXiv:2512.01868v4, §4 ("the uniform measure on the sphere"). -/
noncomputable def uniformLaw : Measure (SSphere d) :=
  (sphereMeasurableSpace_eq_subtype d).symm ▸ Causal.uniformSphere d

/-- `uniformLaw` gives every set the mass `Causal.uniformSphere` gives it. -/
theorem uniformLaw_apply (s : Set (SSphere d)) :
    uniformLaw d s = Causal.uniformSphere d s := by
  have key : ∀ (m₁ m₂ : MeasurableSpace (SSphere d)) (h : m₁ = m₂)
      (μ : @Measure (SSphere d) m₁), (h ▸ μ : @Measure (SSphere d) m₂) s = μ s := by
    intro m₁ m₂ h μ
    subst h
    rfl
  exact key _ _ _ _

/-- `uniformLaw d` is a probability measure in every positive dimension. -/
theorem isProbabilityMeasure_uniformLaw (hd : 1 ≤ d) :
    IsProbabilityMeasure (uniformLaw d) := by
  have := Causal.isProbabilityMeasure_uniformSphere d hd
  exact ⟨by rw [uniformLaw_apply]; exact measure_univ⟩

/-- The hypothesis of `isProbabilityMeasure_uniformLaw` is satisfiable: `d = 2`. -/
example : 1 ≤ 2 := by norm_num

/-- **An `L²` density with nonzero mean.**  For every probability measure `σ` on
`𝕊^{d-1}` there is a probability measure `μ₀ = f₀ · σ`, with `f₀ ≥ 0` bounded,
whose mean `∫ x dμ₀` is nonzero.

If `σ` has a nonzero mean itself, `f₀ = 1`.  Otherwise
`Σ_k ∫ x_k² dσ = ∫ ‖x‖² dσ = 1`, so some coordinate has `∫ x_k² dσ > 0`, and
`f₀ = 1 + x_k` has mass `1 + ∫ x_k dσ = 1` and mean with `k`-th coordinate
`∫ x_k dσ + ∫ x_k² dσ > 0`.  No symmetry of `σ` is used.

Source: arXiv:2512.01868v4, §4, the hypotheses of `thm:mfclust`. -/
theorem exists_density_mean_pos (σ : Measure (SSphere d)) [IsProbabilityMeasure σ] :
    ∃ (μ₀ : Perspective.ProbSphere d) (f₀ : SSphere d → ℝ), (∀ x, 0 ≤ f₀ x) ∧
      MemLp f₀ 2 σ ∧
      (μ₀ : Measure (SSphere d)) = σ.withDensity (fun x => ENNReal.ofReal (f₀ x)) ∧
      0 < ‖∫ x, (x : EucSpace d) ∂(μ₀ : Measure (SSphere d))‖ ^ 2 := by
  classical
  have hnorm : ∀ x : SSphere d, ‖(x : EucSpace d)‖ = 1 :=
    fun x => mem_sphere_zero_iff_norm.mp x.2
  have hcont : Continuous (fun x : SSphere d => (x : EucSpace d)) := continuous_subtype_val
  have hint : Integrable (fun x : SSphere d => (x : EucSpace d)) σ :=
    Integrable.mono' (integrable_const (1 : ℝ)) hcont.aestronglyMeasurable
      (ae_of_all _ fun x => (hnorm x).le)
  by_cases hm : ∫ x, (x : EucSpace d) ∂σ = 0
  swap
  · refine ⟨⟨σ, inferInstance⟩, fun _ => 1, fun _ => zero_le_one, memLp_const 1, ?_, ?_⟩
    · show σ = σ.withDensity (fun _ => ENNReal.ofReal 1)
      rw [ENNReal.ofReal_one]
      exact withDensity_one.symm
    · exact pow_pos (norm_pos_iff.mpr hm) 2
  -- The coordinates, their integrals, and the squares.
  have hcoord_le : ∀ (x : SSphere d) (k : Fin d), |(x : EucSpace d) k| ≤ 1 := by
    intro x k
    have hsq : ‖(x : EucSpace d)‖ ^ 2 = ∑ i, ‖(x : EucSpace d) i‖ ^ 2 :=
      EuclideanSpace.norm_sq_eq _
    rw [hnorm, one_pow] at hsq
    have hk : ‖(x : EucSpace d) k‖ ^ 2 ≤ 1 :=
      le_of_le_of_eq (Finset.single_le_sum (f := fun i => ‖(x : EucSpace d) i‖ ^ 2)
        (fun i _ => by positivity) (Finset.mem_univ k)) hsq.symm
    rw [Real.norm_eq_abs, sq_le_one_iff_abs_le_one, abs_abs] at hk
    exact hk
  have hccont : ∀ k : Fin d, Continuous (fun x : SSphere d => (x : EucSpace d) k) :=
    fun k => (EuclideanSpace.proj k : EucSpace d →L[ℝ] ℝ).continuous.comp hcont
  have hcint : ∀ k : Fin d, Integrable (fun x : SSphere d => (x : EucSpace d) k) σ :=
    fun k => Integrable.mono' (integrable_const (1 : ℝ)) (hccont k).aestronglyMeasurable
      (ae_of_all _ fun x => by rw [Real.norm_eq_abs]; exact hcoord_le x k)
  have hsqint : ∀ k : Fin d, Integrable (fun x : SSphere d => (x : EucSpace d) k ^ 2) σ :=
    fun k => Integrable.mono' (integrable_const (1 : ℝ))
      ((hccont k).pow 2).aestronglyMeasurable
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_pow, sq_le_one_iff_abs_le_one, abs_abs]
        exact hcoord_le x k)
  have hmk : ∀ k : Fin d, ∫ x, (x : EucSpace d) k ∂σ = 0 := by
    intro k
    have : ∫ x, (x : EucSpace d) k ∂σ = (∫ x, (x : EucSpace d) ∂σ) k :=
      (EuclideanSpace.proj k : EucSpace d →L[ℝ] ℝ).integral_comp_comm hint
    rw [this, hm]
    rfl
  -- Some coordinate has a positive second moment.
  have hsum : ∑ k, ∫ x, (x : EucSpace d) k ^ 2 ∂σ = 1 := by
    rw [← integral_finsetSum _ (fun k _ => hsqint k)]
    have : ∀ x : SSphere d, ∑ k, (x : EucSpace d) k ^ 2 = 1 := by
      intro x
      have hsq := EuclideanSpace.norm_sq_eq (x : EucSpace d)
      rw [hnorm, one_pow] at hsq
      have : ∑ k, (x : EucSpace d) k ^ 2 = ∑ k, ‖(x : EucSpace d) k‖ ^ 2 := by
        simp [Real.norm_eq_abs, sq_abs]
      linarith
    simp [this]
  obtain ⟨k, hk⟩ : ∃ k : Fin d, 0 < ∫ x, (x : EucSpace d) k ^ 2 ∂σ := by
    by_contra h
    push Not at h
    have : ∑ k, ∫ x, (x : EucSpace d) k ^ 2 ∂σ ≤ 0 := Finset.sum_nonpos fun k _ => h k
    linarith
  -- The density `1 + x_k`.
  set f₀ : SSphere d → ℝ := fun x => 1 + (x : EucSpace d) k with hf₀
  have hf0 : ∀ x, 0 ≤ f₀ x := fun x => by
    have := hcoord_le x k
    rw [abs_le] at this
    simp only [hf₀]
    linarith
  have hf₀cont : Continuous f₀ := continuous_const.add (hccont k)
  have hf₀int : Integrable f₀ σ := (integrable_const 1).add (hcint k)
  have hmass : ∫ x, f₀ x ∂σ = 1 := by
    rw [hf₀, integral_add (integrable_const 1) (hcint k), hmk k]
    simp
  have hmeas : Measurable fun x => ENNReal.ofReal (f₀ x) :=
    hf₀cont.measurable.ennreal_ofReal
  have hprob : IsProbabilityMeasure (σ.withDensity fun x => ENNReal.ofReal (f₀ x)) :=
    ⟨by rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
        ← ofReal_integral_eq_lintegral_ofReal hf₀int (ae_of_all _ hf0), hmass,
        ENNReal.ofReal_one]⟩
  refine ⟨⟨_, hprob⟩, f₀, hf0, ?_, rfl, ?_⟩
  · refine MemLp.of_bound hf₀cont.aestronglyMeasurable 2 (ae_of_all _ fun x => ?_)
    have := hcoord_le x k
    rw [abs_le] at this
    rw [Real.norm_eq_abs, abs_le]
    simp only [hf₀]
    constructor <;> linarith
  · -- The `k`-th coordinate of the mean is `∫ x_k² dσ > 0`.
    show 0 < ‖∫ x, (x : EucSpace d) ∂(σ.withDensity fun x => ENNReal.ofReal (f₀ x))‖ ^ 2
    rw [integral_withDensity_eq_integral_toReal_smul hmeas
      (ae_of_all _ fun _ => ENNReal.ofReal_lt_top)]
    have hfun : (fun x : SSphere d => (ENNReal.ofReal (f₀ x)).toReal • (x : EucSpace d))
        = fun x => f₀ x • (x : EucSpace d) :=
      funext fun x => by rw [ENNReal.toReal_ofReal (hf0 x)]
    rw [hfun]
    set v := ∫ x, f₀ x • (x : EucSpace d) ∂σ with hv
    have hvint : Integrable (fun x : SSphere d => f₀ x • (x : EucSpace d)) σ :=
      Integrable.mono' (integrable_const (2 : ℝ))
        (hf₀cont.smul hcont).aestronglyMeasurable
        (ae_of_all _ fun x => by
          rw [norm_smul, hnorm, mul_one, Real.norm_eq_abs, abs_of_nonneg (hf0 x)]
          have := hcoord_le x k
          rw [abs_le] at this
          simp only [hf₀]
          linarith)
    have hvk : v k = ∫ x, (x : EucSpace d) k ^ 2 ∂σ := by
      have : ∫ x, (f₀ x • (x : EucSpace d)) k ∂σ = v k :=
        (EuclideanSpace.proj k : EucSpace d →L[ℝ] ℝ).integral_comp_comm hvint
      rw [← this]
      have : ∀ x : SSphere d, (f₀ x • (x : EucSpace d)) k
          = (x : EucSpace d) k + (x : EucSpace d) k ^ 2 := by
        intro x
        simp only [hf₀, PiLp.smul_apply, smul_eq_mul]
        ring
      simp_rw [this]
      rw [integral_add (hcint k) (hsqint k), hmk k, zero_add]
    have hv0 : v ≠ 0 := by
      intro h
      rw [h] at hvk
      simp only [PiLp.zero_apply] at hvk
      linarith
    exact pow_pos (norm_pos_iff.mpr hv0) 2

end MeanField
end Transformer
