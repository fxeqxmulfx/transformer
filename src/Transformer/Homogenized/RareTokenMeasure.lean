/-
# Homogenized Transformers — transporting a rare antipodal token

The measures `(1-ε)δ_{-e₁} + εδ_{e₁}` are actual probability measures on
the unit circle. Moving the rare atom onto `-e₁` gives an explicit transport
plan of cost `4ε`, hence `W₂² ≤ 4ε`. The second basis vector is the query
used to read the sparse Gaussian-product attention entry.

These measures test the uniformity over all sphere-supported token laws in
arXiv:2604.01978v1, `prop:satisfying_MF`, `eq:Lipschitz_measure`.
-/

import Transformer.Homogenized.ProductHeadLaw
import Transformer.Wasserstein.Basic

open scoped ENNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The first basis vector, supporting the two antipodal token atoms. -/
noncomputable def rareAtom : EucSpace 2 := EuclideanSpace.single 0 1

/-- The unit query perpendicular to both token atoms. -/
noncomputable def rareQuery : EucSpace 2 := EuclideanSpace.single 1 1

@[simp] theorem norm_rareAtom : ‖rareAtom‖ = 1 := by simp [rareAtom]

@[simp] theorem norm_rareQuery : ‖rareQuery‖ = 1 := by simp [rareQuery]

/-- The two token locations are a distance exactly `2` apart. -/
theorem dist_rareAtom_neg : dist rareAtom (-rareAtom) = 2 := by
  rw [dist_eq_norm, sub_neg_eq_add, ← two_smul ℝ rareAtom, norm_smul, norm_rareAtom]
  norm_num

/-- The actual two-atom token law, with rare mass `ε` at the positive atom. -/
noncomputable def rareTokenLaw (ε : ℝ) : Measure (EucSpace 2) :=
  ENNReal.ofReal (1 - ε) • Measure.dirac (-rareAtom) +
    ENNReal.ofReal ε • Measure.dirac rareAtom

/-- The two nonnegative masses add to one. -/
theorem rareTokenLaw_weights {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    ENNReal.ofReal (1 - ε) + ENNReal.ofReal ε = 1 := by
  rw [← ENNReal.ofReal_add (sub_nonneg.mpr hε.2) hε.1]
  simp

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- These are genuine probability measures, not merely finite weighted
measures. Source: arXiv:2604.01978v1, `eq:Lipschitz_measure`. -/
theorem isProbabilityMeasure_rareTokenLaw {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    IsProbabilityMeasure (rareTokenLaw ε) := by
  constructor
  simpa [rareTokenLaw] using rareTokenLaw_weights hε

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- Both atoms are unit vectors, so every mass is carried by the sphere. -/
theorem rareTokenLaw_sphere (ε : ℝ) : rareTokenLaw ε {z : EucSpace 2 | ‖z‖ = 1}ᶜ = 0 := by
  simp [rareTokenLaw, Measure.dirac_apply]

/-- Integration against the token law is its exact weighted two-point sum.
Source: arXiv:2604.01978v1, `eq:Lipschitz_measure`. -/
theorem integral_rareTokenLaw {ε : ℝ} (hε : ε ∈ Set.Icc 0 1)
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (f : EucSpace 2 → E) :
    ∫ z, f z ∂rareTokenLaw ε = (1 - ε) • f (-rareAtom) + ε • f rareAtom := by
  rw [rareTokenLaw, integral_add_measure
    ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top)
    ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac,
    ENNReal.toReal_ofReal (sub_nonneg.mpr hε.2), ENNReal.toReal_ofReal hε.1]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- Every finite-valued integrand is integrable at this finite atomic law. -/
theorem integrable_rareTokenLaw (ε : ℝ)
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f : EucSpace 2 → E) : Integrable f (rareTokenLaw ε) := by
  apply Integrable.add_measure
  · exact (integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top
  · exact (integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top

/-- The transport plan keeps the majority atom fixed and moves the rare atom
to the majority atom. -/
noncomputable def rareTokenCoupling (ε : ℝ) : Measure (EucSpace 2 × EucSpace 2) :=
  ENNReal.ofReal (1 - ε) • Measure.dirac (-rareAtom, -rareAtom) +
    ENNReal.ofReal ε • Measure.dirac (rareAtom, -rareAtom)

/-- The displayed plan has exactly the prescribed source and target laws.
Source: arXiv:2604.01978v1, `eq:Lipschitz_measure`. -/
theorem isCoupling_rareTokenLaw {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Wasserstein.IsCoupling (rareTokenLaw ε) (Measure.dirac (-rareAtom))
      (rareTokenCoupling ε) := by
  constructor
  · simp [rareTokenCoupling, rareTokenLaw, Measure.map_add _ _ measurable_fst,
      Measure.map_smul _ measurable_fst.aemeasurable]
  · simp only [rareTokenCoupling, Measure.map_add _ _ measurable_snd,
      Measure.map_smul _ measurable_snd.aemeasurable, Measure.map_dirac]
    rw [← add_smul, rareTokenLaw_weights hε, one_smul]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The cost of the actual plan is exactly the rare mass times `2²`. -/
theorem integral_rareTokenCoupling_cost {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    ∫ p, dist p.1 p.2 ^ 2 ∂rareTokenCoupling ε = 4 * ε := by
  rw [rareTokenCoupling, integral_add_measure
    ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top)
    ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top)]
  simp [dist_rareAtom_neg, ENNReal.toReal_ofReal hε.1]
  ring

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The two token measures approach in `W₂` at least as fast as `√ε`, with
the constant fixed by their actual locations. Source:
arXiv:2604.01978v1, `prop:satisfying_MF`, `eq:Lipschitz_measure`. -/
theorem W2_rareTokenLaw_sq_le {ε : ℝ} (hε : ε ∈ Set.Icc 0 1) :
    Wasserstein.W2 (rareTokenLaw ε) (Measure.dirac (-rareAtom)) ^ 2 ≤ 4 * ε := by
  have h := Wasserstein.W2_le_of_coupling (rareTokenCoupling ε) (isCoupling_rareTokenLaw hε)
  rw [integral_rareTokenCoupling_cost hε] at h
  calc Wasserstein.W2 (rareTokenLaw ε) (Measure.dirac (-rareAtom)) ^ 2
      ≤ Real.sqrt (4 * ε) ^ 2 :=
        pow_le_pow_left₀ (Wasserstein.W2_nonneg _ _) h 2
    _ = 4 * ε := Real.sq_sqrt (mul_nonneg (by norm_num) hε.1)

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The exponentially small perturbation used at Gaussian threshold `u`. -/
noncomputable def rareMass (u : ℝ) : ℝ := Real.exp (-6 * u ^ 2)

/-- Every threshold gives a positive mass at most one. -/
theorem rareMass_mem (u : ℝ) : rareMass u ∈ Set.Ioc 0 1 := by
  refine ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr ?_⟩
  nlinarith [sq_nonneg u]

/-- At the positive Gaussian thresholds both token atoms have positive mass.
Source: arXiv:2604.01978v1, auxiliary to `eq:Lipschitz_measure`. -/
theorem rareMass_lt_one {u : ℝ} (hu : 1 ≤ u) : rareMass u < 1 := by
  apply Real.exp_lt_one_iff.mpr
  have hu0 : 0 < u := zero_lt_one.trans_le hu
  nlinarith [sq_pos_of_pos hu0]

example : (1 : ℝ) ≤ 1 := le_rfl

/-- Every threshold supplies a genuine circle-supported probability law.
Source: arXiv:2604.01978v1, auxiliary to `prop:satisfying_MF`. -/
theorem isProbabilityMeasure_rareTokenLaw_mass (u : ℝ) :
    IsProbabilityMeasure (rareTokenLaw (rareMass u)) :=
  isProbabilityMeasure_rareTokenLaw ⟨(rareMass_mem u).1.le, (rareMass_mem u).2⟩

end Transformer.Homogenized
