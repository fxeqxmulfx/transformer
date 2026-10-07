/-
# Two equally weighted atoms and their interaction energy

The atomic energy in the proof of `thm: bound`, §3.2 of
arXiv:2601.21366v2, evaluated for two atoms of mass `1/2`. With zero
perceptron weights, its only variable term is
`exp(β ⟨x,y⟩)`. For `β > 0` this term is smallest when the two atoms
are antipodal. The masses are fixed throughout this comparison.

This is a minimum among two-atom configurations, not a minimum among
all probability measures. It is nevertheless enough for the source's
second variations along Monge perturbations: a pushforward of two atoms
still has the same two masses. `Section3_Antipodal` uses this fact to
establish SOPD without presupposing any atomicity theorem. The energy
along these curves is twice continuously differentiable.
-/

import Transformer.Perceptron.Atoms
import Transformer.Perceptron.GeodesicCurve
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.ContDiff.Deriv

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer.Perceptron

variable {d : ℕ}

/-- Two atoms of mass `1/2`, the `N = 2` specialization of
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
noncomputable def twoAtomProb (x y : SSphere d) : Perspective.ProbSphere d :=
  ⟨(1 / 2 : ℝ≥0∞) • Measure.dirac x + (1 / 2 : ℝ≥0∞) • Measure.dirac y,
    ⟨by simpa [Measure.add_apply, Measure.smul_apply] using ENNReal.inv_two_add_inv_two⟩⟩

/-- The measure underlying the two-atom constructor;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
@[simp] theorem coe_twoAtomProb (x y : SSphere d) :
    (twoAtomProb x y : Measure (SSphere d)) =
      (1 / 2 : ℝ≥0∞) • Measure.dirac x + (1 / 2 : ℝ≥0∞) • Measure.dirac y := rfl

/-- Integration against the two-atom measure is the average of the two
values; arXiv:2601.21366v2, §3.2, the atomic energy in the proof of
`thm: bound`. No continuity of the integrand is required. -/
theorem integral_twoAtomProb {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E]
    (x y : SSphere d) (f : SSphere d → E) :
    ∫ z, f z ∂(twoAtomProb x y : Measure (SSphere d)) =
      (1 / 2 : ℝ) • f x + (1 / 2 : ℝ) • f y := by
  have hx : Integrable f (Measure.dirac x) := integrable_dirac (by finiteness)
  have hy : Integrable f (Measure.dirac y) := integrable_dirac (by finiteness)
  rw [coe_twoAtomProb,
    integral_add_measure (hx.smul_measure (by norm_num)) (hy.smul_measure (by norm_num)),
    integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac]
  norm_num

/-- The support of the two-atom measure contains no other points;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
theorem mem_support_twoAtomProb {x y z : SSphere d}
    (hz : z ∈ (twoAtomProb x y : Measure (SSphere d)).support) : z = x ∨ z = y := by
  rw [coe_twoAtomProb, Measure.support_add] at hz
  rcases hz with hz | hz
  · exact Or.inl (Interpolation.eq_of_mem_support_dirac
      (Measure.smul_absolutelyContinuous.support_mono hz))
  · exact Or.inr (Interpolation.eq_of_mem_support_dirac
      (Measure.smul_absolutelyContinuous.support_mono hz))

/-- The support hypothesis is satisfiable at either atom;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
example : (basePoint 0 : SSphere 1) ∈
    (twoAtomProb (basePoint 0) (basePoint 0) : Measure (SSphere 1)).support := by
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : (basePoint 0 : SSphere 1) ∈ U := mem_of_mem_nhds hU
  simp [Measure.add_apply, Measure.smul_apply, hx]

/-- A Monge perturbation preserves the two masses;
arXiv:2601.21366v2, §2.2 and §3.2, proof of `thm: bound`. -/
theorem map_twoAtomProb (x y : SSphere d) (T : SSphere d → SSphere d)
    (hT : Measurable T) :
    (twoAtomProb x y : Measure (SSphere d)).map T =
      (twoAtomProb (T x) (T y) : Measure (SSphere d)) := by
  rw [coe_twoAtomProb, coe_twoAtomProb, Measure.map_add _ _ hT]
  simp only [Measure.map_smul _ hT.aemeasurable, Measure.map_dirac]

/-- The map hypothesis holds for the identity;
arXiv:2601.21366v2, §2.2, Monge perturbations. -/
example : Measurable (id : SSphere 1 → SSphere 1) := measurable_id

/-- The interaction energy of two equally weighted atoms;
arXiv:2601.21366v2, §3.2, proof of `thm: bound`.
The self-interactions equal `exp β` since the atoms lie on the unit sphere. -/
theorem interactionEnergy_twoAtomProb (β : ℝ) (x y : SSphere d) :
    Perspective.interactionEnergy d β (twoAtomProb x y) =
      (2 * β)⁻¹ * ((Real.exp β +
        Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) / 2) := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  rw [Perspective.interactionEnergy]
  simp only [integral_twoAtomProb, smul_eq_mul, real_inner_self_eq_norm_sq,
    hx, hy, one_pow, mul_one]
  rw [real_inner_comm (y : EucSpace d) (x : EucSpace d)]
  ring

/-- With zero perceptron weights, the coupled energy has the same explicit
two-atom formula; arXiv:2601.21366v2, §2.3 and §3.2, `thm: bound`. -/
theorem energy_twoAtomProb_zero (β : ℝ) (x y : SSphere d) :
    energy β (0 : ℝ → ℝ) (0 : Idx d → ℝ) (0 : Idx d → EucSpace d)
      (twoAtomProb x y) = (2 * β)⁻¹ * ((Real.exp β +
        Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) / 2) := by
  rw [energy]
  simp only [potential, Pi.zero_apply, Finset.sum_const_zero,
    integral_zero, mul_zero, add_zero]
  exact interactionEnergy_twoAtomProb β x y

/-- At positive temperature, an antipodal pair minimizes the energy among
all configurations with two fixed equal masses;
arXiv:2601.21366v2, §3.2, the atomic energy in the proof of `thm: bound`.
This comparison will certify SOPD for the counterexample to the claimed
exclusion of a single cluster for every `β > 0`. -/
theorem energy_twoAtomProb_antipodal_le (β : ℝ) (hβ : 0 < β)
    (x y z : SSphere d) :
    energy β 0 0 0 (twoAtomProb x (Perspective.antipode d x)) ≤
      energy β 0 0 0 (twoAtomProb y z) := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have hz : ‖(z : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp z.2
  have hinner : -1 ≤ inner (𝕜 := ℝ) (y : EucSpace d) (z : EucSpace d) := by
    have h := abs_real_inner_le_norm (y : EucSpace d) (z : EucSpace d)
    rw [hy, hz, mul_one, abs_le] at h
    exact h.1
  have he : Real.exp (-β) ≤
      Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace d) (z : EucSpace d)) := by
    apply Real.exp_le_exp.mpr
    nlinarith
  rw [energy_twoAtomProb_zero, energy_twoAtomProb_zero]
  simp only [Perspective.antipode, inner_neg_right, real_inner_self_eq_norm_sq,
    hx, one_pow, mul_neg, mul_one]
  exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)

/-- The positive-temperature hypothesis is met at `β = 1`;
arXiv:2601.21366v2, §3.2, `thm: bound`. -/
example : (0 : ℝ) < 1 := one_pos

/-- The zero-weight two-atom energy is `C²` along every Monge geodesic;
arXiv:2601.21366v2, §2.2, `eq:2order` and §3.2, the atomic energy in
`thm: bound`. Fixed atom velocities give explicit great-circle curves. -/
theorem contDiff_energy_twoAtom_geodesic (β : ℝ) (φ : ℝ → ℝ) (x y : SSphere d)
    (ξ : SSphere d → EucSpace d) (ν : ℝ → Perspective.ProbSphere d)
    (hν : IsGeodesicFrom ξ (twoAtomProb x y) ν) :
    ContDiff ℝ 2 (fun t => energy β φ 0 0 (ν t)) := by
  choose T hTm hT hmap using hν
  have hcurve (t : ℝ) : ν t = twoAtomProb (T t x) (T t y) :=
    ProbabilityMeasure.toMeasure_injective
      ((hmap t).trans (map_twoAtomProb x y (T t) (hTm t)))
  have hE : (fun t => energy β φ 0 0 (ν t)) = fun t =>
      (2 * β)⁻¹ * ((Real.exp β + Real.exp (β * inner (𝕜 := ℝ)
        (sphereExp (x : EucSpace d) (t • ξ x))
        (sphereExp (y : EucSpace d) (t • ξ y)))) / 2) := by
    funext t
    rw [hcurve t]
    have hz : energy β φ 0 0 (twoAtomProb (T t x) (T t y)) =
        energy β 0 0 0 (twoAtomProb (T t x) (T t y)) := by simp [energy, potential]
    rw [hz, energy_twoAtomProb_zero, hT t x, hT t y]
  have hc (z : SSphere d) :
      ContDiff ℝ 2 (fun t : ℝ => sphereExp (z : EucSpace d) (t • ξ z)) := by
    simp_rw [sphereExp_smul]
    have hm : ContDiff ℝ 2 (fun t : ℝ => t * ‖ξ z‖) := contDiff_id.mul contDiff_const
    exact (hm.cos.smul_const (z : EucSpace d)).add
      ((hm.sin.div_const ‖ξ z‖).smul_const (ξ z))
  rw [hE]
  exact contDiff_const.mul ((contDiff_const.add
    (contDiff_const.mul ((hc x).inner ℝ (hc y))).exp).div_const 2)

/-- The second variation exists, as the source's SOPD definition requires;
arXiv:2601.21366v2, §2.2, `eq:2order`. Zero output weights remove the
primitive term, so no regularity assumption on `φ` is needed here. -/
theorem exists_secondDeriv_energy_twoAtom_geodesic (β : ℝ) (φ : ℝ → ℝ)
    (x y : SSphere d) (ξ : SSphere d → EucSpace d)
    (ν : ℝ → Perspective.ProbSphere d)
    (hν : IsGeodesicFrom ξ (twoAtomProb x y) ν) :
    ∃ H : ℝ, HasDerivAt (deriv fun t => energy β φ 0 0 (ν t)) H 0 := by
  have hc := contDiff_energy_twoAtom_geodesic β φ x y ξ ν hν
  exact ⟨_, (hc.differentiable_deriv_two 0).hasDerivAt⟩

/-- Both geodesic hypotheses hold for a constant two-atom curve;
arXiv:2601.21366v2, §2.2, Monge perturbations. -/
example (x y : SSphere d) :
    IsGeodesicFrom (fun _ : SSphere d => (0 : EucSpace d))
      (twoAtomProb x y) (fun _ => twoAtomProb x y) := fun _ =>
  ⟨id, measurable_id, fun z => by rw [smul_zero, sphereExp_zero]; rfl, Measure.map_id.symm⟩

end Transformer.Perceptron
