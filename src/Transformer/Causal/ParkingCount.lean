/-
# Causal attention — Counting strong R'enyi centers (§ "R'enyi centers vs
strong R'enyi centers" of 2411.04990v2)

The meta-stable cluster nuclei of `Causal/Metastability.lean` are the *strong*
R'enyi centers of the initial configuration, and the survey counts them: for an
i.i.d. sequence on `𝕊^{d-1}` whose spherical caps all carry the same mass, the
expected number of strong centers at separation `δ` is the reciprocal of that
cap mass.

The unlabelled lemma of that subsection is stated here and not proved, together
with the cap masses it specializes to in dimensions two and three.
-/

import Transformer.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Constructions.HaarToSphere
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (d : ℕ)

/-- The **geodesic distance** on the unit sphere: the angle between two unit
vectors, `arccos ⟨x, y⟩`.  This is the `dist` of the survey's R'enyi parking
definitions, as opposed to the ambient distance used in
`Transformer.Causal.isStrongRenyiCenters`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def geoDist (x y : SSphere d) : ℝ :=
  Real.arccos (inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))

/-- The **probability that the `k`-th point of an i.i.d. sequence is a strong
R'enyi center** at separation `δ`: it is farther than `δ` from every one of its
predecessors.

The event involves only the first `k + 1` points, so it is written against the
finite product measure `μ^{⊗(k+1)}` rather than an infinite product.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def strongCenterProb
    (μ : Measure (SSphere d)) (δ : ℝ) (k : ℕ) : ℝ≥0∞ :=
  (Measure.pi fun _ : Fin (k + 1) => μ)
    { x : Fin (k + 1) → SSphere d |
        ∀ i : Fin (k + 1), (i : ℕ) < k → δ < geoDist d (x (Fin.last k)) (x i) }

/-- **Lemma (strong R'enyi parking count).**

For an infinite i.i.d. sequence on `𝕊^{d-1}` sampled from a distribution `μ`
all of whose geodesic caps of radius `δ` carry the same mass
`σ^{d-1}(B_δ) > 0` — the survey's "spherically harmonic" distributions, the
uniform measure among them — the average number of points chosen by strong
R'enyi parking is

  `1 / σ^{d-1}(B_δ)`.

The average is the sum over `k` of the probabilities that the `k`-th point is
chosen, which is what the survey's proof computes; the constancy of the cap
mass is carried as the hypothesis `hharm` rather than derived from a symmetry
assumption on `μ`.

Not proved here.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem strong_renyi_expected_count
    (μ : Measure (SSphere d)) [IsProbabilityMeasure μ] (δ capMass : ℝ)
    (hcap : 0 < capMass)
    (hharm : ∀ x : SSphere d,
      μ { y : SSphere d | geoDist d x y ≤ δ } = ENNReal.ofReal capMass) :
    ∑' k : ℕ, strongCenterProb d μ δ k = ENNReal.ofReal capMass⁻¹ := by
  sorry

/-- The unit vector `e_0` of `EucSpace 1`, as a point of the sphere. -/
noncomputable def northPole : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The hypotheses of `strong_renyi_expected_count` are satisfiable: at
`δ = π` every geodesic cap is the whole sphere, so any probability measure —
here a Dirac mass on `𝕊^0` — has constant cap mass `1`. -/
example :
    (0 : ℝ) < 1 ∧ ∀ x : SSphere 1,
      (Measure.dirac northPole) { y : SSphere 1 | geoDist 1 x y ≤ Real.pi }
        = ENNReal.ofReal 1 := by
  refine ⟨one_pos, fun x => ?_⟩
  have hset : { y : SSphere 1 | geoDist 1 x y ≤ Real.pi } = Set.univ := by
    ext y
    simp [geoDist, Real.arccos_le_pi]
  rw [hset, ENNReal.ofReal_one]
  exact measure_univ

/-- The **uniform probability measure on `𝕊^{d-1}`**: the spherical part of
Lebesgue measure, normalized to unit mass.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def uniformSphere : Measure (SSphere d) :=
  ((volume : Measure (EucSpace d)).toSphere Set.univ)⁻¹ •
    (volume : Measure (EucSpace d)).toSphere

/-- **Cap mass in dimension two.**

On the circle `𝕊^1`, the uniform geodesic cap of radius `δ` has mass `δ / π`,
so the expected number of strong R'enyi centers is `π δ^{-1}`.

Not proved here.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_two (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 2,
      uniformSphere 2 { y : SSphere 2 | geoDist 2 x y ≤ δ }
        = ENNReal.ofReal (δ / Real.pi) := by
  sorry

/-- The hypothesis of `uniform_cap_mass_two` is satisfiable: `δ = 0`. -/
example : (0 : ℝ) ∈ Set.Icc 0 Real.pi :=
  ⟨le_rfl, Real.pi_pos.le⟩

/-- **Cap mass in dimension three.**

On `𝕊^2` the uniform geodesic cap of radius `δ` has mass `sin²(δ/2)`, so the
expected number of strong R'enyi centers is `(sin²(δ/2))^{-1}`.

The survey writes this cap mass as `3 sin²(δ/2)`; that cannot be a normalized
cap mass, since at `δ = π` the cap is the whole sphere and the value would be
`3`.  What is stated here is the cap mass itself, `(1 - cos δ) / 2 =
sin²(δ/2)`, which agrees with the two-dimensional case in being a probability.

Not proved here.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem uniform_cap_mass_three (δ : ℝ) (hδ : δ ∈ Set.Icc 0 Real.pi) :
    ∀ x : SSphere 3,
      uniformSphere 3 { y : SSphere 3 | geoDist 3 x y ≤ δ }
        = ENNReal.ofReal (Real.sin (δ / 2) ^ 2) := by
  sorry

/-- The hypothesis of `uniform_cap_mass_three` is satisfiable: `δ = π`, where
the cap is the whole sphere and `sin²(π/2) = 1`. -/
example : Real.pi ∈ Set.Icc 0 Real.pi ∧ Real.sin (Real.pi / 2) ^ 2 = 1 := by
  refine ⟨⟨Real.pi_pos.le, le_rfl⟩, ?_⟩
  simp

end Causal
end Transformer
