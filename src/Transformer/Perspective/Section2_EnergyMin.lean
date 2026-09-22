/-
# §3.2 — The minimiser of the interaction energy

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2,
`prop: existence.uniqueness.energy`, minimiser half.

`𝖤_β` is invariant under the linear isometries of `ℝ^d`
(`interactionEnergy_map_sphereMap`), since `⟨Ux, Ux'⟩ = ⟨x, x'⟩`.  If a
continuous `V` is invariant too, the image of the minimiser of `𝖤_β + ∫ V` is
again a minimiser, hence the minimiser itself, which is unique
(`Section2_EnergyConvex`): `map_sphereMap_eq_of_isMin`.

`V = 0` is the minimiser half of the proposition
(`existence_uniqueness_energy_min`).  `Perceptron.MinMax` takes the
perceptron's potential for `V`.
-/

import Transformer.Perspective.Section2_EnergyConvex
import Transformer.Perspective.SphereInvariant

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- **`𝖤_β` is invariant under the linear isometries of `ℝ^d`:**
`𝖤_β[U_# μ] = 𝖤_β[μ]`, since `⟨Ux, Ux'⟩ = ⟨x, x'⟩`. -/
theorem interactionEnergy_map_sphereMap (β : ℝ) (μ : ProbSphere d)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    interactionEnergy d β (μ.map (sphereMap d U)) = interactionEnergy d β μ := by
  rw [interactionEnergy_eq_integral_prod, interactionEnergy_eq_integral_prod,
    ProbabilityMeasure.toMeasure_map, Measure.map_prod_map _ _ (measurable_sphereMap d U)
      (measurable_sphereMap d U),
    integral_map ((measurable_sphereMap d U).prodMap (measurable_sphereMap d U)).aemeasurable
      (continuous_expInner_prod β).aestronglyMeasurable]
  simp [sphereMap]

/-- **A minimiser of `𝖤_β + ∫ V` is invariant under the isometries that fix
`V`**, for `β > 0` and continuous `V`: its image has the same energy and the
same potential, so it is a minimiser too, and there is only one
(`eq_of_isMin_interactionEnergy_add`).

Source: arXiv:2312.10794v5, §3.2, `prop: existence.uniqueness.energy`, the
case `V = 0`. -/
theorem map_sphereMap_eq_of_isMin (β : ℝ) (hβ : 0 < β) {V : SSphere d → ℝ}
    (hV : Continuous V) {μ₀ : ProbSphere d}
    (h₀ : ∀ μ : ProbSphere d, interactionEnergy d β μ₀ + ∫ x, V x ∂(μ₀ : Measure (SSphere d)) ≤
      interactionEnergy d β μ + ∫ x, V x ∂(μ : Measure (SSphere d)))
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (hU : ∀ x, V (sphereMap d U x) = V x) :
    (μ₀ : Measure (SSphere d)).map (sphereMap d U) = μ₀ := by
  have hF : interactionEnergy d β (μ₀.map (sphereMap d U)) +
      ∫ x, V x ∂((μ₀.map (sphereMap d U) : ProbSphere d) : Measure (SSphere d)) =
      interactionEnergy d β μ₀ + ∫ x, V x ∂(μ₀ : Measure (SSphere d)) := by
    rw [interactionEnergy_map_sphereMap, ProbabilityMeasure.toMeasure_map,
      integral_map (measurable_sphereMap d U).aemeasurable hV.aestronglyMeasurable]
    simp_rw [hU]
  exact congrArg (fun ν : ProbSphere d => (ν : Measure (SSphere d)))
    (eq_of_isMin_interactionEnergy_add β hβ hV (fun μ => hF.trans_le (h₀ μ)) h₀)

/-- The hypotheses of `map_sphereMap_eq_of_isMin` are satisfiable: `β = 1`,
`V = 0`, `μ₀` a minimiser of `𝖤_1` (`exists_isMin_interactionEnergy_add`) and
`U = -id`. -/
example : ∃ μ₀ : ProbSphere 1, (0 : ℝ) < 1 ∧ Continuous (fun _ : SSphere 1 => (0 : ℝ)) ∧
    (∀ μ : ProbSphere 1, interactionEnergy 1 1 μ₀ + ∫ _, (0 : ℝ) ∂(μ₀ : Measure (SSphere 1)) ≤
      interactionEnergy 1 1 μ + ∫ _, (0 : ℝ) ∂(μ : Measure (SSphere 1))) ∧
    ∀ x : SSphere 1, (fun _ : SSphere 1 => (0 : ℝ)) (sphereMap 1 (LinearIsometryEquiv.neg ℝ) x)
      = (fun _ : SSphere 1 => (0 : ℝ)) x := by
  obtain ⟨μ₀, h₀⟩ := exists_isMin_interactionEnergy_add 1 le_rfl
    (continuous_const : Continuous fun _ : SSphere 1 => (0 : ℝ))
  exact ⟨μ₀, one_pos, continuous_const, h₀, fun _ => rfl⟩

/-- **Proposition (prop: existence.uniqueness.energy), first half.**  For
`β > 0` and `d ≥ 1`, `𝖤_β` has a unique global minimiser over
`𝒫(𝕊^{d-1})`, and that minimiser is the uniform measure `σ_d` on the sphere.

The minimiser is pinned down here by rotation invariance rather than by
name — a probability measure on `𝕊^{d-1}` invariant under every linear
isometry of `ℝ^d` *is* `σ_d` — so that no Haar-measure machinery is needed to
state the proposition.

The source assumes `d ≥ 2`; `d ≥ 1` suffices, and is what is proved.  On
`𝕊^0 = {±1}` the isometries are `±id`, and the invariant measure is
`(δ_1 + δ_{-1})/2`, the uniform measure there.  At `d = 0` the sphere is
empty and there is no minimiser.

The source's proof expands `e^{βt}` in Gegenbauer polynomials; this one
expands it in monomials (`Section2_EnergyConvex`), and gets the invariance
from the uniqueness.

The second half — every global maximiser is a Dirac mass `δ_{x⋆}` — is
`exists_eq_dirac_of_isMaxEnergy`, in `Perspective.Section2_EnergyMax`.

Source: arXiv:2312.10794v5, §3.2, `prop: existence.uniqueness.energy`. -/
theorem existence_uniqueness_energy_min (d : ℕ) (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d) :
    (∃! μ₀ : ProbSphere d, ∀ μ : ProbSphere d,
        interactionEnergy d β μ₀ ≤ interactionEnergy d β μ) ∧
    (∀ μ₀ : ProbSphere d,
        (∀ μ : ProbSphere d, interactionEnergy d β μ₀ ≤ interactionEnergy d β μ) →
        ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
          (μ₀ : Measure (SSphere d)).map (sphereMap d U)
            = (μ₀ : Measure (SSphere d))) := by
  have hV : Continuous fun _ : SSphere d => (0 : ℝ) := continuous_const
  have h := existsUnique_isMin_interactionEnergy_add β hβ hd hV
  simp only [integral_zero, add_zero] at h
  refine ⟨h, fun μ₀ h₀ U => map_sphereMap_eq_of_isMin β hβ hV (fun μ => ?_) U fun _ => rfl⟩
  simpa using h₀ μ

/-- The hypotheses of `existence_uniqueness_energy_min` are satisfiable:
`β = 1` and `d = 1`. -/
example : (0 : ℝ) < 1 ∧ 1 ≤ 1 := ⟨one_pos, le_rfl⟩

end Perspective
end Transformer
