/-
# Metastability — the energy window of §4 can be empty

The problem of `sec: energy.levels` (arXiv:2410.06833v1, §4) asks for `1 > c₂ > c₁ > 0`, depending
on `β`, such that every configuration with

  `c₂ ≥ 𝖤_β(x) - 𝔼[𝖤_β(U)] ≥ c₁`

is metastable.  The source does not say how large the window `[c₁, c₂]` has to be, and read
literally the question has the trivial answer "yes, an empty window": the printed energy is at
most `1/(2β)` (`Eβ_le`) and the mean is nonnegative, so for `β > 1/2` a `c₁ ≥ 1/(2β)` leaves no
configuration in the window (`printed_energy_level_trivial`).  This is why
`energy_level_metastability` asks in addition that the window be nonempty.
-/

import Transformer.Metastability.EnergyScale
import Transformer.Metastability.InitialUniform

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- **With the printed energy the window of `sec: energy.levels` can be chosen empty.**

For `β > 1/2` there are `1 > c₂ > c₁ > 0` such that *no* configuration `x` has
`c₁ ≤ 𝖤_β(x) - 𝔼[𝖤_β(U)] ≤ c₂`: `𝖤_β(x) ≤ 1/(2β) < c₁` (`Eβ_le`), and the mean of the
nonnegative `𝖤_β` is nonnegative.  Any statement of the form "every configuration of the window
is metastable" is then true for this window without saying anything about the dynamics.

Source: arXiv:2410.06833v1, §4, `sec: energy.levels`, with `𝖤_β` of §1. -/
theorem printed_energy_level_trivial (d n : ℕ) (ν : Measure (SSphere d)) {β : ℝ}
    (hβ : 1 / 2 < β) :
    ∃ c₁ c₂ : ℝ, 0 < c₁ ∧ c₁ < c₂ ∧ c₂ < 1 ∧ ∀ X₀ : SphereTuple d n,
      ¬ (c₁ ≤ Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ∧
        Eβ d n β X₀ - ∫ U, Eβ d n β U ∂(iidSphere d n ν) ≤ c₂) := by
  have hβ0 : 0 < β := by linarith
  have h1 : 1 / (2 * β) < 1 := by
    rw [div_lt_one (by positivity)]
    linarith
  have h2 : 0 < 1 / (2 * β) := by positivity
  refine ⟨(1 / (2 * β) + 1) / 2, ((1 / (2 * β) + 1) / 2 + 1) / 2, by positivity, by linarith,
    by linarith, fun X₀ h => ?_⟩
  have hint : 0 ≤ ∫ U, Eβ d n β U ∂(iidSphere d n ν) :=
    integral_nonneg fun U => Eβ_nonneg d n hβ0 U
  have hle := Eβ_le d n hβ0 X₀
  linarith [h.1]

/-- The hypothesis of `printed_energy_level_trivial` is satisfiable: `β = 1`. -/
example : (1 : ℝ) / 2 < 1 := by norm_num

end Metastability
end Transformer
