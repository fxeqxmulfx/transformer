/-
# Homogenized Transformers — the sphere `𝕊⁰`

In dimension `1` the unit sphere is the two-point set `{±1}`, its tangent
space at either point is `{0}`, and `Proj_x` is the zero map.  So the diffusion
kernel `G_μ(x,θ) = Proj_x ξ_θ[μ](x)` of `eq:G_def` vanishes there **at every
head law**, and not only at the degenerate `ρ* = δ_0` of `MvGenerator.lean`.

That is what makes `d = 1` the witness the satisfiability of
`thm:large_beta_meta` needs: the theorem's hypotheses include a coupled pair of
`eq:non_linear_SDE_common` at a Gaussian head law with `σ_V² = 1/d` and
`σ_A > 0`, and in dimension `1` such a pair exists — frozen, by `Frozen.lean` —
while in dimension `d ≥ 2` producing one is exactly `thm:PoC_wellposedness`,
which is not proved here.

The degeneracy is real and not a trick: `eq:non_linear_SDE_common` on `𝕊⁰` is
`dx = 0`, the overlap `R(t) = ⟨x⁽¹⁾(t), x⁽²⁾(t)⟩` is constant, and the source's
own conclusion — that `R` moves by `O(T/d)` plus a martingale of bracket
`O(T/d)` — is satisfied, with room to spare, at `d = 1`.
-/

import Transformer.Homogenized.Frozen

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- A unit vector of `ℝ¹` has `x_0² = 1`. -/
theorem mul_self_eq_one_of_norm_eq_one {x : EucSpace 1} (hx : ‖x‖ = 1) : x 0 * x 0 = 1 := by
  have h : inner (𝕜 := ℝ) x x = 1 := by
    rw [real_inner_self_eq_norm_sq, hx]; norm_num
  rwa [PiLp.inner_apply, Fin.sum_univ_one, RCLike.inner_apply, starRingEnd_apply,
    star_trivial] at h

/-- **The tangent space of `𝕊⁰` is trivial.**  At a unit vector of `ℝ¹`,
`Proj_x y = y - ⟨x,y⟩x = y - y x_0² = 0`. -/
theorem proj_one {x : EucSpace 1} (hx : ‖x‖ = 1) (y : EucSpace 1) : proj 1 x y = 0 := by
  have hsq := mul_self_eq_one_of_norm_eq_one hx
  ext i
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  simp only [proj, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul, PiLp.inner_apply,
    Fin.sum_univ_one, RCLike.inner_apply, starRingEnd_apply, star_trivial,
    PiLp.zero_apply]
  linear_combination (- y 0) * hsq

/-- **`eq:G_def` vanishes in dimension `1`**, at every temperature, every head
law, every measure, every point of `𝕊⁰` and every head. -/
theorem GfieldOf_one (β : ℝ) (ρ : Measure (HeadParam 1)) (μ : Measure (EucSpace 1))
    {x : EucSpace 1} (hx : ‖x‖ = 1) (θ : HeadParam 1) : GfieldOf β ρ μ x θ = 0 :=
  proj_one hx _

/-- Hence **every** head law has vanishing field in dimension `1` — in
particular the Gaussian laws of assumption (G), which `δ_0` is not. -/
theorem hasVanishingField_one (β : ℝ) (ρ : Measure (HeadParam 1)) :
    HasVanishingField β ρ :=
  fun μ x hx => Filter.Eventually.of_forall (GfieldOf_one β ρ μ (x := x) hx)

/-- **A coupled pair at an arbitrary head law, whose conditional law is the
uniform measure on `𝕊⁰`.**

This is `isCoupledPair_uniformAmbient` at `d = 1`, where the hypothesis
`HasVanishingField` costs nothing.  Together with
`isLowTemperature_uniformAmbient` it satisfies every hypothesis of
`thm:large_beta_meta` that concerns the dynamics. -/
theorem isCoupledPair_uniformAmbient_one (β T : ℝ) (ρ : Measure (HeadParam 1))
    {e : EucSpace 1} (he : ‖e‖ = 1) :
    IsCoupledPair β T ρ
      ((uniformAmbient 1).prod (uniformAmbient 1))
      (Filtration.const ℝ _ le_rfl) (Filtration.const ℝ ⊥ bot_le)
      (uniformAmbient 1)
      (fun _ ω => clampSphere e ω.1) (fun _ ω => clampSphere e ω.2)
      (fun _ _ => uniformAmbient 1) :=
  isCoupledPair_uniformAmbient one_pos β T ρ (hasVanishingField_one β ρ) he

/-- The hypotheses of `isCoupledPair_uniformAmbient_one` are satisfiable: `𝕊⁰`
is not empty. -/
example : ∃ e : EucSpace 1, ‖e‖ = 1 := exists_norm_eq_one one_pos

end Homogenized
end Transformer
