/-
# Perceptrons and attention's mean-field landscape — analyticity on the sphere

`thm: circle` and `thm: any.d` (i) are about weights `ϑ` for which the
perceptron potential `v_ϑ` is *not* real-analytic on `𝕊^{d-1}`; this file says
what that means, and refutes it for the ReLU perceptron, so those statements
have a witness.

**What the source says and what is carried here.**

* "`v_ϑ` is real-analytic on `𝕊^{d-1}`" is carried as: `v_ϑ` agrees on the
  sphere with a function analytic on a neighbourhood of it.  For a compact
  real-analytic submanifold the two are the same — the restriction of an
  ambient analytic function is analytic in charts, and an analytic function on
  a compact analytic submanifold extends analytically to a neighbourhood — and
  this reading needs no analytic-manifold structure on `Metric.sphere`, which
  Mathlib does not have.

* The footnote of `thm: circle` ("this discards `v_ϑ ≡ 0` and the weight
  symmetries making `v_ϑ` a quadratic trigonometric polynomial") is not a
  further hypothesis: those are exactly the cases in which `v_ϑ` *is*
  analytic, which `IsAnalyticOnSphere` already excludes.

Source: arXiv:2601.21366v2, `thm: circle`, `thm: any.d`.
-/

import Transformer.Perceptron.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.Analytic.Uniqueness

open scoped BigOperators
open Real

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Great circles -/

/-- The great circle through the orthonormal pair `u, w`:
`θ ↦ cos θ · u + sin θ · w`. -/
noncomputable def greatCircle (u w : EucSpace d) (θ : ℝ) : EucSpace d :=
  Real.cos θ • u + Real.sin θ • w

/-- A great circle through an orthonormal pair lies on the sphere. -/
theorem norm_greatCircle {u w : EucSpace d} (hu : ‖u‖ = 1) (hw : ‖w‖ = 1)
    (huw : inner (𝕜 := ℝ) u w = 0) (θ : ℝ) : ‖greatCircle u w θ‖ = 1 := by
  have h0 : inner (𝕜 := ℝ) (Real.cos θ • u) (Real.sin θ • w) = (0 : ℝ) := by
    rw [real_inner_smul_left, real_inner_smul_right, huw]; ring
  have hsq : ‖greatCircle u w θ‖ ^ 2 = 1 := by
    rw [greatCircle, norm_add_sq_real, h0, norm_smul, norm_smul, hu, hw,
      Real.norm_eq_abs, Real.norm_eq_abs, mul_one, mul_one, sq_abs, sq_abs]
    linarith [Real.sin_sq_add_cos_sq θ]
  have hfac : (‖greatCircle u w θ‖ - 1) * (‖greatCircle u w θ‖ + 1) = 0 := by
    linear_combination hsq
  rcases mul_eq_zero.mp hfac with h | h
  · linarith
  · linarith [norm_nonneg (greatCircle u w θ)]

/-- The angle read off a great circle: `⟪u, cos θ · u + sin θ · w⟫ = cos θ`. -/
theorem inner_greatCircle {u w : EucSpace d} (hu : ‖u‖ = 1)
    (huw : inner (𝕜 := ℝ) u w = 0) (θ : ℝ) :
    inner (𝕜 := ℝ) u (greatCircle u w θ) = Real.cos θ := by
  have huu : inner (𝕜 := ℝ) u u = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, hu]; norm_num
  rw [greatCircle, inner_add_right, real_inner_smul_right, real_inner_smul_right, huu, huw]
  ring

/-- A great circle is a real-analytic curve. -/
theorem analyticOnNhd_greatCircle (u w : EucSpace d) :
    AnalyticOnNhd ℝ (greatCircle u w) Set.univ := fun _ _ =>
  (Real.analyticAt_cos.smul analyticAt_const).add
    (Real.analyticAt_sin.smul analyticAt_const)

/-! ### Analyticity of the potential on the sphere -/

/-- **`v_ϑ` is real-analytic on `𝕊^{d-1}`**: it agrees on the sphere with a
function analytic on a neighbourhood of the sphere.

Source: arXiv:2601.21366v2, `thm: circle`. -/
def IsAnalyticOnSphere (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) : Prop :=
  ∃ g : EucSpace d → ℝ, AnalyticOnNhd ℝ g {y : EucSpace d | ‖y‖ = 1} ∧
    ∀ y : EucSpace d, ‖y‖ = 1 → g y = potential φ ω a y

/-- If `v_ϑ` is analytic on the sphere then it is analytic along every great
circle: the restriction is the composition of an analytic function with an
analytic curve. -/
theorem analyticOnNhd_potential_greatCircle {φ : ℝ → ℝ} {ω : Idx d → ℝ}
    {a : Idx d → EucSpace d} (h : IsAnalyticOnSphere φ ω a) {u w : EucSpace d}
    (hu : ‖u‖ = 1) (hw : ‖w‖ = 1) (huw : inner (𝕜 := ℝ) u w = 0) :
    AnalyticOnNhd ℝ (fun θ : ℝ => potential φ ω a (greatCircle u w θ)) Set.univ := by
  obtain ⟨g, hg, hgv⟩ := h
  have hmaps : Set.MapsTo (greatCircle u w) Set.univ {y : EucSpace d | ‖y‖ = 1} :=
    fun θ _ => norm_greatCircle hu hw huw θ
  have hcomp : AnalyticOnNhd ℝ (g ∘ greatCircle u w) Set.univ :=
    hg.comp (analyticOnNhd_greatCircle u w) hmaps
  refine hcomp.congr isOpen_univ fun θ _ => ?_
  exact hgv _ (norm_greatCircle hu hw huw θ)

/-- **The ReLU perceptron is not analytic on the sphere.**  For the one-neuron
weights `ω = e_{j₀}`, `a_{j₀} = -u` and the ReLU primitive `φ(s) = (s_+)²`, the
potential is `((-⟪u,·⟫)_+)²`: along the great circle through `u` it is
`((-cos θ)_+)²`, which vanishes on `(-π/2, π/2)` and equals `1` at `θ = π`, so
no analytic function can agree with it.

This is the witness for the hypotheses of `thm: circle` and `thm: any.d` (i).

Source: arXiv:2601.21366v2, `thm: circle`. -/
theorem not_isAnalyticOnSphere_relu (j₀ : Idx d) {u w : EucSpace d} (hu : ‖u‖ = 1)
    (hw : ‖w‖ = 1) (huw : inner (𝕜 := ℝ) u w = 0) :
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-u)) := by
  intro h
  have hval : ∀ θ : ℝ,
      potential (fun s => max s 0 ^ 2) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-u))
        (greatCircle u w θ) = max (-Real.cos θ) 0 ^ 2 := by
    intro θ
    rw [potential, Finset.sum_eq_single j₀]
    · simp only [Pi.single_eq_same, inner_neg_left, inner_greatCircle hu huw]
      ring
    · intro j _ hj
      simp [Pi.single_eq_of_ne hj]
    · intro hj
      exact absurd (Finset.mem_univ j₀) hj
  have hana : AnalyticOnNhd ℝ (fun θ : ℝ =>
      potential (fun s => max s 0 ^ 2) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-u))
        (greatCircle u w θ)) Set.univ :=
    analyticOnNhd_potential_greatCircle h hu hw huw
  have hnear : (fun θ : ℝ =>
      potential (fun s => max s 0 ^ 2) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-u))
        (greatCircle u w θ)) =ᶠ[nhds 0] 0 := by
    have hmem : Set.Ioo (-(π / 2)) (π / 2) ∈ nhds (0 : ℝ) :=
      Ioo_mem_nhds (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos])
    filter_upwards [hmem] with θ hθ
    have hcos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo hθ
    rw [hval θ, max_eq_right (by linarith)]
    simp
  have hzero := hana.eqOn_zero_of_preconnected_of_eventuallyEq_zero isPreconnected_univ
    (Set.mem_univ 0) hnear
  have hpi := hzero (Set.mem_univ π)
  simp only [hval, Real.cos_pi, Pi.zero_apply] at hpi
  norm_num at hpi

end Perceptron
end Transformer
