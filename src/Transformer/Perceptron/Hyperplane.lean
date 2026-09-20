/-
# Perceptrons and attention's mean-field landscape — hyperplane sections

`rem: ext` (i) of arXiv:2601.21366v2 puts the biases `b_j` back inside the
perceptron and asks that at least one of the hyperplanes
`{x : a_j · x + b_j = 0}` meet `𝕊^{d-1}` in more than one point,
"equivalently, `|b_j| < ‖a_j‖`".  That equivalence is what this file proves;
the extension itself is `Bias.lean`.

**What the source says and what is carried here.**

* `a ≠ 0` is added to the source's parenthesis.  `{x : a·x + b = 0}` is a
  hyperplane only then, and at `a = b = 0` it is the whole space — meeting the
  sphere in more than one point while `|b| < ‖a‖` reads `0 < 0`, so the
  equivalence is false as written without it.

* `d ≥ 2` is added for the same reason: on `𝕊⁰ = {±1}` the equation
  `a x + b = 0` has at most one solution, so the left-hand side never holds
  while the right-hand side can.

* "in more than one point" is `¬ Set.Subsingleton`: not every two points of the
  section coincide.

Source: arXiv:2601.21366v2, `rem: ext` (i).
-/

import Transformer.Perceptron.Analytic
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional

open scoped BigOperators
open Real

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### A unit normal direction -/

/-- In dimension `d ≥ 2` every nonzero vector has a unit vector orthogonal to
it: the orthogonal complement of its span has dimension `d - 1 ≥ 1`. -/
theorem exists_unit_inner_eq_zero (hd : 2 ≤ d) {a : EucSpace d} (ha : a ≠ 0) :
    ∃ w : EucSpace d, ‖w‖ = 1 ∧ inner (𝕜 := ℝ) a w = 0 := by
  have hadd := Submodule.finrank_add_finrank_orthogonal (K := (ℝ ∙ a))
  rw [finrank_span_singleton ha, finrank_euclideanSpace_fin] at hadd
  have hpos : 0 < Module.finrank ℝ ((ℝ ∙ a)ᗮ : Submodule ℝ (EucSpace d)) := by omega
  obtain ⟨w, hw⟩ := Module.finrank_pos_iff_exists_ne_zero.mp hpos
  have hw0 : (w : EucSpace d) ≠ 0 := fun h => hw (Subtype.ext h)
  have hwn : ‖(w : EucSpace d)‖ ≠ 0 := norm_ne_zero_iff.mpr hw0
  refine ⟨‖(w : EucSpace d)‖⁻¹ • (w : EucSpace d), ?_, ?_⟩
  · rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hwn]
  · rw [real_inner_smul_right,
      (Submodule.mem_orthogonal _ _).mp w.2 a (Submodule.mem_span_singleton_self a),
      mul_zero]

/-- The hypotheses of `exists_unit_inner_eq_zero` are satisfiable: `d = 2` and
the first standard basis vector, which is nonzero because its norm is `1`. -/
example : (2 : ℕ) ≤ 2 ∧ ((basePoint 1 : SSphere 2) : EucSpace 2) ≠ 0 :=
  ⟨le_rfl, fun h => by
    simpa [h] using mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

/-! ### The equality case of Cauchy–Schwarz -/

/-- A unit vector realizing `⟪a, x⟫ = ‖a‖` is the direction of `a`, in the
form `‖a‖ • x = a` that avoids dividing by `‖a‖`. -/
theorem smul_eq_of_inner_eq_norm {a x : EucSpace d} (hx : ‖x‖ = 1)
    (h : inner (𝕜 := ℝ) a x = ‖a‖) : ‖a‖ • x = a := by
  have h' : inner (𝕜 := ℝ) a x = ‖a‖ * ‖x‖ := by rw [h, hx, mul_one]
  have hxa := inner_eq_norm_mul_iff_real.mp h'
  rw [hx, one_smul] at hxa
  exact hxa.symm

/-- The hypotheses of `smul_eq_of_inner_eq_norm` are satisfiable: `a = x` a
unit vector, where `⟪x, x⟫ = 1 = ‖x‖`. -/
example :
    ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ = 1 ∧
      inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2)
          ((basePoint 1 : SSphere 2) : EucSpace 2)
        = ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ := by
  have h1 : ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨h1, ?_⟩
  rw [real_inner_self_eq_norm_mul_norm, h1, mul_one]

/-! ### The section of the sphere by a hyperplane -/

/-- The section `{x ∈ 𝕊^{d-1} : a · x + b = 0}` of the sphere by the
hyperplane of the biased perceptron's `j`-th neuron.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
def sphereHyperplane (a : EucSpace d) (b : ℝ) : Set (SSphere d) :=
  {x : SSphere d | inner (𝕜 := ℝ) a (x : EucSpace d) + b = 0}

/-- **Remark (rem: ext) (i), the parenthesis.**  For `d ≥ 2` and `a ≠ 0`, the
hyperplane `{x : a·x + b = 0}` meets `𝕊^{d-1}` in more than one point if and
only if `|b| < ‖a‖`.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
theorem not_subsingleton_sphereHyperplane_iff (hd : 2 ≤ d) {a : EucSpace d}
    (ha : a ≠ 0) (b : ℝ) : ¬ (sphereHyperplane a b).Subsingleton ↔ |b| < ‖a‖ := by
  have hna : 0 < ‖a‖ := norm_pos_iff.mpr ha
  constructor
  · intro hns
    by_contra hle
    refine hns fun x hx y hy => ?_
    have hx' : inner (𝕜 := ℝ) a (x : EucSpace d) = -b := by
      have := hx; simp only [sphereHyperplane, Set.mem_ofPred_eq] at this; linarith
    have hy' : inner (𝕜 := ℝ) a (y : EucSpace d) = -b := by
      have := hy; simp only [sphereHyperplane, Set.mem_ofPred_eq] at this; linarith
    have hxn : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
    have hyn : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
    have hb : |b| = ‖a‖ := by
      have : |(-b)| ≤ ‖a‖ := by
        calc |(-b)| = |inner (𝕜 := ℝ) a (x : EucSpace d)| := by rw [hx']
          _ ≤ ‖a‖ * ‖(x : EucSpace d)‖ := abs_real_inner_le_norm a _
          _ = ‖a‖ := by rw [hxn, mul_one]
      rw [abs_neg] at this
      linarith [not_lt.mp hle]
    rcases (abs_eq hna.le).mp hb with hbpos | hbneg
    · have hxa : ‖-a‖ • (x : EucSpace d) = -a :=
        smul_eq_of_inner_eq_norm hxn (by rw [inner_neg_left, hx', norm_neg, hbpos]; ring)
      have hya : ‖-a‖ • (y : EucSpace d) = -a :=
        smul_eq_of_inner_eq_norm hyn (by rw [inner_neg_left, hy', norm_neg, hbpos]; ring)
      exact Subtype.ext (smul_right_injective (EucSpace d)
        (by rw [norm_neg]; exact hna.ne') (hxa.trans hya.symm))
    · have hxa : ‖a‖ • (x : EucSpace d) = a :=
        smul_eq_of_inner_eq_norm hxn (by rw [hx', hbneg, neg_neg])
      have hya : ‖a‖ • (y : EucSpace d) = a :=
        smul_eq_of_inner_eq_norm hyn (by rw [hy', hbneg, neg_neg])
      exact Subtype.ext (smul_right_injective (EucSpace d) hna.ne' (hxa.trans hya.symm))
  · intro hlt hsub
    obtain ⟨w, hw1, haw⟩ := exists_unit_inner_eq_zero hd ha
    obtain ⟨u, hu1, huw, hau⟩ : ∃ u : EucSpace d, ‖u‖ = 1 ∧ inner (𝕜 := ℝ) u w = 0 ∧
        ‖a‖ • u = a :=
      ⟨‖a‖⁻¹ • a, by rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hna.ne'],
        by rw [real_inner_smul_left, haw, mul_zero],
        by rw [smul_smul, mul_inv_cancel₀ hna.ne', one_smul]⟩
    set t : ℝ := -b / ‖a‖ with ht
    have ht1 : |t| < 1 := by
      rw [ht, abs_div, abs_neg, abs_of_pos hna]
      exact (div_lt_one hna).mpr hlt
    obtain ⟨htl, htr⟩ := abs_lt.mp ht1
    have hcos : Real.cos (Real.arccos t) = t := Real.cos_arccos htl.le htr.le
    have hsin : Real.sin (Real.arccos t) ≠ 0 := by
      rw [Real.sin_arccos]
      exact ne_of_gt (Real.sqrt_pos.mpr (by nlinarith))
    -- the two points of the section, at the angles `±arccos t` of the great
    -- circle through `u` and `w`
    have hmem : ∀ s : ℝ, greatCircle u w s ∈ Metric.sphere (0 : EucSpace d) 1 := fun s =>
      mem_sphere_zero_iff_norm.mpr (norm_greatCircle hu1 hw1 huw s)
    have hsec : ∀ s : ℝ, Real.cos s = t →
        (⟨greatCircle u w s, hmem s⟩ : SSphere d) ∈ sphereHyperplane a b := by
      intro s hs
      have : inner (𝕜 := ℝ) a (greatCircle u w s) = -b := by
        rw [← hau, real_inner_smul_left, inner_greatCircle hu1 huw, hs, ht]
        field_simp
      simp only [sphereHyperplane, Set.mem_ofPred_eq]
      rw [this]
      ring
    have hwu : inner (𝕜 := ℝ) w u = 0 := by rw [real_inner_comm]; exact huw
    have hww : inner (𝕜 := ℝ) w w = (1 : ℝ) := by
      rw [real_inner_self_eq_norm_mul_norm, hw1]; ring
    have hinner_w : ∀ s : ℝ, inner (𝕜 := ℝ) w (greatCircle u w s) = Real.sin s := by
      intro s
      rw [greatCircle, inner_add_right, real_inner_smul_right, hwu, real_inner_smul_right, hww]
      ring
    have heq := hsub (hsec _ hcos) (hsec (-Real.arccos t) (by rw [Real.cos_neg, hcos]))
    have hval : Real.sin (Real.arccos t) = Real.sin (-Real.arccos t) := by
      rw [← hinner_w, ← hinner_w]
      exact congrArg (fun z : SSphere d => inner (𝕜 := ℝ) w (z : EucSpace d)) heq
    rw [Real.sin_neg] at hval
    exact hsin (by linarith)

/-- The hypotheses of `not_subsingleton_sphereHyperplane_iff` are satisfiable:
`d = 2` and the first standard basis vector. -/
example : (2 : ℕ) ≤ 2 ∧ ((basePoint 1 : SSphere 2) : EucSpace 2) ≠ 0 :=
  ⟨le_rfl, fun h => by
    simpa [h] using mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

end Perceptron
end Transformer
