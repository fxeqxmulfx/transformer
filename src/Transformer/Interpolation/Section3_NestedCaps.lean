module

public import Transformer.Interpolation.Section3_TwoRidgeFields

/-!
# A hemispherical cap inside an arbitrary proper activation cap

Auxiliary to arXiv:2411.04551v3, Appendix, the constant-control
construction toward `lem: two.balls` and `eq: activation.ball`.

Given a cap center and any target in its cap, the normalized sum of
the two unit axes gives a midpoint direction. A slightly enlarged cap
about that midpoint contains both axes strictly and has its entire
closed cap inside the original active region, with an explicit positive
margin. The original cap may be larger than a hemisphere.

The proof uses only squared norms and spherical inner products. A second
estimate bounds `1+⟨x,ω⟩` away from zero on the inner cap, which is the
factor needed in the actual squared-distance contraction equation.
-/

@[expose] public section

open Real Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A common cap supplies a lower bound on the pairwise inner product.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem inner_lower_of_common_cap (z w x : SSphere d) (h : ℝ)
    (hpos : 0 ≤ h + inner ℝ (z : EucSpace d) (w : EucSpace d))
    (hx : h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d)) :
    (h + inner ℝ (z : EucSpace d) (w : EucSpace d)) ^ 2 / 2 - 1 ≤
      inner ℝ (x : EucSpace d) (z : EucSpace d) := by
  have hn := abs_real_inner_le_norm ((x : EucSpace d) + (z : EucSpace d)) (w : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp w.2, mul_one] at hn
  have hi : h + inner ℝ (z : EucSpace d) (w : EucSpace d) ≤ ‖(x : EucSpace d) + (z : EucSpace d)‖ := by
    calc
      _ ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) + inner ℝ (z : EucSpace d) (w : EucSpace d) :=
        add_le_add hx le_rfl
      _ = inner ℝ ((x : EucSpace d) + (z : EucSpace d)) (w : EucSpace d) := (inner_add_left _ _ _).symm
      _ ≤ _ := (le_abs_self _).trans hn
  have hs := (sq_le_sq₀ hpos (norm_nonneg ((x : EucSpace d) + (z : EucSpace d)))).mpr hi
  rw [norm_add_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp z.2] at hs
  nlinarith


/-- The inner cap controls the factor in the original dissipation law.
Auxiliary for arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`. -/
theorem one_add_inner_lower_of_common_cap (ω w x : SSphere d) (h : ℝ)
    (hpos : 0 ≤ h + inner ℝ (ω : EucSpace d) (w : EucSpace d))
    (hx : h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d)) :
    (h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2 ≤
      1 + inner ℝ (x : EucSpace d) (ω : EucSpace d) := by
  have hh := inner_lower_of_common_cap ω w x h hpos hx
  linarith

/-- Every proper original cap has a hemispherical cap containing its center
and the chosen target, with a positive uniform activation margin on its closure.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_nested_hemispherical_cap (z ω : SSphere d) (c : ℝ) (hc : -1 ≤ c)
    (hω : c < inner ℝ (ω : EucSpace d) (z : EucSpace d)) :
    ∃ (w : SSphere d) (h m : ℝ), 0 ≤ h ∧ h < inner ℝ (z : EucSpace d) (w : EucSpace d) ∧
      h < inner ℝ (ω : EucSpace d) (w : EucSpace d) ∧ 0 < m ∧
      ∀ x : SSphere d, h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d) →
        m ≤ inner ℝ (x : EucSpace d) (z : EucSpace d) - c := by
  let N := ‖(z : EucSpace d) + (ω : EucSpace d)‖
  have hN2 : N ^ 2 = 2 + 2 * inner ℝ (ω : EucSpace d) (z : EucSpace d) := by
    dsimp only [N]
    rw [norm_add_sq_real, mem_sphere_zero_iff_norm.mp z.2, mem_sphere_zero_iff_norm.mp ω.2,
      real_inner_comm (ω : EucSpace d) (z : EucSpace d)]
    ring
  have hN : 0 < N := by have hn : 0 ≤ N := norm_nonneg _; nlinarith
  let w : SSphere d := ⟨N⁻¹ • ((z : EucSpace d) + (ω : EucSpace d)), mem_sphere_zero_iff_norm.mpr (by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hN)]
    exact inv_mul_cancel₀ hN.ne')⟩
  let a := N / 2
  have haz : inner ℝ (z : EucSpace d) (w : EucSpace d) = a := by
    change inner ℝ (z : EucSpace d) (N⁻¹ • ((z : EucSpace d) + (ω : EucSpace d))) = N / 2
    rw [inner_smul_right, inner_add_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp z.2,
      real_inner_comm (ω : EucSpace d) (z : EucSpace d)]
    field_simp
    nlinarith
  have haw : inner ℝ (ω : EucSpace d) (w : EucSpace d) = a := by
    change inner ℝ (ω : EucSpace d) (N⁻¹ • ((z : EucSpace d) + (ω : EucSpace d))) = N / 2
    rw [inner_smul_right, inner_add_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp ω.2]
    field_simp
    nlinarith
  have ha : 0 < a := by dsimp only [a]; positivity
  have ha1 : a ≤ 1 := by
    have hh := abs_real_inner_le_norm (z : EucSpace d) (w : EucSpace d)
    rw [haz, mem_sphere_zero_iff_norm.mp z.2, mem_sphere_zero_iff_norm.mp w.2, mul_one] at hh
    exact (le_abs_self a).trans hh
  have hrel : inner ℝ (ω : EucSpace d) (z : EucSpace d) = 2 * a ^ 2 - 1 := by
    dsimp only [a]
    nlinarith
  let g := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  have hg : 0 < g := sub_pos.mpr hω
  let δ := min (a / 2) (g / 8)
  have hδ : 0 < δ := lt_min (by positivity) (by positivity)
  have hδa : δ ≤ a / 2 := min_le_left _ _
  have hδg : δ ≤ g / 8 := min_le_right _ _
  let h := a - δ
  have hh : 0 ≤ h := by dsimp only [h]; linarith
  have hha : h < a := by dsimp only [h]; linarith
  refine ⟨w, h, g / 2, hh, by rwa [haz], by rwa [haw], by positivity, ?_⟩
  intro x hx
  have hp : 0 ≤ h + inner ℝ (z : EucSpace d) (w : EucSpace d) := by rw [haz]; linarith
  have hb := inner_lower_of_common_cap z w x h hp hx
  rw [haz] at hb
  have hm := mul_le_mul_of_nonneg_right ha1 hδ.le
  dsimp only [h] at hb
  dsimp only [g] at hδg ⊢
  nlinarith [sq_nonneg δ]


/-- The entire closed inner cap lies in the original open activation cap.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_closed_cap_inside_cap (z ω : SSphere d) (c : ℝ) (hc : -1 ≤ c)
    (hω : c < inner ℝ (ω : EucSpace d) (z : EucSpace d)) :
    ∃ (w : SSphere d) (h : ℝ), 0 ≤ h ∧ h < inner ℝ (z : EucSpace d) (w : EucSpace d) ∧
      h < inner ℝ (ω : EucSpace d) (w : EucSpace d) ∧
      {x : SSphere d | h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d)} ⊆
        {x : SSphere d | c < inner ℝ (x : EucSpace d) (z : EucSpace d)} := by
  obtain ⟨w, h, m, hh, hz, hωw, hm, hbound⟩ := exists_nested_hemispherical_cap z ω c hc hω
  refine ⟨w, h, hh, hz, hωw, fun x hx => ?_⟩
  have hb := hbound x hx
  change c < inner ℝ (x : EucSpace d) (z : EucSpace d)
  linarith

/-- Both actual inward activations vanish outside the outer cap when
the closed inner cap is contained in it. Auxiliary for arXiv:2411.04551v3,
Appendix, the fixed region in `lem: two.balls`. -/
theorem inward_caps_inactive_outside (z w : SSphere d) (c h : ℝ)
    (hsub : {x : SSphere d | h ≤ inner ℝ (x : EucSpace d) (w : EucSpace d)} ⊆
      {x : SSphere d | c < inner ℝ (x : EucSpace d) (z : EucSpace d)})
    (x : SSphere d) (hx : ¬c < inner ℝ (x : EucSpace d) (z : EucSpace d)) :
    -c - inner ℝ (x : EucSpace d) (-(z : EucSpace d)) ≤ 0 ∧
      -h - inner ℝ (x : EucSpace d) (-(w : EucSpace d)) ≤ 0 := by
  have hz := not_lt.mp hx
  have hw : inner ℝ (x : EucSpace d) (w : EucSpace d) < h := lt_of_not_ge (fun hh => hx (hsub hh))
  rw [inner_neg_right, inner_neg_right]
  constructor <;> linarith

/-- The common-cap premises hold jointly on a genuine unit sphere. -/
example : (0 : ℝ) ≤ 0 + inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) ∧
    (0 : ℝ) ≤ inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) := by
  norm_num [basePoint]

/-- Orthogonal unit axes lie in the cap at threshold `-1/2`, a cap larger
than a hemisphere. Its threshold and strict target hypothesis are genuine. -/
example : ∃ z ω : SSphere 2, (-1 : ℝ) ≤ -1 / 2 ∧
    -1 / 2 < inner ℝ (ω : EucSpace 2) (z : EucSpace 2) := by
  refine ⟨basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩, by norm_num, ?_⟩
  norm_num [basePoint, EuclideanSpace.inner_single_left]

/-- A strict inner hemisphere cap has a genuine inactive point in the
outer complement, so the inactive-complement premises are satisfiable. -/
example : ∃ z w x : SSphere 2,
    ({y : SSphere 2 | (1 / 2 : ℝ) ≤ inner ℝ (y : EucSpace 2) (w : EucSpace 2)} ⊆
      {y : SSphere 2 | 0 < inner ℝ (y : EucSpace 2) (z : EucSpace 2)}) ∧
    ¬0 < inner ℝ (x : EucSpace 2) (z : EucSpace 2) := by
  let a : SSphere 2 := basePoint 1
  refine ⟨a, a, antipode 2 a, fun y hy => ?_, ?_⟩
  · change 0 < inner ℝ (y : EucSpace 2) (a : EucSpace 2)
    change (1 / 2 : ℝ) ≤ inner ℝ (y : EucSpace 2) (a : EucSpace 2) at hy
    linarith
  · norm_num [a, antipode, basePoint]

end Transformer.Interpolation
