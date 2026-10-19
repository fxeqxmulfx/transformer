module

public import Transformer.Metastability.Section3_ForceDerivative

/-!
# Gradient differences control the width of a separated angular cap

ArXiv:2410.06833v1, §3.2, `lem: PL.borjan` and `eq: Ht.second.lb`.
The common scalar force has derivative at most `-1/(4n²)` throughout
the cap chart. The mean value theorem therefore bounds the separation
of any two chart angles by `4n²` times their force difference.
At token angles those force values are the actual gradient components.
-/

@[expose] public section

open Set Real

namespace Transformer.Metastability

/-- Squaring a force-difference separation bound controls the gap by
the two squared force values. Source: arXiv:2410.06833v1, §3.2,
the gradient-norm comparison in `lem: PL.borjan`. -/
theorem square_gap_le_of_force_bound (n x y a b : ℝ)
    (h : |x - y| ≤ 4 * n ^ 2 * |a - b|) :
    (x - y) ^ 2 ≤ 32 * n ^ 4 * (a ^ 2 + b ^ 2) := by
  have hs := pow_le_pow_left₀ (abs_nonneg (x - y)) h 2
  rw [mul_pow, sq_abs, sq_abs] at hs
  have hg : (a - b) ^ 2 ≤ 2 * (a ^ 2 + b ^ 2) := by nlinarith [sq_nonneg (a + b)]
  have hm := mul_le_mul_of_nonneg_left hg (sq_nonneg (4 * n ^ 2))
  nlinarith

/-- The squared-gap input is realized by two distinct positions and
two distinct force values at `n=1`. -/
example : |(0 : ℝ) - 1| ≤ 4 * (1 : ℝ) ^ 2 * |(0 : ℝ) - 1| := by norm_num

/-- A scalar derivative bounded above by `-1/(4n²)` controls separation
by the absolute force difference. Source: arXiv:2410.06833v1, §3.2,
the within-cap Hessian bound in `eq: Ht.second.lb`. -/
theorem ordered_gap_le_force_difference (n : ℕ) (hn : 0 < n) (f : ℝ → ℝ)
    (hf : Differentiable ℝ f) (a b : ℝ) (hab : a ≤ b)
    (hbound : ∀ u ∈ Icc a b, deriv f u ≤ -(1 / (4 * (n : ℝ) ^ 2))) :
    b - a ≤ 4 * (n : ℝ) ^ 2 * |f b - f a| := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have h := (convex_Icc a b).image_sub_le_mul_sub_of_deriv_le hf.continuous.continuousOn
    (fun u _ => (hf u).differentiableWithinAt)
    (fun u hu => by rw [interior_Icc] at hu; exact hbound u ⟨hu.1.le, hu.2.le⟩)
    a ⟨le_rfl, hab⟩ b ⟨hab, le_rfl⟩ hab
  have hquarter : (1 / (4 * (n : ℝ) ^ 2)) * (b - a) ≤ |f b - f a| := by
    linarith [neg_abs_le (f b - f a)]
  have hs := mul_le_mul_of_nonneg_left hquarter (by positivity : 0 ≤ 4 * (n : ℝ) ^ 2)
  have heq : 4 * (n : ℝ) ^ 2 * ((1 / (4 * (n : ℝ) ^ 2)) * (b - a)) = b - a := by
    field_simp
  rwa [heq] at hs

/-- All scalar separation inputs hold for `f(u)=-u`, `n=1`, on `[0,1]`. -/
example : 0 < 1 ∧ Differentiable ℝ (fun u : ℝ => -u) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ u ∈ Icc (0 : ℝ) 1, deriv (fun v : ℝ => -v) u ≤ -(1 / (4 * (1 : ℝ) ^ 2))) := by
  refine ⟨one_pos, fun u => (hasDerivAt_neg u).differentiableAt, zero_le_one, fun u _ => ?_⟩
  rw [(hasDerivAt_neg u).deriv]
  norm_num

/-- **The actual angular force is antitone on a separated small cap.**
This is the qualitative consequence of its uniform negative derivative.
Source: arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem angularForce_antitoneOn_cap (n k : ℕ) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (Θ : Idx n → ℝ) (q : Idx k) (i₀ : Idx n)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hchart : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) → |Θ i - ω q| ≤ π)
    (hmem₀ : 1 - 2 * τ ≤ Real.cos (Θ i₀ - ω q)) :
    AntitoneOn (angularForce n β Θ)
      {u : ℝ | |u - ω q| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (u - ω q)} := by
  have hd := angularForce_deriv_le_on_cap n k β τ δ α ω Θ q i₀
    hβ hτ hτ16 hδ hδ1 hcaps hα hγ hsmall hchart hmem₀
  intro a ha b hb hab
  have hcap := interval_mem_cos_cap τ (ω q) a b ha.1 hb.1 ha.2 hb.2
  have hmono : AntitoneOn (angularForce n β Θ) (Icc a b) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc a b)
    · exact (continuous_iff_continuousAt.mpr
        fun u => (hasDerivAt_angularForce n β Θ u).continuousAt).continuousOn
    · intro u _
      exact (hasDerivAt_angularForce n β Θ u).differentiableAt.differentiableWithinAt
    · intro u hu
      rw [interior_Icc] at hu
      have hc := hcap u ⟨hu.1.le, hu.2.le⟩
      exact (hd u hc.1 hc.2).trans (neg_nonpos.mpr (by positivity))
  exact hmono ⟨le_rfl, hab⟩ ⟨hab, le_rfl⟩ hab

/-- **Any two chart angles in the cap are controlled by their true force difference.**
Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`, using the within-cap
Hessian estimate `eq: Ht.second.lb`. -/
theorem angularForce_gap_bound (n k : ℕ) (β τ δ α : ℝ)
    (ω : Idx k → ℝ) (Θ : Idx n → ℝ) (q : Idx k) (i₀ : Idx n)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hδ : 0 ≤ δ) (hδ1 : δ < 1)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hchart : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) → |Θ i - ω q| ≤ π)
    (hmem₀ : 1 - 2 * τ ≤ Real.cos (Θ i₀ - ω q)) :
    ∀ a b : ℝ, |a - ω q| ≤ π → 1 - 2 * τ ≤ Real.cos (a - ω q) →
      |b - ω q| ≤ π → 1 - 2 * τ ≤ Real.cos (b - ω q) →
      |a - b| ≤ 4 * (n : ℝ) ^ 2 * |angularForce n β Θ a - angularForce n β Θ b| := by
  have hn : 0 < n := by have := i₀.isLt; omega
  have hd := angularForce_deriv_le_on_cap n k β τ δ α ω Θ q i₀
    hβ hτ hτ16 hδ hδ1 hcaps hα hγ hsmall hchart hmem₀
  have hordered (a b : ℝ) (ha : |a - ω q| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (a - ω q))
      (hb : |b - ω q| ≤ π ∧ 1 - 2 * τ ≤ Real.cos (b - ω q)) (hab : a ≤ b) :
      b - a ≤ 4 * (n : ℝ) ^ 2 * |angularForce n β Θ b - angularForce n β Θ a| := by
    apply ordered_gap_le_force_difference n hn (angularForce n β Θ)
      (fun u => (hasDerivAt_angularForce n β Θ u).differentiableAt) a b hab
    intro u hu
    have hcap := interval_mem_cos_cap τ (ω q) a b ha.1 hb.1 ha.2 hb.2 u hu
    exact hd u hcap.1 hcap.2
  intro a b ha hca hb hcb
  rcases le_total a b with hab | hba
  · rw [abs_of_nonpos (sub_nonpos.mpr hab), neg_sub, abs_sub_comm]
    exact hordered a b ⟨ha, hca⟩ ⟨hb, hcb⟩ hab
  · rw [abs_of_nonneg (sub_nonneg.mpr hba)]
    exact hordered b a ⟨hb, hcb⟩ ⟨ha, hca⟩ hba

/-- The cap hypotheses of both force consequences are jointly satisfied
by the same full three-token witness as the derivative theorem. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - 2 * τ ≤ Real.cos (Θ i - ω p)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * (3 : ℝ) ^ 2 / τ) ∧
    (∀ u v : ℝ, |u - ω 0| ≤ π → |v - ω 0| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω 0) → 1 - 2 * τ ≤ Real.cos (v - ω 0) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω 0) → |Θ i - ω 0| ≤ π) ∧
    1 - 2 * τ ≤ Real.cos (Θ 0 - ω 0) := by
  intro τ α ω Θ
  obtain ⟨hcaps, hα, hγ, hsmall, -, -, -, hchart, hmem, -⟩ := circle_witness
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    hcaps, hα, hγ, hsmall 0, ?_, ?_⟩
  · intro i hi
    obtain ⟨j, rfl⟩ := (hmem i).mp hi
    exact hchart j
  · norm_num [Θ, ω, τ]

end Transformer.Metastability
