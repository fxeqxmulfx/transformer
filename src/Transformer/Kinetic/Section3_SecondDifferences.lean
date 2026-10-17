/-
# Second differences of the flow

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 3, `eq:est-Glaub-2nd`:
moving token `l` by `h` and token `m` by `h'`, the mixed second difference of
token `a`,

  `Δ_a(t) = θ_a^{h,h'}(t) - θ_a^{h}(t) - θ_a^{h'}(t) + θ_a(t)`,

obeys the triangular inequality of Step 1 forced by
`L₂ |h||h'| e^{2L₁t} Σ_{b<a} ω_{a,b} (Ĝ^m_a + Ĝ^m_b)(Ĝ^l_a + Ĝ^l_b)`, where `L₂` is
a Lipschitz constant of `w_β''`. The source integrates it in time; here Duhamel's
formula along the profile gives the supersolution
`t Σ_c σ_c Ĝ^c_a(t)`, with `σ_c` the forcing frozen at the final time `T`
(`crossForcing`, `crossProfile`).

The four-point inequality `abs_four_point_le` is the deterministic form of
`eq:chain-rule-Glaub-2nd`.
-/

import Transformer.Kinetic.Section3_FirstDifferences

open Real

namespace Transformer
namespace Kinetic

/-- **Four-point inequality.** If `|g'| ≤ L₁` and `g'` is `L₂`-Lipschitz,
`|g A - g B - g C + g D| ≤ L₁ |A - B - C + D| + L₂ |A - B| |A - C|`. -/
theorem abs_four_point_le {g g' : ℝ → ℝ} {L₁ L₂ : ℝ} (hg : ∀ x, HasDerivAt g (g' x) x)
    (h₁ : ∀ x y, |g x - g y| ≤ L₁ * |x - y|) (h₂ : ∀ x y, |g' x - g' y| ≤ L₂ * |x - y|)
    (A B C D : ℝ) :
    |g A - g B - g C + g D| ≤ L₁ * |A - B - C + D| + L₂ * (|A - B| * |A - C|) := by
  set u := A - B
  have hk : |(g A - g (A - u)) - (g C - g (C - u))| ≤ L₂ * |u| * |A - C| := by
    have := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (s := Set.univ)
      (f := fun s => g s - g (s - u)) (f' := fun s => g' s - g' (s - u))
      (fun s _ => ((hg s).sub ((hg (s - u)).comp s ((hasDerivAt_id s).sub_const u))).congr_deriv
        (by simp) |>.hasDerivWithinAt)
      (fun s _ => by simpa [Real.norm_eq_abs] using h₂ s (s - u)) convex_univ
      (Set.mem_univ C) (Set.mem_univ A)
    simpa [Real.norm_eq_abs] using this
  have hd := h₁ D (C - u)
  have hAB : A - u = B := by simp [u]
  rw [hAB] at hk
  calc |g A - g B - g C + g D| = |((g A - g B) - (g C - g (C - u))) + (g D - g (C - u))| := by
        ring_nf
    _ ≤ |(g A - g B) - (g C - g (C - u))| + |g D - g (C - u)| := abs_add_le _ _
    _ ≤ L₂ * |u| * |A - C| + L₁ * |D - (C - u)| := add_le_add hk hd
    _ = L₁ * |A - B - C + D| + L₂ * (|A - B| * |A - C|) := by
        rw [show D - (C - u) = A - B - C + D by simp only [u]; ring]
        ring

/-- The forcing of token `c` by the pair `l, m`, frozen at time `T` and without
the factor `2e^{2|λ|}` of the weights:
`σ_c = (c+1)⁻¹ Σ_{b<c} (Ĝ^m_c + Ĝ^m_b)(Ĝ^l_c + Ĝ^l_b)`. -/
noncomputable def crossForcing (α γ T : ℝ) (l m c : ℕ) : ℝ :=
  (∑ b ∈ Finset.range c, (profile α γ m c T + profile α γ m b T) *
    (profile α γ l c T + profile α γ l b T)) / ((c : ℝ) + 1)

/-- Duhamel's supersolution `t Σ_{c<n} σ_c Ĝ^c_a(t)`. -/
noncomputable def crossProfile (α γ T : ℝ) (n l m a : ℕ) (t : ℝ) : ℝ :=
  t * ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profile α γ c a t

theorem crossForcing_nonneg {α γ T : ℝ} (hα : 0 ≤ α) (hγ : 0 < γ) (hT : 0 ≤ T) (l m c : ℕ) :
    0 ≤ crossForcing α γ T l m c := by
  have hp := fun b a => profile_nonneg hα hγ b a hT
  exact div_nonneg (Finset.sum_nonneg fun b _ => mul_nonneg (add_nonneg (hp _ _) (hp _ _))
    (add_nonneg (hp _ _) (hp _ _))) (by positivity)

theorem crossProfile_nonneg {α γ T : ℝ} (hα : 0 ≤ α) (hγ : 0 < γ) (hT : 0 ≤ T) (n l m a : ℕ)
    {t : ℝ} (ht : 0 ≤ t) : 0 ≤ crossProfile α γ T n l m a t :=
  mul_nonneg ht (Finset.sum_nonneg fun c _ => mul_nonneg (crossForcing_nonneg hα hγ hT l m c)
    (profile_nonneg hα hγ c a ht))

/-- The supersolution grows in time. -/
theorem crossProfile_mono {α γ T : ℝ} (hα : 0 ≤ α) (hγ : 0 < γ) (hT : 0 ≤ T) (n l m a : ℕ)
    {s t : ℝ} (hs : 0 ≤ s) (hst : s ≤ t) :
    crossProfile α γ T n l m a s ≤ crossProfile α γ T n l m a t := by
  unfold crossProfile
  refine mul_le_mul hst (Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left
    (profile_mono hα hγ c a hst) (crossForcing_nonneg hα hγ hT l m c))
    (Finset.sum_nonneg fun c _ => mul_nonneg (crossForcing_nonneg hα hγ hT l m c)
      (profile_nonneg hα hγ c a hs)) (hs.trans hst)

theorem hasDerivAt_crossProfile (α : ℝ) {γ : ℝ} (hγ : γ ≠ 0) (T : ℝ) (n l m a : ℕ) (t : ℝ) :
    HasDerivAt (crossProfile α γ T n l m a)
      (∑ c ∈ Finset.range n, crossForcing α γ T l m c * profile α γ c a t +
        t * ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profileDeriv α γ c a t) t := by
  have hS : HasDerivAt (fun t => ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profile α γ c a t)
      (∑ c ∈ Finset.range n, crossForcing α γ T l m c * profileDeriv α γ c a t) t :=
    HasDerivAt.fun_sum fun c _ => (hasDerivAt_profile α hγ c a t).const_mul _
  have := (hasDerivAt_id' t).mul hS
  rw [one_mul] at this
  exact this

/-- **Duhamel's inequality.** `V_a' ≥ σ_a + α/(a+1) Σ_{b<a} V_b` for a token `a < n`. -/
theorem crossProfile_super {α γ T : ℝ} (hα : 0 ≤ α) (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) (hT : 0 ≤ T)
    {n : ℕ} (l m : ℕ) {a : ℕ} (ha : a < n) {t : ℝ} (ht : 0 ≤ t) :
    crossForcing α γ T l m a + α / ((a : ℝ) + 1) *
        ∑ b ∈ Finset.range a, crossProfile α γ T n l m b t ≤
      ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profile α γ c a t +
        t * ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profileDeriv α γ c a t := by
  have hσ := crossForcing_nonneg hα hγ0 hT l m
  have h1 : crossForcing α γ T l m a ≤
      ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profile α γ c a t := by
    have := Finset.single_le_sum (f := fun c => crossForcing α γ T l m c * profile α γ c a t)
      (fun c _ => mul_nonneg (hσ c) (profile_nonneg hα hγ0 c a ht)) (Finset.mem_range.2 ha)
    simpa [profile] using this
  have h2 : α / ((a : ℝ) + 1) * ∑ b ∈ Finset.range a, crossProfile α γ T n l m b t ≤
      t * ∑ c ∈ Finset.range n, crossForcing α γ T l m c * profileDeriv α γ c a t := by
    simp only [crossProfile, ← Finset.mul_sum]
    rw [Finset.sum_comm, mul_left_comm, Finset.mul_sum]
    refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun c _ => ?_) ht
    rw [← Finset.mul_sum, mul_left_comm]
    exact mul_le_mul_of_nonneg_left (sum_profile_le hα hγ0 hγ1 c a ht) (hσ c)
  linarith

/-- The hypotheses of `crossProfile_super` are satisfiable. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 := ⟨zero_le_one, one_pos, le_rfl,
  Nat.one_pos⟩

end Kinetic
end Transformer
