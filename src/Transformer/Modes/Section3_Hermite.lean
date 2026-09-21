/-
# The number of modes of a Gaussian KDE — Hermite polynomials and whitening

§3.1 of arXiv:2412.09080v3 (`sec:error-3`): the standard normal density on
`ℝ²`, the multivariate Hermite polynomials of order three, and the
standardization `Σ^{-1/2}` of `eq:Yi`.

**What the source says and what is carried here.**

* Points of `ℝ²` are pairs, as in `IsKacRiceField`; since the norm of `ℝ × ℝ`
  in Mathlib is the sup norm, the Euclidean norm is written out, `eucl`.

* `H^α(x) = (-1)^{|α|} φ(x)⁻¹ ∂^α φ(x)`.  For `φ` the standard normal density
  on `ℝ²` this factors as `He_{α₁}(x₁) He_{α₂}(x₂)`, the probabilists' Hermite
  polynomials; only `|α| = 3` enters, so `He_0, …, He_3` are written out.

* **"Trivially, `|H^{(k,3-k)}(x)| ≲ ‖x‖³`" is false.**  `H^{(3,0)}(x) =
  x₁³ - 3x₁ ≈ -3x₁` near `0`, which no `C‖x‖³` bounds:
  `not_exists_hermite_le_cube`, proved.  What holds is
  `|H^{(k,3-k)}(x)| ≤ 3(‖x‖ + ‖x‖³)` (`abs_hermite_le`, proved), and that is the
  bound `lem:error-3` has to be run with.

* `Σ_t^{-1/2}` is realised by the explicit whitening `whiten`, the
  lower-triangular `L` with `L Σ Lᵀ = I`, in place of the symmetric square
  root.  The two differ by a rotation, which leaves `φ`, `‖·‖` and every
  integral of §3 unchanged; `eucl_whiten_sq` proves
  `‖L z‖² = zᵀ Σ⁻¹ z = quadForm`, the identity everything is read through.

Source: arXiv:2412.09080v3, §3.1, `eq:Yi`, `eq:psi`, `eq:H3-bound`.
-/

import Transformer.Modes.Section2_MainIntPhi

open Real

namespace Transformer
namespace Modes

/-- The Euclidean norm of a point of `ℝ²`. -/
noncomputable def eucl (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)

/-- `φ`, the density of `N(0, I₂)`: `φ(x) = (2π)⁻¹ e^{-‖x‖²/2}`.

Source: arXiv:2412.09080v3, §2.2, before `eq:approx`. -/
noncomputable def phi2 (x : ℝ × ℝ) : ℝ := (2 * π)⁻¹ * Real.exp (-(x.1 ^ 2 + x.2 ^ 2) / 2)

/-- The probabilists' Hermite polynomials `He_0, …, He_3`; `0` above order 3,
where they are not used. -/
def hermite1 : ℕ → ℝ → ℝ
  | 0, _ => 1
  | 1, x => x
  | 2, x => x ^ 2 - 1
  | 3, x => x ^ 3 - 3 * x
  | _, _ => 0

/-- `H^{(k, 3-k)}(x) = He_k(x₁) He_{3-k}(x₂)`, the Hermite polynomials of order
three on `ℝ²`.

Source: arXiv:2412.09080v3, §3.1, the definition of `H^α`. -/
def hermite3 (k : ℕ) (x : ℝ × ℝ) : ℝ := hermite1 k x.1 * hermite1 (3 - k) x.2

/-! ### The cubic bound and its failure -/

/-- **The bound "`|H^{(k,3-k)}(x)| ≲ ‖x‖³`" of §3.1 is false**: for every `C`,
`H^{(3,0)}` exceeds `C‖x‖³` at some `x`.

Source: arXiv:2412.09080v3, §3.1, before `eq:H3-bound` (refuted). -/
theorem not_exists_hermite_le_cube :
    ¬ ∃ C : ℝ, ∀ x : ℝ × ℝ, |hermite3 3 x| ≤ C * eucl x ^ 3 := by
  rintro ⟨C, hC⟩
  set e : ℝ := min 1 (1 / (|C| + 1)) with he
  have he0 : 0 < e := lt_min one_pos (by positivity)
  have he1 : e ≤ 1 := min_le_left _ _
  have heC : e * (|C| + 1) ≤ 1 := by
    calc e * (|C| + 1) ≤ 1 / (|C| + 1) * (|C| + 1) := by gcongr; exact min_le_right _ _
      _ = 1 := by field_simp
  have h := hC (e, 0)
  have hn : eucl (e, 0) = e := by simp [eucl, Real.sqrt_sq he0.le]
  simp only [hermite3, hermite1, Nat.sub_self, hn] at h
  have he2 : e ^ 2 ≤ 1 := by nlinarith
  have hneg : e ^ 3 - 3 * e < 0 := by nlinarith
  rw [mul_one, abs_of_neg hneg] at h
  have hCe : C * e ^ 3 ≤ |C| * e ^ 3 := by gcongr; exact le_abs_self C
  have hbig : (|C| + 1) * e ^ 3 ≤ e ^ 2 := by
    calc (|C| + 1) * e ^ 3 = e * (|C| + 1) * e ^ 2 := by ring
      _ ≤ 1 * e ^ 2 := by gcongr
      _ = e ^ 2 := one_mul _
  have hee : e ^ 2 ≤ e := by nlinarith
  linarith

/-- **What holds instead:** `|H^{(k,3-k)}(x)| ≤ 3(‖x‖ + ‖x‖³)`.

Source: arXiv:2412.09080v3, §3.1, the bound before `eq:H3-bound`, corrected. -/
theorem abs_hermite_le (k : ℕ) (hk : k ≤ 3) (x : ℝ × ℝ) :
    |hermite3 k x| ≤ 3 * (eucl x + eucl x ^ 3) := by
  have h1 : |x.1| ≤ eucl x := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
  have h2 : |x.2| ≤ eucl x := by
    rw [eucl, ← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (by nlinarith)
  have hr : 0 ≤ eucl x := Real.sqrt_nonneg _
  have a1 := abs_nonneg x.1
  have a2 := abs_nonneg x.2
  interval_cases k <;> simp only [hermite3, hermite1, abs_mul, one_mul, mul_one]
  · -- `He_3(x₂)`
    calc |x.2 ^ 3 - 3 * x.2| ≤ |x.2| ^ 3 + 3 * |x.2| := by
          refine (abs_sub _ _).trans ?_; rw [abs_pow, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 3)]
      _ ≤ _ := by nlinarith [pow_le_pow_left₀ a2 h2 3, pow_nonneg hr 3]
  · -- `He_1(x₁) He_2(x₂)`
    calc |x.1| * |x.2 ^ 2 - 1| ≤ |x.1| * (|x.2| ^ 2 + 1) := by
          gcongr; refine (abs_sub _ _).trans ?_; rw [abs_pow, abs_one]
      _ ≤ eucl x * (eucl x ^ 2 + 1) := by gcongr
      _ ≤ _ := by nlinarith [pow_nonneg hr 3]
  · -- `He_2(x₁) He_1(x₂)`
    calc |x.1 ^ 2 - 1| * |x.2| ≤ (|x.1| ^ 2 + 1) * |x.2| := by
          gcongr; refine (abs_sub _ _).trans ?_; rw [abs_pow, abs_one]
      _ ≤ (eucl x ^ 2 + 1) * eucl x := by gcongr
      _ ≤ _ := by nlinarith [pow_nonneg hr 3]
  · -- `He_3(x₁)`
    calc |x.1 ^ 3 - 3 * x.1| ≤ |x.1| ^ 3 + 3 * |x.1| := by
          refine (abs_sub _ _).trans ?_; rw [abs_pow, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 3)]
      _ ≤ _ := by nlinarith [pow_le_pow_left₀ a1 h1 3, pow_nonneg hr 3]

/-- The hypothesis of `abs_hermite_le` is satisfiable. -/
example : (0 : ℕ) ≤ 3 := Nat.zero_le 3

/-! ### Whitening -/

/-- The lower-triangular whitening `L` of `Σ = [[a, b], [b, d]]`, `D = ad - b²`:
`L z = (z₁/√a, (a z₂ - b z₁)/√(aD))`, so that `L Σ Lᵀ = I`.  It stands for
`Σ^{-1/2}` of `eq:Yi`. -/
noncomputable def whiten (a b d : ℝ) (z : ℝ × ℝ) : ℝ × ℝ :=
  (z.1 / Real.sqrt a, (a * z.2 - b * z.1) / Real.sqrt (a * (a * d - b ^ 2)))

/-- **`‖L z‖² = zᵀ Σ⁻¹ z`.**  For `a > 0` and `ad - b² > 0`. -/
theorem eucl_whiten_sq {a b d : ℝ} (ha : 0 < a) (hD : 0 < a * d - b ^ 2) (z : ℝ × ℝ) :
    eucl (whiten a b d z) ^ 2 = quadForm a b d z.1 z.2 := by
  have hs : 0 ≤ (z.1 / Real.sqrt a) ^ 2
      + ((a * z.2 - b * z.1) / Real.sqrt (a * (a * d - b ^ 2))) ^ 2 := by positivity
  rw [eucl, whiten, Real.sq_sqrt hs, div_pow, div_pow, Real.sq_sqrt ha.le,
    Real.sq_sqrt (mul_pos ha hD).le, quadForm]
  generalize hDdef : a * d - b ^ 2 = D at hD ⊢
  have hd : d = (D + b ^ 2) / a := by rw [← hDdef]; field_simp; ring
  subst hd
  field_simp
  ring

/-- The hypotheses of `eucl_whiten_sq` are satisfiable: `Σ = I₂`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 := by norm_num

end Modes
end Transformer
