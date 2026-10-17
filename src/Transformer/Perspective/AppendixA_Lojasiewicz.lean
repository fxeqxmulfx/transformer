/-
# Appendix A — a Łojasiewicz inequality for `𝖤_0`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, proof of `p:beta0`.

The proof of `p:beta0` invokes Łojasiewicz's theorem: a trajectory of the
gradient flow of a real-analytic function on a compact manifold converges to a
single critical point.  For `𝖤_0 = ‖s‖² / n`, `s = Σ_j x_j`, the inequality
behind that theorem is explicit.  With `a_i = ⟨x_i, s⟩`, `σ_i = ±1` its sign
and `m = Σ_i σ_i`, the bounds `‖s‖ |a_i - σ_i ‖s‖| ≤ ‖s‖² - a_i² =
‖Proj_{x_i} s‖²` and `Σ_i a_i = ‖s‖²` give

  `‖s‖² |‖s‖² - m²| ≤ 2n Σ_i ‖Proj_{x_i} s‖²`   (`exists_int_mul_abs_sub_le`):

away from `s = 0` the squared gradient bounds the distance from `‖s‖²` to the
finitely many squares `m²`, `|m| ≤ n`, which keep a positive distance from any
real number other than themselves (`exists_pos_le_abs_sq_sub`).  Along
`e:gradfl`, `d/dt ‖s‖² = (4/n) Σ_i ‖Proj_{x_i} s‖²` (`hasDerivAt_norm_sum_sq`)
while `‖s‖² ≤ n²` (`bddAbove_norm_sum_sq`).
-/

import Transformer.Perspective.AppendixA_Barycenter

open Filter Set
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `Σ_i ‖Proj_{x_i} s‖²`, `s = Σ_j x_j`: the squared length of the Riemannian
gradient of `𝖤_0`, up to the factor `(2/n)²`. -/
noncomputable def tangentSq (X : SphereTuple d n) : ℝ :=
  ∑ i : Idx n, ‖proj d (X i) (∑ j : Idx n, (X j : EucSpace d))‖ ^ 2

theorem tangentSq_nonneg (X : SphereTuple d n) : 0 ≤ tangentSq X :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem continuous_tangentSq : Continuous (tangentSq (d := d) (n := n)) := by
  unfold tangentSq proj
  fun_prop

/-- A tuple with no tangential gradient is a critical point of `𝖤_0`. -/
theorem isCriticalE0_of_tangentSq_eq_zero {X : SphereTuple d n} (h : tangentSq X = 0) :
    IsCriticalE0 d n X := fun i =>
  norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp
    ((Finset.sum_eq_zero_iff_of_nonneg fun _ _ => sq_nonneg _).mp h i (Finset.mem_univ i)))

/-- The hypothesis of `isCriticalE0_of_tangentSq_eq_zero` holds for the empty
tuple. -/
example : tangentSq (fun i : Idx 0 => i.elim0 : SphereTuple 2 0) = 0 := by
  simp [tangentSq]

/-- `‖Proj_x v‖² = ‖v‖² - ⟨x, v⟩²` for a unit vector `x`. -/
theorem norm_proj_sq {x : EucSpace d} (hx : ‖x‖ = 1) (v : EucSpace d) :
    ‖proj d x v‖ ^ 2 = ‖v‖ ^ 2 - inner (𝕜 := ℝ) x v ^ 2 := by
  rw [← inner_proj_self d hx]
  simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
  ring

/-- The hypothesis of `norm_proj_sq` holds for a basis vector of `ℝ²`. -/
example : ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 := by simp

/-- The barycentre of `n` unit vectors has length at most `n`. -/
theorem norm_sum_coe_le (X : SphereTuple d n) : ‖∑ j : Idx n, (X j : EucSpace d)‖ ≤ n :=
  (norm_sum_le _ _).trans (by simp)

/-- **The gradient inequality of `𝖤_0`.**  For some integer `m`, `|m| ≤ n`,

  `‖s‖² |‖s‖² - m²| ≤ 2n Σ_i ‖Proj_{x_i} s‖²`,  `s = Σ_j x_j`;

`m = Σ_i σ_i` for the signs `σ_i` of `⟨x_i, s⟩`. -/
theorem exists_int_mul_abs_sub_le (X : SphereTuple d n) :
    ∃ m : ℤ, |m| ≤ n ∧ ‖∑ j : Idx n, (X j : EucSpace d)‖ ^ 2 *
      |‖∑ j : Idx n, (X j : EucSpace d)‖ ^ 2 - (m : ℝ) ^ 2| ≤ 2 * n * tangentSq X := by
  set s := ∑ j : Idx n, (X j : EucSpace d) with hs
  set ρ := ‖s‖ with hρ
  let σ : Idx n → ℤ := fun i => if inner (𝕜 := ℝ) (X i : EucSpace d) s < 0 then -1 else 1
  refine ⟨∑ i, σ i, ?_, ?_⟩
  · refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
    simp [σ, apply_ite]
  set M : ℝ := ((∑ i, σ i : ℤ) : ℝ) with hM
  have hρ0 : 0 ≤ ρ := norm_nonneg _
  have hρn : ρ ≤ n := norm_sum_coe_le X
  have hMn : |M| ≤ n := by
    rw [hM, ← Int.cast_abs]
    have h : |∑ i, σ i| ≤ ∑ i, |σ i| := Finset.abs_sum_le_sum_abs _ _
    have h' : ∑ i, |σ i| = n := by simp [σ, apply_ite]
    exact_mod_cast h.trans h'.le
  have key : ∀ i : Idx n,
      ρ * |inner (𝕜 := ℝ) (X i : EucSpace d) s - σ i * ρ| ≤ ‖proj d (X i) s‖ ^ 2 := by
    intro i
    have hx : ‖(X i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X i).2
    rw [norm_proj_sq hx]
    have ha : |inner (𝕜 := ℝ) (X i : EucSpace d) s| ≤ ρ := by
      simpa [hx] using abs_real_inner_le_norm (X i : EucSpace d) s
    obtain ⟨ha1, ha2⟩ := abs_le.mp ha
    simp only [σ]
    split_ifs with h
    · push_cast
      rw [abs_of_nonneg (by linarith)]
      nlinarith
    · push_cast
      rw [abs_of_nonpos (by linarith)]
      nlinarith
  have hsum : ∑ i, (inner (𝕜 := ℝ) (X i : EucSpace d) s - σ i * ρ) = ρ ^ 2 - M * ρ := by
    rw [Finset.sum_sub_distrib, ← sum_inner, ← hs, real_inner_self_eq_norm_sq, hM,
      Int.cast_sum, Finset.sum_mul]
  have h1 : ρ * |ρ ^ 2 - M * ρ| ≤ tangentSq X := by
    rw [← hsum]
    calc ρ * |∑ i, (inner (𝕜 := ℝ) (X i : EucSpace d) s - σ i * ρ)|
        ≤ ρ * ∑ i, |inner (𝕜 := ℝ) (X i : EucSpace d) s - σ i * ρ| :=
          mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) hρ0
      _ = ∑ i, ρ * |inner (𝕜 := ℝ) (X i : EucSpace d) s - σ i * ρ| := Finset.mul_sum _ _ _
      _ ≤ tangentSq X := Finset.sum_le_sum fun i _ => key i
  have hρM : |ρ + M| ≤ 2 * n :=
    (abs_add_le _ _).trans (by rw [abs_of_nonneg hρ0]; linarith)
  have heq : ρ ^ 2 * |ρ ^ 2 - M ^ 2| = ρ * |ρ ^ 2 - M * ρ| * |ρ + M| := by
    rw [show ρ ^ 2 - M ^ 2 = (ρ - M) * (ρ + M) by ring,
      show ρ ^ 2 - M * ρ = ρ * (ρ - M) by ring, abs_mul, abs_mul, abs_of_nonneg hρ0]
    ring
  rw [heq]
  calc ρ * |ρ ^ 2 - M * ρ| * |ρ + M| ≤ tangentSq X * (2 * n) :=
        mul_le_mul h1 hρM (abs_nonneg _) (tangentSq_nonneg X)
    _ = 2 * n * tangentSq X := by ring

/-- **The barycentre grows at rate `(4/n) Σ_i ‖Proj_{x_i} s‖²`** along
`e:gradfl`: `‖s‖² = n 𝖤_0` and `hasDerivAt_E0_ascent`. -/
theorem hasDerivAt_norm_sum_sq (hn : 0 < n) {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) (t : ℝ) :
    HasDerivAt (fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)
      (4 * (n : ℝ)⁻¹ * tangentSq (X t)) t := by
  have hn' : (n : ℝ) ≠ 0 := by positivity
  have h := (hasDerivAt_E0_ascent d n X hX t).const_mul (n : ℝ)
  have hf : (fun s => (n : ℝ) * E0 d n (X s))
      = fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2 := by
    funext s
    rw [E0_eq_norm_sum_sq, ← mul_assoc, mul_inv_cancel₀ hn', one_mul]
  rw [hf] at h
  convert h using 1
  rw [tangentSq]
  field_simp
  ring

/-- The hypotheses of `hasDerivAt_norm_sum_sq` hold for a consensus of three
particles on the circle, a stationary solution. -/
example (x : SSphere 2) : 0 < 3 ∧ E0GradientAscent 2 3 (fun _ => signedTuple x ∅) :=
  ⟨by norm_num, E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅)⟩

/-- `‖s‖² ≤ n²` along every path of tuples. -/
theorem bddAbove_norm_sum_sq (X : ℝ → SphereTuple d n) :
    BddAbove (range fun t => ‖∑ j : Idx n, (X t j : EucSpace d)‖ ^ 2) :=
  ⟨(n : ℝ) ^ 2, by
    rintro _ ⟨t, rfl⟩
    exact pow_le_pow_left₀ (norm_nonneg _) (norm_sum_coe_le _) 2⟩

/-- A real number keeps a positive distance from the squares `m²`, `|m| ≤ n`,
other than itself. -/
theorem exists_pos_le_abs_sq_sub (n : ℕ) (q : ℝ) :
    ∃ γ > 0, ∀ m : ℤ, |m| ≤ n → (m : ℝ) ^ 2 ≠ q → γ ≤ |(m : ℝ) ^ 2 - q| := by
  set S : Finset ℝ := ((Finset.Icc (-(n : ℤ)) n).image fun m : ℤ => (m : ℝ) ^ 2).erase q
  obtain ⟨γ, hγ, hball⟩ := Metric.isOpen_iff.mp S.isClosed.isOpen_compl q (by simp [S])
  refine ⟨γ, hγ, fun m hm hmq => not_lt.mp fun hlt => ?_⟩
  have hmS : (m : ℝ) ^ 2 ∈ S := Finset.mem_erase.mpr
    ⟨hmq, Finset.mem_image.mpr ⟨m, Finset.mem_Icc.mpr (abs_le.mp hm), rfl⟩⟩
  exact hball (by rwa [Metric.mem_ball, Real.dist_eq]) hmS

end Perspective
end Transformer
