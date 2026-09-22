/-
# Metastability — Two obstructions to being `(β, ε)`-separated

Both read off `γ(β) = 1 - α - 8ε - β⁻¹ log(2n²/ε)` of `eq: gamma`.

* `α ≥ -1`, so `γ(β) ≤ 2 - β⁻¹ log(2n²/ε)`, which is negative as soon as
  `n ≥ e^{2β}`: past that size no configuration is separated at all
  (`not_isSeparated_of_large`).
* Two points with `1 - 8ε ≤ ⟨x_i, x_j⟩ < 1 - 4ε` spoil every choice of
  centres: a single cap of height `1 - ε` forces `⟨x_i, x_j⟩ ≥ 1 - 4ε`, and two
  distinct caps force `α ≥ ⟨x_i, x_j⟩ ≥ 1 - 8ε`, hence `γ(β) ≤ 0`
  (`not_isSeparated_of_inner`).

Source: arXiv:2410.06833v1, `hyp: init`, `eq: gamma`, `eq: alpha.dist`.
-/

import Transformer.Metastability.UniformCap
import Mathlib.Analysis.Complex.ExponentialBounds

open Real

namespace Transformer
namespace Metastability

variable {d : ℕ}

/-- Two points of one cap `𝒮_w(ε)` have `⟨x, y⟩ ≥ 1 - 4ε`: each is within
`√(2ε)` of `w`, so they are within `2√(2ε)` of each other. -/
theorem inner_ge_of_mem_sphericalCap (w x y : SSphere d) (ε : ℝ)
    (hx : x ∈ sphericalCap d w ε) (hy : y ∈ sphericalCap d w ε) :
    1 - 4 * ε ≤ inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) := by
  have hx' : 1 - ε ≤ inner (𝕜 := ℝ) (x : EucSpace d) (w : EucSpace d) := hx
  have hy' : 1 - ε ≤ inner (𝕜 := ℝ) (y : EucSpace d) (w : EucSpace d) := hy
  have ha := norm_sub_sq_sphere w x
  have hb := norm_sub_sq_sphere w y
  have hxy := norm_sub_sq_sphere y x
  have htri := norm_sub_le_norm_sub_add_norm_sub (x : EucSpace d) (w : EucSpace d)
    (y : EucSpace d)
  rw [norm_sub_rev (w : EucSpace d)] at htri
  set a := ‖(x : EucSpace d) - (w : EucSpace d)‖
  set b := ‖(y : EucSpace d) - (w : EucSpace d)‖
  set c := ‖(x : EucSpace d) - (y : EucSpace d)‖
  have hsq : c ^ 2 ≤ (a + b) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) htri 2
  nlinarith [sq_nonneg (a - b)]

/-- A set of reals inside `[-1, 1]` has supremum at least `-1`, `sSup ∅ = 0`
included. -/
theorem neg_one_le_sSup {S : Set ℝ} (h : ∀ c ∈ S, -1 ≤ c ∧ c ≤ 1) : -1 ≤ sSup S := by
  rcases S.eq_empty_or_nonempty with rfl | ⟨c, hc⟩
  · simp
  · exact (h c hc).1.trans (le_csSup ⟨1, fun c hc => (h c hc).2⟩ hc)

/-- The set whose supremum is `αDist` lies in `[-1, 1]`. -/
theorem mem_Icc_of_mem_αDist {k : ℕ} {w : Idx k → SSphere d} {ε c : ℝ}
    (hc : ∃ i j : Idx k, i ≠ j ∧
      ∃ x ∈ sphericalCap d (w i) (2 * ε), ∃ y ∈ sphericalCap d (w j) (2 * ε),
        c = inner (𝕜 := ℝ) ((x : EucSpace d)) ((y : EucSpace d))) :
    -1 ≤ c ∧ c ≤ 1 := by
  obtain ⟨-, -, -, x, -, y, -, rfl⟩ := hc
  exact inner_mem_Icc_sphere x y

/-- **Past `n ≥ e^{2β}` nothing is separated.**  `α ≥ -1` bounds
`γ(β) ≤ 2 - 8ε - β⁻¹ log(2n²/ε)`, and `2n²/ε ≥ n ≥ e^{2β}`. -/
theorem not_isSeparated_of_large {n : ℕ} {β ε : ℝ} (hβ : 0 < β) (hε : 0 < ε) (hε1 : ε < 1)
    (hn : Real.exp (2 * β) ≤ n) (X : SphereTuple d n) : ¬ isSeparated d n β ε X := by
  rintro ⟨k, -, w, -, hγ⟩
  have hα : -1 ≤ αDist d k w ε := neg_one_le_sSup fun c hc => mem_Icc_of_mem_αDist hc
  have hn1 : (1 : ℝ) ≤ n := le_trans (by simp [hβ.le]) hn
  have hlog : 2 * β ≤ Real.log (2 * (n : ℝ) ^ 2 / ε) := by
    rw [← Real.log_exp (2 * β)]
    refine Real.log_le_log (Real.exp_pos _) (hn.trans ?_)
    rw [le_div_iff₀ hε]
    nlinarith
  have h2 : 2 ≤ β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / ε) := by
    rw [le_inv_mul_iff₀ hβ]; linarith
  unfold γβ at hγ
  linarith

/-- The hypotheses of `not_isSeparated_of_large` are satisfiable:
`β = 1/2`, `ε = 1/2`, `n = 3 ≥ e`. -/
example : Real.exp (2 * (1 / 2 : ℝ)) ≤ ((3 : ℕ) : ℝ) := by
  rw [show 2 * (1 / 2 : ℝ) = 1 by norm_num]
  exact Real.exp_one_lt_three.le.trans (by norm_num)

/-- **An inner product in `[1 - 8ε, 1 - 4ε)` spoils separation.**  If the two
points share a cap, `inner_ge_of_mem_sphericalCap` contradicts the upper end;
if not, they lie in distinct caps of height `1 - 2ε`, so `α ≥ ⟨x_i, x_j⟩ ≥
1 - 8ε` and `γ(β) ≤ -β⁻¹ log(2n²/ε) ≤ 0`. -/
theorem not_isSeparated_of_inner {n : ℕ} {β ε : ℝ} (hβ : 0 < β) (hε : 0 < ε) (hε1 : ε < 1)
    (X : SphereTuple d n) (i j : Idx n)
    (hlo : 1 - 8 * ε ≤ inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d))
    (hhi : inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d) < 1 - 4 * ε) :
    ¬ isSeparated d n β ε X := by
  rintro ⟨k, -, w, hcov, hγ⟩
  obtain ⟨p, hp⟩ := hcov i
  obtain ⟨q, hq⟩ := hcov j
  by_cases hpq : p = q
  · subst hpq
    linarith [inner_ge_of_mem_sphericalCap (w p) (X i) (X j) ε hp hq]
  have hwide : ∀ r x, x ∈ sphericalCap d (w r) ε → x ∈ sphericalCap d (w r) (2 * ε) := by
    intro r x hx
    have hx' : 1 - ε ≤ inner (𝕜 := ℝ) (x : EucSpace d) (w r : EucSpace d) := hx
    show 1 - 2 * ε ≤ inner (𝕜 := ℝ) (x : EucSpace d) (w r : EucSpace d)
    linarith
  have hα : inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d) ≤ αDist d k w ε :=
    le_csSup ⟨1, fun c hc => (mem_Icc_of_mem_αDist hc).2⟩
      ⟨p, q, hpq, X i, hwide p _ hp, X j, hwide q _ hq, rfl⟩
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast i.pos
  have hlog : 0 ≤ β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / ε) := by
    refine mul_nonneg (inv_nonneg.2 hβ.le) (Real.log_nonneg ?_)
    rw [le_div_iff₀ hε]
    nlinarith
  unfold γβ at hγ
  linarith

/-- The hypotheses of `not_isSeparated_of_inner` are satisfiable: at
`ε = 1/8` the window is `[0, 1/2)`, met by two orthogonal points of `𝕊^1`. -/
example : (1 : ℝ) - 8 * (1 / 8) ≤ inner (𝕜 := ℝ)
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) < (1 : ℝ) - 4 * (1 / 8) := by
  simp [EuclideanSpace.inner_single_left]
  norm_num

end Metastability
end Transformer
