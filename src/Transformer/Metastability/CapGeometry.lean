/-
# Metastability — the geometry of the spherical caps

Elementary facts about the caps `𝒮_q(c) = {x ∈ 𝕊^{d-1} : ⟨x, w_q⟩ ≥ 1 - c}` and the
separation constant `α = αDist d k w ε` of arXiv:2410.06833v1, used by the direct proof of
`thm: metastability`:

* `inner_ge_of_mem_cap` — two points of one cap `𝒮_w(c)` have `⟨x, y⟩ ≥ 1 - 4c`;
* `norm_sq_sub_eq` — `‖x - y‖² = 2 - 2 ⟨x, y⟩` on the sphere;
* `inner_le_alphaDist` — points of two different caps `𝒮_q(2ε)`, `𝒮_{q'}(2ε)` have
  `⟨x, y⟩ ≤ α`;
* `eight_eps_lt_one_sub_alphaDist` — `γ(β) > 0` forces `8ε < 1 - α`;
* `cap_eq_of_mem` — the caps `𝒮_q(2ε)` are pairwise disjoint when `α < 1`.
-/

import Transformer.Metastability.Basic
import Transformer.Metastability.AttnFlow

open scoped BigOperators InnerProductSpace
open Real

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- `‖x - y‖² = 2 - 2 ⟨x, y⟩` for points of the sphere. -/
theorem norm_sq_sub_eq (x y : SSphere d) :
    ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 = 2 - 2 * ⟪(x : EucSpace d), (y : EucSpace d)⟫_ℝ := by
  rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2]
  ring

/-- **Two points of a cap are close.**  If `⟨x, w⟩ ≥ 1 - c` and `⟨y, w⟩ ≥ 1 - c` for unit
vectors, then `⟨x, y⟩ ≥ 2⟨x,w⟩ + 2⟨y,w⟩ - 3 ≥ 1 - 4c`, because
`0 ≤ ‖x + y - 2w‖² = 6 + 2⟨x,y⟩ - 4⟨x,w⟩ - 4⟨y,w⟩`. -/
theorem inner_ge_of_mem_cap {x y w : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (hw : ‖w‖ = 1) {c : ℝ} (hxw : 1 - c ≤ ⟪x, w⟫_ℝ) (hyw : 1 - c ≤ ⟪y, w⟫_ℝ) :
    1 - 4 * c ≤ ⟪x, y⟫_ℝ := by
  have h : 0 ≤ ‖x + y - (2 : ℝ) • w‖ ^ 2 := sq_nonneg _
  simp only [norm_sub_sq_real, norm_add_sq_real, norm_smul, hx, hy, hw, inner_add_left,
    real_inner_smul_right] at h
  norm_num at h
  linarith

/-- **The separation constant bounds the inner products between different caps.**  For
`x ∈ 𝒮_q(2ε)` and `y ∈ 𝒮_{q'}(2ε)` with `q ≠ q'`, `⟨x, y⟩ ≤ α(ε)`. -/
theorem inner_le_alphaDist {k : ℕ} {w : Idx k → SSphere d} {ε : ℝ} {q q' : Idx k}
    (hqq : q ≠ q') {x y : SSphere d} (hx : x ∈ sphericalCap d (w q) (2 * ε))
    (hy : y ∈ sphericalCap d (w q') (2 * ε)) :
    ⟪(x : EucSpace d), (y : EucSpace d)⟫_ℝ ≤ αDist d k w ε :=
  le_csSup ⟨1, by
      rintro c ⟨i, j, -, x, -, y, -, rfl⟩
      exact inner_sphere_le_one x y⟩
    ⟨q, q', hqq, x, hx, y, hy, rfl⟩

/-- **`γ(β) > 0` forces `8ε < 1 - α`**: the term `β⁻¹ log(2n²/ε)` of `γ` is nonnegative
because `2n²/ε ≥ 1`. -/
theorem eight_eps_lt_one_sub_alphaDist {β ε α : ℝ} (hβ : 0 < β) (hε : 0 < ε)
    (hε' : ε < 1 / 16) (hn : 1 ≤ n) (hγ : 0 < γβ n β α ε) : 8 * ε < 1 - α := by
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hlog : 0 ≤ Real.log (2 * (n : ℝ) ^ 2 / ε) := by
    apply Real.log_nonneg
    rw [le_div_iff₀ hε]
    nlinarith
  have h := mul_nonneg (inv_nonneg.2 hβ.le) hlog
  unfold γβ at hγ
  linarith

/-- **The caps `𝒮_q(2ε)` are pairwise disjoint when `α < 1`**: a point of two of them would
have `1 = ⟨x, x⟩ ≤ α`. -/
theorem cap_eq_of_mem {k : ℕ} {w : Idx k → SSphere d} {ε : ℝ}
    (hα : αDist d k w ε < 1) {x : SSphere d} {q q' : Idx k}
    (hx : x ∈ sphericalCap d (w q) (2 * ε)) (hx' : x ∈ sphericalCap d (w q') (2 * ε)) :
    q = q' := by
  by_contra hqq
  have h := inner_le_alphaDist hqq hx hx'
  rw [inner_sphere_self] at h
  linarith

end Metastability
end Transformer
