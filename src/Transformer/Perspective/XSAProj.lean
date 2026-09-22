/-
# The projection of exclusive self-attention

`P_v^⊥ y = y - (⟨y, v⟩ / ‖v‖²) v` (`xsaProj`), the one operation exclusive
self-attention adds to a head (`eq:xsa`, arXiv:2603.09078v1, §3): the output
`y_i` of the head with its component along its own value `v_i` removed.  It is
`Transformer.XSA.XSAOutput` read as a function of the two vectors
(`xsaOutput_eq_xsaProj`).

Four properties, and the fourth is the one a residual block needs.  It removes
exactly what it claims (`inner_xsaProj_self`), for every `v`, `0` included; it
does not lengthen (`norm_xsaProj_le`); it is homogeneous in `y`
(`smul_xsaProj`), so a gate on the output of a head is a gate on its weights;
and — since it forgets the direction `v` entirely (`xsaProj_add_smul_self`) —
it turns a head into a difference: `P_{u_i}^⊥ (Σ_j a_{ij} u_j)` is
`P_{u_i}^⊥ (Σ_j a_{ij} (u_j - u_i))`, of norm at most `A r` whenever the values
lie within `r` of `u_i` and `Σ_j |a_{ij}| ≤ A` (`norm_xsaProj_sum_le`).  The
rows are free there: no common sum, which `norm_sum_smul_sub_sum_smul_le` needs
for a head without the projection.
-/

import Transformer.XSA

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The projection of exclusive self-attention:** `y` with its component along
`v` removed,

  `P_v^⊥ y = y - (⟨y, v⟩ / ‖v‖²) v`.

Source: `eq:xsa` of Zhai, "Exclusive Self Attention", arXiv:2603.09078v1, §3,
`z_i = y_i - (y_i^T v_i) v_i / ‖v_i‖²`, as a function of `v_i` and `y_i` alone.
At `v = 0` the paper's quotient is `0 / 0`; here it is `0`, so `P_0^⊥ y = y`,
which is what the paper's own Algorithm 1 computes, and parameter-golf with it:
`F.normalize` divides by `max(‖v‖, 10⁻¹²)`, so the code and `eq:xsa` differ for
`0 < ‖v‖ < 10⁻¹²` only. -/
noncomputable def xsaProj {e : ℕ} (v y : EucSpace e) : EucSpace e :=
  y - (inner (𝕜 := ℝ) y v / ‖v‖ ^ 2) • v

/-- **`eq:xsa` is this projection of the output of the head on its own value.**

Source: `XSA.XSAOutput` (arXiv:2603.09078v1, `eq:xsa`), unfolded. -/
theorem xsaOutput_eq_xsaProj (Q K V : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n) :
    XSA.XSAOutput d n Q K V x i = xsaProj (V (x i)) (XSA.SAOutput d n Q K V x i) :=
  rfl

/-- **What the projection removes, it removes exactly:** `⟨P_v^⊥ y, v⟩ = 0`, for
every `v`, `v = 0` included.

Source: arXiv:2603.09078v1, §3, "XSA's output `z_i` no longer contains `v_i`";
`XSA.xsa_output_orthogonal_to_value` is the same for `v ≠ 0`. -/
theorem inner_xsaProj_self {e : ℕ} (v y : EucSpace e) :
    inner (𝕜 := ℝ) (xsaProj v y) v = 0 := by
  rcases eq_or_ne v 0 with rfl | hv
  · simp [xsaProj]
  · have hnorm : ‖v‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hv)
    rw [xsaProj, inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
    field_simp
    ring

/-- **The projection forgets the direction it projects off:**
`P_v^⊥ (y + c v) = P_v^⊥ y`.

Source: none — posed here; `eq:xsa` (arXiv:2603.09078v1, §3) is a linear
projection onto `v^⊥`, and this is the half of that which a block uses. -/
theorem xsaProj_add_smul_self {e : ℕ} (v y : EucSpace e) (c : ℝ) :
    xsaProj v (y + c • v) = xsaProj v y := by
  rcases eq_or_ne v 0 with rfl | hv
  · simp [xsaProj]
  · have hnorm : ‖v‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hv)
    have hc : inner (𝕜 := ℝ) (y + c • v) v / ‖v‖ ^ 2
        = inner (𝕜 := ℝ) y v / ‖v‖ ^ 2 + c := by
      rw [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
      field_simp
    simp only [xsaProj, hc, add_smul]
    abel

/-- **A gate on the output of a head is a gate on its weights:**
`c P_v^⊥ y = P_v^⊥ (c y)`.

Source: none — posed here; `eq:xsa` (arXiv:2603.09078v1, §3) is linear in `y`,
and parameter-golf multiplies the output of each head by a per-head, per-token
gate after the projection (`CausalSelfAttention.forward`). -/
theorem smul_xsaProj {e : ℕ} (v y : EucSpace e) (c : ℝ) :
    c • xsaProj v y = xsaProj v (c • y) := by
  simp only [xsaProj, real_inner_smul_left, smul_sub, smul_smul, mul_div_assoc]

/-- **The projection does not lengthen:** `‖P_v^⊥ y‖ ≤ ‖y‖`.

Source: none — posed here; `eq:xsa` (arXiv:2603.09078v1, §3) and Pythagoras. -/
theorem norm_xsaProj_le {e : ℕ} (v y : EucSpace e) : ‖xsaProj v y‖ ≤ ‖y‖ := by
  have hy : y = xsaProj v y + (inner (𝕜 := ℝ) y v / ‖v‖ ^ 2) • v := by simp [xsaProj]
  have horth : inner (𝕜 := ℝ) (xsaProj v y) ((inner (𝕜 := ℝ) y v / ‖v‖ ^ 2) • v) = 0 := by
    rw [real_inner_smul_right, inner_xsaProj_self, mul_zero]
  have h := norm_add_sq_eq_norm_sq_add_norm_sq_real horth
  rw [← hy] at h
  refine (mul_self_le_mul_self_iff (norm_nonneg _) (norm_nonneg _)).2 ?_
  rw [h]
  exact le_add_of_nonneg_right (mul_self_nonneg _)

/-- **An exclusive head is as small as the spread of its values:** if
`Σ_j |b_j| ≤ A` and every `u_j` lies within `r` of `u₀`, then
`‖P_{u₀}^⊥ (Σ_j b_j u_j)‖ ≤ A r`.

Source: none — posed here; `eq:xsa` (arXiv:2603.09078v1, §3) applied to
`y_i = Σ_j a_{ij} v_j` at `u₀ = v_i`.  Unlike `norm_sum_smul_sub_sum_smul_le`,
which compares two rows of equal sum, one row suffices and its sum is free: the
projection kills `(Σ_j b_j) u₀` whatever `Σ_j b_j` is. -/
theorem norm_xsaProj_sum_le {e : ℕ} {b : Idx n → ℝ} {u : Idx n → EucSpace e}
    (u₀ : EucSpace e) {A r : ℝ} (hb : ∑ j, |b j| ≤ A) (hr : 0 ≤ r)
    (hu : ∀ j, ‖u j - u₀‖ ≤ r) : ‖xsaProj u₀ (∑ j, b j • u j)‖ ≤ A * r := by
  have hsplit : ∑ j, b j • u j = ∑ j, b j • (u j - u₀) + (∑ j, b j) • u₀ := by
    simp only [smul_sub, Finset.sum_sub_distrib, Finset.sum_smul]
    abel
  rw [hsplit, xsaProj_add_smul_self]
  calc ‖xsaProj u₀ (∑ j, b j • (u j - u₀))‖ ≤ ‖∑ j, b j • (u j - u₀)‖ := norm_xsaProj_le _ _
    _ ≤ ∑ j, ‖b j • (u j - u₀)‖ := norm_sum_le _ _
    _ ≤ ∑ j, |b j| * r := Finset.sum_le_sum fun j _ => by
        rw [norm_smul, Real.norm_eq_abs]
        exact mul_le_mul_of_nonneg_left (hu j) (abs_nonneg _)
    _ = (∑ j, |b j|) * r := by rw [Finset.sum_mul]
    _ ≤ A * r := mul_le_mul_of_nonneg_right hb hr

/-- The hypotheses of `norm_xsaProj_sum_le` are satisfiable, and not only at
`r = 0`: two weights `1/2` on two values at distance `1` from their midpoint,
`A = 1`, `r = 1`. -/
example : ∑ _j : Idx 2, |(1 / 2 : ℝ)| ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    ∀ j : Idx 2, ‖![(EuclideanSpace.single 0 1 : EucSpace 2),
      -EuclideanSpace.single 0 1] j - 0‖ ≤ 1 := by
  refine ⟨by norm_num, zero_le_one, fun j => ?_⟩
  fin_cases j <;> simp

end Perspective
end Transformer
