module

public import Transformer.ScheduleFree.Section2_Averaging

/-!
# Momentum and recovery of linear decay

arXiv:2405.15682v4, Appendix B, Theorem 4 (`thm:reformulation`).
The printed formula divides by individual weights, so those weights and
the relevant total are explicitly nonzero. The main conversion theorem
does not need individual weights to be positive after the first one.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Solve a nonsingular scalar multiple of a vector; used to undo the weights. -/
theorem solve_smul {a b : ℝ} {x y : E} (ha : a ≠ 0) (h : a • x = b • y) :
    x = (b / a) • y := by
  calc
    x = a⁻¹ • (a • x) := by rw [smul_smul, inv_mul_cancel₀ ha, one_smul]
    _ = a⁻¹ • (b • y) := by rw [h]
    _ = (b / a) • y := by rw [smul_smul]; congr 1; ring

example (x : E) : x = ((1 : ℝ) / 1) • x := solve_smul one_ne_zero rfl

/-- The weighted-average recursion in displacement form, Appendix B's proof. -/
theorem balance_update {U w : ℝ} {p x z : E} (htotal : U + w ≠ 0)
    (hb : (U + w) • x = U • p + w • z) :
    x - p = (w / (U + w)) • (z - p) := by
  apply solve_smul htotal
  rw [smul_sub, hb]
  module

example (x : E) : x - x = ((1 : ℝ) / (0 + 1)) • (x - x) :=
  balance_update (by norm_num) (by simp)

/-- Appendix B, Theorem 4, first formula: the momentum recursion.
`U=w_{1:t-1}`, `w=w_t`, and `v=w_{t+1}`. -/
theorem momentum_reformulation {U w v : ℝ} {p x xn z zn : E}
    (hw : w ≠ 0) (htotal : U + w + v ≠ 0)
    (hprev : (U + w) • x = U • p + w • z)
    (hnext : (U + w + v) • xn = (U + w) • x + v • zn) :
    xn - x = (v * U / (w * (U + w + v))) • (x - p) +
      (v / (U + w + v)) • (zn - z) := by
  have hz : z - x = (U / w) • (x - p) :=
    solve_smul hw (balance_displacement hprev).symm
  have hn := balance_update htotal hnext
  rw [show zn - x = (zn - z) + (z - x) by module, hz] at hn
  rw [hn]
  have hc : v / (U + w + v) * (U / w) = v * U / (w * (U + w + v)) := by
    field_simp
  rw [smul_add, smul_smul, hc, add_comm]

example (x : E) : x - x = ((1 : ℝ) * 0 / (1 * (0 + 1 + 1))) • (x - x) +
    ((1 : ℝ) / (0 + 1 + 1)) • (x - x) :=
  momentum_reformulation (U := 0) (w := 1) (v := 1) (p := x) (x := x)
    (xn := x) (z := x) (zn := x) one_ne_zero (by norm_num) (by simp) (by simp; module)

/-- Appendix B, Theorem 4, second formula: updates at the gradient location. -/
theorem query_reformulation {W v β βnext : ℝ} {x xn z zn : E}
    (hv : v ≠ 0) (hnext : (W + v) • xn = W • x + v • zn) :
    mix (1 - βnext) xn zn - mix (1 - β) x z =
      (β + (β - βnext) * W / v) • (xn - x) + (1 - β) • (zn - z) := by
  have hz : zn - xn = (W / v) • (xn - x) :=
    solve_smul hv (balance_displacement hnext).symm
  calc
    _ = β • (xn - x) + (1 - β) • (zn - z) + (β - βnext) • (zn - xn) := by
      unfold mix
      module
    _ = _ := by rw [hz]; module

example (β βnext : ℝ) (x : E) :
    mix (1 - βnext) x x - mix (1 - β) x x =
      (β + (β - βnext) * 1 / 1) • (x - x) + (1 - β) • (x - x) :=
  query_reformulation one_ne_zero (by simp; module)

/-- Appendix B: constant interpolation gives `u_t=β m_t+(1-β)Δ_t`. -/
theorem constant_momentum (β : ℝ) (x xn z zn : E) :
    mix (1 - β) xn zn - mix (1 - β) x z =
      β • (xn - x) + (1 - β) • (zn - z) := by
  unfold mix
  module

/-- Appendix B.1: the coefficient of momentum vanishes for `β_t=t/T`. -/
theorem linear_decay_cancellation (t T : ℝ) (hT : T ≠ 0) :
    t / T + t * (t / T - (t + 1) / T) = 0 := by
  field_simp
  ring

example (t : ℝ) : t / 1 + t * (t / 1 - (t + 1) / 1) = 0 :=
  linear_decay_cancellation t 1 one_ne_zero

/-- Appendix B.1: weighted linear decay cancels the momentum term as well. -/
theorem weighted_decay_cancellation (W v T : ℝ) (hv : v ≠ 0) (hT : T ≠ 0) :
    W / T + (W / T - (W + v) / T) * W / v = 0 := by
  field_simp
  ring

example (W : ℝ) : W / 1 + (W / 1 - (W + 1) / 1) * W / 1 = 0 :=
  weighted_decay_cancellation W 1 1 one_ne_zero one_ne_zero

/-- Appendix B.1: `β=W/T` produces the exact scheduled update. -/
theorem weighted_linear_decay {W v T : ℝ} {x xn z zn : E}
    (hv : v ≠ 0) (hT : T ≠ 0) (hnext : (W + v) • xn = W • x + v • zn) :
    mix (1 - (W + v) / T) xn zn - mix (1 - W / T) x z =
      (1 - W / T) • (zn - z) := by
  rw [query_reformulation hv hnext, weighted_decay_cancellation W v T hv hT,
    zero_smul, zero_add]

example (x : E) :
    mix (1 - (1 + 1) / 2) x x - mix (1 - 1 / 2) x x =
      (1 - (1 : ℝ) / 2) • (x - x) :=
  weighted_linear_decay one_ne_zero (by norm_num) (by simp; module)

/-- Appendix B.1: the scheduled last query is exactly the returned average. -/
theorem final_query (x z : E) : mix (1 - 1) x z = x := by
  simp [mix]

end Transformer.ScheduleFree
