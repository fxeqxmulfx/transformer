module

public import Transformer.SequenceModel.Attention.Adaptive.Basic

/-!
# Dimension-aware activation and product propagation

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M2.
Coordinate multipliers are bounded at recorded inputs. Euclidean errors
are aggregated once, rather than converting every coordinate back to a
full vector error at every multiplication.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Adaptive
open GPTMini Stability Execution Landscape.Neural
open scoped BigOperators

/-- M2: coordinatewise relative bounds transfer to the Euclidean norm. -/
theorem relative_norm {d : ℕ} (x y : EucSpace d) (L : ℝ) (hL : 0 ≤ L)
    (h : ∀ c, |x c| ≤ L * |y c|) : ‖x‖ ≤ L * ‖y‖ := by
  have hs : ‖x‖ ^ 2 ≤ L ^ 2 * ‖y‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro c hc
    have ht := h c
    have ha := abs_nonneg (x c)
    have hb := mul_nonneg hL (abs_nonneg (y c))
    nlinarith [sq_abs (x c), sq_abs (y c)]
  nlinarith [norm_nonneg x, norm_nonneg y, mul_nonneg hL (norm_nonneg y)]

/-- M2: SiLU movement uses a recorded gate bound, and adds its local residual. -/
theorem silu_step {d : ℕ} (x y z : EucSpace d) (R E δ : ℝ) (hR : 0 ≤ R)
    (hbound : ∀ c, |x c| ≤ R) (hx : ‖x - y‖ ≤ E)
    (hz : ‖z - WithLp.toLp 2 (fun c => silu (x c))‖ ≤ δ) :
    ‖z - WithLp.toLp 2 (fun c => silu (y c))‖ ≤ (1 + R) * E + δ := by
  have hm : ‖WithLp.toLp 2 (fun c => silu (x c)) -
      WithLp.toLp 2 (fun c => silu (y c))‖ ≤ (1 + R) * E := by
    refine (relative_norm _ (x - y) (1 + R) (by linarith) ?_).trans
      (mul_le_mul_of_nonneg_left hx (by linarith))
    intro c
    change |silu (x c) - silu (y c)| ≤ (1 + R) * |x c - y c|
    simpa only [abs_sub_comm] using silu_error (y c) (x c) R (hbound c)
  exact (norm_sub_le_norm_sub_add_norm_sub z (WithLp.toLp 2 (fun c => silu (x c))) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

/-- M2: a Hadamard product keeps two input radii and the cross term. -/
theorem product_step {d : ℕ} (a v a₀ v₀ z : EucSpace d) (A V E F δ : ℝ)
    (hA : 0 ≤ A) (hV : 0 ≤ V) (ha : ∀ c, |a c| ≤ A) (hv : ∀ c, |v c| ≤ V)
    (he : ‖a - a₀‖ ≤ E) (hf : ‖v - v₀‖ ≤ F)
    (hz : ‖z - WithLp.toLp 2 (fun c => a c * v c)‖ ≤ δ) :
    ‖z - WithLp.toLp 2 (fun c => a₀ c * v₀ c)‖ ≤ V * E + (A + E) * F + δ := by
  have hE := (norm_nonneg _).trans he
  have ha₀ c : |a₀ c| ≤ A + E := by
    have hd : |a c - a₀ c| ≤ E := (PiLp.norm_apply_le (a - a₀) c).trans he
    have ht := abs_sub_le (a₀ c) (a c) 0
    rw [sub_zero, sub_zero, abs_sub_comm (a₀ c) (a c)] at ht
    linarith [ha c]
  have h₁ : ‖WithLp.toLp 2 (fun c => (a c - a₀ c) * v c)‖ ≤ V * E := by
    refine (relative_norm _ (a - a₀) V hV ?_).trans (mul_le_mul_of_nonneg_left he hV)
    intro c
    change |(a c - a₀ c) * v c| ≤ V * |a c - a₀ c|
    rw [abs_mul, mul_comm]
    exact mul_le_mul_of_nonneg_right (hv c) (abs_nonneg _)
  have h₂ : ‖WithLp.toLp 2 (fun c => a₀ c * (v c - v₀ c))‖ ≤ (A + E) * F := by
    refine (relative_norm _ (v - v₀) (A + E) (by linarith) ?_).trans
      (mul_le_mul_of_nonneg_left hf (by linarith))
    intro c
    change |a₀ c * (v c - v₀ c)| ≤ (A + E) * |v c - v₀ c|
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right (ha₀ c) (abs_nonneg _)
  have hm : ‖WithLp.toLp 2 (fun c => a c * v c) -
      WithLp.toLp 2 (fun c => a₀ c * v₀ c)‖ ≤ V * E + (A + E) * F := by
    rw [show WithLp.toLp 2 (fun c => a c * v c) - WithLp.toLp 2 (fun c => a₀ c * v₀ c) =
      WithLp.toLp 2 (fun c => (a c - a₀ c) * v c) +
      WithLp.toLp 2 (fun c => a₀ c * (v c - v₀ c)) by
        ext c
        change a c * v c - a₀ c * v₀ c = (a c - a₀ c) * v c + a₀ c * (v c - v₀ c)
        ring]
    exact (norm_add_le _ _).trans (add_le_add h₁ h₂)
  exact (norm_sub_le_norm_sub_add_norm_sub z (WithLp.toLp 2 (fun c => a c * v c)) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

example := relative_norm (EuclideanSpace.single 0 2 : EucSpace 4)
  (EuclideanSpace.single 0 1) 2 (by norm_num) (by intro c; fin_cases c <;> norm_num)
example := silu_step (d := 4) 0 0 0 0 0 0 (by norm_num) (by intro c; simp)
  (by simp) (by simp [silu]; rfl)
example := product_step (d := 4) 0 0 0 0 0 0 0 0 0 0 (by norm_num) (by norm_num)
  (by intro c; simp) (by intro c; simp) (by simp) (by simp)
  (by simp; rfl)

end Transformer.SequenceModel.Attention.Adaptive
