module

public import Transformer.SequenceModel.Attention.Stability.Vector
public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Source SwiGLU sensitivity on the normalized input ball

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B2;
diff_gpt `b98c665b`, `feed_forward.py` and `block.py`. The gate and value
matrices are independent and the activation is the actual SiLU product.
Constants are explicit sufficient bounds, not sharp estimates.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini Landscape.Neural

/-- B2: sigmoid is globally 1-Lipschitz, by its actual derivative. -/
theorem sigmoid_error (x y : ℝ) : |Real.sigmoid x - Real.sigmoid y| ≤ |x - y| := by
  have h : LipschitzWith 1 Real.sigmoid :=
    lipschitzWith_of_nnnorm_deriv_le differentiable_sigmoid (fun t => by
      change ‖deriv Real.sigmoid t‖ ≤ (1 : ℝ)
      rw [Real.deriv_sigmoid, Real.norm_eq_abs,
        abs_of_nonneg (mul_nonneg (Real.sigmoid_nonneg t) (sub_nonneg.mpr (Real.sigmoid_le_one t)))]
      nlinarith [Real.sigmoid_nonneg t, Real.sigmoid_le_one t, sq_nonneg (Real.sigmoid t)])
  simpa only [Real.dist_eq, NNReal.coe_one, one_mul] using h.dist_le_mul x y

/-- B2: the actual SiLU gate never exceeds its input magnitude. -/
theorem silu_abs_le (x : ℝ) : |silu x| ≤ |x| := by
  rw [silu, div_eq_mul_inv, ← Real.sigmoid_def, abs_mul, abs_of_nonneg (Real.sigmoid_nonneg x)]
  simpa only [mul_one] using mul_le_mul_of_nonneg_left (Real.sigmoid_le_one x) (abs_nonneg x)

/-- B2: on a bounded gate input, SiLU sensitivity follows from sigmoid,
with no linearization or activation approximation. -/
theorem silu_error (x y R : ℝ) (hy : |y| ≤ R) :
    |silu x - silu y| ≤ (1 + R) * |x - y| := by
  have he : silu x - silu y =
      (x - y) * Real.sigmoid x + y * (Real.sigmoid x - Real.sigmoid y) := by
    simp only [silu, Real.sigmoid_def, div_eq_mul_inv]
    ring
  rw [he]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_mul, abs_of_nonneg (Real.sigmoid_nonneg x)]
  have h1 := mul_le_mul_of_nonneg_left (Real.sigmoid_le_one x) (abs_nonneg (x - y))
  have h2 := mul_le_mul hy (sigmoid_error x y) (abs_nonneg _) ((abs_nonneg y).trans hy)
  nlinarith

/-- B2: a scalar SwiGLU product retains errors from both independent inputs. -/
theorem swiglu_error (g v g' v' Rg Rv eg ev : ℝ)
    (hg : |g'| ≤ Rg) (hv : |v| ≤ Rv)
    (hge : |g - g'| ≤ eg) (hve : |v - v'| ≤ ev) :
    |swiglu g v - swiglu g' v'| ≤ Rv * (1 + Rg) * eg + Rg * ev := by
  have hRg := (abs_nonneg _).trans hg
  have hRv := (abs_nonneg _).trans hv
  have he : swiglu g v - swiglu g' v' = (silu g - silu g') * v + silu g' * (v - v') := by
    unfold swiglu
    ring
  have hs : |silu g - silu g'| ≤ (1 + Rg) * eg :=
    (silu_error g g' Rg hg).trans (mul_le_mul_of_nonneg_left hge (by linarith))
  have hb : |silu g'| ≤ Rg := (silu_abs_le g').trans hg
  rw [he]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_mul]
  have h1 := mul_le_mul hs hv (abs_nonneg _) ((abs_nonneg _).trans hs)
  have h2 := mul_le_mul hb hve (abs_nonneg _) hRg
  nlinarith

/-- Explicit bound for the real SwiGLU map on the radius-sqrt(W) ball.
The sqrt(FFN width) factor comes from coordinatewise Euclidean aggregation. -/
def ffnLip (cfg : Config) (B : BlockParams cfg) : ℝ :=
  ‖B.output‖ * Real.sqrt (cfg.d_ff : ℝ) *
    (‖B.value‖ * Real.sqrt (cfg.d_model : ℝ) *
      (1 + ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ)) * ‖B.gate‖ +
      ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ) * ‖B.value‖)

/-- B2: the actual source FFN is Lipschitz on its pre-normalized input
ball, with an explicit constant depending on all three learned matrices. -/
theorem feedForward_error (cfg : Config) (B : BlockParams cfg)
    (x y : EucSpace cfg.d_model)
    (hx : ‖x‖ ≤ Real.sqrt (cfg.d_model : ℝ)) (hy : ‖y‖ ≤ Real.sqrt (cfg.d_model : ℝ)) :
    ‖feedForward cfg B x - feedForward cfg B y‖ ≤ ffnLip cfg B * ‖x - y‖ := by
  have hcoord (A : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff)
      (z : EucSpace cfg.d_model) (c : Fin cfg.d_ff) : |A z c| ≤ ‖A‖ * ‖z‖ :=
    (PiLp.norm_apply_le (A z) c).trans (A.le_opNorm z)
  have hdiff (A : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff) (c : Fin cfg.d_ff) :
      |A x c - A y c| ≤ ‖A‖ * ‖x - y‖ := by
    simpa only [map_sub, PiLp.sub_apply] using hcoord A (x - y) c
  let C := ‖B.value‖ * Real.sqrt (cfg.d_model : ℝ) *
    (1 + ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ)) * ‖B.gate‖ +
      ‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ) * ‖B.value‖
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hp c := swiglu_error (B.gate x c) (B.value x c) (B.gate y c) (B.value y c)
    (‖B.gate‖ * Real.sqrt (cfg.d_model : ℝ)) (‖B.value‖ * Real.sqrt (cfg.d_model : ℝ))
    (‖B.gate‖ * ‖x - y‖) (‖B.value‖ * ‖x - y‖)
    ((hcoord B.gate y c).trans (mul_le_mul_of_nonneg_left hy (norm_nonneg _)))
    ((hcoord B.value x c).trans (mul_le_mul_of_nonneg_left hx (norm_nonneg _)))
    (hdiff B.gate c) (hdiff B.value c)
  have hv : ‖WithLp.toLp 2 (fun c => swiglu (B.gate x c) (B.value x c)) -
      WithLp.toLp 2 (fun c => swiglu (B.gate y c) (B.value y c))‖ ≤
      Real.sqrt (cfg.d_ff : ℝ) * (C * ‖x - y‖) := by
    apply norm_le_coordinates _ _ (mul_nonneg hC (norm_nonneg _))
    intro c
    exact (hp c).trans_eq (by dsimp [C]; ring)
  unfold feedForward
  rw [← map_sub]
  refine (B.output.le_opNorm _).trans ?_
  exact (mul_le_mul_of_nonneg_left hv (norm_nonneg _)).trans_eq (by unfold ffnLip; dsimp [C]; ring)

example := silu_error 2 3 3 (by norm_num)
example := swiglu_error 2 3 4 5 4 3 2 2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example (B : BlockParams witnessConfig) := feedForward_error witnessConfig B
  (WithLp.toLp 2 ![1, 0, 0, 0]) (WithLp.toLp 2 ![0, 1, 0, 0])
  (by norm_num [witnessConfig, EuclideanSpace.norm_eq, Fin.sum_univ_succ])
  (by norm_num [witnessConfig, EuclideanSpace.norm_eq, Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Stability
