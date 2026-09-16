/-
# ReLU² Feed-Forward Network

Formalization of the `ReLU2_FFN` class from `reference/model.py`:

```python
class ReLU2_FFN(nn.Module):
    def forward(self, x):
        return self.w_out(F.relu(self.w_in(x)).square())
```

Mathematically:

  `FFN(x) = W_out · ReLU(W_in x)²`,

acting coordinate-wise on `ReLU(z)² = max(0, z)²`.

This is a piecewise-polynomial function of degree 2:
  - on `{z > 0}`: acts as `z²`,
  - on `{z ≤ 0}`: acts as `0`.

The FFN as a whole is therefore piecewise polynomial of degree ≤ 2 on the
polyhedral subdivision determined by the signs of the pre-activations.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Mathlib.Analysis.SpecificLimits.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- The scalar ReLU² activation:

  `relu2(z) = max(0, z)² = (relu z)²`. -/
noncomputable def relu2 (z : ℝ) : ℝ := (max 0 z)^2

/-- `relu2` is non-negative. -/
theorem relu2_nonneg (z : ℝ) : 0 ≤ relu2 z := by
  unfold relu2
  exact sq_nonneg _

/-- `relu2` on the positive ray. -/
theorem relu2_of_pos (z : ℝ) (hz : 0 < z) : relu2 z = z^2 := by
  unfold relu2
  simp [le_of_lt hz]

/-- `relu2` on the non-positive ray. -/
theorem relu2_of_nonpos (z : ℝ) (hz : z ≤ 0) : relu2 z = 0 := by
  unfold relu2
  simp [hz]

/-- Piecewise polynomial bound:  `relu2 z ≤ z²`. -/
theorem relu2_le_sq (z : ℝ) : relu2 z ≤ z^2 := by
  unfold relu2
  by_cases h : z ≤ 0
  · simp [h]; positivity
  · push Not at h
    simp [le_of_lt h]

/-- **Continuity of `relu2`.** -/
theorem continuous_relu2 : Continuous relu2 := by
  unfold relu2
  exact (continuous_const.max continuous_id).pow 2

/-- The component-wise ReLU² activation on `EucSpace d`. -/
noncomputable def relu2Vec {d : ℕ} (x : EucSpace d) : EucSpace d :=
  EuclideanSpace.equiv _ ℝ |>.symm (fun i =>
    relu2 ((EuclideanSpace.equiv _ ℝ x) i))

/-- The full `ReLU²` feed-forward:

  `FFN(x) = W_out · relu2Vec(W_in x)`. -/
noncomputable def relu2FFN
    {d d_ff : ℕ}
    (W_in  : EucSpace d    →L[ℝ] EucSpace d_ff)
    (W_out : EucSpace d_ff →L[ℝ] EucSpace d)
    (x : EucSpace d) : EucSpace d :=
  W_out (relu2Vec (W_in x))

/-- **Non-negativity of activations.**  Every coordinate of `relu2Vec(z)`
is non-negative. -/
theorem relu2Vec_coord_nonneg
    {d : ℕ} (x : EucSpace d) (i : Fin d) :
    0 ≤ (EuclideanSpace.equiv _ ℝ (relu2Vec x)) i := by
  unfold relu2Vec
  simp
  exact relu2_nonneg _

/-- Coordinates of `relu2Vec` are the coordinates of `relu2`. -/
theorem relu2Vec_apply {d : ℕ} (x : EucSpace d) (i : Fin d) :
    relu2Vec x i = relu2 (x i) := rfl

/-- **Bounded growth.**

  `‖relu2Vec z‖ ≤ ‖z‖²`  (componentwise `relu2 ≤ sq`). -/
theorem relu2Vec_norm_bound
    {d : ℕ} (x : EucSpace d) :
    ‖relu2Vec x‖ ≤ ‖x‖^2 := by
  have hcoord : ∀ i : Fin d, ‖relu2Vec x i‖ ≤ ‖x i‖ ^ 2 := by
    intro i
    rw [relu2Vec_apply, Real.norm_eq_abs, abs_of_nonneg (relu2_nonneg _)]
    calc relu2 (x i) ≤ (x i) ^ 2 := relu2_le_sq _
      _ = ‖x i‖ ^ 2 := by rw [Real.norm_eq_abs, sq_abs]
  have hx2 : ‖x‖ ^ 2 = ∑ i, ‖x i‖ ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  rw [EuclideanSpace.norm_eq, hx2,
    show (∑ i, ‖x i‖ ^ 2) = Real.sqrt ((∑ i, ‖x i‖ ^ 2) ^ 2) from
      (Real.sqrt_sq (by positivity)).symm]
  refine Real.sqrt_le_sqrt ?_
  calc ∑ i, ‖relu2Vec x i‖ ^ 2 ≤ ∑ i, (‖x i‖ ^ 2) ^ 2 := by gcongr with i _; exact hcoord i
    _ ≤ (∑ i, ‖x i‖ ^ 2) ^ 2 :=
        Finset.sum_sq_le_sq_sum_of_nonneg fun i _ => by positivity

/-- A coordinatewise bound on `EucSpace d` upgrades to a bound on the norms. -/
theorem euclidean_norm_le_of_coord_le {d : ℕ} (C : ℝ) (hC : 0 ≤ C)
    (w z : EucSpace d) (h : ∀ i, ‖w i‖ ≤ C * ‖z i‖) : ‖w‖ ≤ C * ‖z‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq,
    show C * Real.sqrt (∑ i, ‖z i‖ ^ 2) = Real.sqrt (C ^ 2 * ∑ i, ‖z i‖ ^ 2) by
      rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hC]]
  refine Real.sqrt_le_sqrt ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  calc ‖w i‖ ^ 2 ≤ (C * ‖z i‖) ^ 2 := by gcongr; exact h i
    _ = C ^ 2 * ‖z i‖ ^ 2 := by ring

/-- `relu2` is Lipschitz with constant `|a| + |b|` between `a` and `b`. -/
theorem abs_relu2_sub_le (a b : ℝ) : |relu2 a - relu2 b| ≤ (|a| + |b|) * |a - b| := by
  have hfactor : relu2 a - relu2 b = (max 0 a - max 0 b) * (max 0 a + max 0 b) := by
    simp only [relu2]; ring
  have hlip : |max 0 a - max 0 b| ≤ |a - b| := by
    simpa [max_comm (0 : ℝ)] using abs_max_sub_max_le_abs a b (0 : ℝ)
  have hbound : |max 0 a + max 0 b| ≤ |a| + |b| := by
    rw [abs_of_nonneg (by positivity)]
    gcongr
    · exact max_le (abs_nonneg a) (le_abs_self a)
    · exact max_le (abs_nonneg b) (le_abs_self b)
  rw [hfactor, abs_mul, mul_comm (|a| + |b|)]
  exact mul_le_mul hlip hbound (abs_nonneg _) (abs_nonneg _)

/-- **Local Lipschitz bound for `relu2Vec`.** -/
theorem relu2Vec_lipschitz {d : ℕ} (u v : EucSpace d) :
    ‖relu2Vec u - relu2Vec v‖ ≤ (‖u‖ + ‖v‖) * ‖u - v‖ := by
  refine euclidean_norm_le_of_coord_le _ (by positivity) _ _ fun i => ?_
  have hu : |u i| ≤ ‖u‖ := by simpa using PiLp.norm_apply_le u i
  have hv : |v i| ≤ ‖v‖ := by simpa using PiLp.norm_apply_le v i
  have hcoord : (relu2Vec u - relu2Vec v) i = relu2 (u i) - relu2 (v i) := rfl
  rw [hcoord, Real.norm_eq_abs]
  refine (abs_relu2_sub_le (u i) (v i)).trans ?_
  have hsub : ‖(u - v) i‖ = |u i - v i| := by
    rw [show (u - v) i = u i - v i from rfl, Real.norm_eq_abs]
  rw [hsub]
  exact mul_le_mul_of_nonneg_right (by linarith) (abs_nonneg _)

/-- **Lipschitz constant of `relu2FFN` on a bounded ball.**

For `‖x‖ ≤ R`, `relu2FFN` has Lipschitz constant
`‖W_out‖_op · 2R · ‖W_in‖_op²`. -/
theorem relu2FFN_lipschitz_on_ball
    {d d_ff : ℕ} (R : ℝ) (hR : 0 < R)
    (W_in  : EucSpace d    →L[ℝ] EucSpace d_ff)
    (W_out : EucSpace d_ff →L[ℝ] EucSpace d) :
    ∀ x y : EucSpace d, ‖x‖ ≤ R → ‖y‖ ≤ R →
      ‖relu2FFN W_in W_out x - relu2FFN W_in W_out y‖
        ≤ ‖W_out‖ * (2 * R) * ‖W_in‖^2 * ‖x - y‖ := by
  intro x y hx hy
  have hin : ∀ z : EucSpace d, ‖z‖ ≤ R → ‖W_in z‖ ≤ ‖W_in‖ * R := fun z hz =>
    (W_in.le_opNorm z).trans (by gcongr)
  have hsum : ‖W_in x‖ + ‖W_in y‖ ≤ 2 * R * ‖W_in‖ := by
    have := hin x hx
    have := hin y hy
    nlinarith
  have hdiff : ‖W_in x - W_in y‖ ≤ ‖W_in‖ * ‖x - y‖ := by
    rw [← map_sub]; exact W_in.le_opNorm _
  have hmain : ‖relu2Vec (W_in x) - relu2Vec (W_in y)‖
      ≤ (2 * R * ‖W_in‖) * (‖W_in‖ * ‖x - y‖) :=
    (relu2Vec_lipschitz _ _).trans
      (mul_le_mul hsum hdiff (norm_nonneg _) (by positivity))
  calc ‖relu2FFN W_in W_out x - relu2FFN W_in W_out y‖
      = ‖W_out (relu2Vec (W_in x) - relu2Vec (W_in y))‖ := by
        rw [relu2FFN, relu2FFN, ← map_sub]
    _ ≤ ‖W_out‖ * ‖relu2Vec (W_in x) - relu2Vec (W_in y)‖ := W_out.le_opNorm _
    _ ≤ ‖W_out‖ * ((2 * R * ‖W_in‖) * (‖W_in‖ * ‖x - y‖)) := by
        gcongr
    _ = ‖W_out‖ * (2 * R) * ‖W_in‖^2 * ‖x - y‖ := by ring

end GPTMini
end Transformer
