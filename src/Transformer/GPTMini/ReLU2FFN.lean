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

/-- **Bounded growth.**

  `‖relu2Vec z‖ ≤ ‖z‖²`  (componentwise `relu2 ≤ sq`). -/
theorem relu2Vec_norm_bound
    {d : ℕ} (x : EucSpace d) :
    ‖relu2Vec x‖ ≤ ‖x‖^2 := by
  sorry

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
  sorry

end GPTMini
end Transformer
