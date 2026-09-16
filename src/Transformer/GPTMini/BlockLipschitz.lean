/-
# How far apart one sub-layer sends two residual streams

`Block.attnSubLayer_bounded` and `Block.ffnSubLayer_bounded` say how *large* a
sub-layer output is; the Lipschitz estimates say how much it *moves* when the
stream moves.  The two are independent: a bounded map need not be Lipschitz,
and the constants here are not the constants there.

The FFN sub-layer is the easy half, because it acts position by position:

  `ffnSubLayer(x) i = relu2FFN(W_in, W_out, rmsNormEps eps (x i))`,

so its constant is the product of

  - `rmsNormEps_lipschitz`, a factor `2 / √eps`,
  - `relu2FFN_lipschitz_on_ball` on the ball of radius `√d_model`, which is
    where `rmsNormEps` lands, a factor `‖W_out‖ · 2√d_model · ‖W_in‖²`.

The attention sub-layer mixes positions and is treated separately.
-/

import Transformer.GPTMini.Block

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- **The FFN sub-layer is Lipschitz**, with constant

  `4 ‖W_out‖ ‖W_in‖² √d_model / √eps`.

Both factors are sharp in their own right: the `√d_model` is the radius of the
ball `rmsNormEps` maps into, where the quadratic `relu2` has slope `2√d_model`,
and the `1/√eps` is the steepness of the normalization itself.  Source:
`reference/model.py` (`Block.forward`, second sub-layer), through
`GPTMini.rmsNormEps_lipschitz` and `GPTMini.relu2FFN_lipschitz_on_ball`. -/
theorem ffnSubLayer_lipschitz
    (cfg : Config) (params : FFNParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (x y : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖ffnSubLayer cfg params eps x i - ffnSubLayer cfg params eps y i‖
      ≤ 4 * ‖params.W_out‖ * ‖params.W_in‖ ^ 2
          * Real.sqrt (cfg.d_model : ℝ) / Real.sqrt eps * ‖x i - y i‖ := by
  have hd : (0 : ℝ) < Real.sqrt (cfg.d_model : ℝ) :=
    Real.sqrt_pos.mpr (Nat.cast_pos.mpr cfg.d_model_pos)
  have hx : ‖rmsNormEps eps (x i)‖ ≤ Real.sqrt (cfg.d_model : ℝ) :=
    rmsNormEps_norm_le eps heps cfg.d_model_pos _
  have hy : ‖rmsNormEps eps (y i)‖ ≤ Real.sqrt (cfg.d_model : ℝ) :=
    rmsNormEps_norm_le eps heps cfg.d_model_pos _
  calc ‖ffnSubLayer cfg params eps x i - ffnSubLayer cfg params eps y i‖
      = ‖relu2FFN params.W_in params.W_out (rmsNormEps eps (x i))
          - relu2FFN params.W_in params.W_out (rmsNormEps eps (y i))‖ := rfl
    _ ≤ ‖params.W_out‖ * (2 * Real.sqrt (cfg.d_model : ℝ)) * ‖params.W_in‖ ^ 2
          * ‖rmsNormEps eps (x i) - rmsNormEps eps (y i)‖ :=
        relu2FFN_lipschitz_on_ball _ hd params.W_in params.W_out _ _ hx hy
    _ ≤ ‖params.W_out‖ * (2 * Real.sqrt (cfg.d_model : ℝ)) * ‖params.W_in‖ ^ 2
          * (2 / Real.sqrt eps * ‖x i - y i‖) := by
        gcongr
        exact rmsNormEps_lipschitz eps heps cfg.d_model_pos (x i) (y i)
    _ = 4 * ‖params.W_out‖ * ‖params.W_in‖ ^ 2
          * Real.sqrt (cfg.d_model : ℝ) / Real.sqrt eps * ‖x i - y i‖ := by ring

/-- The hypothesis is satisfiable: the `eps = 10⁻⁶` of `reference/model.py`. -/
example (params : FFNParams Config.default)
    (x y : Fin 1 → EucSpace Config.default.d_model) :
    ‖ffnSubLayer Config.default params 1e-6 x 0
        - ffnSubLayer Config.default params 1e-6 y 0‖
      ≤ 4 * ‖params.W_out‖ * ‖params.W_in‖ ^ 2
          * Real.sqrt (Config.default.d_model : ℝ) / Real.sqrt 1e-6 * ‖x 0 - y 0‖ :=
  ffnSubLayer_lipschitz Config.default params 1e-6 (by norm_num) x y 0

end GPTMini
end Transformer
