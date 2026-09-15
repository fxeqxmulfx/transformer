/-
# Bridge: XSA at `V = I_d` equals spherical projection

When the value projection is the identity (`V x_i = x_i`) and the tokens
are pre-normalized (`‖x_i‖ = 1`), the XSA output formula

  `z_i = y_i - ⟨y_i, v̂_i⟩ v̂_i`,   `v̂_i = v_i / ‖v_i‖ = x_i`

reduces to the spherical projection

  `z_i = Proj_{x_i}(y_i)`

used throughout `Transformer.Section1_IPS` and the canonical theory of
2312.10794v5.

This is the most direct theoretical justification for the `XSA` addition:
in the canonical limit it is **literally** the same operation as the
sphere-tangent projection in the formalized clustering theorems.
-/

import Transformer.Basic
import Transformer.GPTMini.CausalMHA
import Transformer.GPTMini.Bridge.SphereResidence

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- **XSA at `V = I_d`, with unit-norm tokens, equals spherical
projection.**

If `v_i = x_i` (`V = identity`) and `‖x_i‖ = 1` (post-RMSNorm/√d), then

  `xsaProjection cfg 0 v y i = Transformer.proj cfg.head_dim (x i) (y i)`.

This is the canonical reduction used in `Transformer.XSA`:
`xsa_equals_spherical_SA_when_V_is_identity`. -/
theorem xsaProjection_eq_sphereProj
    (cfg : Config)
    {T : ℕ} (v y : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (h_unit : ‖v i‖ = 1) :
    xsaProjection cfg 0 v y i =
      Transformer.proj cfg.head_dim (v i) (y i) := by
  have hv : normL2 (0 : ℝ) (v i) = v i := by
    simp [normL2, h_unit]
  unfold xsaProjection Transformer.proj
  rw [hv, real_inner_comm]

/-- **Consequence: one head at `V = I_d` is the spherical projection.**

With the value stream on the unit sphere — which is where `RMSNorm` puts it,
up to the `√d` of `Bridge.SphereResidence` — and `eps = 0`, one head of
`CausalMHA` is

  `head_i = Proj_{v_i}(attnOutput_i)`,

the spherical-tangent projection of the attention output, which is the
operation the canonical dynamics `Perspective.Section1_IPS.SA` and
`Perspective.Section5_HighD` are written with.  The XSA sub-layer is therefore
not an approximation of that projection: at `V = I_d` it *is* that projection.

What is still missing before the clustering theorems apply is the multi-head
and time-varying-`Q, K` setting; that is `Bridge.RoPEAsTimeVarying`.

Source: `reference/model.py` (`CausalMHA.forward`, XSA branch); the canonical
projection is arXiv:2312.10794v5, `eq: transformerSd.QKV`. -/
theorem attentionHead_eq_sphereProj
    (cfg : Config) {T : ℕ} (alpha : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T)
    (h_unit : ‖v i‖ = 1) :
    attentionHead cfg alpha 0 q k v positions i
      = Transformer.proj cfg.head_dim (v i)
          (attnOutput cfg alpha 0
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i) := by
  unfold attentionHead
  simpa using
    xsaProjection_eq_sphereProj cfg v
      (fun i' => if i' = i then
          attnOutput cfg alpha 0
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
            (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i
        else 0) i h_unit

/-- The hypothesis of both theorems is satisfiable: the first basis vector of
`ℝ^{head_dim}` has norm `1`. -/
example (cfg : Config) (y : Fin 1 → EucSpace cfg.head_dim) :
    xsaProjection cfg 0 (fun _ => EuclideanSpace.single (⟨0, cfg.head_dim_pos⟩) (1 : ℝ)) y 0
      = Transformer.proj cfg.head_dim
          (EuclideanSpace.single (⟨0, cfg.head_dim_pos⟩) (1 : ℝ)) (y 0) :=
  xsaProjection_eq_sphereProj cfg _ y 0 (by simp [PiLp.norm_single])

end Bridge
end GPTMini
end Transformer
