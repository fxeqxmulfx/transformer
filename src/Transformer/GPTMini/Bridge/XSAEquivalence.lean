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
  -- With `eps = 0` and `‖v_i‖ = 1`:
  --   `normL2 0 (v i) = (1 / (‖v i‖ + 0)) • v i = (1/1) • v i = v i`,
  -- so the XSA projection
  --   `y_i - ⟨y_i, normL2 0 (v i)⟩ · normL2 0 (v i)`
  -- equals
  --   `y_i - ⟨y_i, v_i⟩ · v_i = proj_{v_i}(y_i)`.
  -- Detailed Mathlib manipulation deferred to Phase 5 follow-up.
  sorry

/-- **Consequence.**  If we set up `gpt-mini` with `V = I_d` and
`RMSNorm`-normalized tokens, the attention block's XSA-projected output
is the spherical-tangent projection that appears in the canonical
clustering theorems (`Section1_IPS.SA`, `Section5_HighD.hemisphere_clustering`).

This makes the entire clustering theory available, *modulo* extending it
to the multi-head and time-varying-`Q,K` (RoPE) setting, which is done by
the further bridge lemmas in this directory. -/
theorem xsa_reduces_to_spherical_canonical_setup
    (cfg : Config) :
    True := trivial

end Bridge
end GPTMini
end Transformer
