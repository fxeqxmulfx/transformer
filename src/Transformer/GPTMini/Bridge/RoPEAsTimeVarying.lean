/-
# Bridge: RoPE = time-varying `Q, K` in `Transformer.Perspective.Section1_IPS`

The Lean type `Transformer.Basic.TimeParam d := ℝ → ParamMatrix d` already
allows time-varying parameter matrices.  RoPE, which rotates `Q` and `K` by a
position-dependent orthogonal matrix `R(t)`, is a *specific* instance of this
generality, and this file exhibits it as one:

  - `ropeIsometry` bundles `GPTMini.applyRope` as a linear isometry — the
    rotation is linear, and `applyRope_isometry` says it preserves the norm;
  - `rope_timeParam Q θ t = R(t) ∘ Q` is then a genuine `TimeParam`, whose
    operator norm does not depend on the position (`rope_timeParam_norm_preserved`);
  - `rope_score_relative` is what makes this parametrization worth the trouble:
    the attention score of two tokens is a function of their *relative*
    position only.

What this file proves is the parametrization, not the dynamics.  The file used
to carry `rope_clustering` as well, the claim that the resulting RoPE-attention
dynamics send *every* initial sequence to a common point; that claim is false,
and `Bridge.RoPENoClustering` refutes it.
-/

import Transformer.Basic
import Transformer.GPTMini.RoPE
import Transformer.GPTMini.CausalMHA
import Transformer.Perspective.Section1_IPS
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Analysis.Normed.Operator.NormedSpace

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- The RoPE rotation is additive: it acts coordinatewise by a matrix. -/
theorem applyRope_add (d_head : ℕ) (theta t : ℝ) (x y : EucSpace d_head) :
    applyRope d_head theta t (x + y)
      = applyRope d_head theta t x + applyRope d_head theta t y := by
  ext i
  rcases hi : ropeSplit d_head i with (k | k) | j <;>
    simp [applyRope, ropeCoord, hi] <;> ring

/-- The RoPE rotation is homogeneous. -/
theorem applyRope_smul (d_head : ℕ) (theta t : ℝ) (c : ℝ) (x : EucSpace d_head) :
    applyRope d_head theta t (c • x) = c • applyRope d_head theta t x := by
  ext i
  rcases hi : ropeSplit d_head i with (k | k) | j <;>
    simp [applyRope, ropeCoord, hi] <;> ring

/-- **RoPE as a linear isometry `R(t)`.**

The rotation of `reference/model.py` (`apply_rope`) bundled as a linear
isometry of `ℝ^{d_head}`: linear by `applyRope_add`/`applyRope_smul`,
norm-preserving by `GPTMini.applyRope_isometry`. -/
noncomputable def ropeIsometry (d_head : ℕ) (theta t : ℝ) :
    EucSpace d_head →ₗᵢ[ℝ] EucSpace d_head where
  toFun := applyRope d_head theta t
  map_add' := applyRope_add d_head theta t
  map_smul' := applyRope_smul d_head theta t
  norm_map' := applyRope_isometry d_head theta t

@[simp] theorem ropeIsometry_apply (d_head : ℕ) (theta t : ℝ) (x : EucSpace d_head) :
    ropeIsometry d_head theta t x = applyRope d_head theta t x := rfl

/-- **RoPE as a `TimeParam`.**

Given a constant matrix `Q : ParamMatrix d`, the position-dependent
"RoPE-rotated" Q-projection `t ↦ R(t) ∘ Q` is a `TimeParam d`, so that
RoPE attention is the `Perspective.transformerODE` of `eq: transformerSd.QKV`
at those particular `Q(t), K(t)`. -/
noncomputable def rope_timeParam
    {d_head : ℕ} (Q : ParamMatrix d_head) (theta : ℝ) :
    Transformer.TimeParam d_head :=
  fun t => (ropeIsometry d_head theta t).toContinuousLinearMap.comp Q

/-- **Isometry preservation.**

For any constant `Q` and position `t`, the operator norm of `R(t) ∘ Q` equals
that of `Q`, because `R(t)` is an isometry.  This is what keeps a RoPE
`TimeParam` bounded uniformly in the position — the hypothesis under which
the time-varying results of `Perspective.Section1_IPS` are stated. -/
theorem rope_timeParam_norm_preserved
    {d_head : ℕ} (Q : ParamMatrix d_head) (theta : ℝ) (t : ℝ) :
    ‖rope_timeParam Q theta t‖ = ‖Q‖ :=
  LinearIsometry.norm_toContinuousLinearMap_comp (ropeIsometry d_head theta t)

/-- **The RoPE attention score is relative.**

  `⟨R(t) Q x, R(s) K y⟩ = ⟨Q x, R(s - t) K y⟩`:

the score of a token at position `t` against a token at position `s` depends
on `s - t` and not on `t` and `s` separately.  This is the property that
`rope_timeParam` is built for, and it is `GPTMini.applyRope_relative` read
through the parametrization.  Source: `reference/model.py` (`apply_rope`);
Su et al., *RoFormer*, eq. (14). -/
theorem rope_score_relative
    {d_head : ℕ} (Q K : ParamMatrix d_head) (theta : ℝ) (t s : ℝ)
    (x y : EucSpace d_head) :
    inner (𝕜 := ℝ) (rope_timeParam Q theta t x) (rope_timeParam K theta s y)
      = inner (𝕜 := ℝ) (Q x) (applyRope d_head theta (s - t) (K y)) :=
  applyRope_relative d_head theta t s (Q x) (K y)

end Bridge
end GPTMini
end Transformer
