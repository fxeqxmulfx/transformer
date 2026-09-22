/-
# Bridge: RoPE is a relative, pair-dependent key matrix — not a depth-varying one

RoPE rotates the query of the token at position `p_i` by `R(p_i)` and the key
of the token at `p_j` by `R(p_j)`, both orthogonal.  This file bundles the
rotation and reads off what that does to the score:

  - `ropeIsometry` bundles `GPTMini.applyRope` as a linear isometry — the
    rotation is linear, and `applyRope_isometry` says it preserves the norm;
  - `ropeRotated Q θ p = R(p) ∘ Q` is the family of rotated projections,
    indexed by the *position* `p`, and its operator norm does not depend on
    `p` (`ropeRotated_norm`);
  - `rope_score_relative`: `⟨R(p_i) Q x, R(p_j) K y⟩ = ⟨Q x, R(p_j - p_i) K y⟩`,
    so the pair `(i, j)` sees the key matrix `R(p_j - p_i) K`.

What RoPE is *not* is an instance of the time-varying `Q(t), K(t)` of
`Perspective.transformerODE` (`eq: transformerSd.QKV`).  There `t` is the
depth, one clock shared by every token; here `p` is a token's own position,
different for each token and fixed in depth.  The honest reading of RoPE
attention is a depth-constant key matrix that depends on the *pair* of tokens,
`K_{ij} = R(p_j - p_i) K`, which none of the survey's dynamics allow.  The
file used to name `ropeRotated` a `TimeParam` and claim the RoPE head is
`transformerODE` at `Q(t) = R(t) Q`, `K(t) = R(t) K`; that identification
conflates the two clocks and is withdrawn.
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

/-- **The RoPE-rotated projections, indexed by position.**

Given a constant matrix `Q : ParamMatrix d`, `p ↦ R(p) ∘ Q` is the projection
a token at position `p` is read through.  The index is the token's position,
not the depth: see the module header for why this family is not the
`Q(t)` of `Perspective.transformerODE`.

Source: `reference/model.py` (`apply_rope`); Su et al., *RoFormer*, eq. (14). -/
noncomputable def ropeRotated
    {d_head : ℕ} (Q : ParamMatrix d_head) (theta : ℝ) :
    ℝ → ParamMatrix d_head :=
  fun t => (ropeIsometry d_head theta t).toContinuousLinearMap.comp Q

/-- **Isometry preservation.**

For any constant `Q` and position `t`, the operator norm of `R(t) ∘ Q` equals
that of `Q`, because `R(t)` is an isometry: the rotated projections are
bounded uniformly in the position.

Source: `reference/model.py` (`apply_rope`); norm preservation is
`GPTMini.applyRope_isometry`. -/
theorem ropeRotated_norm
    {d_head : ℕ} (Q : ParamMatrix d_head) (theta : ℝ) (t : ℝ) :
    ‖ropeRotated Q theta t‖ = ‖Q‖ :=
  LinearIsometry.norm_toContinuousLinearMap_comp (ropeIsometry d_head theta t)

/-- **The RoPE attention score is relative.**

  `⟨R(t) Q x, R(s) K y⟩ = ⟨Q x, R(s - t) K y⟩`:

the score of a token at position `t` against a token at position `s` depends
on `s - t` and not on `t` and `s` separately: the pair sees the key matrix
`R(s - t) K`.  It is `GPTMini.applyRope_relative` read through `ropeRotated`.  Source: `reference/model.py` (`apply_rope`);
Su et al., *RoFormer*, eq. (14). -/
theorem rope_score_relative
    {d_head : ℕ} (Q K : ParamMatrix d_head) (theta : ℝ) (t s : ℝ)
    (x y : EucSpace d_head) :
    inner (𝕜 := ℝ) (ropeRotated Q theta t x) (ropeRotated K theta s y)
      = inner (𝕜 := ℝ) (Q x) (applyRope d_head theta (s - t) (K y)) :=
  applyRope_relative d_head theta t s (Q x) (K y)

end Bridge
end GPTMini
end Transformer
