/-
# The emergence of clusters in self-attention dynamics — the case `V = I_d`

§3 of arXiv:2305.05465v6 continued: the rescaled dynamics at `V = I_d`, which
is `e:dynIdzi`, and the clustering theorem `t:Idcase11int` stated on it.

**What the source says and what is carried here.**

* At `V = I_d` the flow is the scalar `e^t` (`expTime_one_apply`), so the
  exponent of `e:Rres` becomes `e^{2t}⟨Az_i,Az_j⟩` for the square root
  `A = (QᵀK)^{1/2}` of `eq: A`, and the drift becomes `z_j - z_i`.  That is
  `e:dynIdzi`, and `rescaledDynamics_one_iff` proves it is the same equation
  as `e:Rres` at `V = 1` — the source introduces it as a "shorthand
  notation", and this is what makes that literally true.

* `e:dynIdzi` is displayed in §8, where `t:Idcase11` is proved; it is placed
  here because it is `e:Rres` at `V = I_d` and nothing more.

* "There exists a convex polytope `K ⊂ ℝ^d`" is `convexHull ℝ ↑S` for a
  `Finset`: a convex polytope is the convex hull of finitely many points, and
  Mathlib has no separate type for one.

* "`z_i(t)` converges either to `0` or to some point on `∂K`" is the
  disjunction below; the source's own summary sentence is the weaker form of
  `t:Idcase11`, whose full statement (the limit set `𝒮`, the vertices `𝒱`)
  is carried with §8.

Source: arXiv:2305.05465v6, `e:dynIdzi`, `eq: A`, `t:Idcase11int`.
-/

import Transformer.Clusters.Section3_Rescaled
import Mathlib.Analysis.Convex.Combination

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### `e:dynIdzi` -/

/-- The self-attention matrix of `e:dynIdzi`: the softmax of the scores
`c⟨Az_i, Az_l⟩`, where `c = e^{2t}` is the time factor left over by the
rescaling.

Source: arXiv:2305.05465v6, `e:dynIdzi`. -/
noncomputable def scaledAttention (A : ParamMatrix d) (c : ℝ) (Z : Idx n → EucSpace d)
    (i j : Idx n) : ℝ :=
  Perspective.softmaxWeight (fun l : Idx n => c * inner (𝕜 := ℝ) (A (Z i)) (A (Z l))) j

/-- **Equation (e:dynIdzi).**  The rescaled dynamics at `V = I_d`:

  `ż_i(t) = Σ_j ( e^{e^{2t}⟨Az_i,Az_j⟩} / Σ_k e^{e^{2t}⟨Az_i,Az_k⟩} ) (z_j(t) - z_i(t))`.

Source: arXiv:2305.05465v6, `e:dynIdzi`. -/
def IdRescaledDynamics (A : ParamMatrix d) (Z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    HasDerivAt (fun s => Z s i)
      (∑ j : Idx n, scaledAttention A (Real.exp (2 * t)) (Z t) i j • (Z t j - Z t i)) t

/-- The attention coefficients of `e:Rres` at `V = I_d` are those of
`e:dynIdzi`: the flow contributes a factor `e^t` to each of the two slots of
`⟨Q·, K·⟩`, and `eq: A` replaces `⟨Qu, Kv⟩` by `⟨Au, Av⟩`. -/
theorem attentionMatrix_expTime_one {Q K A : ParamMatrix d} (h : IsAttentionRoot Q K A)
    (t : ℝ) (Z : Idx n → EucSpace d) (i j : Idx n) :
    attentionMatrix Q K (fun l => expTime (1 : ParamMatrix d) t (Z l)) i j
      = scaledAttention A (Real.exp (2 * t)) Z i j := by
  unfold attentionMatrix scaledAttention
  congr 1
  funext l
  simp only [expTime_one_apply, map_smul, real_inner_smul_left, real_inner_smul_right, h.2.2]
  rw [← mul_assoc, ← Real.exp_add, two_mul]

/-- **`e:dynIdzi` is `e:Rres` at `V = I_d`.**  Under `eq: A`, the two
equations have the same solutions. -/
theorem rescaledDynamics_one_iff {Q K A : ParamMatrix d} (h : IsAttentionRoot Q K A)
    (Z : ℝ → Idx n → EucSpace d) :
    RescaledDynamics Q K 1 Z ↔ IdRescaledDynamics A Z := by
  have key : ∀ (t : ℝ) (i : Idx n),
      (∑ j : Idx n, attentionMatrix Q K (fun l => expTime (1 : ParamMatrix d) t (Z t l)) i j •
          (1 : ParamMatrix d) (Z t j - Z t i))
        = ∑ j : Idx n, scaledAttention A (Real.exp (2 * t)) (Z t) i j • (Z t j - Z t i) := by
    intro t i
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [attentionMatrix_expTime_one h]
    rfl
  constructor
  · intro H t i
    rw [← key t i]
    exact H t i
  · intro H t i
    rw [key t i]
    exact H t i

/-- **A configuration of equal tokens is stationary.**  At `V = I_d` the drift
is `Σ_j P_ij (z_j - z_i)`, which vanishes when all the tokens coincide; this
is the solution in closed form that witnesses the hypotheses below. -/
theorem rescaledDynamics_one_const (Q K : ParamMatrix d) (z : EucSpace d) :
    RescaledDynamics (n := n) Q K 1 (fun _ _ => z) := by
  intro t i
  simpa using hasDerivAt_const t z

/-! ### `t:Idcase11int` -/

/-- **Theorem (t:Idcase11int).**  Let `V = I_d` and `QᵀK ≻ 0`.  Then for any
initial sequence of tokens there is a convex polytope `K ⊂ ℝ^d` such that each
`z_i(t)` converges, as `t → +∞`, either to `0` or to a point of `∂K`.

Not proved here.

Source: arXiv:2305.05465v6, `t:Idcase11int`. -/
theorem idCase_tendsto_zero_or_frontier (Q K : ParamMatrix d) (hQK : IsPosDefQK Q K)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K 1 Z) :
    ∃ S : Finset (EucSpace d), ∀ i : Idx n,
      Filter.Tendsto (fun t => Z t i) Filter.atTop (nhds 0) ∨
        ∃ p ∈ frontier (convexHull ℝ (S : Set (EucSpace d))),
          Filter.Tendsto (fun t => Z t i) Filter.atTop (nhds p) := by
  sorry

/-- The hypotheses of `idCase_tendsto_zero_or_frontier` are satisfiable:
`Q = K = I_d` is positive definite, and the configuration in which all `n`
tokens sit at one point solves `e:Rres` at `V = I_d`. -/
example (z : EucSpace d) :
    IsPosDefQK (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      RescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) 1 (fun _ _ => z) :=
  ⟨isPosDefQK_of_isAttentionRoot (isAttentionRoot_id d), rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer
