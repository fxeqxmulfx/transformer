module

public import Transformer.Clusters.Section3_Rescaled
public import Transformer.Clusters.Section8_AttentionRoot
public import Mathlib.Analysis.Convex.Combination

/-!
# The emergence of clusters in self-attention dynamics — the case `V = I_d`

§3 of arXiv:2305.05465v6 continued: the rescaled dynamics at `V = I_d`,
their exact identity-root form, and continuity of the actual solutions.

**What the source says and what is carried here.**

* At `V = I_d` the flow is the scalar `e^t` (`expTime_one_apply`), so the
  exponent of `e:Rres` becomes `e^{2t}⟨Az_i,Az_j⟩` for the square root
  `A = (QᵀK)^{1/2}` of `eq: A`, and the drift becomes `z_j - z_i`. That is
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
  is proved with §8 in `Section8_IdConvergence.lean`.

Source: arXiv:2305.05465v6, `e:dynIdzi`, `eq: A`, `t:Idcase11int`.
-/

@[expose] public section

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

/-! ### The positive-root form and continuity -/

/-- Positive definiteness is equivalent to existence of the positive
attention root. Source: arXiv:2305.05465v6, §3, `t:Idcase11int`, and §8.1,
`eq: A`. The square-root notation adds no hypothesis to the dynamics. -/
theorem isPosDefQK_iff_exists_attentionRoot (Q K : ParamMatrix d) :
    IsPosDefQK Q K ↔ ∃ A : ParamMatrix d, IsAttentionRoot Q K A := by
  constructor
  · exact IsPosDefQK.exists_attentionRoot
  · rintro ⟨A, hA⟩
    exact isPosDefQK_of_isAttentionRoot hA

/-- Every solution under the original positive-definiteness hypothesis
has the identity-root form `e:dynIdzi`. Source: arXiv:2305.05465v6, §3,
`e:Rres`, and §8.1, `eq: A`. The root is constructed from `Q` and `K`. -/
theorem RescaledDynamics.exists_identity_root (Q K : ParamMatrix d)
    (hQK : IsPosDefQK Q K) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K 1 Z) :
    ∃ A : ParamMatrix d, IsAttentionRoot Q K A ∧ IdRescaledDynamics A Z := by
  obtain ⟨A, hA⟩ := hQK.exists_attentionRoot
  exact ⟨A, hA, (rescaledDynamics_one_iff hA Z).mp hZ⟩

/-- The complete positivity and ODE inputs hold simultaneously for identity
query and key maps and stationary tokens; arXiv:2305.05465v6, §3, `e:Rres`. -/
example (z : EucSpace d) :
    IsPosDefQK (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      RescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) 1 (fun _ _ => z) :=
  ⟨isPosDefQK_of_isAttentionRoot (isAttentionRoot_id d), rescaledDynamics_one_const _ _ z⟩

/-- Any token path in the rescaled dynamics is continuous, by its
specified derivative. Source: arXiv:2305.05465v6, §3, `e:Rres`.
This does not require positive definiteness or identity value maps. -/
theorem RescaledDynamics.continuous_token (Q K V : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    Continuous (fun t => Z t i) :=
  continuous_iff_continuousAt.mpr fun t => (hZ t i).continuousAt

/-- An actual stationary solution satisfies the entire ODE input.
Source: arXiv:2305.05465v6, §3, `e:Rres`. -/
example (z : EucSpace 1) : RescaledDynamics (1 : ParamMatrix 1) 1 1
    (fun _ (_ : Idx 2) => z) := rescaledDynamics_one_const 1 1 z

/-- The complete rescaled token tuple is continuous.
Source: arXiv:2305.05465v6, §3, `e:Rres`; finite products preserve continuity. -/
theorem RescaledDynamics.continuous_tuple (Q K V : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) : Continuous Z :=
  continuous_pi fun i => hZ.continuous_token Q K V Z i

/-- Tuple continuity is based on an actual solution, here two stationary
unit tokens. Source: arXiv:2305.05465v6, §3, `e:Rres`. -/
example : RescaledDynamics (1 : ParamMatrix 1) 1 1
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := rescaledDynamics_one_const 1 1 _

end Clusters
end Transformer
