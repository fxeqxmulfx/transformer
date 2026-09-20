/-
# The emergence of clusters in self-attention dynamics — Step 2'

§10 of arXiv:2305.05465v6: Claim `cl:gamma'12` and Step 2', the lower bound on
the growth of `‖π_ℱ(Az_i(t))‖²` away from `𝒮_δ`, which replace Step 2
(`e:eq17`) of the proof of `t:Idcase11`.

**What the source says and what is carried here.**

* `𝒮_δ` "denotes the set of all points in `𝒦` at distance `≤ δ` to some point
  of `𝒮`"; that is `candidatesThickening`, with `𝒮` the `projLimitCandidates`
  of `Section10_ProjHull` seen in `ℝ^d`.

* Both statements are conditioned on `z_i(t) ∉ 𝒮_δ × 𝒢`, which through the
  splitting is `π_ℱ(z_i(t)) ∉ 𝒮_δ`.

* `𝒦` is not re-derived: the convergence of the projected hull, the sorried
  half of `exists_polytope_tendsto_image_tokenHull`, is taken as an explicit
  hypothesis, so that `𝒦` is the polytope the proof means and nothing here
  rests on an unproved result.

* The first equality of `e:derivativenorm123`,
  `½ d/dt ‖π_ℱ(Az_i(t))‖² = ⟨π_ℱ(Aż_i(t)), π_ℱ(Az_i(t))⟩`, is proved; it is
  the chain rule on `e:Rres`.  What Step 2' adds to it — the lower bound — is
  the sorried part.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`,
`cl:gamma'12`, Step 2', `e:derivativenorm123`.
-/

import Transformer.Clusters.Section10_Remainder
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The first equality of `e:derivativenorm123` -/

/-- **The first line of `e:derivativenorm123`.**

  `½ d/dt ‖π_ℱ(Az_i(t))‖² = ⟨π_ℱ(Aż_i(t)), π_ℱ(Az_i(t))⟩`,

with `ż_i(t)` the drift of `e:Rres`.  This is the chain rule, and it is what
Step 2' bounds from below.

Source: arXiv:2305.05465v6, `e:derivativenorm123`. -/
theorem hasDerivAt_norm_sq_proj (Q K V A proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => ‖proj (A (Z s i))‖ ^ 2)
      (2 * inner (𝕜 := ℝ) (proj (A (∑ j : Idx n,
          attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i))))
        (proj (A (Z t i)))) t := by
  have h1 : HasDerivAt (fun s => A (Z s i))
      (A (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i))) t :=
    (A.hasFDerivAt (x := Z t i)).comp_hasDerivAt t (hZ t i)
  have hu : HasDerivAt (fun s => proj (A (Z s i)))
      (proj (A (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i)))) t :=
    (proj.hasFDerivAt (x := A (Z t i))).comp_hasDerivAt t h1
  have h := hu.inner ℝ hu
  have hfun : (fun s => ‖proj (A (Z s i))‖ ^ 2)
      = fun s => inner (𝕜 := ℝ) (proj (A (Z s i))) (proj (A (Z s i))) := by
    funext s
    rw [real_inner_self_eq_norm_sq]
  rw [hfun]
  convert h using 1
  rw [real_inner_comm (proj (A (Z t i)))]
  ring

/-- The hypothesis of `hasDerivAt_norm_sq_proj` is satisfiable, by the
stationary configuration of `rescaledDynamics_one_const`. -/
example (z : EucSpace d) :
    RescaledDynamics (n := n) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  rescaledDynamics_one_const _ _ z

/-! ### `𝒮_δ` -/

/-- **`𝒮_δ`**, the points of `𝒦` at distance `≤ δ` from `𝒮`.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
def candidatesThickening (A proj : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) (δ : ℝ) : Set (EucSpace d) :=
  {w ∈ polytopeIn F S |
    Metric.infDist w (((↑) : F → EucSpace d) '' projLimitCandidates A proj F S) ≤ δ}

/-- Every point of `𝒦` lies in `𝒮_δ` for `δ` large, so the family is not
empty of content; at `δ < 0` it is empty, which is why Step 2' quantifies over
`δ > 0`. -/
theorem candidatesThickening_subset (A proj : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) (δ : ℝ) : candidatesThickening A proj F S δ ⊆ polytopeIn F S :=
  fun _ hw => hw.1

/-! ### `cl:gamma'12` and Step 2' -/

/-- **Claim (cl:gamma'12).**  There is `γ' > 0`, depending only on the
geometry of `𝒦`, such that for every `δ ∈ (0, δ₀]` there is `T = T(δ) > 0`
with: if `t ≥ T` and `z_i(t) ∉ 𝒮_δ × 𝒢`, then `a_j(t) ≥ γ'δ` for some
`j ∈ [n]`.

Not proved here.

Source: arXiv:2305.05465v6, `cl:gamma'12`. -/
theorem exists_gamma_projScore (Q K V A proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hQKV : IsGoodTripleMulti Q K V F G lam) (hA : IsAttentionRoot Q K A)
    (hproj : IsProjOnto proj F G) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (S : Finset F)
    (hS : Tendsto (fun t => Metric.hausdorffDist (proj '' tokenHull (Z t)) (polytopeIn F S))
      atTop (nhds 0)) :
    ∃ γ δ₀ : ℝ, 0 < γ ∧ 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      ∃ T : ℝ, 0 < T ∧ ∀ t : ℝ, T ≤ t → ∀ i : Idx n,
        proj (Z t i) ∉ candidatesThickening A proj F S δ →
          ∃ j : Idx n, γ * δ ≤ projScore A proj (Z t) i j := by
  sorry

/-- **Step 2'.**  There is `γ = γ(𝒦) > 0` such that for every `δ ∈ (0, δ₀]`
there is `T = T(δ) > 0` with: if `t ≥ T` and `π_ℱ(z_i(t)) ∉ 𝒮_δ`, then

  `d/dt ‖π_ℱ(Az_i(t))‖² ≥ γδ`.

Not proved here.  This replaces `e:eq17`, Step 2 of the proof of
`t:Idcase11`.

Source: arXiv:2305.05465v6, Step 2'. -/
theorem exists_gamma_deriv_norm_sq_proj (Q K V A proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hQKV : IsGoodTripleMulti Q K V F G lam) (hA : IsAttentionRoot Q K A)
    (hproj : IsProjOnto proj F G) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (S : Finset F)
    (hS : Tendsto (fun t => Metric.hausdorffDist (proj '' tokenHull (Z t)) (polytopeIn F S))
      atTop (nhds 0)) :
    ∃ γ δ₀ : ℝ, 0 < γ ∧ 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      ∃ T : ℝ, 0 < T ∧ ∀ t : ℝ, T ≤ t → ∀ i : Idx n,
        proj (Z t i) ∉ candidatesThickening A proj F S δ →
          ∃ D : ℝ, HasDerivAt (fun s => ‖proj (A (Z s i))‖ ^ 2) D t ∧ γ * δ ≤ D := by
  sorry

/-- The hypotheses of `exists_gamma_projScore` and `exists_gamma_deriv_norm_sq_proj`
are satisfiable: the triple `(I_d, I_d, I_d)` with `ℱ = ℝ^d`, `𝒢 = 0`, the
stationary configuration, and `𝒦 = {z}`, which the constant hull equals. -/
example (d : ℕ) (z : EucSpace d) (hz : z ∈ (⊤ : Submodule ℝ (EucSpace d))) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 ∧
      IsAttentionRoot (1 : ParamMatrix d) 1 1 ∧
      IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
      RescaledDynamics (n := n + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) ∧
      Tendsto (fun _ : ℝ => Metric.hausdorffDist
          ((1 : ParamMatrix d) '' tokenHull ((fun _ (_ : Idx (n + 1)) => z) 0))
          (polytopeIn ⊤ ({⟨z, hz⟩} : Finset (⊤ : Submodule ℝ (EucSpace d)))))
        atTop (nhds 0) := by
  refine ⟨isGoodTripleMulti_one d, isAttentionRoot_id d, isProjOnto_one d,
    rescaledDynamics_one_const _ _ z, ?_⟩
  have hset : (1 : ParamMatrix d) '' tokenHull ((fun _ (_ : Idx (n + 1)) => z) 0)
      = polytopeIn ⊤ ({⟨z, hz⟩} : Finset (⊤ : Submodule ℝ (EucSpace d))) := by
    have hrange : Set.range (fun _ : Idx (n + 1) => z) = {z} := by
      ext x
      simp [eq_comm]
    simp [tokenHull, polytopeIn, hrange, one_apply_eq_self, Set.image_id']
  rw [hset]
  simp

end Clusters
end Transformer
