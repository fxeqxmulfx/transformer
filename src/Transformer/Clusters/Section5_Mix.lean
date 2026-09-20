/-
# The emergence of clusters in self-attention dynamics — a mix of hyperplanes
  and polytopes

§5 of arXiv:2305.05465v6: the good triple with multiplicity `d:goodmulti`, and
`t:multiplicity`, clustering toward `(∂𝒦 ∪ {0}) × 𝒢`.

**What the source says and what is carried here.**

* `d:goodmulti` (ii) calls `V` *paranormal* when `ℝ^d` splits into two
  `V`-invariant subspaces `ℱ ⊕ 𝒢` with `V|_ℱ = λ Id`, `λ > 0`, and
  `ρ(V|_𝒢) < λ`.  The spectral radius is taken through Gelfand's formula, as
  in `Section4_Hyperplanes` — see `IsSpectralRadiusLtOn`.  `ℱ` and `𝒢` are
  parameters of the definition rather than existentials, because the
  conclusion of `t:multiplicity` names them.

* `t:multiplicity` sets `ℋ := (∂𝒦 ∪ {0}) × 𝒢` for a bounded convex polytope
  `𝒦 ⊂ ℱ`.  The product is read through the splitting: `ℋ` is the set of
  `x ∈ ℝ^d` whose `ℱ`-part lies in `∂𝒦 ∪ {0}`, that is `x - p ∈ 𝒢` for some
  `p ∈ ∂𝒦 ∪ {0}`.

* `∂𝒦` is the frontier **in `ℱ`**, not in `ℝ^d`: `𝒦` is a subset of a proper
  subspace, so its frontier in `ℝ^d` would be all of `𝒦` and the conclusion
  would say strictly less.  `𝒦` therefore lives in `↥ℱ`, where `frontier` is
  the relative one.  "Bounded convex polytope" is the convex hull of a finite
  set, which is bounded and convex outright.

Source: arXiv:2305.05465v6, `d:goodmulti`, `t:multiplicity`.
-/

import Transformer.Clusters.Section4_Hyperplanes

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **Definition (d:goodmulti) (ii), paranormality.**  `V` acts as `λ Id` on
`ℱ`, leaves `𝒢` invariant, and has spectral radius `< λ` there.

Source: arXiv:2305.05465v6, `d:goodmulti` (ii). -/
def IsParanormalOn (V : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ) : Prop :=
  (∀ w ∈ F, V w ∈ F) ∧ (∀ w ∈ G, V w ∈ G) ∧ IsCompl F G ∧ 0 < lam ∧
    (∀ w ∈ F, V w = lam • w) ∧ IsSpectralRadiusLtOn G V lam

/-- **Definition (d:goodmulti).**  A good triple with multiplicity: `QᵀK ≻ 0`
and `V` paranormal.

Source: arXiv:2305.05465v6, `d:goodmulti`. -/
def IsGoodTripleMulti (Q K V : ParamMatrix d) (F G : Submodule ℝ (EucSpace d))
    (lam : ℝ) : Prop :=
  IsPosDefQK Q K ∧ IsParanormalOn V F G lam

/-- `QᵀK ≻ 0` holds for `Q = K = I_d`. -/
theorem isPosDefQK_one (d : ℕ) : IsPosDefQK (1 : ParamMatrix d) 1 := by
  refine ⟨fun u v => ?_, fun u hu => ?_⟩
  · simpa using real_inner_comm v u
  · simpa using real_inner_self_pos.mpr hu

/-- **`(I_d, I_d, I_d)` is a good triple with multiplicity**, with `ℱ = ℝ^d`,
`𝒢 = 0` and `λ = 1`: the case `V = I_d` of §3, which `t:multiplicity`
specializes to `t:Idcase11int`.  This witnesses `d:goodmulti`. -/
theorem isGoodTripleMulti_one (d : ℕ) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 := by
  refine ⟨isPosDefQK_one d, fun w _ => Submodule.mem_top, fun w hw => by simpa using hw,
    isCompl_top_bot, one_pos, fun w _ => (one_smul ℝ w).symm,
    ⟨1, 0, one_pos, le_rfl, one_pos, ?_⟩⟩
  intro k w hw
  have hw0 : w = 0 := by simpa using hw
  rw [hw0]
  simp

/-- **`ℋ = (∂𝒦 ∪ {0}) × 𝒢`.**  The points of `ℝ^d` whose `ℱ`-component, taken
along `𝒢`, lies on the relative boundary of the polytope `𝒦` or at the origin.

Source: arXiv:2305.05465v6, `t:multiplicity`. -/
def polytopeTimesSubspace (F G : Submodule ℝ (EucSpace d)) (T : Finset F) :
    Set (EucSpace d) :=
  {x | ∃ p ∈ frontier (convexHull ℝ (T : Set F)) ∪ {0}, x - (p : EucSpace d) ∈ G}

/-- **Theorem (t:multiplicity), clustering for `λ₁` with multiplicity.**  For a
good triple with multiplicity and any initial sequence there is a bounded
convex polytope `𝒦 ⊂ ℱ` such that every rescaled token approaches
`(∂𝒦 ∪ {0}) × 𝒢`.

Not proved here.

Source: arXiv:2305.05465v6, `t:multiplicity`. -/
theorem multiplicity_dist_tendsto_zero (Q K V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hQKV : IsGoodTripleMulti Q K V F G lam) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ T : Finset F, ∀ i : Idx n,
      Tendsto (fun t => Metric.infDist (Z t i) (polytopeTimesSubspace F G T))
        atTop (nhds 0) := by
  sorry

/-- The hypotheses of `multiplicity_dist_tendsto_zero` are satisfiable: the
triple `(I_d, I_d, I_d)` with `ℱ = ℝ^d`, `𝒢 = 0`, together with the stationary
configuration of `rescaledDynamics_one_const`. -/
example (d : ℕ) (z : EucSpace d) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  ⟨isGoodTripleMulti_one d, rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer
