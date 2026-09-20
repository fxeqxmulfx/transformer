/-
# The emergence of clusters in self-attention dynamics — beyond `QᵀK ≻ 0`

§12 of arXiv:2305.05465v6, `sec: first.new`: the two open problems the source
raises by dropping the positive-definiteness of `QᵀK` from `t:Idcase11int` and
`t:multiplicity`.

**What the source says and what is carried here.**

* "None of the conclusions of Theorems `t:Idcase11int` or `t:multiplicity`
  seem to change for generic choices of `QᵀK` […] which we leave as open
  problems."  The two theorems are carried here with `IsPosDefQK` deleted and
  nothing else touched: `V = I_d` in the first, `V` paranormal in the second,
  and the conclusions verbatim from `Section3_IdCase` and `Section5_Mix`.

* "Generic" is read as: outside a Lebesgue-null set of matrices `QᵀK`, as in
  §2's "for almost all initial sequences".  The source's own illustration is a
  matrix "with entries drawn from the uniform distribution on `[-1,1]`", and a
  property witnessed by sampling is one that holds off a null set.  The
  exceptional set is allowed to depend on `d` and on the number of tokens.

* The parameter that genericity is asserted of is the matrix `QᵀK`, not the
  pair `(Q,K)`: `qkMatrix` is that matrix, `(QᵀK)_{ij} = ⟨Qe_i, Ke_j⟩`, and
  `attentionMatrix_congr_qkMatrix` proves that the dynamics sees `(Q,K)` only
  through it — so the exceptional set is genuinely a set of matrices, and
  `Idx d → Idx d → ℝ` is where it carries Lebesgue measure.

Source: arXiv:2305.05465v6, `sec: first.new`, Figures `f: first.new` and
`f: second.new`; `t:Idcase11int`, `t:multiplicity`.
-/

import Transformer.Clusters.Section3_IdCase
import Transformer.Clusters.Section5_Mix
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The matrix `QᵀK` -/

/-- **The matrix `QᵀK`.**  Its entries are `(QᵀK)_{ij} = ⟨Qe_i, Ke_j⟩`, the
form `(u,v) ↦ ⟨Qu, Kv⟩` of `eq:P` read in the standard basis.

Source: arXiv:2305.05465v6, `sec: first.new`. -/
noncomputable def qkMatrix (Q K : ParamMatrix d) (i j : Idx d) : ℝ :=
  inner (𝕜 := ℝ) (Q (EuclideanSpace.single i (1 : ℝ))) (K (EuclideanSpace.single j (1 : ℝ)))

/-- `Q = K = I_d` gives the identity matrix. -/
theorem qkMatrix_one (d : ℕ) (i j : Idx d) :
    qkMatrix (1 : ParamMatrix d) 1 i j = if i = j then 1 else 0 := by
  simp [qkMatrix, one_apply_eq_self, EuclideanSpace.inner_single_left, PiLp.single_apply]

/-- A vector is the sum of its coordinates against the standard basis. -/
theorem sum_smul_single (x : EucSpace d) :
    ∑ i : Idx d, x i • EuclideanSpace.single i (1 : ℝ) = x := by
  ext j
  simp [Pi.single_apply, mul_ite]

/-- **The form of `eq:P` is `QᵀK` read in coordinates**: `⟨Qu, Kv⟩` is the
bilinear form of the matrix `qkMatrix Q K`. -/
theorem inner_eq_sum_qkMatrix (Q K : ParamMatrix d) (u v : EucSpace d) :
    inner (𝕜 := ℝ) (Q u) (K v) = ∑ i : Idx d, ∑ j : Idx d, u i * v j * qkMatrix Q K i j := by
  conv_lhs => rw [← sum_smul_single u, ← sum_smul_single v]
  rw [map_sum, map_sum, sum_inner]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [inner_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [map_smul, map_smul, real_inner_smul_left, real_inner_smul_right, qkMatrix]
  ring

/-- **The dynamics sees `(Q,K)` only through `QᵀK`**: two pairs with the same
matrix have the same self-attention matrix, hence the same trajectories.  This
is what makes "generic `QᵀK`" a condition on the dynamics. -/
theorem attentionMatrix_congr_qkMatrix {Q K Q' K' : ParamMatrix d}
    (h : qkMatrix Q K = qkMatrix Q' K') (X : Idx n → EucSpace d) :
    attentionMatrix Q K X = attentionMatrix Q' K' X := by
  funext i j
  refine congrArg (fun f => Perspective.softmaxWeight f j) ?_
  funext l
  rw [inner_eq_sum_qkMatrix, inner_eq_sum_qkMatrix, h]

/-! ### `t:Idcase11int` for a generic `QᵀK` -/

/-- **Open problem (`sec: first.new`), `t:Idcase11int` beyond `QᵀK ≻ 0`.**
For `V = I_d` and a generic `QᵀK` — one outside a Lebesgue-null set of
matrices, the positive-definiteness of `t:Idcase11int` being dropped — does
every rescaled token still converge either to `0` or to a point of the
boundary of a convex polytope?

Open: "Nonetheless, the clustering pattern entailed by Theorem
`t:Idcase11int` persists", illustrated in Figure `f: first.new` with `QᵀK` a
matrix of uniform entries on `[-1,1]`; "which we leave as open problems".

Source: arXiv:2305.05465v6, `sec: first.new`, Figure `f: first.new`. -/
theorem idCase_tendsto_zero_or_frontier_generic (d n : ℕ) :
    ∃ N : Set (Idx d → Idx d → ℝ), volume N = 0 ∧
      ∀ Q K : ParamMatrix d, qkMatrix Q K ∉ N →
        ∀ Z : ℝ → Idx n → EucSpace d, RescaledDynamics Q K 1 Z →
          ∃ S : Finset (EucSpace d), ∀ i : Idx n,
            Tendsto (fun t => Z t i) atTop (nhds 0) ∨
              ∃ p ∈ frontier (convexHull ℝ (S : Set (EucSpace d))),
                Tendsto (fun t => Z t i) atTop (nhds p) := by
  sorry

/-! ### `t:multiplicity` for a generic `QᵀK` -/

/-- **Open problem (`sec: first.new`), `t:multiplicity` beyond `QᵀK ≻ 0`.**
For `V` paranormal — the second half of `d:goodmulti`, the first half being
what is dropped — and a generic `QᵀK`, does every rescaled token still
approach `(∂𝒦 ∪ {0}) × 𝒢` for some bounded convex polytope `𝒦 ⊂ ℱ`?

Open: "Nonetheless, the clustering pattern entailed by Theorem
`t:multiplicity` persists", illustrated in Figure `f: second.new` with `V`
paranormal and `QᵀK` a matrix of uniform entries on `[-1,1]`; "which we leave
as open problems".

Source: arXiv:2305.05465v6, `sec: first.new`, Figure `f: second.new`. -/
theorem multiplicity_dist_tendsto_zero_generic (d n : ℕ) (V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ) (hV : IsParanormalOn V F G lam) :
    ∃ N : Set (Idx d → Idx d → ℝ), volume N = 0 ∧
      ∀ Q K : ParamMatrix d, qkMatrix Q K ∉ N →
        ∀ Z : ℝ → Idx n → EucSpace d, RescaledDynamics Q K V Z →
          ∃ T : Finset F, ∀ i : Idx n,
            Tendsto (fun t => Metric.infDist (Z t i) (polytopeTimesSubspace F G T))
              atTop (nhds 0) := by
  sorry

/-- The hypothesis of `multiplicity_dist_tendsto_zero_generic` is satisfiable:
`V = I_d` is paranormal with `ℱ = ℝ^d`, `𝒢 = 0` and `λ = 1`, which is the
case `t:multiplicity` specializes to `t:Idcase11int`. -/
example (d : ℕ) : IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 :=
  (isGoodTripleMulti_one d).2

end Clusters
end Transformer
