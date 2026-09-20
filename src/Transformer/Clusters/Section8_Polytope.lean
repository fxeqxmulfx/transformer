/-
# The emergence of clusters in self-attention dynamics — the convex polytope

§8 of arXiv:2305.05465v6, `sec: clustering.polytopes`: the full form of the
`V = I_d` clustering theorem, and the shrinking convex hull it rests on.

**What the source says and what is carried here.**

* `Cx({z_i(t)}_{i∈[n]})` is `tokenHull (Z t)`, the convex hull of the range of
  the configuration.  It is a convex polytope for every `t`, being the hull of
  finitely many points.

* `p:noninc` says `t ↦ Cx({z_i(t)})` is non-increasing for inclusion.  Its
  proof uses only that the attention weights are non-negative and that the
  drift is `Σ_j P_ij (z_j - z_i)`; the hypotheses `V = I_d`, `QᵀK ≻ 0` are
  carried as the source states them.

* `c:theziunifbdd` is deduced from `p:noninc` in one line in the source.  It
  is stated here with the conclusion of `p:noninc` as an explicit hypothesis,
  so that it is genuinely proved and its dependence on the unproved
  proposition is legible in its own signature.

* `t:Idcase11` is the full statement whose simplified form `t:Idcase11int` is
  carried in `Section3_IdCase.lean`.  "The convex hull converges to a convex
  polytope `𝒦`" is convergence in Hausdorff distance — the hulls are nested
  compact sets, so this is the convergence to their intersection.  The
  vertices `𝒱` are `Set.extremePoints ℝ`, and `‖Ax‖² = max_{j∈[m]}⟨Ax,Av_j⟩`
  is `IsGreatest`, which records both that the maximum is attained at a vertex
  and that it dominates every vertex.

Source: arXiv:2305.05465v6, `p:noninc`, `c:theziunifbdd`, `t:Idcase11`,
`e:dynIdzi`, `eq: A`.
-/

import Transformer.Clusters.Section3_IdCase
import Mathlib.Analysis.Convex.Topology
import Mathlib.Analysis.Convex.Extreme
import Mathlib.Topology.MetricSpace.HausdorffDistance

open scoped BigOperators
open Real Filter

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The convex hull of a configuration -/

/-- `Cx({z_i}_{i∈[n]})`: the convex hull of the tokens of a configuration.  It
is a convex polytope, being the hull of the finite set `range Z`.

Source: arXiv:2305.05465v6, `p:noninc`. -/
def tokenHull (Z : Idx n → EucSpace d) : Set (EucSpace d) :=
  convexHull ℝ (Set.range Z)

/-- Every token lies in the hull of the configuration it belongs to. -/
theorem mem_tokenHull (Z : Idx n → EucSpace d) (i : Idx n) : Z i ∈ tokenHull Z :=
  subset_convexHull ℝ _ (Set.mem_range_self i)

/-- The hull of a configuration is compact: it is the convex hull of a finite
set. -/
theorem isCompact_tokenHull (Z : Idx n → EucSpace d) : IsCompact (tokenHull Z) :=
  (Set.finite_range Z).isCompact_convexHull ℝ

/-- **A configuration of equal tokens solves `e:dynIdzi`.**  The drift
`Σ_j P_ij (z_j - z_i)` vanishes when all the tokens coincide; this is the
solution in closed form that witnesses the hypotheses below. -/
theorem idRescaledDynamics_const (A : ParamMatrix d) (z : EucSpace d) :
    IdRescaledDynamics (n := n) A (fun _ _ => z) := by
  intro t i
  simpa using hasDerivAt_const t z

/-! ### `p:noninc` and `c:theziunifbdd` -/

/-- **Proposition (p:noninc).**  Let `V = I_d` and `QᵀK ≻ 0`.  Then the
solution of `e:dynIdzi` is such that `t ↦ Cx({z_i(t)}_{i∈[n]})` is
non-increasing in the sense of set-inclusion.

Not proved here.

Source: arXiv:2305.05465v6, `p:noninc`. -/
theorem tokenHull_antitone (Q K A : ParamMatrix d) (hA : IsAttentionRoot Q K A)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) {s t : ℝ}
    (hs : 0 ≤ s) (hst : s ≤ t) :
    tokenHull (Z t) ⊆ tokenHull (Z s) := by
  sorry

/-- The hypotheses of `tokenHull_antitone` are satisfiable: `Q = K = A = I_d`
is an attention root, and the configuration in which all `n` tokens sit at one
point solves `e:dynIdzi`. -/
example (z : EucSpace d) :
    IsAttentionRoot (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      IdRescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d)) (fun _ _ => z) ∧
      (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 :=
  ⟨isAttentionRoot_id d, idRescaledDynamics_const _ z, le_rfl, zero_le_one⟩

/-- **Corollary (c:theziunifbdd), first half.**  For any `i ∈ [n]` and
`t ≥ 0`, `z_i(t) ∈ Cx({z_i(0)}_{i∈[n]})`.

The conclusion of `p:noninc` is taken as an explicit hypothesis.

Source: arXiv:2305.05465v6, `c:theziunifbdd`. -/
theorem mem_tokenHull_zero (Z : ℝ → Idx n → EucSpace d)
    (hmono : ∀ ⦃s t : ℝ⦄, 0 ≤ s → s ≤ t → tokenHull (Z t) ⊆ tokenHull (Z s))
    (i : Idx n) {t : ℝ} (ht : 0 ≤ t) :
    Z t i ∈ tokenHull (Z 0) :=
  hmono le_rfl ht (mem_tokenHull (Z t) i)

/-- **Corollary (c:theziunifbdd), second half.**  Hence `z_i(·)` is uniformly
bounded in time: one radius serves every token and every `t ≥ 0`.

Source: arXiv:2305.05465v6, `c:theziunifbdd`. -/
theorem exists_bound_of_tokenHull_antitone (Z : ℝ → Idx n → EucSpace d)
    (hmono : ∀ ⦃s t : ℝ⦄, 0 ≤ s → s ≤ t → tokenHull (Z t) ⊆ tokenHull (Z s)) :
    ∃ R : ℝ, ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖Z t i‖ ≤ R := by
  obtain ⟨R, hR⟩ := (isCompact_tokenHull (Z 0)).isBounded.subset_closedBall 0
  exact ⟨R, fun i t ht => by
    simpa using hR (mem_tokenHull_zero Z hmono i ht)⟩

/-- The hypothesis of `mem_tokenHull_zero` is satisfiable: a configuration
that does not move has a constant hull. -/
example (z : EucSpace d) :
    ∀ ⦃s t : ℝ⦄, 0 ≤ s → s ≤ t →
      tokenHull ((fun _ _ => z : ℝ → Idx n → EucSpace d) t)
        ⊆ tokenHull ((fun _ _ => z : ℝ → Idx n → EucSpace d) s) :=
  fun _ _ _ _ => subset_rfl

/-! ### `t:Idcase11` -/

/-- The defining condition of the candidate limit set `𝒮` of `t:Idcase11`:
`‖Ax‖² = max_{j∈[m]}⟨Ax, Av_j⟩`, the maximum being over the vertices of the
polytope.  `IsGreatest` records both halves of the equality — that the value
is attained at a vertex, and that it dominates every vertex.

Source: arXiv:2305.05465v6, `e:Ax15`. -/
def IsAttentionMax (A : ParamMatrix d) (P : Set (EucSpace d)) (x : EucSpace d) : Prop :=
  IsGreatest ((fun v => inner (𝕜 := ℝ) (A x) (A v)) '' P.extremePoints ℝ) (‖A x‖ ^ 2)

/-- The candidate limit set `𝒮` of `t:Idcase11`.

Source: arXiv:2305.05465v6, `e:Ax15`. -/
def limitCandidates (A : ParamMatrix d) (P : Set (EucSpace d)) : Set (EucSpace d) :=
  {x ∈ P | IsAttentionMax A P x}

/-- **Theorem (t:Idcase11).**  Let `V = I_d` and `QᵀK ≻ 0`.  Then for any
initial datum the solution of `e:dynIdzi` has a convex hull converging to some
convex polytope `𝒦 ⊂ ℝ^d`.  Moreover the set

  `𝒮 = {x ∈ 𝒦 : ‖Ax‖² = max_{j∈[m]}⟨Ax, Av_j⟩}`,

where `𝒱 = {v_1, …, v_m}` are the vertices of `𝒦`, is finite and satisfies
`𝒱 ⊆ 𝒮 ⊆ ∂𝒦 ∪ {0}`; and every `z_i(t)` converges to a point of `𝒮`.

Not proved here.

Source: arXiv:2305.05465v6, `t:Idcase11`. -/
theorem idCase_tendsto_limitCandidates (Q K A : ParamMatrix d) (hA : IsAttentionRoot Q K A)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) :
    ∃ (P : Set (EucSpace d)) (S : Finset (EucSpace d)),
      P = convexHull ℝ (S : Set (EucSpace d)) ∧
      Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t)) P) atTop (nhds 0) ∧
      (limitCandidates A P).Finite ∧
      P.extremePoints ℝ ⊆ limitCandidates A P ∧
      limitCandidates A P ⊆ frontier P ∪ {0} ∧
      ∀ i : Idx n, ∃ p ∈ limitCandidates A P, Tendsto (fun t => Z t i) atTop (nhds p) := by
  sorry

/-- The hypotheses of `idCase_tendsto_limitCandidates` are satisfiable, by the
same witness as `tokenHull_antitone`. -/
example (z : EucSpace d) :
    IsAttentionRoot (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      IdRescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d)) (fun _ _ => z) :=
  ⟨isAttentionRoot_id d, idRescaledDynamics_const _ z⟩

end Clusters
end Transformer
