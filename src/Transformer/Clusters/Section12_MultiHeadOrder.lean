/-
# Order and separation for actual discrete multi-head trajectories

ArXiv:2305.05465v6, §12, `sec:conclusion`, and §7, `l:distnondec`.
For the original discrete update, every scalar head has monotone drift
under the original positive query-key and value hypotheses. Multiplying
its increment by a nonnegative step size preserves the current order and
can only increase an ordered gap. Induction therefore preserves the
initial gap, strictly ordered data, and the absence of collisions.

These are assertions about solutions of the original vector iteration,
using its proved exact coordinate equation. They do not assume a
continuous-time interpolation or convergence of any attention matrix.
The joint witness has two distinct nonzero tokens, two positive heads,
and is the actual recursively constructed original trajectory.
-/

import Transformer.Clusters.Section12_MultiHeadIteration
import Transformer.Clusters.Section7_Unbounded

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {n H : ℕ}

/-- The exact original scalar update increases every already ordered gap.
Source: arXiv:2305.05465v6, §7, `l:distnondec`, for §12's discrete head sum. -/
theorem multiHeadScalarDrift_step_gap (c v : Idx H → ℝ) (hc : ∀ h, 0 ≤ c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx n → ℝ) (Δt : ℝ) (hΔt : 0 ≤ Δt)
    (r s : ℝ) (hrs : r ≤ s) :
    s - r ≤ (s + Δt * multiHeadScalarDrift c v y s) -
      (r + Δt * multiHeadScalarDrift c v y r) := by
  have hmono := multiHeadScalarDrift_monotone c v hc hv y hrs
  have hinc := mul_nonneg hΔt (sub_nonneg.mpr hmono)
  nlinarith

section Original

variable {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1} {X : ℕ → Idx n → EucSpace 1}
variable (hΔt : 0 ≤ Δt) (hV : ∀ h, IsPosDefOp (V h))
variable (hQK : ∀ h, IsPosDefQK (Q h) (K h)) (hX : MultiHeadTransformer Δt Q K V X)

include hΔt hV hQK hX

/-- Every actual step increases a currently ordered coordinate gap.
Source: arXiv:2305.05465v6, §7, `l:distnondec`, and §12's original update. -/
theorem multiHeadTransformer_gap_step (k : ℕ) (i j : Idx n)
    (hij : X k i 0 ≤ X k j 0) :
    X k j 0 - X k i 0 ≤ X (k + 1) j 0 - X (k + 1) i 0 := by
  rw [multiHeadTransformer_coord hX k i, multiHeadTransformer_coord hX k j]
  exact multiHeadScalarDrift_step_gap _ _
    (fun h => (scalar_head_parameters_pos _ _ _ (hV h) (hQK h)).1.le)
    (fun h => (scalar_head_parameters_pos _ _ _ (hV h) (hQK h)).2.le)
    _ Δt hΔt _ _ hij

/-- The actual coordinate separation always dominates its original ordered gap.
Source: arXiv:2305.05465v6, §7, `e:infdist`, for the discrete heads of §12. -/
theorem multiHeadTransformer_initial_gap (i j : Idx n) (hij : X 0 i 0 ≤ X 0 j 0) :
    ∀ k, X 0 j 0 - X 0 i 0 ≤ X k j 0 - X k i 0 := by
  intro k
  induction k with
  | zero => exact le_rfl
  | succ k ih =>
      have horder : X k i 0 ≤ X k j 0 := by linarith
      exact ih.trans (multiHeadTransformer_gap_step hΔt hV hQK hX k i j horder)

/-- Every initially ordered pair has a nondecreasing actual discrete gap.
Source: arXiv:2305.05465v6, §7, `l:distnondec`, under §12's positive head hypotheses. -/
theorem multiHeadTransformer_gap_monotone (i j : Idx n) (hij : X 0 i 0 ≤ X 0 j 0) :
    Monotone (fun k => X k j 0 - X k i 0) := by
  apply monotone_nat_of_le_succ
  intro k
  have hsep := multiHeadTransformer_initial_gap hΔt hV hQK hX i j hij k
  apply multiHeadTransformer_gap_step hΔt hV hQK hX k i j
  linarith

/-- The original iteration preserves every strict initial coordinate order.
Source: arXiv:2305.05465v6, §7, the ordering before `e:infdist`, extended to §12. -/
theorem multiHeadTransformer_strict_order (i j : Idx n) (hij : X 0 i 0 < X 0 j 0) (k : ℕ) :
    X k i 0 < X k j 0 := by
  have hsep := multiHeadTransformer_initial_gap hΔt hV hQK hX i j hij.le k
  linarith

/-- A sorted original initial configuration remains sorted at every discrete layer.
Source: arXiv:2305.05465v6, §7's ordering convention, for §12's original iteration. -/
theorem multiHeadTransformer_ordered (hord : IsOrderedConfig (X 0)) (k : ℕ) :
    IsOrderedConfig (X k) := by
  intro i j hij
  exact multiHeadTransformer_strict_order hΔt hV hQK hX i j (hord i j hij) k

/-- Distinct original one-dimensional tokens never collide in the actual multi-head iteration.
Source: arXiv:2305.05465v6, §7, `e:infdist`, for the positive discrete heads of §12. -/
theorem multiHeadTransformer_no_collision (i j : Idx n) (hne : X 0 i ≠ X 0 j) (k : ℕ) :
    X k i ≠ X k j := by
  have hc : X 0 i 0 ≠ X 0 j 0 := by
    intro he
    apply hne
    rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j), he]
  intro he
  have hec := congrArg (fun z : EucSpace 1 => z 0) he
  rcases lt_or_gt_of_ne hc with hlt | hgt
  · have hlt' := multiHeadTransformer_strict_order hΔt hV hQK hX i j hlt k
    linarith
  · have hgt' := multiHeadTransformer_strict_order hΔt hV hQK hX j i hgt k
    linarith

/-- Pairwise distinct original tokens remain pairwise distinct at every layer.
Source: arXiv:2305.05465v6, §7, `e:infdist`, for the discrete equation of §12.
This uses exactly the initial distinctness premise of the multi-head limit claim. -/
theorem multiHeadTransformer_distinct
    (hdist : ∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) (k : ℕ) :
    ∀ i j : Idx n, i ≠ j → X k i ≠ X k j := by
  intro i j hij
  exact multiHeadTransformer_no_collision hΔt hV hQK hX i j (hdist i j hij) k

end Original

/-- Two positive original heads and two distinct nonzero initial tokens
jointly realize all trajectory, strict order, and noncollision hypotheses. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1),
    (0 : ℝ) ≤ 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ IsOrderedConfig (X 0) ∧
    X 0 0 0 < X 0 1 0 ∧ X 0 0 ≠ X 0 1 ∧
    (∀ i j : Idx 2, i ≠ j → X 0 i ≠ X 0 j) ∧
    (∀ k, (2 : ℝ) ≤ X k 1 0 - X k 0 0) := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  let X := multiHeadTrajectory 1 Q Q Q Y
  have hV : ∀ h, IsPosDefOp (Q h) := fun _ => isPosDefOp_id 1
  have hQK : ∀ h, IsPosDefQK (Q h) (Q h) :=
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  have hX : MultiHeadTransformer 1 Q Q Q X := multiHeadTrajectory_solution _ _ _ _ _
  have hzero : X 0 = Y := multiHeadTrajectory_zero _ _ _ _ _
  have hlt : X 0 0 0 < X 0 1 0 := by rw [hzero]; norm_num [Y, unit1]
  have hord : IsOrderedConfig (X 0) := by
    rw [hzero]
    intro i j hij
    fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num [Y, unit1]
  have hne : X 0 0 ≠ X 0 1 := by
    intro he
    have hc := congrArg (fun z : EucSpace 1 => z 0) he
    linarith
  refine ⟨Q, Q, Q, X, zero_le_one, hV, hQK, hX, hord, hlt, hne, ?_, ?_⟩
  · intro i j hij he
    fin_cases i <;> fin_cases j <;>
      first | exact (hij rfl).elim | exact hne he | exact hne he.symm
  · intro k
    have hg := multiHeadTransformer_initial_gap zero_le_one hV hQK hX 0 1 hlt.le k
    have hini : X 0 1 0 - X 0 0 0 = 2 := by rw [hzero]; norm_num [Y, unit1]
    rwa [hini] at hg

end Transformer.Clusters
