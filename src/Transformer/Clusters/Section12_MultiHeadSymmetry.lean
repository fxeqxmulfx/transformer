/-
# Original multi-head symmetries and a bounded interior token beside two escaping tokens

Construction for arXiv:2305.05465v6, §12, `sec:conclusion`, using §7's
sign symmetry and the symmetric triple of `Section7_Symmetric`.
The actual finite head sum commutes with every token permutation.
Combining this with its proved sign symmetry and uniqueness preserves
an initially reflected configuration at every original layer.

In particular, the actual recursive trajectory starting at (-R,0,R)
has its middle token identically zero and opposite outer tokens.
Choose R from the already proved self-pull threshold. The outer tokens
then tend to their respective infinities. This constructs actual
original discrete trajectories with a bounded interior token, positive
original heads, and distinct data; no scalar ODE or unproved trajectory
existence is taken as a premise of the witness.
-/

import Transformer.Clusters.Section12_MultiHeadEscape

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {n m H : ℕ}

/-- Permuting the original anchor tokens preserves each original scalar head average.
Source: arXiv:2305.05465v6, §12's normalized finite head sum. -/
theorem scalarHeadAverage_comp_perm (c : ℝ) (y : Idx n → ℝ) (r : ℝ) (σ : Equiv.Perm (Idx n)) :
    scalarHeadAverage c (fun j => y (σ j)) r = scalarHeadAverage c y r := by
  unfold scalarHeadAverage
  change (∑ j, Perspective.softmaxWeight ((fun l => c * r * y l) ∘ σ) j * y (σ j)) = _
  simp_rw [softmaxWeight_comp_perm]
  exact Equiv.sum_comp σ (fun j => Perspective.softmaxWeight (fun l => c * r * y l) j * y j)

/-- The exact original summed drift is invariant under a token permutation.
Source: arXiv:2305.05465v6, §12, the displayed sum over the original heads. -/
theorem multiHeadScalarDrift_comp_perm (c v : Idx H → ℝ) (y : Idx n → ℝ) (r : ℝ)
    (σ : Equiv.Perm (Idx n)) :
    multiHeadScalarDrift c v (fun j => y (σ j)) r = multiHeadScalarDrift c v y r := by
  unfold multiHeadScalarDrift
  simp_rw [scalarHeadAverage_comp_perm]

/-- Token relabelling preserves the actual original vector iteration.
Source: arXiv:2305.05465v6, §7's relabelling convention, for §12's original heads. -/
theorem multiHeadTransformer_comp_perm {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1}
    {X : ℕ → Idx n → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (σ : Equiv.Perm (Idx n)) : MultiHeadTransformer Δt Q K V (fun k i => X k (σ i)) := by
  apply (multiHeadTransformer_iff_coord Δt Q K V _).mpr
  intro k i
  rw [multiHeadScalarDrift_comp_perm _ _ (fun j => X k j 0) _ σ]
  exact multiHeadTransformer_coord hX k (σ i)

/-- Original uniqueness preserves any initial symmetry combining sign reversal and token permutation.
Source: arXiv:2305.05465v6, §7's involution, for §12's original discrete equation. -/
theorem multiHeadTransformer_symmetry_of_initial {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1}
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (σ : Equiv.Perm (Idx (m + 1))) (hinit : ∀ i, X 0 i = -X 0 (σ i)) :
    ∀ k i, X k i = -X k (σ i) := by
  have hY := multiHeadTransformer_neg (multiHeadTransformer_comp_perm hX σ)
  have he := multiHeadTransformer_unique Δt Q K V X (fun k i => -X k (σ i)) hX hY (funext hinit)
  intro k i
  exact congrFun (congrFun he k) i

/-- The actual symmetric triple has the required reflected original initial state.
Source: arXiv:2305.05465v6, §7's involution, realized by (-u,0,u). -/
theorem symTriple_neg_swap (u : ℝ) (i : Idx 3) :
    symTriple u i = -symTriple u (Equiv.swap (0 : Idx 3) 2 i) := by
  fin_cases i <;> simp [symTriple, symSign, Equiv.swap_apply_def]

/-- The middle token of the actual recursive original symmetric trajectory stays exactly at zero.
Source: arXiv:2305.05465v6, §12's original equation, with the symmetric data from §7. -/
theorem multiHeadTrajectory_middle_zero (Δt : ℝ) (Q K V : Idx H → ParamMatrix 1) (u : ℝ) :
    ∀ k, multiHeadTrajectory Δt Q K V (symTriple u) k 1 = 0 := by
  let X := multiHeadTrajectory Δt Q K V (symTriple u)
  have hX : MultiHeadTransformer Δt Q K V X := multiHeadTrajectory_solution _ _ _ _ _
  have hs := multiHeadTransformer_symmetry_of_initial hX (Equiv.swap (0 : Idx 3) 2)
    (symTriple_neg_swap u)
  intro k
  have he := hs k 1
  norm_num [Equiv.swap_apply_def] at he
  have hc := congrArg (fun z : EucSpace 1 => z 0) he
  change X k 1 0 = -(X k 1 0) at hc
  change X k 1 = 0
  rw [eq_smul_unit1 (X k 1)]
  have hz : X k 1 0 = 0 := by linarith
  rw [hz, zero_smul]

/-- Every original head sees the stationary middle token's exact uniform attention row.
Source: arXiv:2305.05465v6, §12's original softmax formula, for the symmetric data from §7. -/
theorem multiHeadTrajectory_middle_attention (Δt : ℝ) (Q K V : Idx H → ParamMatrix 1)
    (u : ℝ) (h : Idx H) (k : ℕ) (j : Idx 3) :
    attentionMatrix (Q h) (K h) (multiHeadTrajectory Δt Q K V (symTriple u) k) 1 j = 1 / 3 := by
  have hz := multiHeadTrajectory_middle_zero Δt Q K V u k
  simp only [attentionMatrix, hz, map_zero, inner_zero_left]
  norm_num [Perspective.softmaxWeight]

/-- A bounded interior token and two escaping outer tokens coexist in an actual original positive-head trajectory.
The heads may have different query-key and value coefficients.
This is a constructed satisfiability witness for §7's bounded-token arguments, for §12's original equation. -/
theorem exists_multiHead_symmetric_escape {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) (h : Idx H) :
    ∃ X : ℕ → Idx 3 → EucSpace 1, MultiHeadTransformer Δt Q K V X ∧
      (∀ k, X k 1 = 0) ∧ Tendsto (fun k => X k 2 0) atTop atTop ∧
      Tendsto (fun k => X k 0 0) atTop atBot ∧ IsOrderedConfig (X 0) := by
  obtain ⟨hΛ, hB⟩ := multiHead_coefficients_pos 3 Q K V hV hQK h
  obtain ⟨R, hR, hg⟩ := exists_discrete_self_growth_threshold Δt
    (multiHeadValueCoefficient V) (multiHeadPullCoefficient 3 Q K V) hΔt hΛ hB
  let X := multiHeadTrajectory Δt Q K V (symTriple R)
  have hX : MultiHeadTransformer Δt Q K V X := multiHeadTrajectory_solution _ _ _ _ _
  have hstep : ∀ k, R ≤ X k 2 0 →
      (1 + Δt * multiHeadValueCoefficient V / 2) * X k 2 0 ≤ X (k + 1) 2 0 := by
    intro k hk
    exact (hg _ hk).trans (multiHeadTransformer_self_pull hΔt.le hV hQK hX k 2 (hR.trans_le hk))
  have ha : 1 < 1 + Δt * multiHeadValueCoefficient V / 2 := by nlinarith [mul_pos hΔt hΛ]
  have hini : R ≤ X 0 2 0 := by change R ≤ symTriple R 2 0; simp [symTriple, unit1]
  have ht := discrete_tendsto_atTop_of_threshold (fun k => X k 2 0) ha hR hstep 0 hini
  have hs := multiHeadTransformer_symmetry_of_initial hX (Equiv.swap (0 : Idx 3) 2)
    (symTriple_neg_swap R)
  have hends : (fun k => X k 0 0) = (fun k => -(X k 2 0)) := by
    funext k
    have he := hs k 0
    rw [Equiv.swap_apply_left] at he
    exact congrArg (fun z : EucSpace 1 => z 0) he
  refine ⟨X, hX, multiHeadTrajectory_middle_zero Δt Q K V R, ht, ?_, ?_⟩
  · rw [hends]
    exact tendsto_neg_atTop_atBot.comp ht
  · change IsOrderedConfig (symTriple R)
    exact isOrderedConfig_symTriple hR

/-- Two positive original identity heads realize every matrix and step premise
of the constructed trajectory with its bounded interior token. -/
example : (0 : ℝ) < 1 ∧ (∀ h : Idx 2, IsPosDefOp ((fun _ => 1 : Idx 2 → ParamMatrix 1) h)) ∧
    (∀ h : Idx 2, IsPosDefQK ((fun _ => 1 : Idx 2 → ParamMatrix 1) h)
      ((fun _ => 1 : Idx 2 → ParamMatrix 1) h)) ∧ Nonempty (Idx 2) :=
  ⟨one_pos, fun _ => isPosDefOp_id 1, fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1), ⟨0⟩⟩

/-- A nonzero symmetric triple, two actual heads, and an actual nontrivial permutation
jointly realize the permutation and reflected-initial-state hypotheses. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1) (σ : Equiv.Perm (Idx 3)),
    MultiHeadTransformer 1 Q K V X ∧ (∀ i, X 0 i = -X 0 (σ i)) ∧ X 0 2 0 = 1 := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  refine ⟨Q, Q, Q, multiHeadTrajectory 1 Q Q Q (symTriple 1), Equiv.swap 0 2,
    multiHeadTrajectory_solution _ _ _ _ _, symTriple_neg_swap 1, ?_⟩
  change symTriple 1 2 0 = 1
  simp [symTriple, unit1]

end Transformer.Clusters
