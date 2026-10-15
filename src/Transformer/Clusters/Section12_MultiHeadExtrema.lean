/-
# Fixed separated extreme tokens of actual discrete multi-head trajectories

ArXiv:2305.05465v6, §7's ordering and `e:infdist`, for the original
multi-head equation of §12, `sec:conclusion`.
A finite family of distinct original coordinates has a strictly positive
minimum separation. Choosing its initial minimum and maximum therefore
gives two actual original token indices and a uniform strict gap to
every other token. The original discrete gap estimate preserves all
these inequalities at every layer, without relabelling any head.

These statements provide the extreme indices required by the original
per-head row estimates and by the rescaled asymptotic argument. The
finite separation is proved directly from the source's distinctness
hypothesis; no uniform initial gap is added as a new hypothesis to
the original multi-head convergence problem.
Combining these indices with the proved escape classification gives
basis-row limits for every original head outside at most one row.
-/

import Transformer.Clusters.Section12_MultiHeadRows
import Transformer.Clusters.Section12_MultiHeadOnlyOne

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {n m H : ℕ}

/-- A finite family of pairwise-distinct original scalar tokens has a positive uniform separation.
Source: arXiv:2305.05465v6, §7, `e:infdist`, without changing the original token labels. -/
theorem scalar_family_exists_gap (y : Idx n → ℝ) (hne : ∀ i j, i ≠ j → y i ≠ y j) :
    ∃ g : ℝ, 0 < g ∧ ∀ i j, i ≠ j → g ≤ |y i - y j| := by
  classical
  let S := Finset.univ.filter (fun p : Idx n × Idx n => p.1 ≠ p.2)
  have hpos : ∀ p ∈ S, 0 < |y p.1 - y p.2| := by
    intro p hp
    exact abs_pos.mpr (sub_ne_zero.mpr (hne p.1 p.2 (Finset.mem_filter.mp hp).2))
  rcases S.eq_empty_or_nonempty with hS | hS
  · refine ⟨1, one_pos, ?_⟩
    intro i j hij
    have hp : (i, j) ∈ S := by simp [S, hij]
    simp [hS] at hp
  · obtain ⟨p, hp, hmin⟩ := Finset.exists_min_image S (fun p => |y p.1 - y p.2|) hS
    refine ⟨|y p.1 - y p.2|, hpos p hp, ?_⟩
    intro i j hij
    exact hmin (i, j) (by simp [S, hij])

/-- Original distinct data have fixed minimum and maximum indices with positive gaps to all other indices.
Source: arXiv:2305.05465v6, §7's ordering before `e:infdist`, expressed without relabelling. -/
theorem scalar_family_extreme_gaps (y : Idx (m + 1) → ℝ) (hne : ∀ i j, i ≠ j → y i ≠ y j) :
    ∃ (L M : Idx (m + 1)) (g : ℝ), 0 < g ∧
      (∀ j, y L ≤ y j ∧ y j ≤ y M) ∧
      (∀ j, j ≠ L → y L + g ≤ y j) ∧ (∀ j, j ≠ M → y j + g ≤ y M) := by
  obtain ⟨g, hg, hgap⟩ := scalar_family_exists_gap y hne
  obtain ⟨L, _, hL⟩ := Finset.exists_min_image Finset.univ y Finset.univ_nonempty
  obtain ⟨M, _, hM⟩ := Finset.exists_max_image Finset.univ y Finset.univ_nonempty
  refine ⟨L, M, g, hg, ?_, ?_, ?_⟩
  · exact fun j => ⟨hL j (Finset.mem_univ j), hM j (Finset.mem_univ j)⟩
  · intro j hj
    have hs := hgap j L hj
    rw [abs_of_nonneg (sub_nonneg.mpr (hL j (Finset.mem_univ j)))] at hs
    linarith
  · intro j hj
    have hs := hgap j M hj
    rw [abs_of_nonpos (sub_nonpos.mpr (hM j (Finset.mem_univ j)))] at hs
    linarith

/-- The actual original discrete trajectory retains its fixed separated extreme token indices.
Source: arXiv:2305.05465v6, §7, `e:infdist`, for §12's original head sum. -/
theorem multiHeadTransformer_extreme_data {Δt : ℝ} (hΔt : 0 ≤ Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j) :
    ∃ (L M : Idx (m + 1)) (g : ℝ), 0 < g ∧
      (∀ k j, X k L 0 ≤ X k j 0 ∧ X k j 0 ≤ X k M 0) ∧
      (∀ k j, j ≠ L → X k L 0 + g ≤ X k j 0) ∧
      (∀ k j, j ≠ M → X k j 0 + g ≤ X k M 0) := by
  have hc : ∀ i j, i ≠ j → X 0 i 0 ≠ X 0 j 0 := by
    intro i j hij he
    apply hdist i j hij
    rw [eq_smul_unit1 (X 0 i), eq_smul_unit1 (X 0 j), he]
  obtain ⟨L, M, g, hg, hord, hmin, hmax⟩ := scalar_family_extreme_gaps (fun i => X 0 i 0) hc
  refine ⟨L, M, g, hg, ?_, ?_, ?_⟩
  · intro k j
    have hL := multiHeadTransformer_initial_gap hΔt hV hQK hX L j (hord j).1 k
    have hM := multiHeadTransformer_initial_gap hΔt hV hQK hX j M (hord j).2 k
    constructor <;> linarith [(hord j).1, (hord j).2]
  · intro k j hj
    have hi := hmin j hj
    have hs := multiHeadTransformer_initial_gap hΔt hV hQK hX L j (by linarith) k
    linarith
  · exact multiHeadTransformer_max_gap hΔt hV hQK hX M hg hmax

/-- Every original head has basis-row limits at all but at most one original token.
Source: arXiv:2305.05465v6, §7, `l:onlyone` and `l:unboundedparticles`, for §12's original head sum. -/
theorem multiHeadTransformer_except_one_basis_rows {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j) (h : Idx H) :
    ∃ L M i₀ : Idx (m + 1), ∀ i, i ≠ i₀ →
      (∀ j, Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = L then 1 else 0))) ∨
      (∀ j, Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = M then 1 else 0))) := by
  obtain ⟨L, M, g, hg, _, hmin, hmax⟩ := multiHeadTransformer_extreme_data hΔt.le hV hQK hX hdist
  obtain ⟨i₀, he⟩ := multiHeadTransformer_except_one_escape hΔt hV hQK hX h hdist
  refine ⟨L, M, i₀, ?_⟩
  intro i hi
  rcases he i hi with hp | hn
  · exact Or.inr fun j => multiHeadTransformer_positive_row hΔt.le hV hQK hX i M hg (hmax 0) h hp j
  · exact Or.inl fun j => multiHeadTransformer_negative_row hΔt.le hV hQK hX i L hg (hmin 0) h hn j

/-- If every original token is unbounded, every original head row has one of the two extreme basis-row limits.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, extended to §12's original positive heads. -/
theorem multiHeadTransformer_all_unbounded_basis_rows {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ h, IsPosDefOp (V h))
    (hQK : ∀ h, IsPosDefQK (Q h) (K h)) {X : ℕ → Idx (m + 1) → EucSpace 1}
    (hX : MultiHeadTransformer Δt Q K V X) (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j)
    (hb : ∀ i, ¬ ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) (h : Idx H) :
    ∃ L M : Idx (m + 1), ∀ i,
      (∀ j, Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = L then 1 else 0))) ∨
      (∀ j, Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (if j = M then 1 else 0))) := by
  obtain ⟨L, M, g, hg, _, hmin, hmax⟩ := multiHeadTransformer_extreme_data hΔt.le hV hQK hX hdist
  obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  refine ⟨L, M, ?_⟩
  intro i
  rcases halt i with hi | hp | hn
  · exact (hb i ⟨R, hi⟩).elim
  · exact Or.inr fun j => multiHeadTransformer_positive_row hΔt.le hV hQK hX i M hg (hmax 0) h hp j
  · exact Or.inl fun j => multiHeadTransformer_negative_row hΔt.le hV hQK hX i L hg (hmin 0) h hn j

/-- Distinct nonzero original scalar tokens jointly realize the finite gap and extremum premises. -/
example : ∃ y : Idx 2 → ℝ, ∀ i j, i ≠ j → y i ≠ y j := by
  refine ⟨![-1, 1], ?_⟩
  intro i j hij he
  fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num at he

/-- Two genuine positive heads and two distinct nonzero initial tokens jointly realize
all original matrix, trajectory, and distinctness premises of the fixed extreme data. -/
example : ∃ (Q K V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 2 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ h, IsPosDefOp (V h)) ∧ (∀ h, IsPosDefQK (Q h) (K h)) ∧
    MultiHeadTransformer 1 Q K V X ∧ (∀ i j, i ≠ j → X 0 i ≠ X 0 j) := by
  let Q : Idx 2 → ParamMatrix 1 := fun _ => 1
  let Y : Idx 2 → EucSpace 1 := ![-unit1, unit1]
  let X := multiHeadTrajectory 1 Q Q Q Y
  refine ⟨Q, Q, Q, X, one_pos, fun _ => isPosDefOp_id 1,
    fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1), multiHeadTrajectory_solution _ _ _ _ _, ?_⟩
  intro i j hij he
  have hc := congrArg (fun z : EucSpace 1 => z 0) he
  change Y i 0 = Y j 0 at hc
  fin_cases i <;> fin_cases j <;> norm_num at hij <;> norm_num [Y, unit1] at hc

/-- A nonzero geometric original singleton trajectory jointly realizes strict original
positivity, distinct data, and unboundedness of every original token. -/
example : ∃ X : ℕ → Idx 1 → EucSpace 1,
    (0 : ℝ) < 1 ∧ IsPosDefOp (1 : ParamMatrix 1) ∧ IsPosDefQK (1 : ParamMatrix 1) 1 ∧
    MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X ∧
    (∀ i j : Idx 1, i ≠ j → X 0 i ≠ X 0 j) ∧
    (∀ i, ¬ ∃ R : ℝ, ∀ k, |X k i 0| ≤ R) := by
  let X := fun k : ℕ => fun _ : Idx 1 => (3 : ℝ) ^ k • unit1
  have hX : MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X := by
    convert multiHeadTransformer_singleton_power (H := 2) 1 (fun _ => (1 : ParamMatrix 1))
      (fun _ => 1) (fun _ => 1) unit1 using 1
    norm_num [X, unit1]
  refine ⟨X, one_pos, isPosDefOp_id 1, isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1),
    hX, fun i j hij => (hij (Subsingleton.elim i j)).elim, ?_⟩
  intro i hb
  obtain ⟨R, hR⟩ := hb
  obtain ⟨K, hK⟩ := eventually_atTop.mp
    (tendsto_atTop.mp (tendsto_pow_atTop_atTop_of_one_lt (by norm_num : (1 : ℝ) < 3)) (R + 1))
  have hl := hK K le_rfl
  have hu := hR K
  have he : X K i 0 = (3 : ℝ) ^ K := by simp [X, unit1]
  rw [he, abs_of_pos (pow_pos (by norm_num : (0 : ℝ) < 3) K)] at hu
  linarith

end Transformer.Clusters
