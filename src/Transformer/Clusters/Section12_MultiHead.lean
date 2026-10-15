/-
# The emergence of clusters in self-attention dynamics — multi-headed
  Transformers

§12 of arXiv:2305.05465v6, `sec:conclusion`: the multi-headed Transformer and
the positive one-dimensional case of the open problem it carries.

**What the source says and what is carried here.**

* The display is written "borrowing the notation from Remark
  `r:discreterescaling`", that is in discrete time: it is `e:discreteequation`
  with the single attention head replaced by a sum over `h ∈ [H]`, each head
  having its own constant `Q_h, K_h, V_h`.  `MultiHeadTransformer` is that
  display, and `multiHeadTransformer_one_iff` proves that at `H = 1` it is
  `e:discreteequation` itself.

* "Proofs regarding clustering or convergence of the self-attention matrix for
  such dynamics is an open problem."  The source attaches no hypotheses to the
  problem, and with none it would be false: already at `H = 1` the
  single-headed conclusion needs them — §2's `t:boolean` asks for `d = 1`,
  `V > 0`, `QK > 0` and pairwise distinct initial tokens, and the source
  itself says of a value matrix whose leading eigenvalue is complex that "we
  do not expect any clustering to occur".  The open problem is therefore
  carried as the multi-headed form of `t:boolean`: those hypotheses, one copy
  per head, and `t:boolean`'s own conclusion, one per head.  The addition of
  the hypotheses is the deviation from the source. This case is proved here
  for the actual discrete dynamics and each original head. `Δt > 0` is among them,
  since at `Δt = 0` no token moves and the attention matrix stays wherever it
  started.

* The conclusion is stated along the iteration, `k → +∞`.  That it is the
  discrete rather than the continuous dynamics is the source's own choice of
  display, and `r:discreterescaling` says the proofs of `t:boolean` and the
  rest "carry through with straightforward modifications" there.

Source: arXiv:2305.05465v6, `sec:conclusion`, the multi-headed display;
`r:discreterescaling`, `t:boolean`.
-/

import Transformer.Clusters.Section2_LowRank
import Transformer.Clusters.Section12_MultiHeadBoundedRows

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The multi-headed Transformer -/

/-- **One head is `e:discreteequation`.**  At `H = 1` the multi-headed
iteration is the discrete-time Transformer of `r:discretetime`.

Source: arXiv:2305.05465v6, `sec:conclusion`, "borrowing the notation from
Remark `r:discreterescaling`". -/
theorem multiHeadTransformer_one_iff (Δt : ℝ) (Q K V : Idx 1 → ParamMatrix d)
    (X : ℕ → Idx n → EucSpace d) :
    MultiHeadTransformer Δt Q K V X ↔ DiscreteTransformer Δt (Q 0) (K 0) (V 0) X := by
  simp [MultiHeadTransformer, DiscreteTransformer]

/-- **The configuration at the origin is stationary**, whatever the heads:
every `V_h` kills it, so no token moves.  This is the solution in closed form
that witnesses the hypotheses below. -/
theorem multiHeadTransformer_zero {H : ℕ} (Δt : ℝ) (Q K V : Idx H → ParamMatrix d) :
    MultiHeadTransformer (n := n) Δt Q K V (fun _ _ => (0 : EucSpace d)) := by
  intro k i
  simp

/-! ### Convergence for the original positive one-dimensional heads -/

/-- Every actual original head has a limiting row, including the possibly bounded token.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, `l:boundedxn`, and `l:boundedother`,
for the original discrete head sum of §12. No rate or escape premise is added. -/
theorem multiHeadTransformer_row_limits {m H : ℕ} {Δt : ℝ} (hΔt : 0 < Δt)
    {Q K V : Idx H → ParamMatrix 1} (hV : ∀ a, IsPosDefOp (V a))
    (hQK : ∀ a, IsPosDefQK (Q a) (K a))
    {X : ℕ → Idx (m + 1) → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X)
    (hdist : ∀ i j, i ≠ j → X 0 i ≠ X 0 j) (h : Idx H) :
    ∀ i, ∃ P : Idx (m + 1) → ℝ, ∀ j,
      Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (P j)) := by
  obtain ⟨_, _, _, _, γ, _, _, hγ, _, _, _, hz⟩ :=
    multiHeadTransformer_exists_token_rates hΔt hV hQK hX hdist h
  obtain ⟨L, M, g, hg, _, hmin, hmax⟩ := multiHeadTransformer_extreme_data hΔt.le hV hQK hX hdist
  obtain ⟨R, _, halt⟩ := multiHeadTransformer_escape_alternatives hΔt hV hQK hX h
  intro i
  rcases halt i with hb | hp | hn
  · have hbi : ∃ A : ℝ, ∀ k, |X k i 0| ≤ A := ⟨R, hb⟩
    have hne : ∀ j, j ≠ i → γ j ≠ 0 := by
      intro j hji he
      have hbj := (hz j).mp he
      have hij := multiHeadTransformer_bounded_initial_eq hΔt hV hQK hX h j i hbj hbi
      exact hdist j i hji hij
    exact multiHeadTransformer_bounded_row_exists hΔt hV hQK hX h i hbi γ hγ hne
  · exact ⟨fun j => if j = M then 1 else 0,
      fun j => multiHeadTransformer_positive_row hΔt.le hV hQK hX i M hg (hmax 0) h hp j⟩
  · exact ⟨fun j => if j = L then 1 else 0,
      fun j => multiHeadTransformer_negative_row hΔt.le hV hQK hX i L hg (hmin 0) h hn j⟩

/-- **The positive one-dimensional case of the multi-head convergence problem.**
With `V_h > 0`, `Q_h K_h > 0`, distinct initial tokens, and a positive time step,
each original head's full self-attention matrix converges to `𝒫` of `e:star`.

The source calls convergence for the general displayed multi-head dynamics an
open problem and attaches no hypotheses. This theorem retains the carried
one-dimensional hypotheses of `t:boolean`, one copy per original head; its
conclusion and discrete dynamics are unchanged. The proof includes the
exceptional bounded row and assumes no limiting rates or escape alternatives.
The condition `0 < n` is retained because `𝒫` has no empty matrix.

Source: arXiv:2305.05465v6, §12, `sec:conclusion`; §2, `t:boolean`, `e:star`;
§3, `r:discreterescaling`; §7, proof of `t:boolean`. -/
theorem multiHead_tendsto_isBooleanLimit {H : ℕ} (Δt : ℝ) (hΔt : 0 < Δt)
    (Q K V : Idx H → ParamMatrix 1) (hV : ∀ h : Idx H, IsPosDefOp (V h))
    (hQK : ∀ h : Idx H, IsPosDefQK (Q h) (K h))
    (X : ℕ → Idx n → EucSpace 1) (hX : MultiHeadTransformer Δt Q K V X)
    (hdist : ∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) (hn : 0 < n) (h : Idx H) :
    ∃ P : Idx n → Idx n → ℝ, IsBooleanLimit P ∧
      ∀ i j : Idx n,
        Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (P i j)) := by
  classical
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  choose P hP using multiHeadTransformer_row_limits hΔt hV hQK hX hdist h
  obtain ⟨L, M, i₀, hrows⟩ := multiHeadTransformer_except_one_basis_rows hΔt hV hQK hX hdist h
  refine ⟨P, ?_, hP⟩
  obtain ⟨hnn, hsum⟩ := attentionMatrix_discrete_row_limit_probability (Nat.succ_pos m)
    (Q h) (K h) X i₀ (P i₀) (hP i₀)
  refine ⟨L, M, i₀, ?_, hnn, hsum⟩
  intro i hi
  rcases hrows i hi with hL | hM
  · left
    funext j
    simpa only [Pi.single_apply] using tendsto_nhds_unique (hP i j) (hL j)
  · right
    funext j
    simpa only [Pi.single_apply] using tendsto_nhds_unique (hP i j) (hM j)

/-- The hypotheses of `multiHead_tendsto_isBooleanLimit` are satisfiable: with
a single token, every head at `Q_h = K_h = V_h = I_1` and the token pinned at
the origin, the distinctness condition is vacuous and the constant sequence
solves the multi-headed iteration because every `V_h` kills the origin. -/
example {H : ℕ} :
    (0 : ℝ) < 1 ∧
      (∀ h : Idx H, IsPosDefOp ((fun _ => 1 : Idx H → ParamMatrix 1) h)) ∧
      (∀ h : Idx H, IsPosDefQK ((fun _ => 1 : Idx H → ParamMatrix 1) h)
        ((fun _ => 1 : Idx H → ParamMatrix 1) h)) ∧
      MultiHeadTransformer (H := H) (n := 1) 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)
        (fun _ _ => (0 : EucSpace 1)) ∧
      (∀ i j : Idx 1, i ≠ j →
        (fun _ _ => (0 : EucSpace 1)) 0 i ≠ (fun _ _ => (0 : EucSpace 1)) 0 j) ∧ 0 < 1 :=
  ⟨one_pos, fun _ => isPosDefOp_id 1, fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1),
    multiHeadTransformer_zero 1 _ _ _, fun i j hij => absurd (Subsingleton.elim i j) hij, one_pos⟩

/-- Two genuine original heads and three distinct actual tokens with a bounded
middle token and both signs of escape jointly realize the full theorem's premises. -/
example : ∃ (V : Idx 2 → ParamMatrix 1) (X : ℕ → Idx 3 → EucSpace 1),
    (0 : ℝ) < 1 ∧ (∀ a, IsPosDefOp (V a)) ∧ (∀ a, IsPosDefQK (V a) (V a)) ∧
    MultiHeadTransformer 1 V V V X ∧ (∀ i j, i ≠ j → X 0 i ≠ X 0 j) ∧
    (∀ k, X k 1 = 0) ∧ Tendsto (fun k => X k 2 0) atTop atTop ∧
    Tendsto (fun k => X k 0 0) atTop atBot := by
  let V : Idx 2 → ParamMatrix 1 := fun _ => 1
  have hV : ∀ a, IsPosDefOp (V a) := fun _ => isPosDefOp_id 1
  have hQK : ∀ a, IsPosDefQK (V a) (V a) := fun _ => isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)
  obtain ⟨X, hX, hz, hp, hn, hord⟩ := exists_multiHead_symmetric_escape one_pos hV hQK 0
  refine ⟨V, X, one_pos, hV, hQK, hX, ?_, hz, hp, hn⟩
  intro i j hij he
  have hc := congrArg (fun z : EucSpace 1 => z 0) he
  rcases lt_or_gt_of_ne hij with hlt | hgt
  · have hs := hord i j hlt; linarith
  · have hs := hord j i hgt; linarith

end Clusters
end Transformer
