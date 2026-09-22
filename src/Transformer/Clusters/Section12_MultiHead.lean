/-
# The emergence of clusters in self-attention dynamics — multi-headed
  Transformers

§12 of arXiv:2305.05465v6, `sec:conclusion`: the multi-headed Transformer and
the open problem it carries.

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
  the hypotheses is the deviation from the source; `Δt > 0` is among them,
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

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The multi-headed Transformer -/

/-- **The multi-headed Transformer of `sec:conclusion`.**  `H` heads, each
with its own constant weights `Q_h, K_h, V_h`, acting in parallel at every
layer:

  `x_i^{[k+1]} = x_i^{[k]} + Δt Σ_{h∈[H]} Σ_j P^h_ij(k) V_h x_j^{[k]}`,

where `P^h` is the self-attention matrix `eq:P` of the head `h`.

Source: arXiv:2305.05465v6, `sec:conclusion`. -/
def MultiHeadTransformer {H : ℕ} (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (X : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (k : ℕ) (i : Idx n),
    X (k + 1) i = X k i + Δt • ∑ h : Idx H, ∑ j : Idx n,
      attentionMatrix (Q h) (K h) (X k) i j • V h (X k j)

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

/-! ### The open problem -/

/-- **Open problem (`sec:conclusion`), convergence of the self-attention
matrix of a multi-headed Transformer.**  With `d = 1`, `V_h > 0` and
`Q_hK_h > 0` for every head and pairwise distinct initial tokens, does the
self-attention matrix of each head converge, as `k → +∞`, to a matrix of the
set `𝒫` of `e:star`?

Open: "Proofs regarding clustering or convergence of the self-attention matrix
for such dynamics is an open problem.  Preliminary numerical investigations
seem to indicate that interesting clustering phenomena also occur in this
context."  The hypotheses are `t:boolean`'s, one copy per head; the source
states none.  `0 < n` is added, as for `t:boolean`: at `n = 0` no matrix is
in `𝒫`.

Source: arXiv:2305.05465v6, `sec:conclusion`; `t:boolean`, `e:star`. -/
theorem multiHead_tendsto_isBooleanLimit {H : ℕ} (Δt : ℝ) (hΔt : 0 < Δt)
    (Q K V : Idx H → ParamMatrix 1) (hV : ∀ h : Idx H, IsPosDefOp (V h))
    (hQK : ∀ h : Idx H, IsPosDefQK (Q h) (K h))
    (X : ℕ → Idx n → EucSpace 1) (hX : MultiHeadTransformer Δt Q K V X)
    (hdist : ∀ i j : Idx n, i ≠ j → X 0 i ≠ X 0 j) (hn : 0 < n) (h : Idx H) :
    ∃ P : Idx n → Idx n → ℝ, IsBooleanLimit P ∧
      ∀ i j : Idx n,
        Tendsto (fun k => attentionMatrix (Q h) (K h) (X k) i j) atTop (nhds (P i j)) := by
  sorry

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

end Clusters
end Transformer
