module

public import Transformer.AttentionResiduals.Section3_Attention

/-!
# Block Attention Residuals and partial-block accumulation

arXiv:2603.15031v1, §3.2, and Kimi K3 technical report, §2.2, Eq. (10).
History includes the embedding as source zero. An empty current block
contributes no source; later sublayers add exactly one partial-sum source.
The partial block stores raw branch outputs, not attention aggregates.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.AttentionResiduals

/-- The history has at least the embedding. Subsequent layers additionally
attend to the current block's partial sum. Sources: arXiv:2603.15031v1,
§3.2, two-case value matrix; Kimi K3, §2.2, Eq. (10).
`n` counts completed blocks excluding the embedding. -/
noncomputable def blockInput {n d : ℕ} (ε : ℝ) (gain query : Fin d → ℝ)
    (history : Fin (n + 1) → (Fin d → ℝ))
    (current : Option (Fin d → ℝ)) : Fin d → ℝ :=
  match current with
  | none => attentionResidual ε gain query history
  | some p => attentionResidual ε gain query (Sum.elim history (fun _ : Unit => p))

/-- Grow the current block using the branch applied to its actual attention
input. Query and RMSNorm gain are allowed to vary with the sublayer.
Sources: arXiv:2603.15031v1, §3.2, `b_n^i`; Kimi K3, §2.2.
The first branch has no partial source. -/
noncomputable def blockPartial {n d : ℕ} (ε : ℝ)
    (gain query : ℕ → Fin d → ℝ) (F : ℕ → (Fin d → ℝ) → (Fin d → ℝ))
    (history : Fin (n + 1) → (Fin d → ℝ)) : ℕ → (Fin d → ℝ)
  | 0 => 0
  | t + 1 =>
    let p := blockPartial ε gain query F history t
    p + F t (blockInput ε (gain t) (query t) history (if t = 0 then none else some p))

/-- The stored partial block is precisely the sum of raw outputs of its
preceding branches, each evaluated at the correct attention input.
Source: arXiv:2603.15031v1, §3.2, intra-block accumulation. -/
theorem blockPartial_unroll {n d : ℕ} (ε : ℝ)
    (gain query : ℕ → Fin d → ℝ) (F : ℕ → (Fin d → ℝ) → (Fin d → ℝ))
    (history : Fin (n + 1) → (Fin d → ℝ)) (t : ℕ) :
    blockPartial ε gain query F history t =
      ∑ k ∈ Finset.range t,
        F k (blockInput ε (gain k) (query k) history
          (if k = 0 then none else some (blockPartial ε gain query F history k))) := by
  induction t with
  | zero => simp [blockPartial]
  | succ t ih =>
    change blockPartial ε gain query F history t + _ = _
    rw [Finset.sum_range_succ, ← ih]

/-- The first stored partial is exactly the first branch's output: neither
the embedding nor a weighted historical aggregate is accumulated into
the raw block representation. Source: arXiv:2603.15031v1, §3.2,
intra-block accumulation; Kimi K3, §2.2. -/
theorem first_partial_output {n d : ℕ} (ε : ℝ)
    (gain query : ℕ → Fin d → ℝ) (F : ℕ → (Fin d → ℝ) → (Fin d → ℝ))
    (history : Fin (n + 1) → (Fin d → ℝ)) :
    blockPartial ε gain query F history 1 =
      F 0 (blockInput ε (gain 0) (query 0) history none) := by
  simp [blockPartial]

/-- At the first sublayer there is no spurious zero source in the softmax.
Source: arXiv:2603.15031v1, §3.2, `i = 1` case. -/
theorem first_block_input {n d : ℕ} (ε : ℝ) (gain query : Fin d → ℝ)
    (history : Fin (n + 1) → (Fin d → ℝ)) :
    blockInput ε gain query history none = attentionResidual ε gain query history := by
  unfold blockInput
  dsimp

/-- Before the first block, a single embedding is returned unchanged.
Source: arXiv:2603.15031v1, §3.2, first layer input `b_0 = h_1`. -/
theorem initial_embedding_input {d : ℕ} (ε : ℝ) (gain query v : Fin d → ℝ) :
    blockInput ε gain query (fun _ : Fin 1 => v) none = v := by
  unfold blockInput attentionResidual
  exact attend_const _ v

/-- First-sublayer aggregation respects any norm bound on stored blocks.
Source: arXiv:2603.15031v1, §3.2, inter-block convex aggregation. -/
theorem blockInput_first_norm_le {n d : ℕ} (ε : ℝ) (gain query : Fin d → ℝ)
    (history : Fin (n + 1) → (Fin d → ℝ)) (R : ℝ)
    (hh : ∀ i, ‖history i‖ ≤ R) : ‖blockInput ε gain query history none‖ ≤ R := by
  exact attentionResidual_norm_le ε gain query history R hh

/-- Nonzero stored blocks satisfy the first-sublayer theorem's hypothesis. -/
example : ∀ i : Fin 3, ‖(fun _ : Fin 3 => (fun _ : Fin 1 => (1 : ℝ))) i‖ ≤ 1 := by
  norm_num

/-- Inside a block the bound must include the entire partial sum, not only
the individual branch outputs. Source: arXiv:2603.15031v1, §3.2. -/
theorem blockInput_inside_norm_le {n d : ℕ} (ε : ℝ) (gain query : Fin d → ℝ)
    (history : Fin (n + 1) → (Fin d → ℝ)) (p : Fin d → ℝ) (R : ℝ)
    (hh : ∀ i, ‖history i‖ ≤ R) (hp : ‖p‖ ≤ R) :
    ‖blockInput ε gain query history (some p)‖ ≤ R := by
  apply attentionResidual_norm_le
  intro i
  cases i with
  | inl j => exact hh j
  | inr j => exact hp

/-- Both the historical and partial source bounds are jointly satisfiable. -/
example : (∀ i : Fin 3, ‖(fun _ : Fin 3 => (fun _ : Fin 1 => (1 : ℝ))) i‖ ≤ 1) ∧
    ‖(fun _ : Fin 1 => (1 : ℝ))‖ ≤ 1 := by
  norm_num

/-- Counterexample to the unqualified sentence “N=1 reduces to standard
residual connections” in arXiv:2603.15031v1, §3.2. With the embedding
isolated, the specified softmax averages it with the current block. Two
equal unit sources give one, whereas their ordinary residual sum is two.
The block formulas are retained exactly; the claimed equivalence is not. -/
theorem one_block_not_standard_residual :
    blockInput 1 (fun _ : Fin 1 => 1) (fun _ => 0)
      (fun _ : Fin 1 => (fun _ : Fin 1 => (1 : ℝ))) (some (fun _ => 1)) ≠
      (fun _ : Fin 1 => (1 : ℝ)) + (fun _ => 1) := by
  have hv : blockInput 1 (fun _ : Fin 1 => 1) (fun _ => 0)
      (fun _ : Fin 1 => (fun _ : Fin 1 => (1 : ℝ))) (some (fun _ => 1)) =
      (fun _ : Fin 1 => (1 : ℝ)) := by
    unfold blockInput attentionResidual
    dsimp only
    have hvalues :
        Sum.elim (fun _ : Fin 1 => (fun _ : Fin 1 => (1 : ℝ)))
          (fun _ : Unit => (fun _ : Fin 1 => (1 : ℝ))) =
          (fun _ : Fin 1 ⊕ Unit => (fun _ : Fin 1 => (1 : ℝ))) := by
      funext i
      cases i <;> rfl
    rw [hvalues]
    exact attend_const _ (fun _ : Fin 1 => (1 : ℝ))
  rw [hv]
  intro h
  have h0 := congrFun h 0
  norm_num at h0

/-- Kimi K3's 93 decoder blocks split at size 12 into eight groups with
nine blocks in the final group. Counting the embedding gives nine stored
sources at output. Source: Kimi K3 technical report, §2.2, and pinned
`config.json`, `num_hidden_layers=93`, `attn_res_block_size=12`.
Here “decoder block” contains both attention and MLP sublayers. -/
theorem kimi_k3_block_counts :
    (93 + 12 - 1) / 12 = (8 : ℕ) ∧ 93 % 12 = (9 : ℕ) ∧
      1 + (93 + 12 - 1) / 12 = (9 : ℕ) := by
  norm_num

end Transformer.AttentionResiduals
