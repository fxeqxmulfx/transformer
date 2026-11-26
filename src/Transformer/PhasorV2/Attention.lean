module

public import Transformer.PhasorV2.QKNorm
public import Transformer.PhasorV2.MHC

/-!
# Phasor v2: from the embedding to the output of attention

The elements of Phasor v2 put together into the attention sublayer. The formulas follow DiffGPT
(`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*, with
`N_i = i + 1`:

    h_i,s^0 = R_D(E[token_i]) for each of four streams s
    z_i = R_D(sum_s read_i,s * h_i,s)
    q_i = LeRoPE_i(R_d(W_Q z_i)) per head
    k_i = LeRoPE_i(R_d(W_K z_i)) per head
    v_i = W_V z_i per head
    Z_ij = sum_c conj(q_i,c)*k_j,c
    p_ij = softmax_j((log(N_i)/d)*Re(Z_ij)), j<=i
    a_i = concatenate_heads(sum_{j<=i} p_ij*v_j)
    h'_i,s = sum_t P_i,s,t*r_s*conj(r_t)*h_i,t + write_i,s*alpha*R_D(W_O a_i)

The score is that of DiffGPT's `complex-phase-mhc-lerope`, without the phase score
`beta*Im(Z_ij)^2/d` of the README. DiffGPT computes in real arithmetic; Phasor v2 is complex
throughout:

* the embedding table holds complex vectors, `E : V → CSpace c` with `D = 2c` (`embed`);
* `W_Q`, `W_K`, `W_V` and `W_O` are complex-linear maps (`Attention`);
* a head is `p` complex bands, `d = 2p`, and `Heads H p` holds all `H` heads, so concatenating
  them is the identity;
* every RMS takes the mean over complex channels, `R_D` over the `c` of the stream (`rms`) and
  `R_d` over the `p` bands of a head (`headRMS`), and the gain is `1/p` instead of `1/d`
  (`Attention.weight`);
* `z_i = R_D(…)` and `R_D(W_O a_i)` are the Peri-norm of `Router.step`, and the branch here is
  `z ↦ W_O a` (`Attention.branch`).

For all weights:

* the score depends on the positions only through the offset `j − i` (`Attention.score_eq`),
  and Q/K RMS commutes with LeRoPE (`turn_headRMS`);
* every logit at the position `i` is at most `ln(i + 1)` (`Attention.abs_logit_le`), and no key
  outweighs another by more than `(i + 1)²` (`Attention.weight_le_sq_mul`);
* the branch is causal (`Attention.isCausal_branch`), so the stack of sublayers is causal;
* from the embedding of any tokens, every stream has RMS at most `1 + 2n|α|` after `n`
  sublayers with the write scale `α` (`rms_stack_embed_le`).
-/

@[expose] public section

open Transformer.Sandwich

namespace Transformer.PhasorV2

variable {c H p : ℕ}

/-- **The embedding** (README: `h_i,s^0 = R_D(E[token_i])` for each of four streams `s`): the
row of the complex table `E` at each token, normalized, on all four streams. -/
noncomputable def embed {V : Type*} (eps : ℝ) (E : V → CSpace c) (tok : ℕ → V) :
    ℕ → State c :=
  fun i _ => rms eps (E (tok i))

/-- **`H` heads of `p` complex bands**, `d = 2p` real coordinates each. -/
abbrev Heads (H p : ℕ) := Fin H → Fin p → ℂ

/-- **The projections of an attention sublayer** (README: "bias-free complex-linear maps"): the
queries, keys and values of all heads from the stream, and the output from all heads. -/
structure Attention (c H p : ℕ) where
  WQ : CSpace c →ₗ[ℂ] Heads H p
  WK : CSpace c →ₗ[ℂ] Heads H p
  WV : CSpace c →ₗ[ℂ] Heads H p
  WO : Heads H p →ₗ[ℂ] CSpace c

/-- **Q/K RMS commutes with LeRoPE**: normalizing the bands before or after the turn is the
same. -/
theorem turn_headRMS (θ : Fin p → ℝ) (t eps : ℝ) (q : Fin p → ℂ) :
    turn θ t (headRMS eps q) = headRMS eps (turn θ t q) := by
  funext m
  simp only [turn, headRMS, sqLen_turn]
  ring

namespace Attention

variable (A : Attention c H p) (eps : ℝ) (α : Fin p → ℝ) (z : ℕ → CSpace c)

/-- **The query** of head `h` at the position `i`: `LeRoPE_i(R_d(W_Q z_i))`, LeRoPE on the base
`10000` with the learned scales `α`. -/
noncomputable def query (i : ℕ) (h : Fin H) : Fin p → ℂ :=
  turn (leropeFrequency 10000 α) i (headRMS eps (A.WQ (z i) h))

/-- **The key** of head `h` at the position `j`: `LeRoPE_j(R_d(W_K z_j))`. -/
noncomputable def key (j : ℕ) (h : Fin H) : Fin p → ℂ :=
  turn (leropeFrequency 10000 α) j (headRMS eps (A.WK (z j) h))

/-- **The weight** of the key `j ≤ i` for the query `i` of head `h`: SSMax over the `N_i = i + 1`
visible keys with the gain `1/p`, `softmax_j((log N_i / p) Re Z_ij)`, where the source divides by
`d = 2p`. -/
noncomputable def weight (i : ℕ) (h : Fin H) (j : Fin (i + 1)) : ℝ :=
  ssmaxWeight (1 / p)
    (fun k : Fin (i + 1) => score (A.query eps α z i h) (A.key eps α z k h)) j

/-- **The read of the heads** `a_i`: head `h` is `∑_{j ≤ i} p_ij v_j` with `v_j = W_V z_j`. -/
noncomputable def read (i : ℕ) : Heads H p :=
  fun h => ∑ j : Fin (i + 1), (A.weight eps α z i h j : ℂ) • A.WV (z j) h

/-- **The branch** `W_O a_i`, before the Peri-norm of `Router.step`. -/
noncomputable def branch (i : ℕ) : CSpace c :=
  A.WO (A.read eps α z i)

/-- **The score depends on the positions only through the offset `j − i`.** -/
theorem score_eq (i j : ℕ) (h : Fin H) :
    score (A.query eps α z i h) (A.key eps α z j h) =
      score (headRMS eps (A.WQ (z i) h))
        (turn (leropeFrequency 10000 α) ((j : ℝ) - i) (headRMS eps (A.WK (z j) h))) :=
  score_turn _ _ _ _ _

theorem weight_pos (i : ℕ) (h : Fin H) (j : Fin (i + 1)) : 0 < A.weight eps α z i h j := by
  unfold weight ssmaxWeight
  exact AttentionResiduals.depthWeight_pos _ _

theorem sum_weight (i : ℕ) (h : Fin H) : ∑ j, A.weight eps α z i h j = 1 := by
  unfold weight ssmaxWeight
  exact AttentionResiduals.depthWeight_sum _

/-- **Every logit at the position `i` is at most `ln(i + 1)`**, for all weights:
`|(ln N_i / p) Re Z_ij| ≤ ln N_i`. -/
theorem abs_logit_le (hp : 0 < p) (heps : 0 < eps) (i j : ℕ) (h : Fin H) :
    |1 / p * Real.log (i + 1) * score (A.query eps α z i h) (A.key eps α z j h)| ≤
      Real.log (i + 1) :=
  PhasorV2.abs_logit_le (abs_mul_score_le hp heps _ _ _ _ _)
    (le_add_of_nonneg_left (Nat.cast_nonneg i))

/-- **No key outweighs another by more than `(i + 1)²`**, for all weights. -/
theorem weight_le_sq_mul (hp : 0 < p) (heps : 0 < eps) (i : ℕ) (h : Fin H) (j k : Fin (i + 1)) :
    A.weight eps α z i h j ≤ ((i : ℝ) + 1) ^ 2 * A.weight eps α z i h k := by
  have hz := ssmaxWeight_le_sq_mul (fun l : Fin (i + 1) => abs_mul_score_le hp heps
    (leropeFrequency 10000 α) i l (A.WQ (z i) h) (A.WK (z l) h)) j k
  rw [Fintype.card_fin, Nat.cast_add, Nat.cast_one] at hz
  exact hz

/-- **The branch is causal**: at the position `i` it reads the stream at `j ≤ i` only. -/
theorem isCausal_branch : IsCausal (A.branch eps α) := by
  intro x y i hxy
  have hk : ∀ k : Fin (i + 1), x k = y k := fun k => hxy k (Nat.lt_succ_iff.1 k.2)
  have hr : A.read eps α x i = A.read eps α y i := by
    funext h
    simp only [read, weight, query, key, hk, hxy i le_rfl]
  rw [branch, branch, hr]

end Attention

/-- The main model: `c = 64`, four heads of `p = 16` bands, `ε = 2^{−23}`, the query at the
position `7` against the key at `3`. -/
example (A : Attention 64 4 16) (α : Fin 16 → ℝ) (z : ℕ → CSpace 64) (h : Fin 4) :=
  A.abs_logit_le (1 / 2 ^ 23) α z (by norm_num) (by norm_num) 7 3 h

/-- No key at the position `7` outweighs another by more than `64`. -/
example (A : Attention 64 4 16) (α : Fin 16 → ℝ) (z : ℕ → CSpace 64) (h : Fin 4)
    (j k : Fin 8) :=
  A.weight_le_sq_mul (1 / 2 ^ 23) α z (by norm_num) (by norm_num) 7 h j k

/-- **The stack of attention sublayers is causal**, here eight of them. -/
example (R : ℕ → Router 64) (A : ℕ → Attention 64 4 16) (α : Fin 16 → ℝ) :=
  isCausal_stack (1 / 2 ^ 23) (1 / Real.sqrt 8) R
    (fun l => (A l).isCausal_branch (1 / 2 ^ 23) α) 8

/-- **From the embedding of any tokens every stream has RMS at most `1 + 2n|α|`** after `n`
sublayers with the write scale `α`, for all weights and branches. -/
theorem rms_stack_embed_le {eps : ℝ} (heps : 0 ≤ eps) (hc : 0 < c) (α : ℝ) (R : ℕ → Router c)
    (F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c) (n : ℕ) {V : Type*} (E : V → CSpace c)
    (tok : ℕ → V) (i : ℕ) (s : Fin 4) :
    ‖stack eps α R F n (embed eps E tok) i s‖ / Real.sqrt c ≤ 1 + 2 * n * |α| :=
  rms_stack_le heps hc α R F n (fun j => E (tok j)) i s

/-- Eight attention sublayers on the main model's widths at `α = 1/√8`. -/
example {V : Type*} (E : V → CSpace 64) (tok : ℕ → V) (R : ℕ → Router 64)
    (A : ℕ → Attention 64 4 16) (α : Fin 16 → ℝ) (i : ℕ) (s : Fin 4) :=
  rms_stack_embed_le (eps := 1 / 2 ^ 23) (by norm_num) (by norm_num) (1 / Real.sqrt 8) R
    (fun l => (A l).branch (1 / 2 ^ 23) α) 8 E tok i s

end Transformer.PhasorV2
