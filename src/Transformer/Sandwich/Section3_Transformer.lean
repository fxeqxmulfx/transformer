module

public import Transformer.BAM.Section2_BayesianAttention
public import Mathlib.Data.Matrix.Mul

/-!
# §3.1 The transformer language model

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, "Dissecting Transformer Length Extrapolation via
the Lens of Receptive Field Analysis" (ACL 2023), §3.1. The TeX source is in
`papers/arXiv-2212.10356/`.

An `R`-layer model with `H` heads turns its inputs `e_m ∈ ℝ^d` into queries, keys and values
`q_m = W_q e_m`, `k_m = W_k e_m`, `v_m = W_v e_m ∈ ℝ^{d/H}` at each layer. The weights `a_{m,n}`
of Eq. (1) are the Softmax over the keys `n ≤ m` of the logits scaled by `1/√(d/H)`. The heads'
outputs `o_m^{(h)} = Σ_n a_{m,n} v_n` are concatenated and multiplied by `W_o`, and
`LayerNorm(o_m)` is the next layer's input. The last layer's `o_m` gives the distribution
`p = Softmax(V o_m)` of Eq. (3) over the next token.

* Positions are `0, 1, 2, …` for the paper's `1, 2, 3, …`, and a sequence is `ℕ → ℝ^d`: the
  model is causal, so the output at `m` reads only the first `m + 1` inputs, whatever the length
  `L`. `attnWeight_eq_softmax` is Eq. (1) over every length `L > m`, with the mask `−∞` after `m`.
* The head dimension is `k`; the paper's is `d/H`, and the scaling is `1/√k`.
* The positional scheme is a bias `b(m − n) ∈ [−∞, ∞)` of the distance, added to the scaled
  logit; a bias `−∞` masks the key. The plain model has none; ALiBi (Eq. 2) and windowed
  attention (§3.3) are in `Section3_Biases`, Sandwich (Eq. 6) in `Section6_Sandwich`.
* LayerNorm (Ba et al., 2016) has a gain `γ`, a bias `β` and `ε ≥ 0` under the root; the paper
  does not specify them.
* As §3.1 says, "After `R` layers of propagation, the last `o_m` is transformed by `V`": the last
  layer has no LayerNorm (`Model.last` is an `Attn`).
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Transformer.BAM

variable {d k H v : ℕ}

/-- A sequence of vectors in `ℝ^d`; position `m` is the paper's `m + 1`. -/
abbrev Embs (d : ℕ) := ℕ → Fin d → ℝ

/-- The parameters of one head (§3.1): `W_q, W_k, W_v ∈ ℝ^{k × d}`, `k = d/H` in the paper. -/
structure Head (d k : ℕ) where
  /-- The query matrix `W_q`. -/
  Wq : Matrix (Fin k) (Fin d) ℝ
  /-- The key matrix `W_k`. -/
  Wk : Matrix (Fin k) (Fin d) ℝ
  /-- The value matrix `W_v`. -/
  Wv : Matrix (Fin k) (Fin d) ℝ

/-- The attention of one layer: `H` heads and `W_o`, which multiplies the concatenation
`o^{(1)} ⊕ ⋯ ⊕ o^{(H)}`, indexed by `(head, coordinate)`. -/
structure Attn (d k H : ℕ) where
  /-- The heads. -/
  head : Fin H → Head d k
  /-- The output matrix `W_o`. -/
  Wo : Matrix (Fin d) (Fin H × Fin k) ℝ

/-- A layer: its attention, then a LayerNorm with gain `gain` and bias `bias`. -/
structure Layer (d k H : ℕ) extends Attn d k H where
  /-- LayerNorm's gain `γ`. -/
  gain : Fin d → ℝ
  /-- LayerNorm's bias `β`. -/
  bias : Fin d → ℝ

/-- The logit of the query at `m` for the key at `n ≤ m`, scaled as in Eq. (1), plus the
positional bias of the distance: `⟨q_m, k_n⟩/√k + b(m − n)`. -/
noncomputable def score (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m n : ℕ) : EReal :=
  (((A.Wq *ᵥ x m) ⬝ᵥ (A.Wk *ᵥ x n) / Real.sqrt k : ℝ) : EReal) + b (m - n)

/-- The weight `a_{m,n}` of Eq. (1) for `n ≤ m`: `exp(s_{m,n}) / Σ_{i ≤ m} exp(s_{m,i})`. -/
noncomputable def attnWeight (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m n : ℕ) : ℝ :=
  weight (score A b x m n) / ∑ i ∈ range (m + 1), weight (score A b x m i)

/-- A head's output `o_m^{(h)} = Σ_{n ≤ m} a_{m,n} v_n` (§3.1). -/
noncomputable def headOut (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m : ℕ) : Fin k → ℝ :=
  ∑ n ∈ range (m + 1), attnWeight A b x m n • (A.Wv *ᵥ x n)

/-- `o_m = W_o (o_m^{(1)} ⊕ ⋯ ⊕ o_m^{(H)})` (§3.1), head `h` with the bias `b h`. -/
noncomputable def Attn.out (A : Attn d k H) (b : Fin H → ℕ → EReal) (x : Embs d) (m : ℕ) :
    Fin d → ℝ :=
  A.Wo *ᵥ fun j => headOut (A.head j.1) (b j.1) x m j.2

/-- LayerNorm's mean `μ = (1/d) Σ_i o_i`. -/
noncomputable def lnMean (o : Fin d → ℝ) : ℝ := (∑ i, o i) / d

/-- LayerNorm's variance `σ² = (1/d) Σ_i (o_i − μ)²`. -/
noncomputable def lnVar (o : Fin d → ℝ) : ℝ := (∑ i, (o i - lnMean o) ^ 2) / d

/-- LayerNorm (Ba et al., 2016): `γ_i (o_i − μ)/√(σ² + ε) + β_i`. -/
noncomputable def layerNorm (ε : ℝ) (γ β o : Fin d → ℝ) : Fin d → ℝ :=
  fun i => γ i * ((o i - lnMean o) / Real.sqrt (lnVar o + ε)) + β i

/-- One layer: the next layer's input `LayerNorm(o_m)` (§3.1). -/
noncomputable def Layer.apply (ε : ℝ) (P : Layer d k H) (b : Fin H → ℕ → EReal) (x : Embs d) :
    Embs d :=
  fun m => layerNorm ε P.gain P.bias (P.toAttn.out b x m)

/-- The input of the layer after `Ps`: `x^{(0)} = e`, `x^{(r)} = LayerNorm(o^{(r)})`. -/
noncomputable def hidden (ε : ℝ) (b : Fin H → ℕ → EReal) : List (Layer d k H) → Embs d → Embs d
  | [], x => x
  | P :: Ps, x => hidden ε b Ps (P.apply ε b x)

/-- A language model of §3.1 with `R = init.length + 1` layers: the layers `init`, each followed
by its LayerNorm, the last attention `last`, LayerNorm's `ε`, and `V ∈ ℝ^{v × d}`. -/
structure Model (d k H v : ℕ) where
  /-- The first `R − 1` layers. -/
  init : List (Layer d k H)
  /-- The last layer's attention. -/
  last : Attn d k H
  /-- LayerNorm's `ε`. -/
  eps : ℝ
  /-- The output matrix `V` of Eq. (3). -/
  V : Matrix (Fin v) (Fin d) ℝ

namespace Model

/-- The number of layers `R`. -/
def depth (M : Model d k H v) : ℕ := M.init.length + 1

/-- The last layer's `o_m` for the inputs `e`. -/
noncomputable def out (M : Model d k H v) (b : Fin H → ℕ → EReal) (e : Embs d) : Embs d :=
  M.last.out b (hidden M.eps b M.init e)

/-- **Eq. (3)**: the distribution `p = Softmax(V o_m)` over the token after position `m`. -/
noncomputable def predict (M : Model d k H v) (b : Fin H → ℕ → EReal) (e : Embs d) (m : ℕ) :
    Fin v → ℝ :=
  softmax fun j => ((M.V *ᵥ M.out b e m) j : EReal)

end Model

/-- **Eq. (1).** For every length `L > m`, the weight `a_{m,n}` of a key `n ≤ m` is the Softmax
over all `L` keys of the scaled logits, the keys after `m` masked by `−∞`. -/
theorem attnWeight_eq_softmax (A : Head d k) (b : ℕ → EReal) (x : Embs d) {L m : ℕ} (hm : m < L)
    (n : Fin L) (hn : (n : ℕ) ≤ m) :
    attnWeight A b x m n =
      softmax (fun i : Fin L => if (i : ℕ) ≤ m then score A b x m i else ⊥) n := by
  have hZ : ∑ i : Fin L, weight (if (i : ℕ) ≤ m then score A b x m i else ⊥) =
      ∑ i ∈ range (m + 1), weight (score A b x m i) := by
    rw [Fin.sum_univ_eq_sum_range (fun i => weight (if i ≤ m then score A b x m i else ⊥)) L]
    simp only [apply_ite weight, weight_bot]
    rw [← Finset.sum_filter]
    congr 1
    ext i
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  simp only [softmax, partition, ite_eq_left hn, attnWeight, hZ]

/-- The hypotheses of `attnWeight_eq_softmax` hold for the key `0` of the query `0` of a length
`1`, with no bias. -/
example (A : Head d k) (x : Embs d) :
    attnWeight A (fun _ => 0) x 0 ((0 : Fin 1) : ℕ) =
      softmax (fun i : Fin 1 => if (i : ℕ) ≤ 0 then score A (fun _ => 0) x 0 i else ⊥) 0 :=
  attnWeight_eq_softmax A (fun _ => 0) x zero_lt_one 0 le_rfl

/-- The weights `a_{m,n}`, `n ≤ m`, sum to `1` when the bias at the distance `0` is finite. -/
theorem sum_attnWeight (A : Head d k) {b : ℕ → EReal} (h₁ : b 0 ≠ ⊥) (h₂ : b 0 ≠ ⊤) (x : Embs d)
    (m : ℕ) : ∑ n ∈ range (m + 1), attnWeight A b x m n = 1 := by
  have hpos : 0 < ∑ i ∈ range (m + 1), weight (score A b x m i) := by
    refine lt_of_lt_of_le ?_ (single_le_sum (fun i _ => weight_nonneg _) (self_mem_range_succ m))
    rw [score, Nat.sub_self, ← EReal.coe_toReal h₂ h₁, ← EReal.coe_add, weight_coe]
    exact Real.exp_pos _
  simp only [attnWeight, ← sum_div]
  exact div_self hpos.ne'

/-- The hypotheses of `sum_attnWeight` hold for the bias `0`. -/
example (A : Head d k) (x : Embs d) (m : ℕ) :
    ∑ n ∈ range (m + 1), attnWeight A (fun _ => 0) x m n = 1 :=
  sum_attnWeight A (b := fun _ => 0) (by simp) (by simp) x m

/-- The weights `a_{m,n}` are nonnegative. -/
theorem attnWeight_nonneg (A : Head d k) (b : ℕ → EReal) (x : Embs d) (m n : ℕ) :
    0 ≤ attnWeight A b x m n :=
  div_nonneg (weight_nonneg _) (sum_nonneg fun _ _ => weight_nonneg _)

end Transformer.Sandwich
