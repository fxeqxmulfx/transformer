module

public import Transformer.Sandwich.Section3_Transformer
public import Transformer.Sandwich.Section5_Locality

/-!
# §3.1 The language model is causal

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §3.1, Eq. (1): the query `m` attends to the keys
`n ≤ m`. Whatever the positional biases, every head, every layer and the prediction at `m` depend
only on the inputs at the positions `n ≤ m` (`Model.isCausal_predict`); only these tokens can
contribute to the prediction (`IsCausal.contributes`).
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Transformer.BAM

variable {d k H v : ℕ}

/-- A head is causal. -/
theorem isCausal_headOut (A : Head d k) (b : ℕ → EReal) : IsCausal (headOut A b) := by
  intro x y m h
  have hs : ∀ n, n ≤ m → score A b x m n = score A b y m n := fun n hn => by
    simp only [score, h n hn, h m le_rfl]
  have hZ : ∑ i ∈ range (m + 1), weight (score A b x m i) =
      ∑ i ∈ range (m + 1), weight (score A b y m i) :=
    sum_congr rfl fun i hi => by rw [hs i (Nat.lt_succ_iff.1 (mem_range.1 hi))]
  refine sum_congr rfl fun n hn => ?_
  have hn' := Nat.lt_succ_iff.1 (mem_range.1 hn)
  simp only [attnWeight, hZ, hs n hn', h n hn']

/-- The attention of a layer is causal. -/
theorem Attn.isCausal_out (A : Attn d k H) (b : Fin H → ℕ → EReal) : IsCausal (A.out b) := by
  intro x y m h
  simp only [Attn.out]
  congr 1
  funext j
  rw [isCausal_headOut _ _ x y m h]

/-- A layer is causal. -/
theorem Layer.isCausal_apply (ε : ℝ) (P : Layer d k H) (b : Fin H → ℕ → EReal) :
    IsCausal (P.apply ε b) :=
  fun x y m h => congrArg (layerNorm ε P.gain P.bias) (P.toAttn.isCausal_out b x y m h)

/-- The first layers are causal. -/
theorem isCausal_hidden (ε : ℝ) (b : Fin H → ℕ → EReal) (Ps : List (Layer d k H)) :
    IsCausal (hidden ε b Ps) := by
  induction Ps with
  | nil => exact fun x y m h => h m le_rfl
  | cons P Ps ih => exact ih.comp (P.isCausal_apply ε b)

/-- The last layer's `o_m` is causal. -/
theorem Model.isCausal_out (M : Model d k H v) (b : Fin H → ℕ → EReal) : IsCausal (M.out b) :=
  (M.last.isCausal_out b).comp (isCausal_hidden M.eps b M.init)

/-- **The prediction is causal** (§3.1): the prediction at `m` depends only on the tokens
`n ≤ m`, whatever the positional biases. -/
theorem Model.isCausal_predict (M : Model d k H v) (b : Fin H → ℕ → EReal) :
    IsCausal (M.predict b) :=
  fun x y m h => congrArg (fun o => softmax fun j => ((M.V *ᵥ o) j : EReal))
    (M.isCausal_out b x y m h)

end Transformer.Sandwich
