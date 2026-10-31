module

public import Transformer.Rotate90.Power.Expressivity

/-!
# Every positive finite depth

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*,
depth extension; the loop in `python/src/lab/infrastructure/nn/peri_rotary.py`,
`PeriRotate90.forward`. A stack has one shared embedding and independent gate, value
and output matrices for each residual block. Each block retains its two RMSNorms;
input and final RMSNorm occur once, outside the loop.

Zero matrices make a block exactly the identity, even with positive epsilon. Thus
the explicit powered-division construction extends to every positive finite depth,
with arbitrary residual and hidden padding as before. This quantifies over finite
models. It does not assume that an infinite-width kernel or an infinite-depth limit
has the same function class or training dynamics as these models.
-/

@[expose] public section

namespace Transformer.Rotate90.Power

open Landscape.Neural

/-- The three ordinary matrices of one block, without duplicating the shared embedding. -/
abbrev BlockWeights (K H : Type*) :=
  (Channels K → H → ℝ) × (Channels K → H → ℝ) × (H → Channels K → ℝ)

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- One iteration of the source's residual loop, with both local normalizers. -/
noncomputable def block (eps : ℝ) (B : BlockWeights K H)
    (x : Channels K → ℝ) : Channels K → ℝ :=
  fun i => x i + rmsNorm eps
    (dense B.2.2 (fun j => swiglu (dense B.1 (rmsNorm eps x) j)
      (dense B.2.1 (rmsNorm eps x) j))) i

/-- The source loop after the initial normalization, before the final normalization. -/
noncomputable def stackState (eps : ℝ) (E : V → Channels K → ℝ)
    (blocks : List (BlockWeights K H)) (a b : V) : Channels K → ℝ :=
  blocks.foldl (fun x B => block eps B x) (rmsNorm eps (tiedInput E a b))

/-- Tied output scores for a finite stack; there is still only one embedding. -/
noncomputable def stackLogit (eps : ℝ) (E : V → Channels K → ℝ)
    (blocks : List (BlockWeights K H)) (a b q : V) : ℝ :=
  tiedReadout E (rmsNorm eps (stackState eps E blocks a b)) q

/-- Extract only the three branch matrices from the one-block parameter vector. -/
def blockOfWeights (w : GatedWeights V K H) : BlockWeights K H :=
  (gateMatrix w, valueMatrix w, outputMatrix w)

/-- A legal parameter choice for an additional residual block. -/
def zeroBlock : BlockWeights K H := (0, 0, 0)

/-- The finite-stack definition agrees with the existing current one-block model. -/
theorem stackState_singleton (eps : ℝ) (w : GatedWeights V K H) (a b : V) :
    stackState eps (gatedEmbedding w) [blockOfWeights w] a b = state eps w a b := by
  unfold stackState
  rw [List.foldl_cons, List.foldl_nil]
  rfl

/-- In particular its tied, fully normalized scores agree, not just its raw branch. -/
theorem stackLogit_singleton (eps : ℝ) (w : GatedWeights V K H) (a b q : V) :
    stackLogit eps (gatedEmbedding w) [blockOfWeights w] a b q = logit eps w a b q := by
  unfold stackLogit
  rw [stackState_singleton]
  rfl

/-- A zero output matrix has zero branch output; RMSNorm leaves that vector zero. -/
theorem block_zero (eps : ℝ) (x : Channels K → ℝ) :
    block eps (zeroBlock (K := K) (H := H)) x = x := by
  funext i
  simp [block, zeroBlock, dense, rmsNorm]

/-- Any finite number of zero blocks preserves the full residual state. -/
theorem fold_zero_blocks (eps : ℝ) (d : ℕ) (x : Channels K → ℝ) :
    (List.replicate d (zeroBlock (K := K) (H := H))).foldl
      (fun y B => block eps B y) x = x := by
  induction d with
  | zero => rfl
  | succ d ih =>
    rw [List.replicate_succ, List.foldl_cons, block_zero, ih]

/-- Appending zero blocks after the active block preserves every score exactly. -/
theorem stackLogit_padded (eps : ℝ) (w : GatedWeights V K H)
    (d : ℕ) (a b q : V) :
    stackLogit eps (gatedEmbedding w)
      (blockOfWeights w :: List.replicate d zeroBlock) a b q = logit eps w a b q := by
  have hstate : stackState eps (gatedEmbedding w)
      (blockOfWeights w :: List.replicate d zeroBlock) a b = state eps w a b := by
    unfold stackState
    rw [List.foldl_cons, fold_zero_blocks]
    rfl
  unfold stackLogit
  rw [hstate]
  rfl

variable {p : ℕ} [Fact p.Prime] {Z : Type*} [Fintype Z]

/-- **Every prime, exponent and positive finite depth.** The quantifiers permit the
weights to depend on the task and architecture but never on an operand pair. Depth
is d+1, width is 14+2*card K, and hidden size is 20+card Z. No infinite limit is used. -/
theorem exists_weights_any_depth {eps : ℝ} (heps : 0 < eps) (k d : ℕ) :
    ∃ E : ZMod p → Channels (PairIndex K) → ℝ,
      ∃ blocks : List (BlockWeights (PairIndex K) ((Fin 10 × Fin 2) ⊕ Z)),
        blocks.length = d + 1 ∧
          ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
            stackLogit eps E blocks a b q < stackLogit eps E blocks a b ((a / b) ^ k) := by
  obtain ⟨w, hw⟩ := exists_weights (p := p) (K := K) (Z := Z) heps k
  refine ⟨gatedEmbedding w, blockOfWeights w :: List.replicate d zeroBlock, ?_, ?_⟩
  · rw [List.length_cons, List.length_replicate]
  · intro a b hb q hq
    simpa only [stackLogit_padded] using hw a b hb q hq

/-- Width 16 and the source's hidden size 43 suffice at every positive finite depth. -/
theorem exists_weights_width16_any_depth {eps : ℝ} (heps : 0 < eps) (k d : ℕ) :
    ∃ E : ZMod p → Channels (PairIndex (Fin 1)) → ℝ,
      ∃ blocks : List (BlockWeights (PairIndex (Fin 1)) ((Fin 10 × Fin 2) ⊕ Fin 23)),
        blocks.length = d + 1 ∧
          ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
            stackLogit eps E blocks a b q < stackLogit eps E blocks a b ((a / b) ^ k) :=
  exists_weights_any_depth heps k d

example := stackState_singleton (1 / 8388608)
  (weights (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 2) 0 2
example := stackLogit_singleton (1 / 8388608)
  (weights (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 2) 0 2 0
example := block_zero (K := Fin 8) (H := Fin 43) (1 / 8388608)
  (fun i => (i.1.val + i.2.val : ℝ))
example := fold_zero_blocks (K := Fin 8) (H := Fin 43) (1 / 8388608) 100
  (fun i => (i.1.val + i.2.val : ℝ))
example := stackLogit_padded (1 / 8388608)
  (weights (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 2) 100 0 2 0

example := exists_weights_any_depth (p := 2) (K := Fin 0) (Z := Fin 0)
  (eps := 1 / 8388608) (by norm_num) 0 0
example := exists_weights_any_depth (p := 3) (K := Fin 100) (Z := Fin 1000)
  (eps := 1 / 8388608) (by norm_num) 5 100

section

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

example := exists_weights_width16_any_depth (p := 97)
  (eps := 1 / 8388608) (by norm_num) 2 0
example := exists_weights_width16_any_depth (p := 97)
  (eps := 1 / 8388608) (by norm_num) 3 3

end

end Transformer.Rotate90.Power
