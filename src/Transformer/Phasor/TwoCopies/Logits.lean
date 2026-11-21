module

public import Transformer.Phasor.TwoCopies.Forward

/-!
# Two copies: the states and the logits of the construction

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

From layer `2` on the layers of `twoModel` are zero (`states_add_two`), so at the position `3`
every depth `L ≥ 2` holds the state that layer `1` leaves (`states_twoModel`): the bias, the four
operands, the planes of the copies and `A ψ(w₀ + w₁)` on the output plane. The logit of `v` is a
positive `κ` times the readout of that state against `e(v)` (`logits_twoModel`, from
`sum_embC_mul_stateC`):

    κ (κ_emb (b² + [v = t₃] + cos(θ_v − θ_{t₃})) + A cos(θ_v − θ_{w₀+w₁})).

The first term is the embedding of the own token, which the residual stream keeps; the second is
the answer.
-/

@[expose] public section

open Finset Real
open Transformer.ComplexFFN
open scoped RealInnerProductSpace

namespace Transformer.Phasor.TwoCopies

variable {D d F p : ℕ} {e : Slot p → EucSpace D} {f : Fin p → EucSpace d}
  {ι₀ : Fin 2 × Fin p ↪ Fin F} {ι₁ : Unit ↪ Fin F} {q : Fin 2 → Fin p → Fin p → Fin p}

/-- The states of layer `0`: the normalized embeddings. -/
theorem states_zero (he : Orthonormal ℝ e) (eps b g s : ℝ) (tokens : ℕ → Fin p) (j : ℕ) :
    (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens 0 j =
      vec e (kEmb D eps b • embC b (tokens j)) := by
  change GPTMini.rmsNormEps eps (vec e (embC b (tokens j))) = _
  rw [rmsNormEps_vec he, sum_embC_sq]; rfl

/-- Main's widths, `D = 128`, `d = 32` and `F = 171`, with real-linear projections, `ε = 2⁻²³`,
`α = 1/√8` and `p = 29`. -/
example (ι₀ : Fin 2 × Fin 29 ↪ Fin 171) (ι₁ : Unit ↪ Fin 171)
    (q : Fin 2 → Fin 29 → Fin 29 → Fin 29) (tokens : ℕ → Fin 29) :=
  states_zero (f := coordCodes (by norm_num : 29 ≤ 32)) (ι₀ := ι₀) (ι₁ := ι₁) (q := q)
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) (1 / 8388608) 1 1
    (1 / Real.sqrt 8) tokens 3

/-- Layers `2, 3, …` change nothing. -/
theorem states_add_three (eps b g s : ℝ) (tokens : ℕ → Fin p) (l i : ℕ) :
    (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens (l + 3) i =
      (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens (l + 2) i := by
  change oneStep eps s (plainFFN (FFN.zero D F).gate (FFN.zero D F).value (FFN.zero D F).outRe
    (FFN.zero D F).outIm) (oneStep eps s (fun _ => (Attention.zero D d 4).output
      ((Attention.zero D d 4).oneHeads eps (dftAlpha d)
        ((twoModel e f ι₀ ι₁ q eps b g).states eps s tokens (l + 2)) i))
      ((twoModel e f ι₀ ι₁ q eps b g).states eps s tokens (l + 2) i)) = _
  rw [oneStep_attention_zero, oneStep_ffn_zero]

theorem states_add_two (eps b g s : ℝ) (tokens : ℕ → Fin p) (l i : ℕ) :
    (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens (l + 2) i =
      (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens 2 i := by
  induction l with
  | zero => rfl
  | succ l ih => exact (states_add_three eps b g s tokens l i).trans ih

/-- **The state at the position `3`** from layer `2` on. -/
theorem states_twoModel (he : Orthonormal ℝ e) (hf : Orthonormal ℝ f) (hd : 4 ≤ d) (hD : 0 < D)
    {eps b : ℝ} (heps : 0 ≤ eps) (hb : 0 < b) (s g : ℝ) (tokens : ℕ → Fin p) {L : ℕ}
    (hL : 2 ≤ L) :
    (twoModel e f ι₀ ι₁ q eps b g).states eps s tokens L 3 =
      vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens) (ampPlanes D eps b s)
        (copyVals q tokens) (ampOut D eps b s g) (copyVals q tokens 0 + copyVals q tokens 1)) := by
  obtain ⟨l, rfl⟩ : ∃ l, L = l + 2 := ⟨L - 2, by omega⟩
  rw [states_add_two]
  exact apply_layer1 he s g tokens ((twoModel e f ι₀ ι₁ q eps b g).states eps s tokens 1)
    (apply_layer0 he hf hd hD heps hb s tokens
      ((twoModel e f ι₀ ι₁ q eps b g).states eps s tokens 0) (states_zero he eps b g s tokens))

/-- The same sizes, at the depth `4`. -/
example (ι₀ : Fin 2 × Fin 29 ↪ Fin 171) (ι₁ : Unit ↪ Fin 171)
    (q : Fin 2 → Fin 29 → Fin 29 → Fin 29) (tokens : ℕ → Fin 29) :=
  states_twoModel (ι₀ := ι₀) (ι₁ := ι₁) (q := q)
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128))
    (orthonormal_coordCodes (by norm_num : 29 ≤ 32)) (by norm_num) (by norm_num)
    (by norm_num : (0 : ℝ) ≤ 1 / 8388608) one_pos (1 / Real.sqrt 8) 1 tokens (L := 4) (by norm_num)

/-- **The logits at the position `3`** from layer `2` on: a positive `κ` times the readout of the
state against `e(v)`, the own token's term plus the answer's. -/
theorem logits_twoModel (he : Orthonormal ℝ e) (hf : Orthonormal ℝ f) (hd : 4 ≤ d) (hD : 0 < D)
    {eps b : ℝ} (heps : 0 ≤ eps) (hb : 0 < b) (s g : ℝ) (tokens : ℕ → Fin p) {L : ℕ}
    (hL : 2 ≤ L) :
    ∃ κ > 0, ∀ v, (twoModel e f ι₀ ι₁ q eps b g).logits eps s tokens L 3 v =
      κ * (kEmb D eps b * (b ^ 2 + (if v = toks tokens 3 then 1 else 0) +
          cos (angle p v - angle p (toks tokens 3))) +
        ampOut D eps b s g * cos (angle p v - angle p (copyVals q tokens 0 + copyVals q tokens 1)))
    := by
  refine ⟨rmsScale D eps (∑ x, stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens)
    (ampPlanes D eps b s) (copyVals q tokens) (ampOut D eps b s g)
      (copyVals q tokens 0 + copyVals q tokens 1) x ^ 2), rmsScale_pos hD heps ?_, fun v => ?_⟩
  · exact sum_pos' (fun x _ => sq_nonneg _)
      ⟨Sum.inl (), mem_univ _, pow_pos (mul_pos (kEmb_pos (b := b) hD heps) hb) 2⟩
  · rw [OneModel.logits, states_twoModel he hf hd hD heps hb s g tokens hL, rmsNormEps_vec he,
      smul_stateC]
    change ⟪vec e (embC b v), vec e _⟫ = _
    rw [inner_vec_vec he, sum_embC_mul_stateC]
    ring

/-- The same sizes. -/
example (ι₀ : Fin 2 × Fin 29 ↪ Fin 171) (ι₁ : Unit ↪ Fin 171)
    (q : Fin 2 → Fin 29 → Fin 29 → Fin 29) (tokens : ℕ → Fin 29) :=
  logits_twoModel (ι₀ := ι₀) (ι₁ := ι₁) (q := q)
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128))
    (orthonormal_coordCodes (by norm_num : 29 ≤ 32)) (by norm_num) (by norm_num)
    (by norm_num : (0 : ℝ) ≤ 1 / 8388608) one_pos (1 / Real.sqrt 8) 1 tokens (L := 4) (by norm_num)

end Transformer.Phasor.TwoCopies
