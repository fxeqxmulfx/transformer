module

public import Transformer.Phasor.Forward
public import Transformer.Phasor.TwoCopies.Slots
public import Transformer.Perspective.Softmax

/-!
# Transporting tied embedding values before the Phasor readout

The construction follows “Transported embedding values for NV11” in `review/flat_spectrum.md`.
The first causal feature attends to the learned embedding table. At the answer
position the value at address `s` is the learned embedding at `s + a*b`.
The final RMS and tied readout act on their weighted sum. The results below
describe the forward map conditional on the embeddings and the attention.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Hidden answer value at modular offset `r`. -/
noncomputable def transportedValue (E : Fin p → EucSpace D) (w : Fin p → ℝ)
    (r : Fin p) : EucSpace D := ∑ s, w s • E (s + r)

/-- The ordinary tied, RMS-normalized readout of the new hidden value. -/
noncomputable def transportedLogit (E : Fin p → EucSpace D) (w : Fin p → ℝ)
    (eps : ℝ) (r c : Fin p) : ℝ :=
  ⟪E c, GPTMini.rmsNormEps eps (transportedValue E w r)⟫

/-- The first-token Phasor scores give the address distribution. The product
of the two input tokens selects the offset in the answer-position value table. -/
noncomputable def Model.valueAttentionLogit [NeZero p] (M : Model D d H F p) (eps scale : ℝ)
    (tokens : ℕ → Fin p) (L : ℕ) : Fin p → ℝ :=
  transportedLogit M.embedding (Perspective.softmaxWeight (M.logits eps scale tokens L 0))
    eps (tokens 0 * tokens 1)

/-- Weighted Gram column of the tied embedding (NV11, transported values). -/
theorem inner_transportedValue (E : Fin p → EucSpace D) (w : Fin p → ℝ) (r c : Fin p) :
    ⟪E c, transportedValue E w r⟫ = ∑ s, w s * ⟪E c, E (s + r)⟫ := by
  simp [transportedValue, inner_sum, real_inner_smul_right]

/-- With orthonormal codes the coefficient at the translated class is exactly
the attention mass at its original address (NV11, transported values). -/
theorem inner_transportedValue_orthonormal [NeZero p] {E : Fin p → EucSpace D}
    (hE : Orthonormal ℝ E) (w : Fin p → ℝ) (r t : Fin p) :
    ⟪E (t + r), transportedValue E w r⟫ = w t := by
  rw [inner_transportedValue]
  simp [orthonormal_iff_ite.mp hE, mul_ite]

/-- All 97 address codes fit in the model's width 128. -/
example (w : Fin 97 → ℝ) (a b : Fin 97) :
    let E := TwoCopies.coordCodes (d := 128) (p := 97) (by decide)
    ⟪E (a^3 + a*b), transportedValue E w (a*b)⟫ = w (a^3) :=
  inner_transportedValue_orthonormal (TwoCopies.orthonormal_coordCodes _) _ _ _

/-- With orthonormal embeddings the tied decision after the final RMS is
exactly the address decision, translated by the offset (NV11, transported values). -/
theorem transportedLogit_lt_iff [NeZero p] {E : Fin p → EucSpace D}
    (hE : Orthonormal ℝ E) (hD : 0 < D) {eps : ℝ} (heps : 0 < eps)
    (w : Fin p → ℝ) (r t u : Fin p) :
    transportedLogit E w eps r (t + r) < transportedLogit E w eps r (u + r) ↔ w t < w u := by
  have hd : (0 : ℝ) < D := Nat.cast_pos.mpr hD
  simp only [transportedLogit, GPTMini.rmsNormEps, real_inner_smul_right,
    inner_transportedValue_orthonormal hE]
  exact mul_lt_mul_iff_right₀ (by positivity)

/-- For every pair, ranking the cubic address ahead of a rival ranks the
NV11 answer ahead of that rival's translation at the experiment's width. -/
example (a b t : Fin 97) (w : Fin 97 → ℝ) :
    let E := TwoCopies.coordCodes (d := 128) (p := 97) (by decide)
    transportedLogit E w 1 (a*b) (t + a*b) < transportedLogit E w 1 (a*b) (a^3 + a*b) ↔
      w t < w (a^3) :=
  transportedLogit_lt_iff (TwoCopies.orthonormal_coordCodes _) (by decide) (by norm_num) _ _ _ _

/-- For arbitrary learned codes, a winning Gram gap `γ` at address `u` and a
worst adverse gap `B` elsewhere suffice when the mass at `u` gives a positive
net gap. RMS preserves that strict decision (NV11, transported values). -/
theorem transportedLogit_margin {E : Fin p → EucSpace D} {w : Fin p → ℝ}
    {eps γ B : ℝ} {r u c : Fin p} (hD : 0 < D) (heps : 0 < eps)
    (hw : ∀ s, 0 ≤ w s) (hsum : ∑ s, w s = 1)
    (hgood : γ ≤ ⟪E (u + r), E (u + r)⟫ - ⟪E c, E (u + r)⟫)
    (hother : ∀ s, s ≠ u → -B ≤ ⟪E (u + r), E (s + r)⟫ - ⟪E c, E (s + r)⟫)
    (hmass : 0 < w u * γ - (1 - w u) * B) :
    transportedLogit E w eps r c < transportedLogit E w eps r (u + r) := by
  have hbound := Finset.sum_le_sum (s := Finset.univ) fun (s : Fin p) _ =>
    mul_le_mul_of_nonneg_left (show (if s = u then γ + B else 0) - B ≤
      ⟪E (u + r), E (s + r)⟫ - ⟪E c, E (s + r)⟫ by
        by_cases hs : s = u
        · simpa [hs] using hgood
        · simpa [hs] using hother s hs) (hw s)
  simp only [mul_sub, mul_ite, mul_zero, Finset.sum_sub_distrib, Fintype.sum_ite_eq',
    ← Finset.sum_mul, hsum, one_mul] at hbound
  have hraw : ⟪E c, transportedValue E w r⟫ < ⟪E (u + r), transportedValue E w r⟫ := by
    rw [inner_transportedValue, inner_transportedValue]
    linarith [show w u * (γ + B) - B = w u * γ - (1 - w u) * B by ring]
  have hd : (0 : ℝ) < D := Nat.cast_pos.mpr hD
  simp only [transportedLogit, GPTMini.rmsNormEps, real_inner_smul_right]
  exact mul_lt_mul_of_pos_left hraw (by positivity)

/-- With mass 3/4 at address 0 and offset 1 the two-address witness selects
class 1. The adverse Gram gap is bounded by 1. -/
example :
    let E := TwoCopies.coords 2
    let w := fun s : Fin 2 => if s = 0 then (3/4 : ℝ) else 1/4
    transportedLogit E w 1 1 0 < transportedLogit E w 1 1 (0 + 1) := by
  dsimp only
  apply transportedLogit_margin (γ := 1) (B := 1) (by decide) (by norm_num)
  · intro s
    split <;> norm_num
  · norm_num [Fin.sum_univ_two]
  · simp only [orthonormal_iff_ite.mp (TwoCopies.orthonormal_coords 2)]
    norm_num
  · intro s _
    simp only [orthonormal_iff_ite.mp (TwoCopies.orthonormal_coords 2)]
    split <;> split <;> norm_num
  · norm_num

end Transformer.Phasor
