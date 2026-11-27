module

public import Transformer.Grokking.FiniteRecovery.Examples
public import Transformer.Grokking.FiniteRecovery.ArgandPhase
public import Transformer.Grokking.FiniteRecovery.ProductCapacity
public import Transformer.Grokking.FlatSpectrum.ProductOneLayer
public import Transformer.Grokking.FlatSpectrum.SharedLayer

/-! Argand's tied readout and the interaction across a word cut impose
different rank bounds. The product results apply when the indicated
separation or product-network hypotheses hold (Flat Spectrum, NV1, NV2,
NV7 and LP1). -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Matrix Transformer.Grokking.FlatSpectrum
open scoped BigOperators InnerProductSpace

namespace Argand

variable {A : Type*} {T H d layers : ℕ}

/-- Across words and output tokens, the real logit table factors through the
real and imaginary coordinates of the tied embedding. This holds at every
depth and for any setting of attention, gates, and FFN parameters. -/
theorem Model.rank_logits_le [Fintype A] (model : Model A H d layers)
    (repeats : ℕ) (i : Fin T) :
    (Matrix.of (fun word token => model.logits repeats word i token)).rank ≤ 2 * (H * d) := by
  let X : Matrix (Word A T) (Fin (H * d) ⊕ Fin (H * d)) ℝ := Matrix.of fun word =>
    let z := model.hidden repeats word i
    Sum.elim (fun j => (z j).re / 2) (fun j => (z j).im / 2)
  let E : Matrix (Fin (H * d) ⊕ Fin (H * d)) A ℝ := Matrix.of fun j token =>
    j.elim (fun k => (model.embedding token k).re) (fun k => (model.embedding token k).im)
  have hfactor : Matrix.of (fun word token => model.logits repeats word i token) = X * E := by
    ext word token
    simp only [Matrix.of_apply, Model.logits, Model.readout, PiLp.inner_apply, RCLike.inner_apply,
      Complex.re_sum, Matrix.mul_apply, X, E, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
      Complex.mul_re, Complex.conj_re, Complex.conj_im, mul_neg, sub_neg_eq_add,
      Finset.sum_add_distrib, add_div]
    congr 1 <;> rw [Finset.sum_div] <;> apply Finset.sum_congr rfl <;> intro j _ <;> ring
  rw [hfactor]
  calc
    (X * E).rank ≤ Fintype.card (Fin (H * d) ⊕ Fin (H * d)) :=
      (rank_mul_le_left X E).trans X.rank_le_card_width
    _ = 2 * (H * d) := by simp [two_mul]

example (model : Model (Fin 3) 4 16 2) :
    (Matrix.of (fun word token => model.logits 3 (word : Word (Fin 3) 8) 7 token)).rank ≤ 128 :=
  model.rank_logits_le 3 7

end Argand

variable {p k T : ℕ} [NeZero p]

/-- An exact factorization of an invertibly coupled phase table across the
specified cut must have at least `p^k` channels (NV1 and NV2). -/
theorem width_le_of_bilinear_factor {D : Type*} [Fintype D]
    (cut : Word (ZMod p) k × Word (ZMod p) k ≃ Word (ZMod p) T)
    (g h : Word (ZMod p) k → ZMod p) {M : Matrix (Fin k) (Fin k) (ZMod p)}
    (hM : IsUnit M) (L : Matrix (Word (ZMod p) k) D ℂ) (R : Matrix D (Word (ZMod p) k) ℂ)
    (hfactor : targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M) = L * R) :
    p ^ k ≤ Fintype.card D := by
  have hrank := (rank_mul_le_left L R).trans L.rank_le_card_width
  rwa [← hfactor, targetKernel_bilinearTask, rank_bilinearKernel g h hM, Fintype.card_fin] at hrank

/-- If the Argand phase table factors across the cut through the complex
stream width, NV2 bounds the error even before the labels are fitted. -/
theorem Argand.phase_error_of_embedding_factor {H d layers : ℕ}
    (model : Argand.Model (ZMod p) H d layers) (repeats : ℕ)
    (cut : Word (ZMod p) k × Word (ZMod p) k ≃ Word (ZMod p) (T + 1))
    (g h : Word (ZMod p) k → ZMod p) {M : Matrix (Fin k) (Fin k) (ZMod p)} (hM : IsUnit M)
    (L : Matrix (Word (ZMod p) k) (Fin (H * d)) ℂ)
    (R : Matrix (Fin (H * d)) (Word (ZMod p) k) ℂ)
    (hfactor : model.phaseKernel repeats cut ZMod.stdAddChar = L * R) :
    (p : ℝ) ^ k * ((p ^ k : ℕ) - H * d) ≤
      frobeniusSq (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M) -
        model.phaseKernel repeats cut ZMod.stdAddChar) := by
  rw [hfactor, targetKernel_bilinearTask]
  have hflat := isFlat_bilinearKernel p g h (vecMul_injective_of_isUnit hM)
  simpa [card_word, ZMod.card] using
    hflat.mul_sub_card_le_frobeniusSq_sub (pow_pos (Nat.cast_pos.mpr (NeZero.pos p)) _) L R

example {D : Type*} [Fintype D]
    (L : Matrix (Word (ZMod 2) 4) D ℂ) (R : Matrix D (Word (ZMod 2) 4) ℂ)
    (hfactor : targetKernel (appendCut (ZMod 2) 4 4) ZMod.stdAddChar
      (bilinearTask (appendCut (ZMod 2) 4 4) 0 0 1) = L * R) :
    16 ≤ Fintype.card D :=
  width_le_of_bilinear_factor _ _ _ isUnit_one L R hfactor

example (model : Argand.Model (ZMod 2) 1 1 1)
    (L : Matrix (Word (ZMod 2) 1) (Fin (1 * 1)) ℂ)
    (R : Matrix (Fin (1 * 1)) (Word (ZMod 2) 1) ℂ)
    (hfactor : model.phaseKernel 1 (appendCut (ZMod 2) 1 1) ZMod.stdAddChar = L * R) :=
  Argand.phase_error_of_embedding_factor model 1 (appendCut (ZMod 2) 1 1) 0 0 isUnit_one L R hfactor

/-- If the readout after one product layer on a separated state realizes the
phase table, its hidden width pays the NV7 rank bound. For Argand's FFN
dimensions one takes `m = 4 * (H * d)`. -/
theorem width_of_product_readout {ι : Type} {m : ℕ}
    (cut : Word (ZMod p) k × Word (ZMod p) k ≃ Word (ZMod p) T)
    (g h : Word (ZMod p) k → ZMod p) {M : Matrix (Fin k) (Fin k) (ZMod p)} (hM : IsUnit M)
    {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F)
    (e f : Word (ZMod p) k → ι → ℂ) (read : (ι → ℂ) →ₗ[ℂ] ℂ)
    (hread : ∀ u v, read (F (e u + f v)) =
      targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M) u v) :
    p ^ k ≤ 2 * m + 2 := by
  simp only [targetKernel_bilinearTask] at hread
  simpa only [Fintype.card_fin] using pow_le_of_productLayer_bilinear hF e f read g h hM hread

example {ι : Type} {m : ℕ} {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F)
    (e f : Word (ZMod 2) 8 → ι → ℂ) (read : (ι → ℂ) →ₗ[ℂ] ℂ)
    (hread : ∀ u v, read (F (e u + f v)) =
      targetKernel (appendCut (ZMod 2) 8 8) ZMod.stdAddChar
        (bilinearTask (appendCut (ZMod 2) 8 8) 0 0 1) u v) :
    256 ≤ 2 * m + 2 :=
  width_of_product_readout _ _ _ isUnit_one hF e f read hread

/-- With one shared token embedding and linear transport into the two
parts of one product layer, NV9 gives an additional embedding-width bound.
The pre-product state is formed from the two transported embeddings and a
fixed vector. -/
theorem embedding_width_of_shared_product_readout {ι : Type} {H d m : ℕ}
    (cut : Word (ZMod p) k × Word (ZMod p) k ≃ Word (ZMod p) T)
    (g h : Word (ZMod p) k → ZMod p) {M : Matrix (Fin k) (Fin k) (ZMod p)} (hM : IsUnit M)
    {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F)
    (E : Matrix (ZMod p) (Fin (H * d)) ℂ)
    (A B : (Fin k × Fin (H * d) → ℂ) →ₗ[ℂ] ι → ℂ) (z : ι → ℂ)
    (read : (ι → ℂ) →ₗ[ℂ] ℂ)
    (hread : ∀ u v, read (F (A (copiesEmbedding k E u) + B (copiesEmbedding k E v) + z)) =
      targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M) u v) :
    p ^ k ≤ k * (H * d) + 3 := by
  by_contra! hsmall
  apply hF.ne_bilinearKernel E A B z read g h hM (by simpa only [Fintype.card_fin] using hsmall)
  ext u v
  rw [of_apply, hread, targetKernel_bilinearTask]

example {ι : Type} {m : ℕ} {F : (ι → ℂ) → ι → ℂ} (hF : IsProductLayer m F)
    (E : Matrix (ZMod 2) (Fin (4 * 16)) ℂ)
    (A B : (Fin 16 × Fin (4 * 16) → ℂ) →ₗ[ℂ] ι → ℂ) (z : ι → ℂ)
    (read : (ι → ℂ) →ₗ[ℂ] ℂ) :
    (of fun u v => read (F (A (copiesEmbedding 16 E u) + B (copiesEmbedding 16 E v) + z))) ≠
      bilinearKernel 2 0 0 1 :=
  hF.ne_bilinearKernel E A B z read 0 0 isUnit_one (by decide)

end Transformer.Grokking.FiniteRecovery
