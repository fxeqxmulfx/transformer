module

public import Transformer.Rotate90.Products

/-!
# A solution of modular division for every prime in the current Rotate90 model

Source: `grokking_abmod/docs/rotate90-swiglu-model.md`, Model, and the user's
2026-10-31 request. `exists_division_weights` constructs a strict correct logit
maximum on EVERY a and every nonzero b, for EVERY prime p and positive epsilon.
The architecture is the actual one-block forward in `Defs`, with its shared
embedding, fixed quarter-turn, SwiGLU, residual and all four RMSNorm calls.

Four active channel pairs and fourteen active hidden units suffice. The generic
types K and Z add unused pairs and hidden units without dropping them from the
model or its RMSNorm denominator. Thus width 16 and hidden size 43, and width
128 and hidden size 341, are explicit instances. This is real-arithmetic
expressivity; no training trajectory or zero cross-entropy is asserted.
-/

@[expose] public section

namespace Transformer.Rotate90

open Landscape.Neural Grokking.FlatSpectrum

variable {p : ℕ} [Fact p.Prime] {K Z : Type*} [Fintype K] [Fintype Z]

/-- The actual normalized residual state, derived from the explicit matrices.
Both raw vectors have squared length four, so their normalizers agree exactly. -/
theorem state_divisionWeights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    state eps (divisionWeights (K := K) (Z := Z) eps χ) a b =
      fun i => inputScale (K := K) eps * (raw χ a b i + branchVector χ a b i) := by
  have hx : input eps (divisionWeights (K := K) (Z := Z) eps χ) a b =
      rmsNorm eps (raw χ a b) := by
    rw [input, embedding_divisionWeights, tiedInput_embedding]
  unfold state
  rw [hx, rmsNorm_twice_of_sum_sq eps _ (raw_sq χ a b), branch_divisionWeights heps,
    rmsNorm_of_sum_sq eps _ (branchVector_sq χ a hb), rmsNorm_of_sum_sq eps _ (raw_sq χ a b)]
  funext i
  ring

/-- The full logits are a positive common scale times one plus the character
inner product. The scale includes the genuine final RMSNorm, with all odd channels. -/
theorem logit_divisionWeights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    ∃ s : ℝ, 0 < s ∧ ∀ q, logit eps (divisionWeights (K := K) (Z := Z) eps χ) a b q =
      s * (1 + characterScore χ (a / b) q) := by
  let v := state eps (divisionWeights (K := K) (Z := Z) eps χ) a b
  let d := Real.sqrt (eps + meanSquare v)
  have hd : 0 < d := Real.sqrt_pos.mpr (rmsNorm_radicand_pos heps v)
  refine ⟨inputScale (K := K) eps / d, div_pos (inputScale_pos heps) hd, fun q => ?_⟩
  have hread : tiedReadout (embedding (K := K) χ) v q =
      inputScale (K := K) eps * (1 + characterScore χ (a / b) q) := by
    dsimp only [v]
    rw [state_divisionWeights heps χ a hb, readout_mul, readout_raw_add_branchVector χ a hb]
  rw [logit, embedding_divisionWeights, readout_rmsNorm]
  change tiedReadout (embedding (K := K) χ) v q / d = _
  rw [hread]
  ring

/-- The correct residue strictly beats every competitor, including answer zero.
This is exact classification by the current architecture's tied argmax. -/
theorem divisionWeights_correct {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (hχ : Function.Injective χ)
    (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) {q : ZMod p} (hq : q ≠ a / b) :
    logit eps (divisionWeights (K := K) (Z := Z) eps χ) a b q <
      logit eps (divisionWeights (K := K) (Z := Z) eps χ) a b (a / b) := by
  obtain ⟨s, hs, hlogit⟩ := logit_divisionWeights (K := K) (Z := Z) heps χ a hb
  rw [hlogit, hlogit, characterScore_self]
  exact mul_lt_mul_of_pos_left (by linarith [characterScore_lt_one χ hχ hq.symm]) hs

/-- **Existence for every prime.** A one-block Rotate90 with `8 + 2*card K`
channels and `14 + card Z` hidden units solves every equation `b*c = a`, b ≠ 0.
All matrices, all normalizations, the residual and the tied embedding are retained. -/
theorem exists_division_weights {eps : ℝ} (heps : 0 < eps) :
    ∃ w : GatedWeights (ZMod p) (PairIndex K) ((Fin 7 × Fin 2) ⊕ Z),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ a / b →
        logit eps w a b q < logit eps w a b (a / b) := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := p)
  exact ⟨divisionWeights eps χ, fun a _ hb _ hq => divisionWeights_correct heps χ hχ a hb hq⟩

/-- The width-16, hidden-43 source model is an instance at every prime. -/
theorem exists_division_width16 {eps : ℝ} (heps : 0 < eps) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 4)) ((Fin 7 × Fin 2) ⊕ Fin 29),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ a / b →
        logit eps w a b q < logit eps w a b (a / b) :=
  exists_division_weights heps

/-- Width eight with the source's default `H = (8W+1)//3 = 21` also suffices. -/
theorem exists_division_width8 {eps : ℝ} (heps : 0 < eps) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 0)) ((Fin 7 × Fin 2) ⊕ Fin 7),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ a / b →
        logit eps w a b q < logit eps w a b (a / b) :=
  exists_division_weights heps

/-- The source's width-128, hidden-341 default is an instance at every prime. -/
theorem exists_division_width128 {eps : ℝ} (heps : 0 < eps) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 60)) ((Fin 7 × Fin 2) ⊕ Fin 327),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ a / b →
        logit eps w a b q < logit eps w a b (a / b) :=
  exists_division_weights heps

/-- Actual parameter count for the documented width-16 instance: 16p + 2064. -/
theorem width16_parameter_count :
    Module.finrank ℝ
      (GatedWeights (ZMod p) (PairIndex (Fin 4)) ((Fin 7 × Fin 2) ⊕ Fin 29)) =
      16 * p + 2064 := by
  rw [finrank_gatedWeights]
  simp [PairIndex, ZMod.card, mul_comm]

example := state_divisionWeights (K := Fin 4) (Z := Fin 29) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 0 (b := 2) (by decide)

example := logit_divisionWeights (K := Fin 4) (Z := Fin 29) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 0 (b := 2) (by decide)

example : ∃ χ : MulChar (ZMod 3) ℂ, Function.Injective χ ∧
    logit (1 / 8388608) (divisionWeights (K := Fin 0) (Z := Fin 0) (1 / 8388608) χ) 0 1 1 <
      logit (1 / 8388608) (divisionWeights (K := Fin 0) (Z := Fin 0) (1 / 8388608) χ) 0 1 0 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 3)
  refine ⟨χ, hχ, ?_⟩
  have h := divisionWeights_correct (p := 3) (K := Fin 0) (Z := Fin 0)
    (eps := 1 / 8388608) (by norm_num) χ hχ 0 (b := 1) (by decide) (q := 1) (by norm_num)
  simpa using h

example := exists_division_weights (p := 2) (K := Fin 0) (Z := Fin 7)
  (eps := 1 / 8388608) (by norm_num)

example := exists_division_width8 (p := 2) (eps := 1 / 8388608) (by norm_num)

section

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

example := exists_division_width16 (p := 97) (eps := 1 / 8388608) (by norm_num)

example := exists_division_width128 (p := 97) (eps := 1 / 8388608) (by norm_num)

example : Module.finrank ℝ
    (GatedWeights (ZMod 97) (PairIndex (Fin 4)) ((Fin 7 × Fin 2) ⊕ Fin 29)) = 3616 := by
  rw [width16_parameter_count]

end

example : width (Fin 4) = 16 ∧ Fintype.card ((Fin 7 × Fin 2) ⊕ Fin 29) = 43 := by
  norm_num [width, Channels, PairIndex]

example : width (Fin 0) = 8 ∧ Fintype.card ((Fin 7 × Fin 2) ⊕ Fin 7) = 21 := by
  norm_num [width, Channels, PairIndex]

end Transformer.Rotate90
