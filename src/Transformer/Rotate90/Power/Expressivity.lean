module

public import Transformer.Rotate90.Power.Products

/-!
# The actual Rotate90 model solves every powered-division task

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*,
expressivity theorem. The finite model is `Rotate90.logit`, with a tied embedding,
fixed quarter-turn, genuine SwiGLU, residual and all four positive-epsilon RMSNorms.
For each prime p and each natural exponent k, explicit weights give the unique correct
logit maximum for every numerator and every nonzero denominator, including zero targets
and k=0. The weights may depend on p and k, never on the queried operand pair.

Fourteen active residual coordinates and twenty active hidden units suffice. Extra
pairs and hidden units are universally quantified, so this is an unbounded family of
finite widths. Width 16 with the source default H=43 is a concrete instance. The
statement is representability, independent of initialization and optimizer convergence.
-/

@[expose] public section

namespace Transformer.Rotate90.Power

open Landscape.Neural Grokking.FlatSpectrum

variable {p : ℕ} [Fact p.Prime] {K Z : Type*} [Fintype K] [Fintype Z]

/-- The raw input and constructed branch have equal squared length six, so their
RMSNorm scales match exactly despite epsilon. This retains the actual residual path. -/
theorem state_weights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    state eps (weights (K := K) (Z := Z) eps χ k) a b =
      fun i => inputScale (K := K) eps * (raw χ k a b i + branchVector χ k a b i) := by
  have hx : input eps (weights (K := K) (Z := Z) eps χ k) a b =
      rmsNorm eps (raw χ k a b) := by
    rw [input, embedding_weights, tiedInput_embedding]
  unfold state
  rw [hx, rmsNorm_twice_of_sum_sq eps _ (raw_sq χ k a b), branch_weights heps χ k a hb,
    rmsNorm_of_sum_sq eps _ (branchVector_sq χ k a b), rmsNorm_of_sum_sq eps _ (raw_sq χ k a b)]
  funext i
  ring

/-- The full tied logits are a positive common scale times one plus the faithful
answer-code score. The scale includes the final RMSNorm and every odd coordinate. -/
theorem logit_weights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    ∃ s : ℝ, 0 < s ∧ ∀ q, logit eps (weights (K := K) (Z := Z) eps χ k) a b q =
      s * (1 + characterScore χ ((a / b) ^ k) q) := by
  let v := state eps (weights (K := K) (Z := Z) eps χ k) a b
  let d := Real.sqrt (eps + meanSquare v)
  have hd : 0 < d := Real.sqrt_pos.mpr (rmsNorm_radicand_pos heps v)
  refine ⟨inputScale (K := K) eps / d, div_pos (inputScale_pos heps) hd, fun q => ?_⟩
  have hread : tiedReadout (embedding (K := K) χ k) v q =
      inputScale (K := K) eps * (1 + characterScore χ ((a / b) ^ k) q) := by
    dsimp only [v]
    rw [state_weights heps χ k a hb, readout_mul, readout_raw_add_branchVector]
  rw [logit, embedding_weights, readout_rmsNorm]
  change tiedReadout (embedding (K := K) χ k) v q / d = _
  rw [hread]
  ring

/-- **Exact classification by actual model weights.** The correct powered quotient
strictly beats every competing residue, including zero and unattainable nonzero labels. -/
theorem weights_correct {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (hχ : Function.Injective χ) (k : ℕ)
    (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) {q : ZMod p} (hq : q ≠ (a / b) ^ k) :
    logit eps (weights (K := K) (Z := Z) eps χ k) a b q <
      logit eps (weights (K := K) (Z := Z) eps χ k) a b ((a / b) ^ k) := by
  obtain ⟨s, hs, hlogit⟩ := logit_weights (K := K) (Z := Z) heps χ k a hb
  rw [hlogit, hlogit, characterScore_self]
  exact mul_lt_mul_of_pos_left (by linarith [characterScore_lt_one χ hχ hq.symm]) hs

/-- **Every prime and every natural exponent, in the actual current architecture.**
Width is 14+2*card K and hidden size 20+card Z. All finite padding sizes are allowed.
The single chosen weight vector solves the entire table, not just observed cells. -/
theorem exists_weights {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ∃ w : GatedWeights (ZMod p) (PairIndex K) ((Fin 10 × Fin 2) ⊕ Z),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        logit eps w a b q < logit eps w a b ((a / b) ^ k) := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := p)
  exact ⟨weights eps χ k, fun a _ hb _ hq => weights_correct heps χ hχ k a hb hq⟩

/-- Width 16 and the source default H=(8W+1)//3=43 suffice for every powered task. -/
theorem exists_weights_width16 {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 1)) ((Fin 10 × Fin 2) ⊕ Fin 23),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        logit eps w a b q < logit eps w a b ((a / b) ^ k) :=
  exists_weights heps k

/-- Even width 14 with its source default H=37 contains the construction. This is
a sufficient size; no lower bound or claim of minimality is asserted. -/
theorem exists_weights_width14 {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 0)) ((Fin 10 × Fin 2) ⊕ Fin 17),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        logit eps w a b q < logit eps w a b ((a / b) ^ k) :=
  exists_weights heps k

/-- Width 128 with the source default H=341 is another instance. -/
theorem exists_weights_width128 {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ∃ w : GatedWeights (ZMod p) (PairIndex (Fin 57)) ((Fin 10 × Fin 2) ⊕ Fin 321),
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        logit eps w a b q < logit eps w a b ((a / b) ^ k) :=
  exists_weights heps k

/-- The width-16 witness has the source parameter count 16p+2064, independent of k. -/
theorem width16_parameter_count :
    Module.finrank ℝ
      (GatedWeights (ZMod p) (PairIndex (Fin 1)) ((Fin 10 × Fin 2) ⊕ Fin 23)) =
      16 * p + 2064 := by
  rw [finrank_gatedWeights]
  simp [PairIndex, ZMod.card, mul_comm]

example := state_weights (K := Fin 1) (Z := Fin 23) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 0 0 (b := 2) (by decide)
example := logit_weights (K := Fin 1) (Z := Fin 23) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 3 0 (b := 2) (by decide)

example : ∃ χ : MulChar (ZMod 3) ℂ, Function.Injective χ ∧
    logit (1 / 8388608) (weights (K := Fin 1) (Z := Fin 23) (1 / 8388608) χ 0) 0 2 0 <
      logit (1 / 8388608) (weights (K := Fin 1) (Z := Fin 23) (1 / 8388608) χ 0) 0 2 1 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 3)
  refine ⟨χ, hχ, ?_⟩
  have h := weights_correct (p := 3) (K := Fin 1) (Z := Fin 23)
    (eps := 1 / 8388608) (by norm_num) χ hχ 0 0 (b := 2) (by decide) (q := 0) (by norm_num)
  simpa using h

example := exists_weights (p := 2) (K := Fin 0) (Z := Fin 0)
  (eps := 1 / 8388608) (by norm_num) 0
example := exists_weights_width14 (p := 3) (eps := 1 / 8388608) (by norm_num) 5
example := exists_weights (p := 3) (K := Fin 100) (Z := Fin 1000)
  (eps := 1 / 8388608) (by norm_num) 5

section

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

example := exists_weights_width16 (p := 97) (eps := 1 / 8388608) (by norm_num) 2
example := exists_weights_width16 (p := 97) (eps := 1 / 8388608) (by norm_num) 3
example := exists_weights_width16 (p := 97) (eps := 1 / 8388608) (by norm_num) 5
example := exists_weights_width128 (p := 97) (eps := 1 / 8388608) (by norm_num) 2

example : Module.finrank ℝ
    (GatedWeights (ZMod 97) (PairIndex (Fin 1)) ((Fin 10 × Fin 2) ⊕ Fin 23)) = 3616 := by
  rw [width16_parameter_count]

end

example : width (Fin 1) = 16 ∧ Fintype.card ((Fin 10 × Fin 2) ⊕ Fin 23) = 43 ∧
    (8 * 16 + 1) / 3 = (43 : ℕ) := by
  norm_num [width, Channels, PairIndex]

end Transformer.Rotate90.Power
