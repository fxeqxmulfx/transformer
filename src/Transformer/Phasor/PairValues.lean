module

public import Transformer.Phasor.TransportedValues

/-!
# Operand-sensitive hidden values for two-token Phasor models

Source: the pair-value construction in `experiments/phasor_value_transport/README.md`.
The product transports the ordinary first-position attention. Independent learned
value tables read the second operand and fixed functions of both operands. Their
sum is taken in the hidden space before the ordinary RMS and tied projection.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Addresses besides the direct second operand. Each power index denotes an
exponent starting at one. Division uses the finite-field inverse extended by
zero on a zero denominator. The distance is on the integer representatives. -/
inductive PairKey
  | difference | distance | quotient
  | sumPowers (i j : Fin 3)
  | productPowers (i j : Fin 2)
  deriving DecidableEq, Fintype

/-- The input addresses are independent of both the supervised task and its labels. -/
def PairKey.address [NeZero p] (a b : Fin p) : PairKey → Fin p
  | .difference => a - b
  | .distance => Fin.ofNat p (max a.val b.val - min a.val b.val)
  | .quotient => if b = 0 then 0 else a * b ^ (p - 2)
  | .sumPowers i j => a ^ (i.val + 1) + b ^ (j.val + 1)
  | .productPowers i j => a ^ (i.val + 1) * b ^ (j.val + 1)

/-- The new trainable parameters. Each table entry has the backbone's hidden width. -/
structure PairTables (D p : ℕ) where
  right : Fin p → EucSpace D
  values : PairKey → Fin p → EucSpace D

/-- The combined residual read from the two operands. -/
noncomputable def PairTables.read [NeZero p] (T : PairTables D p) (a b : Fin p) : EucSpace D :=
  T.right b + ∑ k, T.values k (k.address a b)

/-- Tied logit after adding the residual to the transported hidden value. -/
noncomputable def pairValueLogit (E : Fin p → EucSpace D) (w : Fin p → ℝ) (eps : ℝ)
    (r c : Fin p) (v : EucSpace D) : ℝ :=
  ⟪E c, GPTMini.rmsNormEps eps (transportedValue E w r + v)⟫

/-- Only the second position of a fixed two-token prime-vocabulary model uses
the field tables. Context is a property of the configured model. -/
noncomputable def Model.pairLogits [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (context : ℕ) (eps scale : ℝ) (tokens : ℕ → Fin p) (L i : ℕ) : Fin p → ℝ :=
  if context = 2 ∧ Nat.Prime p ∧ i = 1 then
    fun c => pairValueLogit M.embedding (Perspective.softmaxWeight (M.logits eps scale tokens L 0))
      eps (tokens 0 * tokens 1) c (T.read (tokens 0) (tokens 1))
  else M.logits eps scale tokens L i

/-- The supplemental pair values are zero at initialization, giving exactly the
previous transported readout for every embedding and address distribution. -/
theorem pairValueLogit_zero (E : Fin p → EucSpace D) (w : Fin p → ℝ) (eps : ℝ) (r c : Fin p) :
    pairValueLogit E w eps r c 0 = transportedLogit E w eps r c := by
  simp [pairValueLogit, transportedLogit]

/-- Pair-value decision rule. A residual preserves a transport win precisely when
its adverse projected gap is smaller than the unnormalized transport advantage. -/
theorem pairValueLogit_lt_iff {E : Fin p → EucSpace D} {w : Fin p → ℝ}
    {eps : ℝ} (hD : 0 < D) (heps : 0 < eps) (r c y : Fin p) (v : EucSpace D) :
    pairValueLogit E w eps r c v < pairValueLogit E w eps r y v ↔
      ⟪E c, v⟫ - ⟪E y, v⟫ <
        ⟪E y, transportedValue E w r⟫ - ⟪E c, transportedValue E w r⟫ := by
  have hd : (0 : ℝ) < D := Nat.cast_pos.mpr hD
  simp only [pairValueLogit, GPTMini.rmsNormEps, real_inner_smul_right]
  rw [mul_lt_mul_iff_right₀ (by positivity)]
  simp only [inner_add_right]
  constructor <;> intro h <;> linarith

/-- At NV11 width the margin for orthonormal codes is the learned cubic
address's attention advantage over the rival untranslated address. -/
example (w : Fin 97 → ℝ) (a b t : Fin 97) (v : EucSpace 128) :
    let E := TwoCopies.coordCodes (d := 128) (p := 97) (by decide)
    pairValueLogit E w 1 (a*b) (t + a*b) v < pairValueLogit E w 1 (a*b) (a^3 + a*b) v ↔
      ⟪E (t + a*b), v⟫ - ⟪E (a^3 + a*b), v⟫ < w (a^3) - w t := by
  dsimp only
  rw [pairValueLogit_lt_iff (by decide) (by norm_num),
    inner_transportedValue_orthonormal (TwoCopies.orthonormal_coordCodes _),
    inner_transportedValue_orthonormal (TwoCopies.orthonormal_coordCodes _)]

/-- A single setting of the right-operand table makes every second operand win
on the entire zero row, even with arbitrary address masses in the unit interval.
The translated-value branch remains active in this witness. -/
theorem pair_zero_row_separates [NeZero p] {E : Fin p → EucSpace D} (hE : Orthonormal ℝ E)
    (hD : 0 < D) {eps : ℝ} (heps : 0 < eps) {w : Fin p → ℝ}
    (hw : ∀ s, w s ∈ Set.Icc 0 1) (b c : Fin p) (hcb : c ≠ b) :
    let T : PairTables D p := ⟨fun s => (2 : ℝ) • E s, 0⟩
    pairValueLogit E w eps (0*b) c (T.read 0 b) <
      pairValueLogit E w eps (0*b) b (T.read 0 b) := by
  rw [pairValueLogit_lt_iff hD heps]
  have mass (s : Fin p) : ⟪E s, transportedValue E w 0⟫ = w s := by
    simpa using inner_transportedValue_orthonormal hE w 0 s
  simp [PairTables.read, real_inner_smul_right, orthonormal_iff_ite.mp hE, hcb, mass, hE.norm_eq_one]
  linarith [(hw b).1, (hw c).2]

/-- The separation witness uses the experimental width and a strictly positive
uniform distribution over all 97 addresses. -/
example (b c : Fin 97) (hcb : c ≠ b) :
    let E := TwoCopies.coordCodes (d := 128) (p := 97) (by decide)
    let T : PairTables 128 97 := ⟨fun s => (2 : ℝ) • E s, 0⟩
    pairValueLogit E (fun _ => 1/97) 1 (0*b) c (T.read 0 b) <
      pairValueLogit E (fun _ => 1/97) 1 (0*b) b (T.read 0 b) :=
  pair_zero_row_separates (TwoCopies.orthonormal_coordCodes _) (by decide) (by norm_num)
    (fun _ => by constructor <;> norm_num) b c hcb

/-- For every model, parameter value and sequence position outside the
pair architecture's domain, the original causal readout is exact. -/
theorem Model.pairLogits_of_inactive [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    {context i : ℕ} (h : ¬ (context = 2 ∧ Nat.Prime p ∧ i = 1))
    (eps scale : ℝ) (tokens : ℕ → Fin p) (L : ℕ) :
    M.pairLogits T context eps scale tokens L i = M.logits eps scale tokens L i := by
  simp [Model.pairLogits, h]

/-- A two-position batch inside a long-context model still reads the ordinary
Phasor at position one. -/
example (M : Model D d H F 97) (T : PairTables D 97) (tokens : ℕ → Fin 97) (L : ℕ) :
    M.pairLogits T 64 1 1 tokens L 1 = M.logits 1 1 tokens L 1 :=
  M.pairLogits_of_inactive T (by decide) _ _ _ _

end Transformer.Phasor
