module

public import Transformer.Phasor.SuccessorValues

/-!
# Tied embedding displacement at a causal successor address

Source: the tied successor displacement experiment in
`experiments/phasor_value_transport/README.md`. Its vocabulary read moves a
fixed coefficient from the current token to its most recent completed successor.
The results concern the mathematical forward map at arbitrary lengths.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Shift the coefficients in a vocabulary read from key `b` to value `r`.
The tensor implementation can apply this rule to each row before its embedding
matrix multiply. The resulting coefficients can be signed. -/
def displacementWeight (q : Fin p → ℝ) (strength : ℝ) (b r c : Fin p) : ℝ :=
  q c + (if c = r then strength else 0) - (if c = b then strength else 0)

/-- For every vocabulary and embedding, the coefficient change equals a
hidden-space displacement along the two tied codes. -/
theorem displacementWeight_read (E : Fin p → EucSpace D) (q : Fin p → ℝ)
    (strength : ℝ) (b r : Fin p) :
    ∑ c, displacementWeight q strength b r c • E c =
      (∑ c, q c • E c) + strength • (E r - E b) := by
  simp only [displacementWeight, sub_smul, add_smul, ite_smul, zero_smul,
    Finset.sum_sub_distrib, Finset.sum_add_distrib, Fintype.sum_ite_eq', smul_sub]
  abel

/-- The raw tied-score margin of the retrieved token against its key changes
by exactly the strength times the squared distance of the two embeddings.
This holds for arbitrary learned codes before RMS normalization. -/
theorem displacement_margin (E : Fin p → EucSpace D) (h : EucSpace D)
    (strength : ℝ) (b r : Fin p) :
    ⟪E r, h + strength • (E r - E b)⟫ - ⟪E b, h + strength • (E r - E b)⟫ =
      (⟪E r, h⟫ - ⟪E b, h⟫) + strength * ‖E r - E b‖ ^ 2 := by
  rw [← inner_sub_left, inner_add_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, inner_sub_left]

/-- The successor residual with a tied displacement inside the hidden RMS.
`strength` is shared across positions, prompt lengths and vocabularies. -/
noncomputable def Model.displacedSuccessorLogit [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain eps scale : ℝ) (xs : ℕ → Fin p)
    (L i : ℕ) (c : Fin p) : ℝ :=
  let a := xs (i-1)
  let b := xs i
  let r := successorAddress xs i
  let w := Perspective.softmaxWeight (M.logits eps scale xs L (i-1))
  let raw := transportedValue M.embedding w (a*b) + (T.withSuccessor xs i).prefixRead .xor (0,0) xs i
  ⟪M.embedding c, M.feature eps scale xs L i +
    gain • GPTMini.rmsNormEps eps (raw + strength • (M.embedding r - M.embedding b))⟫

/-- At any position reading its own key the full model agrees with the
incumbent, for all parameters, histories, strengths and classes. -/
theorem Model.displacedSuccessorLogit_eq_of_self [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain eps scale : ℝ) (xs : ℕ → Fin p)
    (L i : ℕ) (hs : successorAddress xs i = xs i) (c : Fin p) :
    M.displacedSuccessorLogit T strength gain eps scale xs L i c =
      M.successorLogit T gain eps scale xs L i c := by
  simp only [displacedSuccessorLogit, hs, sub_self, smul_zero, add_zero, successorLogit,
    prefixLogit]

/-- In particular both positions of any two-token problem, including NV11,
give the incumbent logit as a function of all trainable parameters. -/
theorem Model.displacedSuccessorLogit_start [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain eps scale : ℝ) (xs : ℕ → Fin p)
    (L i : ℕ) (hi : i ≤ 1) (c : Fin p) :
    M.displacedSuccessorLogit T strength gain eps scale xs L i c =
      M.successorLogit T gain eps scale xs L i c :=
  M.displacedSuccessorLogit_eq_of_self T _ _ _ _ xs L i (successorAddress_start xs i hi) c

/-- A repeated query at position five has a completed self-transition.
This exercises the agreement theorem beyond a two-token prompt. -/
example (M : Model 64 16 4 86 4) (T : PairTables 64 4) (L : ℕ) (c : Fin 4) :
    let xs := fun i => (![1, 2, 3, 1, 1, 1] : Fin 6 → Fin 4) (Fin.ofNat 6 i)
    M.displacedSuccessorLogit T 1 (1/2) 1 1 xs L 5 c =
      M.successorLogit T (1/2) 1 1 xs L 5 c := by
  dsimp only
  apply Model.displacedSuccessorLogit_eq_of_self
  decide

example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.displacedSuccessorLogit T (1/4) (1/2) 1 1 xs L 1 c =
      M.successorLogit T (1/2) 1 1 xs L 1 c :=
  M.displacedSuccessorLogit_start T _ _ _ _ xs L _ (by decide) c

end Transformer.Phasor
