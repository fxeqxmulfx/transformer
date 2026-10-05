import Transformer.GPTMini.Sparsemax.InvertibleGram

/-!
# Exact recovery of jointly learned values from attention outputs

Derived change of coordinates for the value sum `attn @ v` at `73f8a0b`,
after actual causal sparsemax arXiv:1602.02068v2, Eq. (1). On the convex
normalized-Gram domain with a positive diagonal floor, the attention
matrix is invertible. Replace the learned value table by its output table
`Z = A * V`; recover the actual values as `A inverse * Z`.

For every feasible learned embedding matrix, this is an exact bijection
between all value tables and all output tables, with no values frozen.
One common value table serves every causal row. The context's token IDs
are distinct; arbitrary repeated-token or multi-context value sharing is
not inferred. The change does not preserve a weight penalty or norm bound
placed on the original values. There is no chosen task loss or FFN here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The actual attention/value product, using one learned value table.
Source: `attn @ v` at `73f8a0b`, after the variational causal sparsemax. -/
def causalGramValueOutput {T D : ℕ} (G : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) : Matrix (Fin T) (Fin D) ℝ :=
  causalGramAttention G * values

/-- Each output coordinate is the ordinary weighted value sum.
Source: the original `attn @ v`, with no surrogate or private row values. -/
theorem causalGramValueOutput_apply {T D : ℕ} (G : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) (i : Fin T) (d : Fin D) :
    causalGramValueOutput G values i d = ∑ j, causalGramAttention G i j * values j d := by
  exact Matrix.mul_apply

/-- Nonlinear decoding of values from the jointly learned Gram and outputs.
Source: the derived invertible change of coordinates for `attn @ v`.
The definition computes a genuine inverse; equality with outputs is proved
only under the structural normalization and positive-floor hypotheses. -/
def recoverGramValues {T D : ℕ} (G : EmbeddingGram T)
    (outputs : Matrix (Fin T) (Fin D) ℝ) : Matrix (Fin T) (Fin D) ℝ :=
  (causalGramAttention G)⁻¹ * outputs

/-- Every output table is realized exactly by recovered learned values.
Source: the derived coordinate change for the actual `attn @ v` at
`73f8a0b`, with invertibility proved from causal diagonal bounds. -/
theorem recoverGramValues_exact {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (outputs : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hG : G ∈ invertibleGramDomain T cap floor) :
    causalGramValueOutput G (recoverGramValues G outputs) = outputs := by
  have hd := causalGramAttention_det_unit cap floor G hf hG
  exact Matrix.mul_nonsing_inv_cancel_left (causalGramAttention G) outputs hd

/-- Nonzero outputs satisfy every recovery hypothesis at an actual Gram. -/
example : causalGramValueOutput normalizedGramUnit
    (recoverGramValues normalizedGramUnit (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ))) =
      (fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ)) :=
  recoverGramValues_exact 1 _ _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- Encoding and then recovering any original learned value table is exact.
Source: the inverse coordinate change for `attn @ v`; values are unrestricted. -/
theorem recoverGramValues_original {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hG : G ∈ invertibleGramDomain T cap floor) :
    recoverGramValues G (causalGramValueOutput G values) = values := by
  have hd := causalGramAttention_det_unit cap floor G hf hG
  exact Matrix.nonsing_inv_mul_cancel_left (causalGramAttention G) values hd

/-- A nonzero value table inhabits the original-coordinate recovery theorem. -/
example : recoverGramValues normalizedGramUnit
    (causalGramValueOutput normalizedGramUnit (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
      (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  recoverGramValues_original 1 _ _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- The value table realizing an output is unique, rather than a relaxed witness.
Source: the derived inverse of the original attention/value product. -/
theorem recoverGramValues_unique {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (values outputs : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hG : G ∈ invertibleGramDomain T cap floor)
    (ho : causalGramValueOutput G values = outputs) : values = recoverGramValues G outputs := by
  rw [← ho]
  exact (recoverGramValues_original cap floor G values hf hG).symm

/-- The uniqueness premises are witnessed by an actual nonzero value output. -/
example : (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) = recoverGramValues normalizedGramUnit
    (causalGramValueOutput normalizedGramUnit (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) :=
  recoverGramValues_unique 1 _ _ _ _ (by norm_num) normalizedGramUnit_mem_invertible rfl

/-- Every value-output map is surjective throughout the convex embedding domain.
Source: exact recovery for the actual `attn @ v`; no attainability hypothesis
about the target outputs is required in this distinct-token context. -/
theorem causalGramValueOutput_surjective {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hf : 0 < floor) (hG : G ∈ invertibleGramDomain T cap floor) :
    Function.Surjective (causalGramValueOutput G :
      Matrix (Fin T) (Fin D) ℝ → Matrix (Fin T) (Fin D) ℝ) := by
  intro outputs
  exact ⟨recoverGramValues G outputs, recoverGramValues_exact cap floor G outputs hf hG⟩

/-- Feasible embeddings inhabit the exact output-surjectivity theorem. -/
example : Function.Surjective (causalGramValueOutput normalizedGramUnit :
    Matrix (Fin 1) (Fin 1) ℝ → Matrix (Fin 1) (Fin 1) ℝ) :=
  causalGramValueOutput_surjective 1 _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- Two original value tables with equal outputs must agree.
Source: the exact invertible attention/value coordinate change. -/
theorem causalGramValueOutput_injective {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hf : 0 < floor) (hG : G ∈ invertibleGramDomain T cap floor) :
    Function.Injective (causalGramValueOutput G :
      Matrix (Fin T) (Fin D) ℝ → Matrix (Fin T) (Fin D) ℝ) := by
  intro values other he
  calc
    values = recoverGramValues G (causalGramValueOutput G values) :=
      (recoverGramValues_original cap floor G values hf hG).symm
    _ = recoverGramValues G (causalGramValueOutput G other) := congrArg (recoverGramValues G) he
    _ = other := recoverGramValues_original cap floor G other hf hG

/-- Actual embeddings inhabit the injectivity hypotheses as well. -/
example : Function.Injective (causalGramValueOutput normalizedGramUnit :
    Matrix (Fin 1) (Fin 1) ℝ → Matrix (Fin 1) (Fin 1) ℝ) :=
  causalGramValueOutput_injective 1 _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- An exact value update when the learned attention Gram changes.
Source: the invertible change of coordinates for the original `attn @ v`.
It uses the old actual output and the new actual attention inverse. -/
def transportGramValues {T D : ℕ} (start stop : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) : Matrix (Fin T) (Fin D) ℝ :=
  recoverGramValues stop (causalGramValueOutput start values)

/-- Learned values can compensate any feasible change of attention exactly.
Source: the derived output-preserving coordinate update for `attn @ v`. -/
theorem transportGramValues_output {T D : ℕ} (cap floor : ℝ) (start stop : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hs : stop ∈ invertibleGramDomain T cap floor) :
    causalGramValueOutput stop (transportGramValues start stop values) =
      causalGramValueOutput start values := by
  exact recoverGramValues_exact cap floor stop _ hf hs

/-- A nonzero common value table inhabits the exact transport hypotheses. -/
example : causalGramValueOutput normalizedGramUnit
    (transportGramValues normalizedGramUnit normalizedGramUnit
      (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
      causalGramValueOutput normalizedGramUnit (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  transportGramValues_output 1 _ _ _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- Transporting values back recovers the original table without relaxation.
Source: the derived bijective value coordinates, with both learned Grams feasible. -/
theorem transportGramValues_back {T D : ℕ} (cap floor : ℝ) (start stop : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hs : start ∈ invertibleGramDomain T cap floor)
    (he : stop ∈ invertibleGramDomain T cap floor) :
    transportGramValues stop start (transportGramValues start stop values) = values := by
  change recoverGramValues start (causalGramValueOutput stop
    (transportGramValues start stop values)) = values
  rw [transportGramValues_output cap floor start stop values hf he]
  exact recoverGramValues_original cap floor start values hf hs

/-- Both feasible-Gram hypotheses hold for an actual value transport. -/
example : transportGramValues normalizedGramUnit normalizedGramUnit
    (transportGramValues normalizedGramUnit normalizedGramUnit
      (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
      (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  transportGramValues_back 1 _ _ _ _ (by norm_num)
    normalizedGramUnit_mem_invertible normalizedGramUnit_mem_invertible

end Transformer.GPTMini.Sparsemax
