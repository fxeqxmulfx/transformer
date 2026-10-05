import Transformer.GPTMini.Sparsemax.GramValues

/-!
# Exact convex coordinates for embeddings, attention and learned values

Derived architecture for sparsemax arXiv:1602.02068v2, Eq. (1), followed
by the original `attn @ v` at `73f8a0b`. Optimize a normalized learned Gram
`G` with a positive self-weight floor and an unconstrained output table
`Z`. Decode the shared learned values by `A(G) inverse * Z`. The actual
attention/value product is proved to equal `Z` on this convex domain.

This is an exact change of value coordinates, not an affine interpolation
of the original value table and not a lifted rank relaxation. Both Q/K
embedding families and values can change. Any future convex objective of
the outputs remains convex; no particular task loss or FFN is selected.
The guarantee covers one distinct-token context with all causal rows.
Original value penalties, repeated-token sharing across contexts, bounded
embedding width and inverse conditioning are not preserved or inferred.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Joint learned Gram/output coordinates; decoded values remain unrestricted.
Source: the derived exact coordinate change for `attn @ v` at `73f8a0b`. -/
def jointGramValueDomain (T D : ℕ) (cap floor : ℝ) :
    Set (EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ) :=
  {p | p.1 ∈ invertibleGramDomain T cap floor}

/-- Evaluate actual sparsemax attention with its decoded learned value table.
Source: the original value sum after Eq. (1). The inverse and multiplication
are evaluated explicitly; output equality is not built into this definition. -/
def jointGramValueForward {T D : ℕ}
    (p : EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ) : Matrix (Fin T) (Fin D) ℝ :=
  causalGramValueOutput p.1 (recoverGramValues p.1 p.2)

/-- Recover the original learned Gram/value parameter pair.
Source: the exact value-coordinate decoder for `attn @ v`. -/
def decodedJointGramValues {T D : ℕ}
    (p : EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ) :
    EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ :=
  (p.1, recoverGramValues p.1 p.2)

/-- All embedding and value-output coordinates vary over a convex domain.
Source: the normalized PSD Gram constraints and free output coordinates. -/
theorem jointGramValueDomain_convex (T D : ℕ) (cap floor : ℝ) :
    Convex ℝ (jointGramValueDomain T D cap floor) := by
  intro p hp q hq a b ha hb hab
  exact invertibleGramDomain_convex T cap floor hp hq ha hb hab

/-- The original attention/value operation realizes every learned output exactly.
Source: the derived invertible value coordinates, with the structural floor
proved sufficient; no original value matrix is held fixed. -/
theorem jointGramValueForward_eq_outputs {T D : ℕ} (cap floor : ℝ)
    (p : EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hp : p ∈ jointGramValueDomain T D cap floor) : jointGramValueForward p = p.2 := by
  exact recoverGramValues_exact cap floor p.1 p.2 hf hp

/-- Nonzero output coordinates inhabit the actual-forward equality theorem. -/
example : jointGramValueForward (normalizedGramUnit,
    fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) =
      (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  jointGramValueForward_eq_outputs 1 _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- Every original feasible Gram/value pair is recovered without relaxation.
Source: the inverse of the exact `Z = A(G) * V` coordinate change. -/
theorem decodedJointGramValues_original {T D : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (values : Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hG : G ∈ invertibleGramDomain T cap floor) :
    decodedJointGramValues (G, causalGramValueOutput G values) = (G, values) := by
  change (G, recoverGramValues G (causalGramValueOutput G values)) = (G, values)
  rw [recoverGramValues_original cap floor G values hf hG]

/-- A nonzero original value table inhabits exact pair recovery. -/
example : decodedJointGramValues (normalizedGramUnit, causalGramValueOutput normalizedGramUnit
    (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ))) =
      (normalizedGramUnit, fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) :=
  decodedJointGramValues_original 1 _ _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- Actual decoded attention outputs are affine on the entire convex domain.
Source: the derived exact Gram/value coordinates. The decoded value matrices
follow nonlinear inverse paths; literal affine value mixing is not asserted. -/
theorem jointGramValueForward_affine {T D : ℕ} (cap floor : ℝ)
    (p q : EmbeddingGram T × Matrix (Fin T) (Fin D) ℝ) (hf : 0 < floor)
    (hp : p ∈ jointGramValueDomain T D cap floor)
    (hq : q ∈ jointGramValueDomain T D cap floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    jointGramValueForward (a • p + b • q) =
      a • jointGramValueForward p + b • jointGramValueForward q := by
  have hm := jointGramValueDomain_convex T D cap floor hp hq ha hb hab
  rw [jointGramValueForward_eq_outputs cap floor _ hf hm,
    jointGramValueForward_eq_outputs cap floor _ hf hp,
    jointGramValueForward_eq_outputs cap floor _ hf hq]
  simp only [Prod.snd_add, Prod.smul_snd]

/-- Distinct nonzero output tables satisfy all joint-affinity premises. -/
example : jointGramValueForward
    ((1 / 2 : ℝ) • (normalizedGramUnit, fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ)) +
      (1 / 2 : ℝ) • (normalizedGramUnit, fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ))) =
      (1 / 2 : ℝ) • jointGramValueForward (normalizedGramUnit,
        fun _ : Fin 1 => fun _ : Fin 1 => (2 : ℝ)) +
      (1 / 2 : ℝ) • jointGramValueForward (normalizedGramUnit,
        fun _ : Fin 1 => fun _ : Fin 1 => (4 : ℝ)) :=
  jointGramValueForward_affine 1 _ _ _ (by norm_num)
    normalizedGramUnit_mem_invertible normalizedGramUnit_mem_invertible _ _
    (by norm_num) (by norm_num) (by norm_num)

/-- Any future convex output objective stays convex in learned Gram and values.
Source: the derived exact coordinate change for `attn @ v`; the convexity of
the chosen objective is an explicit hypothesis, not a task-loss assumption. -/
theorem jointGramValueObjective_convex {T D : ℕ} (cap floor : ℝ)
    (objective : Matrix (Fin T) (Fin D) ℝ → ℝ) (hf : 0 < floor)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (jointGramValueDomain T D cap floor)
      (fun p => objective (jointGramValueForward p)) := by
  refine ⟨jointGramValueDomain_convex T D cap floor, ?_⟩
  intro p hp q hq a b ha hb hab
  change objective (jointGramValueForward (a • p + b • q)) ≤
    a • objective (jointGramValueForward p) + b • objective (jointGramValueForward q)
  rw [jointGramValueForward_affine cap floor p q hf hp hq a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The arbitrary-objective hypothesis has a concrete nonconstant convex
instance. It is an inhabited mathematical example, not a selected task loss. -/
example : ConvexOn ℝ (jointGramValueDomain 1 1 1 (1 / 2))
    (fun p => (jointGramValueForward p 0 0) ^ 2) := by
  apply jointGramValueObjective_convex 1 (1 / 2) (fun Z => (Z 0 0) ^ 2) (by norm_num)
  have hs : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Outputs attained by actual jointly learned Grams and original value tables.
Source: the exact prediction class of the derived causal `attn @ v` model. -/
def jointGramValuePredictionSet (T D : ℕ) (cap floor : ℝ) :
    Set (Matrix (Fin T) (Fin D) ℝ) :=
  {Z | ∃ G ∈ invertibleGramDomain T cap floor,
    ∃ values : Matrix (Fin T) (Fin D) ℝ, causalGramValueOutput G values = Z}

/-- Exact recovery preserves all outputs, once the embedding domain is inhabited.
Source: the derived value-coordinate bijection. This records expressivity
on this one distinct-token context, not a multi-context generalization theorem. -/
theorem jointGramValuePredictionSet_eq_univ {T D : ℕ} (cap floor : ℝ) (hf : 0 < floor)
    (hn : (invertibleGramDomain T cap floor).Nonempty) :
    jointGramValuePredictionSet T D cap floor = Set.univ := by
  ext Z
  constructor
  · intro _
    exact Set.mem_univ Z
  · intro _
    obtain ⟨G, hG⟩ := hn
    exact ⟨G, hG, recoverGramValues G Z, recoverGramValues_exact cap floor G Z hf hG⟩

/-- A concrete embedding matrix inhabits both prediction-class hypotheses. -/
example : jointGramValuePredictionSet 1 1 1 (1 / 2) = Set.univ :=
  jointGramValuePredictionSet_eq_univ 1 _ (by norm_num)
    ⟨normalizedGramUnit, normalizedGramUnit_mem_invertible⟩

end Transformer.GPTMini.Sparsemax
