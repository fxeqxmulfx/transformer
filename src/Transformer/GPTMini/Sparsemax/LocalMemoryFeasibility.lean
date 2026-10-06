import Transformer.GPTMini.Sparsemax.LocalMemoryParameters
import Transformer.GPTMini.Sparsemax.BoundedGramWidth

/-!
# Structural guarantees for compact learned embeddings and actual attention

Derived compact memory for arXiv:1602.02068v2, Eq. (1), followed by common
values at `73f8a0b`. Linear path-weight and independent squared-norm bounds
imply a bounded PSD Gram, probability cross-score rows and the requested
self-weight floor. Actual variational sparsemax fixes these scores. A floor
above one half proves nonsingularity without fixing the learned weights.

Both embedding families are genuine: spectral recovery supplies width 2P,
without imposing a nonconvex rank constraint. Actual attention is affine
jointly in all stored coordinates on the convex parameter domain, while
Q/K squared norms can change independently. This is a restricted path
memory, with fixed same-family orthogonality; unrestricted token attention,
task-loss selection and FFN convexity are not asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Linear compact parameter bounds imply every bounded PSD and memory-score constraint.
Source: the derived compact Gram restriction before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryGram_mem {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ localMemoryParameterDomain N cap floor) :
    localMemoryGram p ∈ memoryGramDomain N cap floor := by
  have hcore := localMemoryCore_mem floor p.1 hf hp.1
  have hcap : 1 ≤ cap := by
    have h := hp.2 (Sum.inl 0)
    linarith
  refine ⟨⟨hcore.1.1.add (Matrix.PosSemidef.diagonal (fun x => (hp.2 x).1)), ?_⟩, ?_, ?_⟩
  · intro x y
    by_cases h : x = y
    · subst y
      rw [localMemoryGram_diagonal]
      have hx := hp.2 x
      constructor <;> linarith
    · rw [localMemoryGram_apply, Matrix.diagonal_apply_ne _ h, add_zero]
      have hb := hcore.1.2 x y
      constructor <;> linarith
  · intro i
    rw [localMemoryGram_scores]
    exact hcore.2.1 i
  · intro i
    rw [localMemoryGram_scores]
    exact hcore.2.2 i

/-- A changed edge and changed norms jointly inhabit the full embedding-domain theorem. -/
example : localMemoryGram localMemoryExampleParameters ∈ memoryGramDomain 1 4 (3 / 4) :=
  localMemoryGram_mem _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- Actual sparsemax equals the affine compact path scores on the proved domain.
Source: variational arXiv:1602.02068v2, Eq. (1), fixes these probability rows. -/
theorem localMemoryAttention_normalized {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ localMemoryParameterDomain N cap floor) :
    memoryGramAttention (localMemoryGram p) = memoryGramScores (localMemoryCore p.1) := by
  rw [memoryGramAttention_normalized cap floor _ (localMemoryGram_mem cap floor p hf hp)]
  exact localMemoryGram_scores p

/-- The actual attention normalization hypotheses hold at a nonidentity parameter point. -/
example : memoryGramAttention (localMemoryGram localMemoryExampleParameters) =
    memoryGramScores (localMemoryCore localMemoryExampleParameters.1) :=
  localMemoryAttention_normalized _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- The independent learned embedding squared norms remain inside their declared cap.
Source: actual diagonal products in the derived arXiv:1602.02068v2, Eq. (1) architecture. -/
theorem localMemoryGram_normBounds {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hp : p ∈ localMemoryParameterDomain N cap floor)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) : 1 ≤ localMemoryGram p x x ∧
      localMemoryGram p x x ≤ cap := by
  rw [localMemoryGram_diagonal]
  have hx := hp.2 x
  constructor <;> linarith

/-- An increased actual query norm inhabits the learned cap bound. -/
example : 1 ≤ localMemoryGram localMemoryExampleParameters (Sum.inl 0) (Sum.inl 0) ∧
    localMemoryGram localMemoryExampleParameters (Sum.inl 0) (Sum.inl 0) ≤ (4 : ℝ) :=
  localMemoryGram_normBounds _ _ _ localMemoryExampleParameters_mem _

/-- The learned key norm also inhabits the independent cap constraint. -/
example : 1 ≤ localMemoryGram localMemoryExampleParameters (Sum.inr 1) (Sum.inr 1) ∧
    localMemoryGram localMemoryExampleParameters (Sum.inr 1) (Sum.inr 1) ≤ (4 : ℝ) :=
  localMemoryGram_normBounds _ _ _ localMemoryExampleParameters_mem _

/-- Every compact learned Gram has exact ordinary Q/K embeddings of width at most 2P.
Source: the derived compact memory before arXiv:1602.02068v2, Eq. (1), and spectral recovery.
The coordinate representation is concluded, not required as an input. -/
theorem localMemoryGram_fixedWidth {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ localMemoryParameterDomain N cap floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      localMemoryGram p = featureGram features :=
  memoryGram_fixedWidth cap floor _ (localMemoryGram_mem cap floor p hf hp)

/-- Changed query/key norms and edges admit a genuine width-four feature realization. -/
example : ∃ features : Fin 4 → Sum (Fin 2) (Fin 2) → ℝ,
    localMemoryGram localMemoryExampleParameters = featureGram features :=
  localMemoryGram_fixedWidth _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- A floor above one half guarantees the actual compact attention has an inverse.
Source: strict diagonal dominance of variational arXiv:1602.02068v2, Eq. (1) memory. -/
theorem localMemoryAttention_det_unit {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 1 / 2 < floor) (hp : p ∈ localMemoryParameterDomain N cap floor) :
    IsUnit (memoryGramAttention (localMemoryGram p)).det :=
  memoryGramAttention_det_unit cap floor _ hf (localMemoryGram_mem cap floor p (by linarith) hp)

/-- Nonsingularity is inhabited by actual changed embeddings and off-diagonal support. -/
example : IsUnit (memoryGramAttention (localMemoryGram localMemoryExampleParameters)).det :=
  localMemoryAttention_det_unit _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- One common original value table exactly decodes every learned output table.
Source: the structural inverse after arXiv:1602.02068v2, Eq. (1), applied to compact parameters.
Values are shared by all queries and are not replaced by per-query parameters. -/
theorem localMemoryValues_exact {N D : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hp : p ∈ localMemoryParameterDomain N cap floor) :
    memoryValueOutput (localMemoryGram p) (recoverMemoryValues (localMemoryGram p) Z) = Z :=
  recoverMemoryValues_exact cap floor _ Z hf (localMemoryGram_mem cap floor p (by linarith) hp)

/-- Changed compact attention and nonconstant outputs inhabit the shared-value decoder premises. -/
example : memoryValueOutput (localMemoryGram localMemoryExampleParameters)
    (recoverMemoryValues (localMemoryGram localMemoryExampleParameters)
      (Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ)))) =
    Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ)) :=
  localMemoryValues_exact _ _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- Actual attention is affine on the whole convex compact parameter domain.
Source: variational arXiv:1602.02068v2, Eq. (1), with the linear structural constraints. -/
theorem localMemoryAttention_affine {N : ℕ} (cap floor : ℝ) (p q : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ localMemoryParameterDomain N cap floor)
    (hq : q ∈ localMemoryParameterDomain N cap floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    memoryGramAttention (localMemoryGram (a • p + b • q)) =
      a • memoryGramAttention (localMemoryGram p) + b • memoryGramAttention (localMemoryGram q) := by
  rw [localMemoryGram_affine p q a b hab]
  exact memoryGramAttention_affine cap floor _ _ (localMemoryGram_mem cap floor p hf hp)
    (localMemoryGram_mem cap floor q hf hq) a b ha hb hab

/-- A joint midpoint with changed norms and support satisfies every affinity premise. -/
example : memoryGramAttention (localMemoryGram ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 1) +
    (1 / 2 : ℝ) • localMemoryExampleParameters)) =
    (1 / 2 : ℝ) • memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 1)) +
      (1 / 2 : ℝ) • memoryGramAttention (localMemoryGram localMemoryExampleParameters) :=
  localMemoryAttention_affine _ _ _ _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    localMemoryExampleParameters_mem _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
