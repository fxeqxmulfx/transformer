import Transformer.GPTMini.Sparsemax.IncidentMemoryParameters
import Transformer.GPTMini.Sparsemax.LocalMemorySupport

/-!
# Learned query/key norms and values under separate incident budgets

Derived extension before arXiv:1602.02068v2, Eq. (1). Separate local
edge constraints suffice for the same affine compact Gram, independently
learned query/key squared norms, exact feature recovery and an attention
inverse. Common original values decode arbitrary output coordinates.

Actual sparsemax retains at most three nonzero routes. The number of
stored embedding coordinates remains 3P-1. The wider feasible domain
permits distant edge weights simultaneously without weakening the PSD,
normalization or strict diagonal-dominance guarantees. Possible path
connections and same-family orthogonality are still fixed restrictions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Linear compact parameter bounds imply every bounded PSD and memory-score constraint.
Source: the derived compact Gram restriction before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryGram_mem {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    localMemoryGram p ∈ memoryGramDomain N cap floor := by
  have hcore := incidentMemoryCore_mem floor p.1 hf hp.1
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
example : localMemoryGram incidentMemoryExampleParameters ∈ memoryGramDomain 3 4 (3 / 4) :=
  incidentMemoryGram_mem _ _ _ (by norm_num) incidentMemoryExampleParameters_mem

/-- Actual sparsemax equals the affine compact path scores on the proved domain.
Source: variational arXiv:1602.02068v2, Eq. (1), fixes these probability rows. -/
theorem incidentMemoryAttention_normalized {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    memoryGramAttention (localMemoryGram p) = memoryGramScores (localMemoryCore p.1) := by
  rw [memoryGramAttention_normalized cap floor _ (incidentMemoryGram_mem cap floor p hf hp)]
  exact localMemoryGram_scores p

/-- The actual attention normalization hypotheses hold at a nonidentity parameter point. -/
example : memoryGramAttention (localMemoryGram incidentMemoryExampleParameters) =
    memoryGramScores (localMemoryCore incidentMemoryExampleParameters.1) :=
  incidentMemoryAttention_normalized _ _ _ (by norm_num) incidentMemoryExampleParameters_mem

/-- The independent learned embedding squared norms remain inside their declared cap.
Source: actual diagonal products in the derived arXiv:1602.02068v2, Eq. (1) architecture. -/
theorem incidentMemoryGram_normBounds {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) : 1 ≤ localMemoryGram p x x ∧
      localMemoryGram p x x ≤ cap := by
  rw [localMemoryGram_diagonal]
  have hx := hp.2 x
  constructor <;> linarith

/-- An increased actual query norm inhabits the learned cap bound. -/
example : 1 ≤ localMemoryGram incidentMemoryExampleParameters (Sum.inl 0) (Sum.inl 0) ∧
    localMemoryGram incidentMemoryExampleParameters (Sum.inl 0) (Sum.inl 0) ≤ (4 : ℝ) :=
  incidentMemoryGram_normBounds _ _ _ incidentMemoryExampleParameters_mem _

/-- The learned key norm also inhabits the independent cap constraint. -/
example : 1 ≤ localMemoryGram incidentMemoryExampleParameters (Sum.inr 1) (Sum.inr 1) ∧
    localMemoryGram incidentMemoryExampleParameters (Sum.inr 1) (Sum.inr 1) ≤ (4 : ℝ) :=
  incidentMemoryGram_normBounds _ _ _ incidentMemoryExampleParameters_mem _

/-- Every compact learned Gram has exact ordinary Q/K embeddings of width at most 2P.
Source: the derived compact memory before arXiv:1602.02068v2, Eq. (1), and spectral recovery.
The coordinate representation is concluded, not required as an input. -/
theorem incidentMemoryGram_fixedWidth {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      localMemoryGram p = featureGram features :=
  memoryGram_fixedWidth cap floor _ (incidentMemoryGram_mem cap floor p hf hp)

/-- Changed query/key norms and edges admit a genuine width-eight feature realization. -/
example : ∃ features : Fin 8 → Sum (Fin 4) (Fin 4) → ℝ,
    localMemoryGram incidentMemoryExampleParameters = featureGram features :=
  incidentMemoryGram_fixedWidth _ _ _ (by norm_num) incidentMemoryExampleParameters_mem

/-- A floor above one half guarantees the actual compact attention has an inverse.
Source: strict diagonal dominance of variational arXiv:1602.02068v2, Eq. (1) memory. -/
theorem incidentMemoryAttention_det_unit {N : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (hf : 1 / 2 < floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    IsUnit (memoryGramAttention (localMemoryGram p)).det :=
  memoryGramAttention_det_unit cap floor _ hf (incidentMemoryGram_mem cap floor p (by linarith) hp)

/-- Nonsingularity is inhabited by actual changed embeddings and off-diagonal support. -/
example : IsUnit (memoryGramAttention (localMemoryGram incidentMemoryExampleParameters)).det :=
  incidentMemoryAttention_det_unit _ _ _ (by norm_num) incidentMemoryExampleParameters_mem

/-- One common original value table exactly decodes every learned output table.
Source: the structural inverse after arXiv:1602.02068v2, Eq. (1), applied to compact parameters.
Values are shared by all queries and are not replaced by per-query parameters. -/
theorem incidentMemoryValues_exact {N D : ℕ} (cap floor : ℝ) (p : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    memoryValueOutput (localMemoryGram p) (recoverMemoryValues (localMemoryGram p) Z) = Z :=
  recoverMemoryValues_exact cap floor _ Z hf (incidentMemoryGram_mem cap floor p (by linarith) hp)

/-- Changed compact attention and nonconstant outputs inhabit the shared-value decoder premises. -/
example : memoryValueOutput (localMemoryGram incidentMemoryExampleParameters)
    (recoverMemoryValues (localMemoryGram incidentMemoryExampleParameters)
      (Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ)))) =
    Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ)) :=
  incidentMemoryValues_exact _ _ _ _ (by norm_num) incidentMemoryExampleParameters_mem

/-- Actual attention is affine on the whole convex compact parameter domain.
Source: variational arXiv:1602.02068v2, Eq. (1), with the separate local structural constraints. -/
theorem incidentMemoryAttention_affine {N : ℕ} (cap floor : ℝ) (p q : LocalMemoryParameters N)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    memoryGramAttention (localMemoryGram (a • p + b • q)) =
      a • memoryGramAttention (localMemoryGram p) + b • memoryGramAttention (localMemoryGram q) := by
  rw [localMemoryGram_affine p q a b hab]
  exact memoryGramAttention_affine cap floor _ _ (incidentMemoryGram_mem cap floor p hf hp)
    (incidentMemoryGram_mem cap floor q hf hq) a b ha hb hab

/-- A joint midpoint with changed norms and support satisfies every affinity premise. -/
example : memoryGramAttention (localMemoryGram ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 3) +
    (1 / 2 : ℝ) • incidentMemoryExampleParameters)) =
    (1 / 2 : ℝ) • memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 3)) +
      (1 / 2 : ℝ) • memoryGramAttention (localMemoryGram incidentMemoryExampleParameters) :=
  incidentMemoryAttention_affine _ _ _ _ (by norm_num)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Actual attention is zero outside the three possible neighboring slots.
Source: the path structure and actual arXiv:1602.02068v2, Eq. (1) on the enlarged domain. -/
theorem incidentMemoryAttention_zero_of_not_mem {N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (hf : 0 ≤ floor)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor) (i j : Fin (N + 1))
    (hj : j ∉ localMemoryNeighbours i) : memoryGramAttention (localMemoryGram p) i j = 0 := by
  rw [incidentMemoryAttention_normalized cap floor p hf hp]
  exact localMemoryCore_zero_of_not_mem p.1 i j hj

/-- Distant slots and changed norms jointly inhabit the structural-zero premises. -/
example : memoryGramAttention (localMemoryGram incidentMemoryExampleParameters) 0 3 = 0 :=
  incidentMemoryAttention_zero_of_not_mem _ _ _ (by norm_num) incidentMemoryExampleParameters_mem
    0 3 (by norm_num [localMemoryNeighbours, localSlotClamp])

/-- The enlarged learned domain still has at most three actual nonzero attention entries per row.
Source: the structural support bound for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryAttention_support_card {N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (hf : 0 ≤ floor)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor) (i : Fin (N + 1)) :
    (Finset.univ.filter (fun j => memoryGramAttention (localMemoryGram p) i j ≠ 0)).card ≤ 3 := by
  classical
  apply (Finset.card_le_card ?_).trans (localMemoryNeighbours_card i)
  intro j hj
  by_contra hn
  exact (Finset.mem_filter.1 hj).2 (incidentMemoryAttention_zero_of_not_mem cap floor p hf hp i j hn)

/-- A point outside the global domain inhabits the actual sparse-support bound. -/
example : (Finset.univ.filter (fun j =>
    memoryGramAttention (localMemoryGram incidentMemoryExampleParameters) 1 j ≠ 0)).card ≤ 3 :=
  incidentMemoryAttention_support_card _ _ _ (by norm_num) incidentMemoryExampleParameters_mem 1

/-- The enlarged-domain witness has three positive actual routes and one distant exact zero.
Source: computed genuine cross products before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryExample_attention_row :
    memoryGramAttention (localMemoryGram incidentMemoryExampleParameters) 1 =
      (fun j => if j = 0 then 1 / 8 else if j = 1 then 3 / 4 else if j = 2 then 1 / 8 else 0) := by
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) incidentMemoryExampleParameters_mem]
  funext j
  rw [localMemoryCore_scores_apply, Fintype.sum_option]
  fin_cases j <;> norm_num [incidentMemoryExampleParameters, localMemoryWeights,
    Fin.sum_univ_three, localMemoryPermutation, Equiv.swap_apply_def]

end Transformer.GPTMini.Sparsemax
