import Transformer.GPTMini.Sparsemax.IncidentJointMemory
import Mathlib.Data.Matrix.Block

/-!
# Jointly convex attention and common-value energy constraints

Derived architecture after arXiv:1602.02068v2, Eq. (1), rather than a
claim made by that paper. The actual sparsemax memory cross block B and
learned output coordinates Z satisfy the affine PSD constraint
`[[B,Z],[Zᵀ,energy I]] ≥ 0`. Values remain the single global decoded table
`B⁻¹Z`; they are not fixed. A prescribed total edge budget prevents the
identity attention from dominating every other path under this constraint.

The forward is still MZ. Its formula alone does not distinguish geometry,
but its feasible outputs now depend on geometry. Subsequent finite witnesses
show ordinary output error selecting different attention edges for different
answers, without an additional geometry criterion or attention labels.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Affine PSD coupling of genuine memory scores and learned output coordinates.
Source: a new energy restriction on the memory chart after arXiv:1602.02068v2, Eq. (1). -/
def energyCoupledMemoryMatrix {N D : ℕ} (energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    Matrix (Fin (N + 1) ⊕ Fin D) (Fin (N + 1) ⊕ Fin D) ℝ :=
  Matrix.fromBlocks (memoryGramScores (localMemoryGram x.1)) x.2 x.2.transpose
    (energy • (1 : Matrix (Fin D) (Fin D) ℝ))

/-- Coupled energy, local feasibility and a prescribed aggregate edge weight.
Source: the derived convex memory restriction after arXiv:1602.02068v2, Eq. (1).
The aggregate budget fixes no individual edge or sparse support. -/
def energyCoupledMemoryDomain (N D : ℕ) (cap floor budget energy : ℝ) :
    Set (LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {x | x.1 ∈ incidentMemoryParameterDomain N cap floor ∧
    (∑ e, x.1.1 e) = budget ∧ (energyCoupledMemoryMatrix energy x).PosSemidef}

/-- The coupling matrix is affine jointly in embeddings and all output coordinates.
Source: the new block energy constraint after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryMatrix_affine {N D : ℕ} (energy : ℝ)
    (x y : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (a b : ℝ) (hab : a + b = 1) :
    energyCoupledMemoryMatrix energy (a • x + b • y) =
      a • energyCoupledMemoryMatrix energy x + b • energyCoupledMemoryMatrix energy y := by
  have hg := localMemoryGram_affine x.1 y.1 a b hab
  ext i j
  cases i with
  | inl i =>
    cases j with
    | inl j =>
      change localMemoryGram (a • x.1 + b • y.1) (Sum.inl i) (Sum.inr j) = _
      rw [hg]
      rfl
    | inr j => rfl
  | inr i =>
    cases j with
    | inl j => rfl
    | inr j =>
      change energy * (1 : Matrix (Fin D) (Fin D) ℝ) i j =
        a * (energy * (1 : Matrix (Fin D) (Fin D) ℝ) i j) +
          b * (energy * (1 : Matrix (Fin D) (Fin D) ℝ) i j)
      rw [← add_mul, hab, one_mul]

/-- Changed values and embeddings satisfy the affine premise. -/
example : energyCoupledMemoryMatrix 6
    ((1 / 2 : ℝ) • ((0 : LocalMemoryParameters 1), (0 : Matrix (Fin 2) (Fin 1) ℝ)) +
      (1 / 2 : ℝ) • (localMemoryExampleParameters,
        Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 : ℝ)))) =
    (1 / 2 : ℝ) • energyCoupledMemoryMatrix 6
      ((0 : LocalMemoryParameters 1), (0 : Matrix (Fin 2) (Fin 1) ℝ)) +
      (1 / 2 : ℝ) • energyCoupledMemoryMatrix 6
        (localMemoryExampleParameters, Matrix.of (fun _ : Fin 2 => fun _ : Fin 1 => (1 : ℝ))) :=
  energyCoupledMemoryMatrix_affine _ _ _ _ _ (by norm_num)

/-- The energy-coupled joint learning domain is convex.
Source: the affine block PSD and linear budget construction after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryDomain_convex (N D : ℕ) (cap floor budget energy : ℝ) :
    Convex ℝ (energyCoupledMemoryDomain N D cap floor budget energy) := by
  intro x hx y hy a b ha hb hab
  refine ⟨incidentMemoryParameterDomain_convex N cap floor hx.1 hy.1 ha hb hab, ?_, ?_⟩
  · change (∑ e, (a * x.1.1 e + b * y.1.1 e)) = budget
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hx.2.1, hy.2.1,
      ← add_mul, hab, one_mul]
  · rw [energyCoupledMemoryMatrix_affine energy x y a b hab]
    exact (hx.2.2.smul ha).add (hy.2.2.smul hb)

/-- The zero-output identity endpoint has exactly the identity coupling matrix.
Source: the genuine identity Gram after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryMatrix_zero (N D : ℕ) :
    energyCoupledMemoryMatrix 1 ((0 : LocalMemoryParameters N),
      (0 : Matrix (Fin (N + 1)) (Fin D) ℝ)) = 1 := by
  have hs : memoryGramScores (localMemoryGram (0 : LocalMemoryParameters N)) = 1 := by
    rw [localMemoryGram_scores]
    change memoryGramScores (localMemoryCore (0 : Fin N → ℝ)) = 1
    rw [localMemoryCore_zero, memoryIdentityGram_scores]
  ext i j
  cases i <;> cases j <;>
    simp [energyCoupledMemoryMatrix, hs, Matrix.one_apply]

/-- Nonempty norm bounds admit the zero-budget, unit-energy endpoint.
Source: a complete feasible instance of the new restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem zero_mem_energyCoupledMemoryDomain (N D : ℕ) (cap floor : ℝ)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    ((0 : LocalMemoryParameters N), (0 : Matrix (Fin (N + 1)) (Fin D) ℝ)) ∈
      energyCoupledMemoryDomain N D cap floor 0 1 := by
  refine ⟨zero_mem_incidentMemoryParameterDomain N cap floor hc hf, ?_, ?_⟩
  · simp
  · rw [energyCoupledMemoryMatrix_zero]
    exact Matrix.PosSemidef.one

/-- The zero-budget domain is inhabited with three real memory slots. -/
example : ((0 : LocalMemoryParameters 2), (0 : Matrix (Fin 3) (Fin 1) ℝ)) ∈
    energyCoupledMemoryDomain 2 1 4 (3 / 4) 0 1 :=
  zero_mem_energyCoupledMemoryDomain _ _ _ _ (by norm_num) (by norm_num)

/-- Exact original shared-value outputs survive the coupling restriction.
Source: the actual inverse chart after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryForward_eq {R N D : ℕ} (cap floor budget energy : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    localJointMemoryForward M x = M * x.2 :=
  incidentJointMemoryForward_eq cap floor M x hf hM hx.1

/-- Genuine memory queries and a feasible energy-coupled table inhabit the forward premises. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j))
    ((0 : LocalMemoryParameters 2), (0 : Matrix (Fin 3) (Fin 1) ℝ)) =
      oneHotContextCodes (fun j : Fin 3 => j) * (0 : Matrix (Fin 3) (Fin 1) ℝ) :=
  energyCoupledMemoryForward_eq 4 (3 / 4) 0 1 _ _ (by norm_num) (oneHotContextCodes_mem _)
    (zero_mem_energyCoupledMemoryDomain _ _ _ _ (by norm_num) (by norm_num))

/-- The actual coupling top block equals actual sparsemax attention on its domain.
Source: probability-score fixed points of arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryMatrix_attention {N D : ℕ} (cap floor budget energy : ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 0 ≤ floor) (hx : x ∈ energyCoupledMemoryDomain N D cap floor budget energy) :
    energyCoupledMemoryMatrix energy x = Matrix.fromBlocks
      (memoryGramAttention (localMemoryGram x.1)) x.2 x.2.transpose
      (energy • (1 : Matrix (Fin D) (Fin D) ℝ)) := by
  unfold energyCoupledMemoryMatrix
  rw [memoryGramAttention_normalized cap floor _
    (incidentMemoryGram_mem cap floor x.1 hf hx.1)]

/-- The coupling premise is inhabited by actual three-slot sparsemax attention. -/
example : energyCoupledMemoryMatrix 1
    ((0 : LocalMemoryParameters 2), (0 : Matrix (Fin 3) (Fin 1) ℝ)) = Matrix.fromBlocks
      (memoryGramAttention (localMemoryGram (0 : LocalMemoryParameters 2))) 0
      (0 : Matrix (Fin 1) (Fin 3) ℝ) (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simpa using energyCoupledMemoryMatrix_attention 4 (3 / 4) 0 1
    ((0 : LocalMemoryParameters 2), (0 : Matrix (Fin 3) (Fin 1) ℝ)) (by norm_num)
    (zero_mem_energyCoupledMemoryDomain _ _ _ _ (by norm_num) (by norm_num))

/-- Convex output losses remain jointly convex under the coupling and aggregate budget.
Source: the exact inverse chart with the new PSD restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem energyCoupledMemoryObjective_convex {R N D : ℕ} (cap floor budget energy : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (energyCoupledMemoryDomain N D cap floor budget energy)
      (fun x => objective (localJointMemoryForward M x)) := by
  refine ⟨energyCoupledMemoryDomain_convex N D cap floor budget energy, ?_⟩
  intro x hx y hy a b ha hb hab
  change objective (localJointMemoryForward M (a • x + b • y)) ≤ _
  rw [incidentJointMemoryForward_affine cap floor M x y hf hM hx.1 hy.1 a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A concrete ordinary output criterion satisfies every abstract joint-convexity premise. -/
example : ConvexOn ℝ (energyCoupledMemoryDomain 2 1 4 (3 / 4) (1 / 8) 6)
    (fun x => (localJointMemoryForward (oneHotContextCodes (fun j : Fin 3 => j)) x 0 0) ^ 2) := by
  apply energyCoupledMemoryObjective_convex 4 (3 / 4) (1 / 8) 6 _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (A 0 0)) (Set.mem_univ (B 0 0)) ha hb hab

end Transformer.GPTMini.Sparsemax
