import Transformer.GPTMini.Sparsemax.LocalMemorySupport

/-!
# Joint convex compact embeddings, attention and common values

Derived architecture for arXiv:1602.02068v2, Eq. (1), and `attn @ v` at
`73f8a0b`. Learn compact path/norm parameters together with a global output
table Z. One common original value table is decoded from the structurally
invertible actual memory attention. For fixed data codes M, the actual
context forward is MZ throughout a convex joint parameter domain.

Any future convex criterion on all outputs stays jointly convex. This
statement does not choose a task loss or cover an FFN. Compactness and
sparse actual routes coexist with the previous gauge freedom: changing
feasible attention parameters while compensating the common values leaves
every output unchanged. An output-only objective consequently cannot
identify those parameters. Additional information or a separate criterion
is needed to select among them; sparsity alone does not remove this freedom.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Compact path/norm coordinates jointly learned with one global output table.
Source: the common-value chart after arXiv:1602.02068v2, Eq. (1). -/
def localJointMemoryDomain (N D : ℕ) (cap floor : ℝ) :
    Set (LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {x | x.1 ∈ localMemoryParameterDomain N cap floor}

/-- Actual attention times the globally decoded common original values.
Source: arXiv:1602.02068v2, Eq. (1), then the shared value product at `73f8a0b`. -/
def localJointMemoryForward {R N D : ℕ} (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    Matrix (Fin R) (Fin D) ℝ := jointContextMemoryForward M (localMemoryGram x.1, x.2)

/-- Every learned compact coordinate and common output table lie in a convex joint domain.
Source: the derived exact common-value chart for arXiv:1602.02068v2, Eq. (1). -/
theorem localJointMemoryDomain_convex (N D : ℕ) (cap floor : ℝ) :
    Convex ℝ (localJointMemoryDomain N D cap floor) := by
  intro x hx y hy a b ha hb hab
  exact localMemoryParameterDomain_convex N cap floor hx hy ha hb hab

/-- The actual compact shared-memory forward is M times the global output coordinates.
Source: the proved structural decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem localJointMemoryForward_eq {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hx : x ∈ localJointMemoryDomain N D cap floor) : localJointMemoryForward M x = M * x.2 :=
  jointContextMemoryForward_eq cap floor M _ hf hM
    (localMemoryGram_mem cap floor x.1 (by linarith) hx)

/-- Changed genuine embeddings and nonconstant shared outputs inhabit the actual forward premises. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
    (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) =
    oneHotContextCodes (fun j : Fin 2 => j) *
      Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)) :=
  localJointMemoryForward_eq 4 (3 / 4) _ _ (by norm_num) (oneHotContextCodes_mem _)
    localMemoryExampleParameters_mem

/-- Actual outputs are affine jointly in compact embeddings and learned common values.
Source: the exact compact chart for arXiv:1602.02068v2, Eq. (1). -/
theorem localJointMemoryForward_affine {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x y : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hx : x ∈ localJointMemoryDomain N D cap floor) (hy : y ∈ localJointMemoryDomain N D cap floor)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    localJointMemoryForward M (a • x + b • y) =
      a • localJointMemoryForward M x + b • localJointMemoryForward M y := by
  have hm := localJointMemoryDomain_convex N D cap floor hx hy ha hb hab
  rw [localJointMemoryForward_eq cap floor M _ hf hM hm,
    localJointMemoryForward_eq cap floor M x hf hM hx,
    localJointMemoryForward_eq cap floor M y hf hM hy]
  simp only [Prod.snd_add, Prod.smul_snd, Matrix.mul_add, Matrix.mul_smul]

/-- Changed compact parameters and changed outputs satisfy every joint midpoint premise. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
    ((1 / 2 : ℝ) • ((0 : LocalMemoryParameters 1),
      Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) +
      (1 / 2 : ℝ) • (localMemoryExampleParameters,
        Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ)))) =
    (1 / 2 : ℝ) • localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
      ((0 : LocalMemoryParameters 1), Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) +
    (1 / 2 : ℝ) • localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
      (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ))) :=
  localJointMemoryForward_affine 4 (3 / 4) _ _ _ (by norm_num) (oneHotContextCodes_mem _)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    localMemoryExampleParameters_mem _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Any convex output criterion remains convex on the compact joint learning domain.
Source: the actual compact shared-value chart after arXiv:1602.02068v2, Eq. (1).
Task-loss selection and FFN are deferred; convexity of the criterion is explicit. -/
theorem localJointMemoryObjective_convex {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (localJointMemoryDomain N D cap floor)
      (fun x => objective (localJointMemoryForward M x)) := by
  refine ⟨localJointMemoryDomain_convex N D cap floor, ?_⟩
  intro x hx y hy a b ha hb hab
  change objective (localJointMemoryForward M (a • x + b • y)) ≤
    a • objective (localJointMemoryForward M x) + b • objective (localJointMemoryForward M y)
  rw [localJointMemoryForward_affine cap floor M x y hf hM hx hy a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A nonconstant squared output criterion inhabits the abstract convexity premise. -/
example : ConvexOn ℝ (localJointMemoryDomain 1 1 4 (3 / 4))
    (fun x => (localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j)) x 0 0) ^ 2) := by
  apply localJointMemoryObjective_convex 4 (3 / 4) _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Compactness and sparse support do not remove the exact output-only attention freedom.
Source: the structural value decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem localJointMemoryForward_parameter_invariant {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hp : p ∈ localMemoryParameterDomain N cap floor)
    (hq : q ∈ localMemoryParameterDomain N cap floor) :
    localJointMemoryForward M (p, Z) = localJointMemoryForward M (q, Z) := by
  rw [localJointMemoryForward_eq cap floor M _ hf hM hp,
    localJointMemoryForward_eq cap floor M _ hf hM hq]

/-- Distinct changed edge/norm parameters and nonconstant outputs inhabit the gauge premises. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
    ((0 : LocalMemoryParameters 1), Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) =
    localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
      (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) :=
  localJointMemoryForward_parameter_invariant 4 (3 / 4) _ _ _ _ (by norm_num)
    (oneHotContextCodes_mem _) (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    localMemoryExampleParameters_mem

/-- Every output-only criterion has the same value after a feasible attention-parameter change.
Source: the exact common-value freedom after arXiv:1602.02068v2, Eq. (1).
Convexity of the output criterion is not needed for this nonidentifiability. -/
theorem localJointMemoryObjective_parameter_invariant {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hp : p ∈ localMemoryParameterDomain N cap floor)
    (hq : q ∈ localMemoryParameterDomain N cap floor) :
    objective (localJointMemoryForward M (p, Z)) =
      objective (localJointMemoryForward M (q, Z)) :=
  congrArg objective (localJointMemoryForward_parameter_invariant cap floor M p q Z hf hM hp hq)

/-- Distinct edge/norm parameters and a nonconstant squared criterion satisfy the same-loss premises. -/
example : (localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
    ((0 : LocalMemoryParameters 1), Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ)))
      0 0) ^ 2 =
    (localJointMemoryForward (oneHotContextCodes (fun j : Fin 2 => j))
      (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val + 1 : ℝ)))
        0 0) ^ 2 :=
  localJointMemoryObjective_parameter_invariant 4 (3 / 4) _ _ _ _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) localMemoryExampleParameters_mem

end Transformer.GPTMini.Sparsemax
