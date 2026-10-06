import Transformer.GPTMini.Sparsemax.IncidentMemoryFeasibility
import Transformer.GPTMini.Sparsemax.LocalJointMemory

/-!
# Convex joint embedding and common-value learning with local budgets

Derived extension of the actual shared-value chart after
arXiv:1602.02068v2, Eq. (1). On the enlarged separate-budget domain,
actual outputs remain MZ. All compact Q/K coordinates and common output
coordinates can be trained jointly with any convex output criterion.

Output-only fitting still leaves the embedding coordinates unidentified.
The separate data criterion added later supplies their selection. The
functional forward is reused unchanged; the new result strengthens its
feasibility domain instead of defining a different prediction function.
Nearest data queries retain at most three actual attention routes.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Compact path/norm coordinates jointly learned with one global output table.
Source: the common-value chart after arXiv:1602.02068v2, Eq. (1). -/
def incidentJointMemoryDomain (N D : ℕ) (cap floor : ℝ) :
    Set (LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {x | x.1 ∈ incidentMemoryParameterDomain N cap floor}

/-- Every learned compact coordinate and common output table lie in a convex joint domain.
Source: the derived exact common-value chart for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentJointMemoryDomain_convex (N D : ℕ) (cap floor : ℝ) :
    Convex ℝ (incidentJointMemoryDomain N D cap floor) := by
  intro x hx y hy a b ha hb hab
  exact incidentMemoryParameterDomain_convex N cap floor hx hy ha hb hab

/-- The actual compact shared-memory forward is M times the global output coordinates.
Source: the proved structural decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem incidentJointMemoryForward_eq {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hx : x ∈ incidentJointMemoryDomain N D cap floor) : localJointMemoryForward M x = M * x.2 :=
  jointContextMemoryForward_eq cap floor M _ hf hM
    (incidentMemoryGram_mem cap floor x.1 (by linarith) hx)

/-- Changed genuine embeddings and nonconstant shared outputs inhabit the actual forward premises. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
    (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) =
    oneHotContextCodes (fun j : Fin 4 => j) *
      Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ)) :=
  incidentJointMemoryForward_eq 4 (3 / 4) _ _ (by norm_num) (oneHotContextCodes_mem _)
    incidentMemoryExampleParameters_mem

/-- Actual outputs are affine jointly in compact embeddings and learned common values.
Source: the exact compact chart for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentJointMemoryForward_affine {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (x y : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hx : x ∈ incidentJointMemoryDomain N D cap floor) (hy : y ∈ incidentJointMemoryDomain N D cap floor)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    localJointMemoryForward M (a • x + b • y) =
      a • localJointMemoryForward M x + b • localJointMemoryForward M y := by
  have hm := incidentJointMemoryDomain_convex N D cap floor hx hy ha hb hab
  rw [incidentJointMemoryForward_eq cap floor M _ hf hM hm,
    incidentJointMemoryForward_eq cap floor M x hf hM hx,
    incidentJointMemoryForward_eq cap floor M y hf hM hy]
  simp only [Prod.snd_add, Prod.smul_snd, Matrix.mul_add, Matrix.mul_smul]

/-- Changed compact parameters and changed outputs satisfy every joint midpoint premise. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
    ((1 / 2 : ℝ) • ((0 : LocalMemoryParameters 3),
      Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) +
      (1 / 2 : ℝ) • (incidentMemoryExampleParameters,
        Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ)))) =
    (1 / 2 : ℝ) • localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
      ((0 : LocalMemoryParameters 3), Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) +
    (1 / 2 : ℝ) • localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
      (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) :=
  incidentJointMemoryForward_affine 4 (3 / 4) _ _ _ (by norm_num) (oneHotContextCodes_mem _)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Any convex output criterion remains convex on the compact joint learning domain.
Source: the actual compact shared-value chart after arXiv:1602.02068v2, Eq. (1).
Task-loss selection and FFN are deferred; convexity of the criterion is explicit. -/
theorem incidentJointMemoryObjective_convex {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (incidentJointMemoryDomain N D cap floor)
      (fun x => objective (localJointMemoryForward M x)) := by
  refine ⟨incidentJointMemoryDomain_convex N D cap floor, ?_⟩
  intro x hx y hy a b ha hb hab
  change objective (localJointMemoryForward M (a • x + b • y)) ≤
    a • objective (localJointMemoryForward M x) + b • objective (localJointMemoryForward M y)
  rw [incidentJointMemoryForward_affine cap floor M x y hf hM hx hy a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A nonconstant squared output criterion inhabits the abstract convexity premise. -/
example : ConvexOn ℝ (incidentJointMemoryDomain 3 1 4 (3 / 4))
    (fun x => (localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j)) x 0 0) ^ 2) := by
  apply incidentJointMemoryObjective_convex 4 (3 / 4) _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Compactness and sparse support do not remove the exact output-only attention freedom.
Source: the structural value decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem incidentJointMemoryForward_parameter_invariant {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor) :
    localJointMemoryForward M (p, Z) = localJointMemoryForward M (q, Z) := by
  rw [incidentJointMemoryForward_eq cap floor M _ hf hM hp,
    incidentJointMemoryForward_eq cap floor M _ hf hM hq]

/-- Distinct changed edge/norm parameters and nonconstant outputs inhabit the gauge premises. -/
example : localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
    ((0 : LocalMemoryParameters 3), Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) =
    localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
      (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) :=
  incidentJointMemoryForward_parameter_invariant 4 (3 / 4) _ _ _ _ (by norm_num)
    (oneHotContextCodes_mem _) (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem

/-- Every output-only criterion has the same value after a feasible attention-parameter change.
Source: the exact common-value freedom after arXiv:1602.02068v2, Eq. (1).
Convexity of the output criterion is not needed for this nonidentifiability. -/
theorem incidentJointMemoryObjective_parameter_invariant {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor) :
    objective (localJointMemoryForward M (p, Z)) =
      objective (localJointMemoryForward M (q, Z)) :=
  congrArg objective (incidentJointMemoryForward_parameter_invariant cap floor M p q Z hf hM hp hq)

/-- Distinct edge/norm parameters and a nonconstant squared criterion satisfy the same-loss premises. -/
example : (localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
    ((0 : LocalMemoryParameters 3), Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ)))
      0 0) ^ 2 =
    (localJointMemoryForward (oneHotContextCodes (fun j : Fin 4 => j))
      (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ)))
        0 0) ^ 2 :=
  incidentJointMemoryObjective_parameter_invariant 4 (3 / 4) _ _ _ _ (fun Y => (Y 0 0) ^ 2)
    (by norm_num) (oneHotContextCodes_mem _)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) incidentMemoryExampleParameters_mem

/-- Nearest observations retain at most three actual attention routes under the separate budgets.
Source: the genuine memory rows before arXiv:1602.02068v2, Eq. (1). -/
theorem contextNearestIncidentAttention_support_card {Key : Type*} {R N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (hf : 0 ≤ floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) (r : Fin R) :
    (Finset.univ.filter (fun j => contextMemoryAttention (localMemoryGram p)
      (nearestPrototypeCodes distance prototypes queries) r j ≠ 0)).card ≤ 3 := by
  rw [contextNearestAttention_row cap floor _ distance prototypes queries
    (incidentMemoryGram_mem cap floor p hf hp)]
  exact incidentMemoryAttention_support_card cap floor p hf hp _

/-- An unseen query and three simultaneous changed edges inhabit the support premises. -/
example : (Finset.univ.filter (fun j => contextMemoryAttention
    (localMemoryGram incidentMemoryExampleParameters)
    (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
      (fun j : Fin 4 => (j.val : ℝ)) (fun _ : Fin 1 => (5 / 4 : ℝ))) 0 j ≠ 0)).card ≤ 3 :=
  contextNearestIncidentAttention_support_card _ _ _ _ _ _ (by norm_num)
    incidentMemoryExampleParameters_mem 0

end Transformer.GPTMini.Sparsemax
