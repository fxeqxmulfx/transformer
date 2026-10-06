import Transformer.GPTMini.Sparsemax.IncidentMemorySelection
import Transformer.GPTMini.Sparsemax.IncidentJointMemory
import Transformer.GPTMini.Sparsemax.LocalRegularizedMemory

/-!
# Additional geometry selection with separate-budget joint convex learning

Derived extension after arXiv:1602.02068v2, Eq. (1). The unchanged
regularized actual shared-value forward remains convex for any fixed
reference, nonnegative coefficient and convex output criterion. Every
attained joint minimum uses the unique constrained geometry projection
when the additional coefficient is positive, even at different outputs.

The reference can now be computed from observations by the adjacent
module. This criterion resolves the exact geometry/value compensation
freedom; output-only fitting still cannot identify those parameters.
Full joint-minimum existence requires a property of the output objective.
Parameter-projection existence is already proved without assuming it.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A preferred geometry outside both separate incident budgets and the norm cap.
Source: an infeasible-reference witness for arXiv:1602.02068v2, Eq. (1). -/
def incidentMemoryOutsideReference : LocalMemoryParameters 3 :=
  ((fun _ => 1), (fun _ => 5))

/-- The enlarged structural domain still excludes this preferred geometry.
Source: the actual norm bound for arXiv:1602.02068v2, Eq. (1) memory. -/
theorem incidentMemoryOutsideReference_not_mem :
    incidentMemoryOutsideReference ∉ incidentMemoryParameterDomain 3 4 (3 / 4) := by
  intro h
  have hb := (h.2 (Sum.inl 0)).2
  norm_num [incidentMemoryOutsideReference] at hb

/-- The additional nonnegative selection term preserves convexity of the entire joint task.
Source: the compact shared-value chart and squared-coordinate criterion for
arXiv:1602.02068v2, Eq. (1). The output criterion's convexity remains explicit. -/
theorem incidentRegularizedMemoryObjective_convex {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference : LocalMemoryParameters N)
    (coefficient : ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N) (hc : 0 ≤ coefficient)
    (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (incidentJointMemoryDomain N D cap floor)
      (localRegularizedMemoryObjective M reference coefficient objective) := by
  refine ⟨incidentJointMemoryDomain_convex N D cap floor, ?_⟩
  intro x hx y hy a b ha hb hab
  have ho := (incidentJointMemoryObjective_convex cap floor M objective hf hM hl).2 hx hy ha hb hab
  have hp := (localMemoryQuadratic_convex reference).2
    (Set.mem_univ x.1) (Set.mem_univ y.1) ha hb hab
  have hs := mul_le_mul_of_nonneg_left hp hc
  change objective (localJointMemoryForward M (a • x + b • y)) ≤
    a * objective (localJointMemoryForward M x) + b * objective (localJointMemoryForward M y) at ho
  change _ ≤ a * localRegularizedMemoryObjective M reference coefficient objective x +
    b * localRegularizedMemoryObjective M reference coefficient objective y
  unfold localRegularizedMemoryObjective
  calc
    _ ≤ a * objective (localJointMemoryForward M x) + b * objective (localJointMemoryForward M y) +
        coefficient * (a * localMemoryQuadratic reference x.1 + b * localMemoryQuadratic reference y.1) :=
      add_le_add ho hs
    _ = _ := by ring

/-- A nonconstant convex output criterion and infeasible reference inhabit all joint-convexity premises. -/
example : ConvexOn ℝ (incidentJointMemoryDomain 3 1 4 (3 / 4))
    (localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
      incidentMemoryOutsideReference 1 (fun Y => (Y 0 0) ^ 2)) := by
  apply incidentRegularizedMemoryObjective_convex _ _ _ _ _ _ (by norm_num)
    (oneHotContextCodes_mem _) (by norm_num)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X _ Y _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (X 0 0)) (Set.mem_univ (Y 0 0)) ha hb hab

/-- Every attained joint minimum also minimizes the additional parameter criterion.
Source: the exact output-only compensation freedom after arXiv:1602.02068v2, Eq. (1).
Strictly positive coefficient is required; convexity of the output criterion is not. -/
theorem incidentRegularizedMemoryObjective_parameter_min {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference : LocalMemoryParameters N)
    (coefficient : ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N) (hc : 0 < coefficient)
    (hx : x ∈ incidentJointMemoryDomain N D cap floor)
    (hm : IsMinOn (localRegularizedMemoryObjective M reference coefficient objective)
      (incidentJointMemoryDomain N D cap floor) x) :
    IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) x.1 := by
  intro p hp
  have hh := hm (show (p, x.2) ∈ incidentJointMemoryDomain N D cap floor from hp)
  change localRegularizedMemoryObjective M reference coefficient objective x ≤
    localRegularizedMemoryObjective M reference coefficient objective (p, x.2) at hh
  unfold localRegularizedMemoryObjective at hh
  rw [incidentJointMemoryForward_eq cap floor M x hf hM hx,
    incidentJointMemoryForward_eq cap floor M (p, x.2) hf hM hp] at hh
  have hmul : coefficient * localMemoryQuadratic reference x.1 ≤
      coefficient * localMemoryQuadratic reference p := by linarith
  exact (mul_le_mul_iff_right₀ hc).1 hmul

/-- All attained joint minima have identical compact attention parameters, even at different outputs.
Source: the extra strictly convex criterion for arXiv:1602.02068v2, Eq. (1).
This is selection by the additional criterion, not output-only identification. -/
theorem incidentRegularizedMemoryObjective_parameters_unique {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference : LocalMemoryParameters N)
    (coefficient : ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (x y : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N) (hc : 0 < coefficient)
    (hx : x ∈ incidentJointMemoryDomain N D cap floor) (hy : y ∈ incidentJointMemoryDomain N D cap floor)
    (hxm : IsMinOn (localRegularizedMemoryObjective M reference coefficient objective)
      (incidentJointMemoryDomain N D cap floor) x)
    (hym : IsMinOn (localRegularizedMemoryObjective M reference coefficient objective)
      (incidentJointMemoryDomain N D cap floor) y) : x.1 = y.1 :=
  localMemoryQuadratic_unique_min reference _ (incidentMemoryParameterDomain_convex N cap floor)
    x.1 y.1 hx hy
    (incidentRegularizedMemoryObjective_parameter_min cap floor M reference coefficient objective x hf hM hc hx hxm)
    (incidentRegularizedMemoryObjective_parameter_min cap floor M reference coefficient objective y hf hM hc hy hym)

/-- Every attained joint minimum uses the proved unique constrained parameter projection.
Source: the additional strictly convex selection after arXiv:1602.02068v2, Eq. (1).
The reference is not required to satisfy the architectural constraints. -/
theorem incidentRegularizedMemoryObjective_selected {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference : LocalMemoryParameters N)
    (coefficient : ℝ) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hcap : 1 ≤ cap) (hf1 : floor ≤ 1) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hc : 0 < coefficient)
    (hx : x ∈ incidentJointMemoryDomain N D cap floor)
    (hm : IsMinOn (localRegularizedMemoryObjective M reference coefficient objective)
      (incidentJointMemoryDomain N D cap floor) x) :
    x.1 = incidentMemorySelectedParameters cap floor reference hcap hf1 :=
  localMemoryQuadratic_unique_min reference _ (incidentMemoryParameterDomain_convex N cap floor)
    x.1 _ hx (incidentMemorySelectedParameters_mem cap floor reference hcap hf1)
    (incidentRegularizedMemoryObjective_parameter_min cap floor M reference coefficient objective x hf hM hc hx hm)
    (incidentMemorySelectedParameters_min cap floor reference hcap hf1)

/-- The nonidentity compact witness attains the joint minimum whenever the first output is zero.
Source: a nonconstant squared-output example for arXiv:1602.02068v2, Eq. (1),
with an explicit additional criterion; no model task loss is selected. -/
theorem incidentMemorySelectionExample_min (Z : Matrix (Fin 4) (Fin 1) ℝ) (hZ : Z 0 0 = 0) :
    IsMinOn (localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
      incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2))
      (incidentJointMemoryDomain 3 1 4 (3 / 4)) (incidentMemoryExampleParameters, Z) := by
  have hz : localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
      incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2) (incidentMemoryExampleParameters, Z) = 0 := by
    unfold localRegularizedMemoryObjective
    rw [incidentJointMemoryForward_eq 4 (3 / 4) _ _ (by norm_num) (oneHotContextCodes_mem _)
      incidentMemoryExampleParameters_mem, oneHotContextCodes_mul,
      (localMemoryQuadratic_eq_zero _ _).2 rfl]
    norm_num [Matrix.of_apply, hZ]
  intro x hx
  rw [hz]
  exact add_nonneg (sq_nonneg _) (mul_nonneg (by norm_num) (localMemoryQuadratic_nonneg _ _))

/-- Nonconstant global outputs inhabit the attained joint-minimum premise. -/
example : IsMinOn (localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
    incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2)) (incidentJointMemoryDomain 3 1 4 (3 / 4))
    (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) :=
  incidentMemorySelectionExample_min _ (by norm_num)

/-- Changed feasible parameters and an attained joint minimum inhabit every parameter-minimum premise. -/
example : IsMinOn (localMemoryQuadratic incidentMemoryExampleParameters)
    (incidentMemoryParameterDomain 3 4 (3 / 4)) incidentMemoryExampleParameters :=
  incidentRegularizedMemoryObjective_parameter_min 4 (3 / 4) (oneHotContextCodes (fun j : Fin 4 => j))
    _ 1 (fun Y => (Y 0 0) ^ 2)
    (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ)))
    (by norm_num) (oneHotContextCodes_mem _) (by norm_num) incidentMemoryExampleParameters_mem
    (incidentMemorySelectionExample_min _ (by norm_num))

/-- Two distinct nonconstant output tables can minimize jointly while their attention parameters coincide. -/
example : (incidentMemoryExampleParameters,
    Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))).1 =
    (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (2 * j.val : ℝ))).1 :=
  incidentRegularizedMemoryObjective_parameters_unique 4 (3 / 4) (oneHotContextCodes (fun j : Fin 4 => j))
    _ 1 (fun Y => (Y 0 0) ^ 2) _ _ (by norm_num) (oneHotContextCodes_mem _) (by norm_num)
    incidentMemoryExampleParameters_mem incidentMemoryExampleParameters_mem
    (incidentMemorySelectionExample_min _ (by norm_num)) (incidentMemorySelectionExample_min _ (by norm_num))

/-- Nonidentity attention and an attained nonconstant-output minimum inhabit every selection premise. -/
example : incidentMemoryExampleParameters = incidentMemorySelectedParameters 4 (3 / 4)
    incidentMemoryExampleParameters (by norm_num) (by norm_num) :=
  incidentRegularizedMemoryObjective_selected 4 (3 / 4) (oneHotContextCodes (fun j : Fin 4 => j))
    _ 1 (fun Y => (Y 0 0) ^ 2)
    (incidentMemoryExampleParameters, Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ)))
    (by norm_num) (by norm_num) (by norm_num) (oneHotContextCodes_mem _) (by norm_num)
    incidentMemoryExampleParameters_mem (incidentMemorySelectionExample_min _ (by norm_num))

end Transformer.GPTMini.Sparsemax
