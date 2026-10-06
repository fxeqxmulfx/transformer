import Transformer.GPTMini.Sparsemax.IncidentRegularizedMemory

/-!
# Quantitative geometry descent in the actual common-value chart

Derived extra criterion after arXiv:1602.02068v2, Eq. (1). At fixed global
output coordinates, a feasible midpoint toward a geometry minimizer lowers
the regularized criterion by at least coefficient/4 times squared parameter
distance. Positive coefficient excludes geometric flatness away from that
minimum, including on support boundaries. No convexity of the output
criterion is needed for this fixed-output descent.

Approximate optimization bounds weighted squared geometry error. An
attained minimum of the output-coordinate objective, together with the
proved parameter projection, supplies an actual joint minimum. This makes
the remaining output-existence requirement explicit without choosing a
model task loss or claiming unconditioned text generalization.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A feasible midpoint toward selected geometry has a certified finite criterion gain.
Source: exact quadratic gap and the shared-value compensation chart after
arXiv:1602.02068v2, Eq. (1). Values are decoded globally at each endpoint. -/
theorem incidentMemoryRegularized_midpoint_gain {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (coefficient : ℝ)
    (objective : Matrix (Fin R) (Fin D) ℝ → ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hc : 0 ≤ coefficient)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor)
    (hm : IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) q) :
    localRegularizedMemoryObjective M reference coefficient objective
      ((1 / 2 : ℝ) • p + (1 / 2 : ℝ) • q, Z) ≤
    localRegularizedMemoryObjective M reference coefficient objective (p, Z) -
      coefficient / 4 * localMemoryQuadratic q p := by
  have hmid := incidentMemoryParameterDomain_convex N cap floor hp hq
    (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num)
  have hmin := hm hp
  change localMemoryQuadratic reference q ≤ localMemoryQuadratic reference p at hmin
  have hmul := mul_le_mul_of_nonneg_left hmin hc
  have hg := congrArg (fun z : ℝ => coefficient * z)
    (localMemoryQuadratic_affine_gap reference p q (1 / 2) (1 / 2) (by norm_num))
  unfold localRegularizedMemoryObjective
  rw [incidentJointMemoryForward_eq cap floor M _ hf hM hmid,
    incidentJointMemoryForward_eq cap floor M _ hf hM hp]
  change objective (M * Z) + coefficient * localMemoryQuadratic reference
    ((1 / 2 : ℝ) • p + (1 / 2 : ℝ) • q) ≤
      objective (M * Z) + coefficient * localMemoryQuadratic reference p -
        coefficient / 4 * localMemoryQuadratic q p
  nlinarith

/-- Changed geometry, nonconstant outputs and a real minimum jointly inhabit the finite-gain premises. -/
example : localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
    incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2)
    ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 3) + (1 / 2 : ℝ) • incidentMemoryExampleParameters,
      Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) ≤
    localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
      incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2)
      ((0 : LocalMemoryParameters 3), Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) -
        1 / 4 * localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) :=
  incidentMemoryRegularized_midpoint_gain 4 (3 / 4) _ _ _ _ _ _ _ (by norm_num)
    (oneHotContextCodes_mem _) (by norm_num)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem (fun p hp => localMemoryQuadratic_reference_min _ (Set.mem_univ p))

/-- Every different feasible geometry has a strict finite descent direction toward a geometry minimum.
Source: positive complete-coordinate curvature after arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryRegularized_strict_descent {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (coefficient : ℝ)
    (objective : Matrix (Fin R) (Fin D) ℝ → ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hc : 0 < coefficient)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor)
    (hm : IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) q)
    (hpq : p ≠ q) :
    localRegularizedMemoryObjective M reference coefficient objective
      ((1 / 2 : ℝ) • p + (1 / 2 : ℝ) • q, Z) <
    localRegularizedMemoryObjective M reference coefficient objective (p, Z) := by
  have hg := incidentMemoryRegularized_midpoint_gain cap floor M reference p q Z coefficient objective
    hf hM (by linarith) hp hq hm
  have hn := localMemoryQuadratic_pos q p hpq
  have hgain : 0 < coefficient / 4 * localMemoryQuadratic q p := mul_pos (by linarith) hn
  linarith

/-- A genuinely different feasible geometry satisfies every strict-descent premise. -/
example : localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
    incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2)
    ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 3) + (1 / 2 : ℝ) • incidentMemoryExampleParameters,
      Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) <
    localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
      incidentMemoryExampleParameters 1 (fun Y => (Y 0 0) ^ 2)
      ((0 : LocalMemoryParameters 3), Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) := by
  apply incidentMemoryRegularized_strict_descent 4 (3 / 4) _ _ _ _ _ _ _ (by norm_num)
    (oneHotContextCodes_mem _) (by norm_num)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem (fun p hp => localMemoryQuadratic_reference_min _ (Set.mem_univ p))
  intro h
  have he := congrArg (fun p : LocalMemoryParameters 3 => p.1 0) h
  norm_num [incidentMemoryExampleParameters] at he

/-- A regularized criterion gap at fixed outputs bounds weighted squared parameter error.
Source: the quadratic growth estimate and actual common-value chart after Eq. (1). -/
theorem incidentMemoryRegularized_suboptimal_error {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference p q : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (coefficient epsilon : ℝ)
    (objective : Matrix (Fin R) (Fin D) ℝ → ℝ) (hf : 1 / 2 < floor)
    (hM : M ∈ contextCodeDomain R N) (hc : 0 ≤ coefficient)
    (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (hq : q ∈ incidentMemoryParameterDomain N cap floor)
    (hm : IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) q)
    (he : localRegularizedMemoryObjective M reference coefficient objective (p, Z) -
      localRegularizedMemoryObjective M reference coefficient objective (q, Z) ≤ epsilon) :
    coefficient * localMemoryQuadratic q p ≤ 2 * epsilon := by
  have hg := mul_le_mul_of_nonneg_left
    (localMemoryQuadratic_min_growth reference p q _ (incidentMemoryParameterDomain_convex N cap floor)
      hp hq hm) hc
  unfold localRegularizedMemoryObjective at he
  rw [incidentJointMemoryForward_eq cap floor M _ hf hM hp,
    incidentJointMemoryForward_eq cap floor M _ hf hM hq] at he
  change objective (M * Z) + coefficient * localMemoryQuadratic reference p -
    (objective (M * Z) + coefficient * localMemoryQuadratic reference q) ≤ epsilon at he
  nlinarith

/-- A positive geometry gap and unchanged nonconstant outputs inhabit the error-bound premises. -/
example : localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) ≤
    2 * localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) := by
  have h := incidentMemoryRegularized_suboptimal_error 4 (3 / 4)
    (oneHotContextCodes (fun j : Fin 4 => j)) incidentMemoryExampleParameters
    (0 : LocalMemoryParameters 3) incidentMemoryExampleParameters
    (Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val + 1 : ℝ))) 1
    (localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3))
    (fun Y => (Y 0 0) ^ 2) (by norm_num) (oneHotContextCodes_mem _) (by norm_num)
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem (fun p hp => localMemoryQuadratic_reference_min _ (Set.mem_univ p))
  simpa only [one_mul] using h (by
    unfold localRegularizedMemoryObjective
    rw [incidentJointMemoryForward_parameter_invariant 4 (3 / 4) _ _ _ _ (by norm_num)
      (oneHotContextCodes_mem _)
      (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) incidentMemoryExampleParameters_mem,
      (localMemoryQuadratic_eq_zero _ _).2 rfl]
    simp only [one_mul, mul_zero, add_zero, add_sub_cancel_left, le_refl])

/-- An attained output-coordinate minimum and the proved geometry projection give an actual joint minimum.
Source: separability of the exact common-value chart after arXiv:1602.02068v2, Eq. (1).
Output-minimum attainment is explicit, while parameter-minimum existence is concluded. -/
theorem incidentMemoryRegularized_min_of_output_min {R N D : ℕ} (cap floor : ℝ)
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (reference : LocalMemoryParameters N)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (coefficient : ℝ)
    (objective : Matrix (Fin R) (Fin D) ℝ → ℝ) (hcap : 1 ≤ cap) (hf1 : floor ≤ 1)
    (hf : 1 / 2 < floor) (hM : M ∈ contextCodeDomain R N) (hc : 0 ≤ coefficient)
    (hZ : IsMinOn (fun W => objective (M * W)) Set.univ Z) :
    IsMinOn (localRegularizedMemoryObjective M reference coefficient objective)
      (incidentJointMemoryDomain N D cap floor)
      (incidentMemorySelectedParameters cap floor reference hcap hf1, Z) := by
  intro x hx
  change localRegularizedMemoryObjective M reference coefficient objective
    (incidentMemorySelectedParameters cap floor reference hcap hf1, Z) ≤
      localRegularizedMemoryObjective M reference coefficient objective x
  unfold localRegularizedMemoryObjective
  rw [incidentJointMemoryForward_eq cap floor M _ hf hM
    (incidentMemorySelectedParameters_mem cap floor reference hcap hf1),
    incidentJointMemoryForward_eq cap floor M x hf hM hx]
  have ho := hZ (Set.mem_univ x.2)
  change objective (M * Z) ≤ objective (M * x.2) at ho
  have hp := incidentMemorySelectedParameters_min cap floor reference hcap hf1 hx
  change localMemoryQuadratic reference (incidentMemorySelectedParameters cap floor reference hcap hf1) ≤
    localMemoryQuadratic reference x.1 at hp
  exact add_le_add ho (mul_le_mul_of_nonneg_left hp hc)

/-- Nonconstant output coordinates and infeasible geometry preferences inhabit actual joint attainment. -/
example : IsMinOn (localRegularizedMemoryObjective (oneHotContextCodes (fun j : Fin 4 => j))
    incidentMemoryOutsideReference 1 (fun Y => (Y 0 0) ^ 2)) (incidentJointMemoryDomain 3 1 4 (3 / 4))
    (incidentMemorySelectedParameters 4 (3 / 4) incidentMemoryOutsideReference (by norm_num) (by norm_num),
      Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) := by
  apply incidentMemoryRegularized_min_of_output_min _ _ _ _ _ _ _
    (by norm_num) (by norm_num) (by norm_num) (oneHotContextCodes_mem _) (by norm_num)
  intro W hW
  change ((oneHotContextCodes (fun j : Fin 4 => j) *
    Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ))) 0 0) ^ 2 ≤
      ((oneHotContextCodes (fun j : Fin 4 => j) * W) 0 0) ^ 2
  rw [oneHotContextCodes_mul, oneHotContextCodes_mul]
  norm_num [Matrix.of_apply]
  all_goals exact sq_nonneg (W 0 0)

end Transformer.GPTMini.Sparsemax
