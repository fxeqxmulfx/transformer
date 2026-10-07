import Transformer.Clusters.Section7_LogSumExp
import Mathlib.LinearAlgebra.Prod
import Mathlib.Tactic

/-!
# Joint structured likelihood at affine energies

New construction motivated by the complete raw Basis semantics and
arXiv:2305.05465v6, §7, Lemma (l:logsumexp). The existing finite Holder
proof there is reused for arbitrary linear parameter maps and offsets.
Each configuration can contain a causal route, an encoder state path,
matching channels and value channels. All their log potentials may be
trained jointly when the complete configuration energy is affine.

The objective is the observed complete-configuration negative log
likelihood. The observed route/states/channels would be supplied by data
semantics during training; inference sums over them. This is additional
supervision, not ordinary label-only cross entropy. Convexity of that
marginal label loss is not asserted. Nor is compact evaluation, a Basis
solver, a drop-in operator or AdamW convergence yet established here.
The parameter space is unrestricted and includes no hidden projection.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators
noncomputable section

variable {E R : Type*} [AddCommGroup E] [Module ℝ E] [Fintype R] [Nonempty R]

/-- A complete configuration's actual affine log energy in all raw parameters.
Source: the new structured construction, extending §7's linear log-sum-exp energies. -/
def energy (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) (z : R) : ℝ :=
  linear z θ + offset z

/-- The full finite Gibbs normalizer, including all unobserved configurations at inference.
Source: §7's positive exponential sum, applied to the complete structured energies. -/
def partition (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) : ℝ :=
  ∑ z, Real.exp (energy linear offset θ z)

/-- The actual normalized probability of one complete configuration.
Source: the new finite Gibbs distribution; observed configurations are not used in its normalizer. -/
def probability (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) (z : R) : ℝ :=
  Real.exp (energy linear offset θ z) / partition linear offset θ

/-- The complete-configuration training objective, with the observed configuration passed only to the loss.
Source: the new structured construction, log partition minus observed affine energy. -/
def jointNLL (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (observed : R) (θ : E) : ℝ :=
  Real.log (partition linear offset θ) - energy linear offset θ observed

omit [Fintype R] [Nonempty R] in
/-- Finite complete-configuration energies preserve actual convex mixtures of raw parameters.
Source: linearity plus offsets; the coefficient sum accounts for the affine constant. -/
theorem energy_mix (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (x y : E) (z : R)
    (a b : ℝ) (hab : a + b = 1) :
    energy linear offset (a • x + b • y) z =
      a * energy linear offset x z + b * energy linear offset y z := by
  simp only [energy, map_add, map_smul, smul_eq_mul]
  have ho : (a + b) * offset z = offset z := by rw [hab, one_mul]
  nlinarith [ho]

example : (1 / 2 : ℝ) + 1 / 2 = 1 := by norm_num

/-- The real Gibbs normalizer is positive for every unrestricted parameter assignment.
Source: the actual finite exponential sum and nonempty configuration domain. -/
theorem partition_pos (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) :
    0 < partition linear offset θ :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- Every complete configuration has positive actual probability at finite raw parameters.
Source: the true exponential and normalizer, without a probability-domain assumption. -/
theorem probability_pos (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) (z : R) :
    0 < probability linear offset θ z :=
  div_pos (Real.exp_pos _) (partition_pos linear offset θ)

/-- All complete configurations normalize together to one.
Source: the actual shared finite partition, not separately normalized route/value declarations. -/
theorem probability_sum (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) :
    ∑ z, probability linear offset θ z = 1 := by
  unfold probability
  rw [← Finset.sum_div]
  exact div_self (partition_pos linear offset θ).ne'

omit [Nonempty R] in
/-- The genuine partition includes each actual configuration's positive exponential weight.
Source: the shared finite sum over the complete inference domain. -/
theorem partition_ge (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) (z : R) :
    Real.exp (energy linear offset θ z) ≤ partition linear offset θ :=
  Finset.single_le_sum (fun _ _ => (Real.exp_pos _).le) (Finset.mem_univ z)

/-- No actual joint probability exceeds one.
Source: the finite positive Gibbs normalizer, including the tested configuration itself. -/
theorem probability_le_one (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E) (z : R) :
    probability linear offset θ z ≤ 1 :=
  (div_le_one (partition_pos linear offset θ)).mpr (partition_ge linear offset θ z)

/-- The complete objective is exactly the negative logarithm of the actual joint probability.
Source: finite Gibbs normalization, with no external encoder correctness or target-logit premise. -/
theorem jointNLL_eq (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (observed : R) (θ : E) :
    jointNLL linear offset observed θ = -Real.log (probability linear offset θ observed) := by
  unfold jointNLL probability
  rw [Real.log_div (Real.exp_pos _).ne' (partition_pos linear offset θ).ne', Real.log_exp]
  ring

omit [Nonempty R] in
/-- Every actual complete-configuration training loss is nonnegative at every raw parameter assignment.
Source: the true partition's observed summand and logarithm monotonicity. -/
theorem jointNLL_nonneg (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (observed : R) (θ : E) :
    0 ≤ jointNLL linear offset observed θ := by
  have h := Real.log_le_log (Real.exp_pos _) (partition_ge linear offset θ observed)
  rw [Real.log_exp] at h
  unfold jointNLL
  linarith

/-- The full actual structured log partition is jointly convex in every raw parameter coordinate.
Source: §7's Holder proof of log-sum-exp, now for arbitrary affine configuration energies. -/
theorem logPartition_convex (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) :
    ConvexOn ℝ Set.univ (fun θ => Real.log (partition linear offset θ)) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hx := partition_pos linear offset x
  have hy := partition_pos linear offset y
  have hs : ∀ z, Real.exp (energy linear offset (a • x + b • y) z) =
      Real.exp (energy linear offset x z) ^ a * Real.exp (energy linear offset y z) ^ b := by
    intro z
    rw [energy_mix linear offset x y z a b hab, Real.exp_add]
    rw [mul_comm a _, mul_comm b _, Real.exp_mul, Real.exp_mul]
  have hl : partition linear offset (a • x + b • y) ≤
      partition linear offset x ^ a * partition linear offset y ^ b := by
    unfold partition
    simp_rw [hs]
    exact Transformer.Clusters.sum_rpow_mul_rpow_le _ _ _
      (fun _ _ => (Real.exp_pos _).le) (fun _ _ => (Real.exp_pos _).le) hx hy ha hb hab
  calc
    Real.log (partition linear offset (a • x + b • y)) ≤
        Real.log (partition linear offset x ^ a * partition linear offset y ^ b) :=
      Real.log_le_log (partition_pos linear offset _) hl
    _ = a • Real.log (partition linear offset x) + b • Real.log (partition linear offset y) := by
      rw [Real.log_mul (by positivity) (by positivity), Real.log_rpow hx, Real.log_rpow hy]
      rfl

/-- Fully observed structured likelihood is convex jointly in all unrestricted trainable parameters.
Source: the actual complete Gibbs objective; subtracting an affine observed energy preserves §7's convexity. -/
theorem jointNLL_convex (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (observed : R) :
    ConvexOn ℝ Set.univ (jointNLL linear offset observed) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  have h := (logPartition_convex linear offset).2 hx hy ha hb hab
  simp only [smul_eq_mul] at h
  unfold jointNLL
  rw [energy_mix linear offset x y observed a b hab]
  simp only [smul_eq_mul]
  linarith

/-- A concrete two-configuration, two-parameter family supplies an unrestricted finite convex training control.
Source: the new joint objective with independent positive/negative first-coordinate energies. -/
example : ConvexOn ℝ Set.univ
    (jointNLL (fun z : Fin 2 => if z = 0 then LinearMap.fst ℝ ℝ ℝ else -LinearMap.fst ℝ ℝ ℝ)
      (fun z => if z = 0 then 0 else 1) 0) := jointNLL_convex _ _ _

end
end Transformer.GPTMini.Convex.Structured
