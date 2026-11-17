module

public import Transformer.Clusters.Section8_SoftmaxEnergy
public import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Actual shared learned content and relative-position retrieval

Source: `review/flat_spectrum.md`, CM111. Query/key embeddings and value/class
embeddings are shared learned maps. All positions and original classes occur
in the forward. Norm restrictions bound capacity; they do not specify labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval

open scoped BigOperators
open Perspective

variable (O E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- Complete learned maps and learned relative potential, CM111. -/
structure Parameters where
  key : O → E
  value : O → E
  scale : ℝ
  relation : ℤ → ℝ

variable {O E}

/-- Both shared maps have unit capacity, CM111. No object roles occur. -/
def Unit (p : Parameters O E) : Prop :=
  (∀ o, ‖p.key o‖ ≤ 1) ∧ ∀ o, ‖p.value o‖ ≤ 1

/-- Actual shared value/readout interaction, CM111. -/
def kernel (p : Parameters O E) (c o : O) : ℝ :=
  (inner ℂ (p.value c) (p.value o)).re

/-- Learned content scores against every past object, CM111. -/
def keyScores {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O) (j : Fin n) : ℝ :=
  p.scale * (inner ℂ (p.key q) (p.key (x j))).re

/-- The same learned relative potential is used at every position, CM111. -/
def relativeScores {n : ℕ} (p : Parameters O E) (j k : Fin n) : ℝ :=
  p.relation ((k : ℤ) - (j : ℤ))

/-- Actual normalized weight of a complete visible two-stage path, CM111. -/
noncomputable def pathMass {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O)
    (jk : Fin n × Fin n) : ℝ :=
  softmaxWeight (keyScores p x q) jk.1 * softmaxWeight (relativeScores p jk.1) jk.2

/-- Physical pooled learned values, CM111. -/
noncomputable def state {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O) : E :=
  ∑ jk, (pathMass p x q jk : ℂ) • p.value (x jk.2)

/-- Complete original-class logit, evaluated through every path, CM111. -/
noncomputable def logit {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q c : O) : ℝ :=
  ∑ jk, pathMass p x q jk * kernel p c (x jk.2)

/-- The path formula is exactly the actual shared complex readout, CM111. -/
theorem readout_state {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q c : O) :
    (inner ℂ (p.value c) (state p x q)).re = logit p x q c := by
  simp only [state, logit, kernel, inner_sum, inner_smul_right, Complex.re_sum,
    Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]

/-- Object sequences remain the sole model input, CM111. -/
noncomputable def score (p : Parameters O E) (xs : List O) (c : O) : ℝ :=
  match xs.reverse with
  | [] => 0
  | q :: ys => logit p (fun j : Fin ys.reverse.length => ys.reverse[j]) q c

/-- Unit learned readouts bound every actual value/class interaction, CM111. -/
theorem kernel_bound (p : Parameters O E) (hp : Unit p) (c o : O) :
    |kernel p c o| ≤ 1 := by
  calc
    |kernel p c o| ≤ ‖inner ℂ (p.value c) (p.value o)‖ := Complex.abs_re_le_norm _
    _ ≤ ‖p.value c‖ * ‖p.value o‖ := norm_inner_le_norm _ _
    _ ≤ 1 := by nlinarith [hp.2 c, hp.2 o, norm_nonneg (p.value c), norm_nonneg (p.value o)]

/-- All actual path weights are nonnegative, CM111. -/
theorem pathMass_nonneg {n : ℕ} (p : Parameters O E) (x : Fin n → O) (q : O)
    (jk : Fin n × Fin n) : 0 ≤ pathMass p x q jk :=
  mul_nonneg (softmaxWeight_nonneg _ _) (softmaxWeight_nonneg _ _)

/-- Both complete softmax normalizations survive composition, CM111. -/
theorem sum_pathMass {n : ℕ} (hn : 0 < n) (p : Parameters O E) (x : Fin n → O) (q : O) :
    ∑ jk, pathMass p x q jk = 1 := by
  simp only [pathMass, Fintype.sum_prod_type, ← Finset.mul_sum,
    sum_softmaxWeight hn, mul_one]

/-- The literal learned forward has unit score capacity at every width and
memory length, including empty memory, CM111. -/
theorem logit_bound {n : ℕ} (p : Parameters O E) (hp : Unit p)
    (x : Fin n → O) (q c : O) : |logit p x q c| ≤ 1 := by
  cases n with
  | zero => simp [logit]
  | succ n =>
    calc
      |logit p x q c| ≤ ∑ jk, |pathMass p x q jk * kernel p c (x jk.2)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ jk, pathMass p x q jk := by
        apply Finset.sum_le_sum
        intro jk hjk
        rw [abs_mul, abs_of_nonneg (pathMass_nonneg p x q jk)]
        exact mul_le_of_le_one_right (pathMass_nonneg p x q jk) (kernel_bound p hp c _)
      _ = 1 := sum_pathMass (Nat.zero_lt_succ _) p x q

/-- The same bound applies to actual raw sequences, CM111. -/
theorem score_bound (p : Parameters O E) (hp : Unit p) (xs : List O) (c : O) :
    |score p xs c| ≤ 1 := by
  cases h : xs.reverse with
  | nil => simp [score, h]
  | cons q ys =>
    simpa only [score, h] using (logit_bound p hp
      (fun j : Fin ys.reverse.length => ys.reverse[j]) q c)

/-- A complete zero parameter point inhabits the capacity domain, CM111. -/
noncomputable def zeroParameters : Parameters O E := ⟨fun _ => 0, fun _ => 0, 0, fun _ => 0⟩

omit [InnerProductSpace ℂ E] in
/-- The complete zero maps satisfy both capacity restrictions, CM111. -/
theorem zero_unit : Unit (zeroParameters : Parameters O E) := by
  simp [Unit, zeroParameters]

/-- Normalize complete complex learned vectors without deleting coordinates,
CM111. This is an exact-real feasibility map. -/
noncomputable def normalize (v : E) : E :=
  (((max 1 ‖v‖)⁻¹ : ℝ) : ℂ) • v

/-- Full-vector normalization enforces unit capacity, CM111. -/
theorem normalize_bound (v : E) : ‖normalize v‖ ≤ 1 := by
  have hd : 0 < max (1 : ℝ) ‖v‖ := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  simpa only [normalize, norm_smul, Complex.norm_real, Real.norm_eq_abs, abs_inv,
    abs_of_pos hd, div_eq_mul_inv, mul_comm] using (div_le_one hd).mpr (le_max_right 1 ‖v‖)

/-- Every already feasible vector is retained exactly, CM111. -/
theorem normalize_fixed (v : E) (hv : ‖v‖ ≤ 1) : normalize v = v := by
  simp [normalize, max_eq_left hv]

/-- Normalize both complete shared maps and retain all other learned data,
CM111. Relative positions and scales are not supplied by a task. -/
noncomputable def project (p : Parameters O E) : Parameters O E :=
  ⟨fun o => normalize (p.key o), fun o => normalize (p.value o), p.scale, p.relation⟩

/-- The full parameter feasibility map satisfies both unit restrictions,
CM111. No floating-point or optimizer claim follows. -/
theorem project_unit (p : Parameters O E) : Unit (project p) :=
  ⟨fun o => normalize_bound (p.key o), fun o => normalize_bound (p.value o)⟩

example : normalize Complex.I = Complex.I := normalize_fixed _ (by simp)

example : |kernel (zeroParameters : Parameters Bool ℂ) true false| ≤ 1 :=
  kernel_bound _ zero_unit _ _

example : ∑ jk, pathMass (zeroParameters : Parameters Bool ℂ)
    (fun _ : Fin 2 => false) true jk = 1 := sum_pathMass (by norm_num) _ _ _

example : |logit (zeroParameters : Parameters Bool ℂ) (fun _ : Fin 2 => false) true false| ≤ 1 :=
  logit_bound _ zero_unit _ _ _

example : |score (zeroParameters : Parameters Bool ℂ) [true, false] true| ≤ 1 :=
  score_bound _ zero_unit _ _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval
