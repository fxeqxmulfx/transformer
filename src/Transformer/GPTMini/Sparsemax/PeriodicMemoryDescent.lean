import Transformer.GPTMini.Sparsemax.PeriodicMemoryForward
import Transformer.GPTMini.Sparsemax.MatrixOutputError
import Transformer.GPTMini.Sparsemax.SquaredSegment
import Mathlib.Analysis.Convex.Extrema

/-!
# General feasible descent for constant-width attention and learned values

New outer-loss guarantee after arXiv:1602.02068v2, §2.5. For any finite
dictionary, observation codes and convex ordinary output criterion, every
better feasible joint endpoint supplies strict descent at all positive
segment times up to one. Every constrained local minimum is global. The
statement needs neither an exactly fitting target nor fixed sparse support.

For ordinary squared error an actually fitting feasible endpoint gives
exact `(1-time)^2` decay, for arbitrary vector answer tables. Values are
decoded globally along the genuine masked sparsemax path. Geometry can
still change at fixed output coordinates in this untied chart; later
modules impose an explicit affine weight tie to remove that freedom.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Ordinary vector-answer error of the actual width-three sparsemax/common-value forward.
Source: the derived outer criterion following arXiv:1602.02068v2, §2.5. -/
def periodicEnergySquaredError {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (target : Matrix (Fin R) (Fin D) ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) : ℝ :=
  matrixOutputError target (periodicEnergyForward code x)

/-- Actual ordinary prediction error is nonnegative even outside the inverse domain.
Source: the genuine outer criterion following sparsemax §2.5. -/
theorem periodicEnergySquaredError_nonneg {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (target : Matrix (Fin R) (Fin D) ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) :
    0 ≤ periodicEnergySquaredError code target x :=
  matrixOutputError_nonneg _ _

/-- On the feasible domain, ordinary error observes exactly the coded output table.
Source: the proved inverse of actual sparsemax after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergySquaredError_eq {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (target : Matrix (Fin R) (Fin D) ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy) :
    periodicEnergySquaredError code target x = matrixOutputError target
      (Matrix.of (fun r => x.2 (code r))) := by
  unfold periodicEnergySquaredError
  rw [periodicEnergyForward_eq floor budget energy code x hf hx]

/-- A genuine nonidentity, nonconstant answer fit inhabits the error-reduction premises. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) =
    matrixOutputError (taskEnergyTarget 0) (taskEnergyTarget 0) :=
  periodicEnergySquaredError_eq (3 / 4) (1 / 8) 6 _ _ _ (by norm_num)
    (taskPeriodicEnergyPair_mem 0)

/-- Ordinary vector-answer error is jointly convex for every finite observation table.
Source: the actual affine masked sparsemax/common-value chart after sparsemax §2.5. -/
theorem periodicEnergySquaredError_convex {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (target : Matrix (Fin R) (Fin D) ℝ)
    (hf : 1 / 2 < floor) :
    ConvexOn ℝ (periodicEnergyDomain N D floor budget energy)
      (periodicEnergySquaredError code target) :=
  periodicEnergyObjective_convex floor budget energy code (matrixOutputError target)
    hf (matrixOutputError_convex target)

/-- Arbitrary ordinary three-slot answer vectors inhabit the joint-convexity premise. -/
example : ConvexOn ℝ (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6)
    (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)) :=
  periodicEnergySquaredError_convex _ _ _ _ _ (by norm_num)

/-- Any better feasible endpoint gives strict arbitrarily short joint descent.
Source: convex outer criteria on the genuine masked sparsemax chart after §2.5.
Targets need not fit exactly; the entire segment remains feasible across support changes. -/
theorem periodicEnergyObjective_strict_descent {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hl : ConvexOn ℝ Set.univ objective)
    (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (hy : y ∈ periodicEnergyDomain N D floor budget energy)
    (hbetter : objective (periodicEnergyForward code y) < objective (periodicEnergyForward code x))
    (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) :
    objective (periodicEnergyForward code ((1 - t) • x + t • y)) <
      objective (periodicEnergyForward code x) := by
  have hc := periodicEnergyObjective_convex floor budget energy code objective hf hl
  have hb := hc.2 hx hy (by linarith : 0 ≤ 1 - t) (by linarith : 0 ≤ t) (by ring)
  change objective (periodicEnergyForward code ((1 - t) • x + t • y)) ≤
    (1 - t) * objective (periodicEnergyForward code x) +
      t * objective (periodicEnergyForward code y) at hb
  exact lt_of_le_of_lt hb (lossSegment_strict_decrease _ _ t hbetter ht)

/-- Opposite edge/answer witnesses inhabit every better-endpoint descent premise. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((1 - (1 / 2 : ℝ)) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0)) <
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) := by
  apply periodicEnergyObjective_strict_descent (3 / 4) (1 / 8) 6 _
    (matrixOutputError (taskEnergyTarget 0)) _ _ (by norm_num)
    (matrixOutputError_convex _) (taskPeriodicEnergyPair_mem 1) (taskPeriodicEnergyPair_mem 0)
  · rw [periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num)
      (taskPeriodicEnergyPair_mem 0), periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _
      (by norm_num) (taskPeriodicEnergyPair_mem 1)]
    change matrixOutputError (taskEnergyTarget 0) (taskEnergyTarget 0) <
      matrixOutputError (taskEnergyTarget 0) (taskEnergyTarget 1)
    rw [(matrixOutputError_eq_zero _ _).mpr rfl]
    norm_num [matrixOutputError, taskEnergyTarget, Fin.sum_univ_succ, Matrix.of_apply]
  · norm_num
  · norm_num

/-- Every constrained local minimum of any convex ordinary output criterion is global.
Source: actual chart affinity plus Mathlib's convex extrema theorem after sparsemax §2.5. -/
theorem periodicEnergyObjective_local_min_global {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hl : ConvexOn ℝ Set.univ objective)
    (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (hm : IsLocalMinOn (fun p => objective (periodicEnergyForward code p))
      (periodicEnergyDomain N D floor budget energy) x) :
    IsMinOn (fun p => objective (periodicEnergyForward code p))
      (periodicEnergyDomain N D floor budget energy) x := by
  exact IsMinOn.of_isLocalMinOn_of_convexOn hx hm
    (periodicEnergyObjective_convex floor budget energy code objective hf hl)

/-- An actual fitting point is a minimum of ordinary output error over all parameter points.
Source: exact ordinary-answer fitting following sparsemax §2.5; no feasibility is inferred. -/
theorem periodicEnergySquaredError_exact_min {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (target : Matrix (Fin R) (Fin D) ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hfit : periodicEnergyForward code x = target) :
    IsMinOn (periodicEnergySquaredError code target) Set.univ x := by
  intro y hy
  change matrixOutputError target (periodicEnergyForward code x) ≤ _
  rw [hfit, (matrixOutputError_eq_zero _ _).mpr rfl]
  exact periodicEnergySquaredError_nonneg code target y

/-- Nonidentity actual attention inhabits the exact-minimum premise. -/
example : IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)) Set.univ
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) :=
  periodicEnergySquaredError_exact_min _ _ _
    (periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num) (taskPeriodicEnergyPair_mem 0))

/-- A nonconstant actual fit supplies the local-minimum premise without assuming existence. -/
example : IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
    (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) := by
  have hm : IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
      (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) := by
    intro y hy
    exact periodicEnergySquaredError_exact_min _ _ _
      (periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num)
        (taskPeriodicEnergyPair_mem 0)) (Set.mem_univ y)
  exact periodicEnergyObjective_local_min_global (3 / 4) (1 / 8) 6 _
    (matrixOutputError (taskEnergyTarget 0)) _ (by norm_num) (matrixOutputError_convex _)
    (taskPeriodicEnergyPair_mem 0) hm.isLocalMinOn

/-- A feasible exact fit gives exact quadratic error decay with all Q/K and values learned jointly.
Source: actual affine outputs and the ordinary squared segment after sparsemax §2.5. -/
theorem periodicEnergySquaredError_target_segment {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (target : Matrix (Fin R) (Fin D) ℝ)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (hy : y ∈ periodicEnergyDomain N D floor budget energy)
    (hfit : periodicEnergyForward code y = target) (t : ℝ) (ht : 0 ≤ t) (hu : t ≤ 1) :
    periodicEnergySquaredError code target ((1 - t) • x + t • y) =
      (1 - t) ^ 2 * periodicEnergySquaredError code target x := by
  unfold periodicEnergySquaredError
  rw [periodicEnergyForward_affine floor budget energy code x y hf hx hy
    (1 - t) t (by linarith) ht (by ring), hfit]
  exact matrixOutputError_target_segment _ _ _

/-- Different learned supports and answers inhabit every exact quadratic-decay premise. -/
example : periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0)) =
    (1 / 2 : ℝ) ^ 2 * periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)
      ((taskEnergyParameters 1).1, taskEnergyTarget 1) := by
  have h := periodicEnergySquaredError_target_segment (3 / 4) (1 / 8) 6
    (fun j : Fin 3 => j) (taskEnergyTarget 0)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1)
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) (by norm_num)
    (taskPeriodicEnergyPair_mem 1) (taskPeriodicEnergyPair_mem 0)
    (periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num)
      (taskPeriodicEnergyPair_mem 0)) (1 / 2) (by norm_num) (by norm_num)
  norm_num only at h
  convert h using 1
  norm_num

end Transformer.GPTMini.Sparsemax
