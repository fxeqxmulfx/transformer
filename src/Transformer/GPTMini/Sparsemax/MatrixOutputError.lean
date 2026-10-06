import Mathlib.Analysis.Convex.Function
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Exact curvature of ordinary finite output error

Derived analytic facts for the outer loss after arXiv:1602.02068v2, §2.5.
The criterion is sum squared error against ordinary vector answers, with
no attention labels, embedding reference or additional penalty. The exact
affine gap is interpolation weight product times output-table distance.

Strict curvature applies to output coordinates. Later modules first prove
the actual masked sparsemax/common-value forward equals those coordinates,
then remove its residual geometry freedom by an explicit architectural
weight tie. This module does not assume that arbitrary Q/K training is
affine or that every answer is attainable under the energy restriction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Ordinary sum squared output error; there is no averaging or factor one half.
Source: the derived outer criterion after arXiv:1602.02068v2, §2.5. -/
def matrixOutputError {R D : ℕ} (target output : Matrix (Fin R) (Fin D) ℝ) : ℝ :=
  ∑ r, ∑ d, (output r d - target r d) ^ 2

/-- Ordinary output error is nonnegative for every finite answer table.
Source: the derived squared outer criterion after sparsemax §2.5. -/
theorem matrixOutputError_nonneg {R D : ℕ} (target output : Matrix (Fin R) (Fin D) ℝ) :
    0 ≤ matrixOutputError target output :=
  Finset.sum_nonneg (fun _ _ => Finset.sum_nonneg (fun _ _ => sq_nonneg _))

/-- Zero ordinary error means agreement in every observed answer coordinate.
Source: the finite squared criterion following arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_eq_zero {R D : ℕ} (target output : Matrix (Fin R) (Fin D) ℝ) :
    matrixOutputError target output = 0 ↔ output = target := by
  constructor
  · intro h
    have hr := (Finset.sum_eq_zero_iff_of_nonneg
      (fun r _ => Finset.sum_nonneg (fun d _ => sq_nonneg (output r d - target r d)))).1 h
    ext r d
    have hd := (Finset.sum_eq_zero_iff_of_nonneg
      (fun d _ => sq_nonneg (output r d - target r d))).1 (hr r (Finset.mem_univ r))
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp (hd d (Finset.mem_univ d)))
  · intro h
    rw [h]
    unfold matrixOutputError
    simp only [sub_self, zero_pow (by decide : (2 : ℕ) ≠ 0), Finset.sum_const_zero]

/-- Distinct answer tables have positive ordinary squared error.
Source: the separating outer criterion after arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_pos {R D : ℕ} (target output : Matrix (Fin R) (Fin D) ℝ)
    (h : output ≠ target) : 0 < matrixOutputError target output := by
  have hn := matrixOutputError_nonneg target output
  by_contra hp
  have hz : matrixOutputError target output = 0 := by linarith
  exact h ((matrixOutputError_eq_zero target output).mp hz)

/-- Nonconstant ordinary answers inhabit the strict positivity premise. -/
example : 0 < matrixOutputError (0 : Matrix (Fin 2) (Fin 1) ℝ)
    (Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ))) := by
  apply matrixOutputError_pos
  intro h
  have he := congrArg (fun Z : Matrix (Fin 2) (Fin 1) ℝ => Z 0 0) h
  norm_num [Matrix.of_apply] at he

/-- The exact affine gap is output distance times the two interpolation weights.
Source: the derived affine-output identity after arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_affine_gap {R D : ℕ} (target A B : Matrix (Fin R) (Fin D) ℝ)
    (a b : ℝ) (hab : a + b = 1) :
    matrixOutputError target (a • A + b • B) =
      a * matrixOutputError target A + b * matrixOutputError target B -
        a * b * matrixOutputError B A := by
  unfold matrixOutputError
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro r hr
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro d hd
  change (a * A r d + b * B r d - target r d) ^ 2 = _
  have ha : a = 1 - b := by linarith
  rw [ha]
  ring

/-- Distinct nonconstant output endpoints inhabit the exact midpoint gap. -/
example : matrixOutputError (0 : Matrix (Fin 2) (Fin 1) ℝ)
    ((1 / 2 : ℝ) • Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ)) + (1 / 2 : ℝ) • 0) =
    (1 / 2 : ℝ) * matrixOutputError 0 (Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ))) +
      (1 / 2 : ℝ) * matrixOutputError (0 : Matrix (Fin 2) (Fin 1) ℝ) 0 -
        (1 / 2 : ℝ) * (1 / 2 : ℝ) *
          matrixOutputError 0 (Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ))) :=
  matrixOutputError_affine_gap _ _ _ _ _ (by norm_num)

/-- Ordinary squared output error is convex on all finite output tables.
Source: the affine gap derived after sparsemax §2.5; no Q/K affinity is assumed here. -/
theorem matrixOutputError_convex {R D : ℕ} (target : Matrix (Fin R) (Fin D) ℝ) :
    ConvexOn ℝ Set.univ (matrixOutputError target) := by
  refine ⟨convex_univ, ?_⟩
  intro A hA B hB a b ha hb hab
  change matrixOutputError target (a • A + b • B) ≤
    a * matrixOutputError target A + b * matrixOutputError target B
  have hg := matrixOutputError_affine_gap target A B a b hab
  have hn := mul_nonneg (mul_nonneg ha hb) (matrixOutputError_nonneg B A)
  linarith

/-- There are no affine flat segments between distinct full observed output tables.
Source: strict ordinary output curvature derived after arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_strictConvex {R D : ℕ} (target : Matrix (Fin R) (Fin D) ℝ) :
    StrictConvexOn ℝ Set.univ (matrixOutputError target) := by
  refine ⟨convex_univ, ?_⟩
  intro A hA B hB hAB a b ha hb hab
  change matrixOutputError target (a • A + b • B) <
    a * matrixOutputError target A + b * matrixOutputError target B
  have hg := matrixOutputError_affine_gap target A B a b hab
  have hn := mul_pos (mul_pos ha hb) (matrixOutputError_pos B A hAB)
  linarith

/-- An exactly fitting affine endpoint gives quadratic decay at every real interpolation time.
Source: the ordinary-output analogue of the derived sparsemax §2.5 segment argument. -/
theorem matrixOutputError_target_segment {R D : ℕ} (target A : Matrix (Fin R) (Fin D) ℝ)
    (t : ℝ) : matrixOutputError target ((1 - t) • A + t • target) =
      (1 - t) ^ 2 * matrixOutputError target A := by
  unfold matrixOutputError
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r hr
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro d hd
  change ((1 - t) * A r d + t * target r d - target r d) ^ 2 = _
  ring

/-- Each individual answer-coordinate error is controlled by the ordinary total error.
Source: the derived finite observation bound after arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_entry_bound {R D : ℕ} (target A : Matrix (Fin R) (Fin D) ℝ)
    (r : Fin R) (d : Fin D) : (A r d - target r d) ^ 2 ≤ matrixOutputError target A := by
  exact (Finset.single_le_sum (fun j _ => sq_nonneg (A r j - target r j))
    (Finset.mem_univ d)).trans (Finset.single_le_sum
      (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg (A i j - target i j)))
      (Finset.mem_univ r))

/-- A constrained minimum has quantitative output-coordinate growth, without exact fit.
Source: midpoint comparison of the ordinary squared-output gap after sparsemax §2.5. -/
theorem matrixOutputError_min_growth {R D : ℕ} (target A B : Matrix (Fin R) (Fin D) ℝ)
    (s : Set (Matrix (Fin R) (Fin D) ℝ)) (hs : Convex ℝ s)
    (hA : A ∈ s) (hB : B ∈ s) (hm : IsMinOn (matrixOutputError target) s B) :
    matrixOutputError B A ≤ 2 * (matrixOutputError target A - matrixOutputError target B) := by
  have hmid := hs hA hB (by norm_num : 0 ≤ (1 / 2 : ℝ))
    (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num)
  have hmin := hm hmid
  change matrixOutputError target B ≤
    matrixOutputError target ((1 / 2 : ℝ) • A + (1 / 2 : ℝ) • B) at hmin
  rw [matrixOutputError_affine_gap target A B _ _ (by norm_num)] at hmin
  linarith

/-- A nonconstant output table and an attained zero answer minimum inhabit the growth premises. -/
example : matrixOutputError (0 : Matrix (Fin 2) (Fin 1) ℝ)
    (Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ))) ≤
    2 * (matrixOutputError 0 (Matrix.of (fun (r : Fin 2) (d : Fin 1) => (r.val + d.val + 1 : ℝ))) -
      matrixOutputError (0 : Matrix (Fin 2) (Fin 1) ℝ) 0) := by
  apply matrixOutputError_min_growth _ _ _ Set.univ convex_univ (Set.mem_univ _) (Set.mem_univ _)
  intro Z hZ
  rw [(matrixOutputError_eq_zero _ _).mpr rfl]
  exact matrixOutputError_nonneg _ Z

end Transformer.GPTMini.Sparsemax
