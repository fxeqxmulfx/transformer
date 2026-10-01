/-
Copyright (c) 2026 Raphael Coelho. All rights reserved.
Released under Apache 2.0 license as described in third_party/Stochastic/LICENSE.
Adapted from proofs by Raphael Coelho.

# The finite adapted Itô isometry for the constructed vector driver

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The finite-sum proof adapts MathFin/Foundations/ItoIsometryAdapted.lean:
https://github.com/formal-applied-math/formal-mathfin.
All coefficients are measurable in the joint vector filtration.
-/

import Transformer.BatchSize.Section4_BrownianMoments

open MeasureTheory ProbabilityTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- A finite left-adapted stochastic sum driven by coordinate k,
Section 4.3 (2)--(3). -/
def brownianItoSum {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (H : ℕ → BrownianSample d → ℝ) (n : ℕ) (ω : BrownianSample d) : ℝ :=
  ∑ j ∈ Finset.range n, H j ω * brownianIncrement k (t j) (t (j + 1)) ω

/-- Adaptedness of the finite scalar sum at its final grid time,
Section 4.3 (2)--(3), relative to the joint vector past. -/
theorem brownianItoSum_adapted {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n)) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (brownianItoSum k t H n) := by
  let : MeasurableSpace (BrownianSample d) := brownianFiltration d (t n)
  apply Finset.stronglyMeasurable_fun_sum
  intro j hj
  have hjn := Finset.mem_range.mp hj
  have hcoef := (hH j).mono ((brownianFiltration d).mono (hmono hjn.le))
  have hd1 : StronglyMeasurable[brownianFiltration d (t n)] (coordinateBrownian k (t (j + 1))) :=
    ((coordinateBrownian_filtered k).stronglyAdapted (t (j + 1))).mono
      ((brownianFiltration d).mono (hmono (by omega)))
  have hd0 : StronglyMeasurable[brownianFiltration d (t n)] (coordinateBrownian k (t j)) :=
    ((coordinateBrownian_filtered k).stronglyAdapted (t j)).mono
      ((brownianFiltration d).mono (hmono hjn.le))
  exact hcoef.mul (hd1.sub hd0)

/-- Earlier stochastic summands have zero mixed moment with a later
one, Section 4.3 (2)--(3). Both coefficient families may depend on all
Brownian coordinates in their past. -/
theorem brownianIto_cross_moment {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H G : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hG : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (G n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d))
    (hGL2 : ∀ n, MemLp (G n) 2 (brownianNoiseLaw d))
    (j l : ℕ) (hjl : j < l) :
    (∫ ω, (H j ω * brownianIncrement k (t j) (t (j + 1)) ω) *
      (G l ω * brownianIncrement k (t l) (t (l + 1)) ω) ∂brownianNoiseLaw d) = 0 := by
  have hΔ : StronglyMeasurable[brownianFiltration d (t l)]
      (brownianIncrement k (t j) (t (j + 1))) :=
    ((coordinateBrownian_filtered k).stronglyAdapted (t (j + 1))).mono
      ((brownianFiltration d).mono (hmono (by omega))) |>.sub
    (((coordinateBrownian_filtered k).stronglyAdapted (t j)).mono
      ((brownianFiltration d).mono (hmono hjl.le)))
  have hprod := (((hH j).mono ((brownianFiltration d).mono (hmono hjl.le))).mul hΔ).mul (hG l)
  have hprodint := (brownianIncrement_adapted_memLp k (t j) (t (j + 1))
    (hmono (Nat.le_succ j)) (hH j) (hHL2 j)).integrable_mul (hGL2 l)
  have hmean := (brownianIncrement_adapted_mean k (t l) (t (l + 1))
    (hmono (Nat.le_succ l)) hprod hprodint).2
  convert hmean using 1
  congr 1
  ext ω
  simp only [Pi.mul_apply]
  ring

/-- Square integrability of the actual finite stochastic sum,
Section 4.3 (2)--(3). -/
theorem brownianItoSum_memLp {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d)) (n : ℕ) :
    MemLp (brownianItoSum k t H n) 2 (brownianNoiseLaw d) := by
  exact memLp_finsetSum _ fun j _ => brownianIncrement_adapted_memLp k
    (t j) (t (j + 1)) (hmono (Nat.le_succ j)) (hH j) (hHL2 j)

/-- Bilinear Itô isometry for genuine finite adapted stochastic sums,
Section 4.3 (2)--(3). It follows from independent Gaussian increments,
not from a postulated stochastic integral. -/
theorem brownianItoSum_bilinear {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H G : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hG : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (G n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d))
    (hGL2 : ∀ n, MemLp (G n) 2 (brownianNoiseLaw d)) (n : ℕ) :
    (∫ ω, brownianItoSum k t H n ω * brownianItoSum k t G n ω ∂brownianNoiseLaw d) =
      ∑ j ∈ Finset.range n, (∫ ω, H j ω * G j ω ∂brownianNoiseLaw d) *
        ((t (j + 1) : ℝ) - t j) := by
  let a : ℕ → BrownianSample d → ℝ := fun j ω => H j ω * brownianIncrement k (t j) (t (j + 1)) ω
  let b : ℕ → BrownianSample d → ℝ := fun j ω => G j ω * brownianIncrement k (t j) (t (j + 1)) ω
  have ha j := brownianIncrement_adapted_memLp k (t j) (t (j + 1))
    (hmono (Nat.le_succ j)) (hH j) (hHL2 j)
  have hb j := brownianIncrement_adapted_memLp k (t j) (t (j + 1))
    (hmono (Nat.le_succ j)) (hG j) (hGL2 j)
  have hint (j l : ℕ) : Integrable (fun ω => a j ω * b l ω) (brownianNoiseLaw d) :=
    (ha j).integrable_mul (hb l)
  have hdiag j : (∫ ω, a j ω * b j ω ∂brownianNoiseLaw d) =
      (∫ ω, H j ω * G j ω ∂brownianNoiseLaw d) * ((t (j + 1) : ℝ) - t j) := by
    have heq : (fun ω => a j ω * b j ω) =
        (fun ω => (H j ω * G j ω) * brownianIncrement k (t j) (t (j + 1)) ω ^ 2) := by
      ext ω
      dsimp [a, b]
      ring
    rw [heq]
    exact (brownianIncrement_adapted_secondMoment k (t j) (t (j + 1))
      (hmono (Nat.le_succ j)) ((hH j).mul (hG j)) ((hHL2 j).integrable_mul (hGL2 j))).2
  have hcross j l (hjl : j ≠ l) : (∫ ω, a j ω * b l ω ∂brownianNoiseLaw d) = 0 := by
    rcases lt_or_gt_of_ne hjl with h | h
    · exact brownianIto_cross_moment k t hmono H G hH hG hHL2 hGL2 j l h
    · rw [show (fun ω => a j ω * b l ω) = (fun ω => b l ω * a j ω) by ext ω; ring]
      exact brownianIto_cross_moment k t hmono G H hG hH hGL2 hHL2 l j h
  calc
    _ = ∫ ω, ∑ j ∈ Finset.range n, ∑ l ∈ Finset.range n, a j ω * b l ω ∂brownianNoiseLaw d := by
      apply integral_congr_ae
      exact Filter.Eventually.of_forall fun ω => Finset.sum_mul_sum _ _ _ _
    _ = ∑ j ∈ Finset.range n, ∑ l ∈ Finset.range n,
        ∫ ω, a j ω * b l ω ∂brownianNoiseLaw d := by
      rw [integral_finsetSum _ (fun j _ => integrable_finsetSum _ (fun l _ => hint j l))]
      exact Finset.sum_congr rfl fun j _ => integral_finsetSum _ (fun l _ => hint j l)
    _ = ∑ j ∈ Finset.range n, ∫ ω, a j ω * b j ω ∂brownianNoiseLaw d := by
      refine Finset.sum_congr rfl fun j hj => ?_
      exact Finset.sum_eq_single j (fun l _ hlj => hcross j l hlj.symm)
        (fun hnot => (hnot hj).elim)
    _ = _ := Finset.sum_congr rfl fun j _ => hdiag j

/-- The finite adapted Itô isometry, Section 4.3 (2)--(3). -/
theorem brownianItoSum_isometry {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ n, StronglyMeasurable[brownianFiltration d (t n)] (H n))
    (hHL2 : ∀ n, MemLp (H n) 2 (brownianNoiseLaw d)) (n : ℕ) :
    (∫ ω, brownianItoSum k t H n ω ^ 2 ∂brownianNoiseLaw d) =
      ∑ j ∈ Finset.range n, (∫ ω, H j ω ^ 2 ∂brownianNoiseLaw d) * ((t (j + 1) : ℝ) - t j) := by
  simpa only [pow_two] using brownianItoSum_bilinear k t hmono H H hH hH hHL2 hHL2 n

/-- Joint nonvacuity of all finite-isometry hypotheses, Section 4.3:
unit time steps and an integrand given by another Brownian coordinate. -/
example : Monotone (fun n : ℕ => (n : ℝ≥0)) ∧
    (∀ n : ℕ, StronglyMeasurable[brownianFiltration 2 n]
      (coordinateBrownian (1 : Fin 2) n)) ∧
    (∀ n : ℕ, MemLp (coordinateBrownian (1 : Fin 2) n) 2 (brownianNoiseLaw 2)) ∧
    (0 : ℕ) < 1 := by
  exact ⟨fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h,
    fun n => (coordinateBrownian_filtered (1 : Fin 2)).stronglyAdapted n,
    fun n => ((coordinateBrownian_isBrownian (1 : Fin 2)).isGaussianProcess.hasGaussianLaw_eval n).memLp_two,
    by norm_num⟩

end Transformer.BatchSize
