module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Coordinates

/-!
# Two linear objectives select a simplex vertex

Source: `review/flat_spectrum.md`, CM21. First maximize the real reading, then
the imaginary reading on the first optimum face. The policy is specified in
observable coordinates; an optimizer or a target eigenvector is not assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix

variable {ι X : Type*} [Fintype ι]

/-- The exact two-stage linear optimization specification, CM21. -/
def LexMaximum (B : Matrix X ι ℂ) (x y : X) (M : Matrix ι ι ℂ) : Prop :=
  Feasible B M ∧
    (∀ N, Feasible B N → (reading B N x y).re ≤ (reading B M x y).re) ∧
    ∀ N, Feasible B N → (reading B N x y).re = (reading B M x y).re →
      (reading B N x y).im ≤ (reading B M x y).im

/-- A finite nonempty set of complex scores has a lexicographic maximum,
CM21. The maximizing index is derived from the scores. -/
theorem exists_lex_vertex [Nonempty ι] (s : ι → ℂ) :
    ∃ i, (∀ j, (s j).re ≤ (s i).re) ∧
      ∀ j, (s j).re = (s i).re → (s j).im ≤ (s i).im := by
  classical
  obtain ⟨a, ha, hmax⟩ := univ.exists_max_image (fun j => (s j).re) univ_nonempty
  let t := univ.filter (fun j => (s j).re = (s a).re)
  have ht : t.Nonempty := ⟨a, by simp [t]⟩
  obtain ⟨i, hi, him⟩ := t.exists_max_image (fun j => (s j).im) ht
  have hir : (s i).re = (s a).re := (mem_filter.mp hi).2
  refine ⟨i, fun j => by rw [hir]; exact hmax j (mem_univ j), ?_⟩
  intro j hj
  exact him j (by simp [t, hj, hir])

/-- A probability average cannot exceed a bound on its nonzero support,
CM21. This includes the second objective on the first optimum face. -/
theorem weighted_bound {w : ι → ℝ} (hw : Weights w) (v : ι → ℝ) (m : ℝ)
    (hv : ∀ j, w j ≠ 0 → v j ≤ m) : ∑ j, w j * v j ≤ m := by
  calc
    _ ≤ ∑ j, w j * m := by
      apply sum_le_sum
      intro j hj
      by_cases hz : w j = 0
      · simp [hz]
      · exact mul_le_mul_of_nonneg_left (hv j hz) (hw.1 j)
    _ = m := by rw [← sum_mul, hw.2, one_mul]

/-- Equality in that average forces every positive-weight score to attain
the bound, CM21. There is no strict-complementarity assumption. -/
theorem weighted_support {w : ι → ℝ} (hw : Weights w) (v : ι → ℝ) (m : ℝ)
    (hv : ∀ j, w j ≠ 0 → v j ≤ m) (he : ∑ j, w j * v j = m) :
    ∀ j, w j ≠ 0 → v j = m := by
  have hn (j : ι) : 0 ≤ w j * (m - v j) := by
    by_cases hz : w j = 0
    · simp [hz]
    · exact mul_nonneg (hw.1 j) (sub_nonneg.mpr (hv j hz))
  have hs : ∑ j, w j * (m - v j) = 0 := by
    simp_rw [mul_sub]
    rw [sum_sub_distrib, ← sum_mul, hw.2, one_mul, he, sub_self]
  have hz := (sum_eq_zero_iff_of_nonneg (fun j hj => hn j)).mp hs
  intro j hj
  have hprod := hz j (mem_univ j)
  exact (sub_eq_zero.mp ((mul_eq_zero.mp hprod).resolve_left hj)).symm

/-- Distinct complex scores force a pure probability vector after both
linear objectives attain the lexicographic maximum, CM21. -/
theorem lex_weights_pure [DecidableEq ι] {w : ι → ℝ} (hw : Weights w)
    (s : ι → ℂ) (hs : Function.Injective s) (i : ι)
    (hr : ∀ j, (s j).re ≤ (s i).re)
    (hi : ∀ j, (s j).re = (s i).re → (s j).im ≤ (s i).im)
    (her : ∑ j, w j * (s j).re = (s i).re)
    (hei : ∑ j, w j * (s j).im = (s i).im) : w = Pi.single i 1 := by
  have hsr := weighted_support hw (fun j => (s j).re) (s i).re (fun j hj => hr j) her
  have hsi := weighted_support hw (fun j => (s j).im) (s i).im
    (fun j hj => hi j (hsr j hj)) hei
  have hout (j : ι) (hj : j ≠ i) : w j = 0 := by
    by_contra hz
    exact hj (hs (Complex.ext (hsr j hz) (hsi j hz)))
  have hwi : w i = 1 := by
    rw [← hw.2]
    exact (Fintype.sum_eq_single i hout).symm
  funext j
  by_cases hj : j = i
  · subst j
    simpa using hwi
  · simp [hj, hout j hj]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
