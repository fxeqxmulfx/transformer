/-
# DASH — turning finite expansions into Clenshaw coefficient arrays

arXiv:2602.02016v2, Appendix A. Sparse finite Chebyshev expansions
and the coefficient lists consumed by the algorithms evaluate identically.
-/

import Transformer.DASH.SectionA_ApproximationExistence
import Mathlib.Algebra.BigOperators.Finsupp.Basic
import Mathlib.Data.List.Indexes

open scoped BigOperators

noncomputable section

namespace Transformer.DASH

/-- The exact indexed finite series evaluated by the list recurrence.
Source: arXiv:2602.02016v2, Appendix A, `Σ c_k T_k(x)`. -/
theorem chebSeriesFrom_eq_mapIdx {R : Type*} [Ring R] (x : R) (cs : List R) (k : ℕ) :
    chebSeriesFrom x k cs = (cs.mapIdx (fun i c => chebT x (k + i) * c)).sum := by
  induction cs generalizing k with
  | nil => rfl
  | cons c cs ih =>
    simp only [chebSeriesFrom, ih, List.mapIdx_cons, List.sum_cons, Nat.add_zero]
    have hindex : (fun i a => chebT x (k + 1 + i) * a) =
        (fun i a => chebT x (k + (i + 1)) * a) := by
      funext i a
      congr 2
      omega
    rw [hindex]

/-- Dense coefficient array through the last supported degree, with holes
filled by zero. Source: arXiv:2602.02016v2, Appendix A, coefficients `c_0,...,c_d`. -/
def denseChebCoefficients (c : ℕ →₀ ℝ) : List ℝ :=
  List.ofFn (fun i : Fin (c.support.sup id + 1) => c i)

/-- Evaluating an indexed coefficient array gives its finite indexed sum.
Source: arXiv:2602.02016v2, Appendix A, scalar Clenshaw output. -/
theorem clenshaw_ofFn (x : ℝ) (N : ℕ) (c : Fin N → ℝ) :
    clenshaw x (List.ofFn c) = ∑ i : Fin N, c i * chebT x i := by
  rw [clenshaw_eq_series, chebSeriesFrom_eq_mapIdx]
  simp only [Nat.zero_add]
  simp only [List.mapIdx_eq_ofFn, List.length_ofFn, List.get_ofFn, List.sum_ofFn]
  apply Finset.sum_congr rfl
  intro i hi
  exact mul_comm _ _

/-- Clenshaw on a dense array evaluates the original finite expansion exactly.
Source: arXiv:2602.02016v2, Appendix A, the coefficient-array representation. -/
theorem clenshaw_dense_coefficients (x : ℝ) (c : ℕ →₀ ℝ) :
    clenshaw x (denseChebCoefficients c) = c.sum (fun k a => a * chebT x k) := by
  rw [denseChebCoefficients, clenshaw_ofFn]
  have hs : c.support ⊆ Finset.range (c.support.sup id + 1) := by
    intro i hi
    exact Finset.mem_range.2 (Nat.lt_succ_of_le (Finset.le_sup (f := id) hi))
  rw [Finsupp.sum_of_support_subset c hs _ (fun _ _ => by simp)]
  exact Fin.sum_univ_eq_sum_range (fun k => c k * chebT x k) _

/-- The existence result has actual finite Clenshaw coefficient arrays,
with the uniform inverse-power accuracy requested by the caller.
Source: arXiv:2602.02016v2, Appendix A, approximation and scalar evaluation. -/
theorem exists_clenshaw_inverse_power_approximation (ε exponent δ : ℝ)
    (hε : 0 < ε) (hδ : 0 < δ) :
    ∃ cs : List ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) 1,
      |clenshaw t cs - (chebFromCoordinate ε (1 + ε) t) ^ exponent| < δ := by
  obtain ⟨c, hc⟩ := exists_chebyshev_inverse_power_approximation ε exponent δ hε hδ
  refine ⟨denseChebCoefficients c, ?_⟩
  intro t ht
  rw [clenshaw_dense_coefficients]
  exact hc t ht

/-- Positive regularization and accuracy assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100 := by norm_num

end Transformer.DASH
