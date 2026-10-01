/-
# Stopping Euler chains and their Brownian interpolation

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The stopped dyadic endpoints at different horizons are evaluations of
the same Brownian interpolation. This is needed for increment estimates.
-/

import Transformer.BatchSize.Section4_DyadicEuler
import Transformer.BatchSize.Section4_EulerPaths

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Euler chains depend only on the grid times preceding their state,
Section 4.3 (2)--(3). -/
theorem eulerChain_congr_grid {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t u : ℕ → ℝ≥0) (x₀ : EucSpace d) (n : ℕ)
    (heq : ∀ j ≤ n, t j = u j) : eulerChain b a t x₀ n = eulerChain b a u x₀ n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hprev := ih (fun j hj => heq j (hj.trans (Nat.le_succ n)))
    funext ω
    simp only [eulerChain, hprev, heq n (Nat.le_succ n), heq (n + 1) le_rfl]

/-- Zero-length steps after a stopped time leave the chain unchanged,
Section 4.3 (2)--(3). -/
theorem eulerChain_eq_of_time_eq {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n N : ℕ) (hn : n ≤ N) (ht : t n = t N) :
    eulerChain b a t x₀ n = eulerChain b a t x₀ N := by
  induction N generalizing n with
  | zero =>
    have hn0 : n = 0 := by omega
    subst n
    rfl
  | succ N ih =>
    by_cases he : n = N + 1
    · subst n; rfl
    have hnN : n ≤ N := by omega
    have hmid : t n = t N := le_antisymm (hmono hnN) (by rw [ht]; exact hmono (Nat.le_succ N))
    have hlast : t N = t (N + 1) := hmid.symm.trans ht
    rw [ih n hnN hmid]
    change eulerChain b a t x₀ N =
      brownianEulerStep b a (t N) (t (N + 1)) (eulerChain b a t x₀ N)
    rw [hlast, brownianEulerStep_same_time]

/-- Stopping the Euler grid at s gives exactly the original chain's
continuous Brownian interpolation evaluated at s, Section 4.3 (2)--(3).
The statement holds pointwise on the actual Brownian sample space. -/
theorem eulerChain_clipped_eq_pathValue {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (s : ℝ≥0) (n : ℕ) (ω : BrownianSample d) :
    eulerChain b a (fun j => min s (t j)) x₀ n ω =
      eulerPathValue b a t x₀ n ω s := by
  induction n with
  | zero => simp [eulerChain, eulerPathValue]
  | succ n ih =>
    have hp : eulerPathValue b a t x₀ (n + 1) ω s = eulerPathValue b a t x₀ n ω s +
        (((min s (t (n + 1)) : ℝ≥0) : ℝ) - min s (t n)) • b (eulerChain b a t x₀ n ω) +
          WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ n ω) k *
            brownianIncrement k (min s (t n)) (min s (t (n + 1))) ω) := by
      simp only [eulerPathValue, Real.toNNReal_coe, Finset.sum_range_succ]
      abel
    rw [hp]
    by_cases hns : t n ≤ s
    · have hgrid : ∀ j ≤ n, min s (t j) = t j :=
        fun j hj => min_eq_right ((hmono hj).trans hns)
      have hchain := eulerChain_congr_grid b a (fun j => min s (t j)) t x₀ n hgrid
      have hpath : eulerPathValue b a t x₀ n ω s = eulerChain b a t x₀ n ω :=
        ih.symm.trans (congrFun hchain ω)
      simp only [eulerChain, hchain, hpath]
    · have hsn : s ≤ t n := (le_of_not_ge hns)
      have hsnext := hsn.trans (hmono (Nat.le_succ n))
      have hzero : WithLp.toLp 2 (fun _ : Fin d => (0 : ℝ)) = (0 : EucSpace d) := by ext k; rfl
      simp only [eulerChain, min_eq_left hsn, min_eq_left hsnext, sub_self,
        zero_smul, brownianIncrement, mul_zero, hzero, add_zero, ih]

/-- Stopped dyadic endpoints at s are evaluations of the Euler path
on any larger horizon T, Section 4.3 (2)--(3). -/
theorem dyadicEuler_eq_pathValue {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (x₀ : EucSpace d) (s T : ℝ≥0)
    (hsT : s ≤ T) (m : ℕ) (ω : BrownianSample d) :
    dyadicEuler b a x₀ s m ω =
      eulerPathValue b a (dyadicGrid T m) x₀ (dyadicEndpointIndex T m) ω s := by
  have hclip : (fun j => min s (dyadicGrid T m j)) = dyadicGrid s m := by
    funext j
    simp only [dyadicGrid, min_comm s, min_assoc, min_eq_right hsT]
  have hind : dyadicEndpointIndex s m ≤ dyadicEndpointIndex T m :=
    Nat.mul_le_mul_left _ (Nat.ceil_mono hsT)
  have ht : dyadicGrid s m (dyadicEndpointIndex s m) =
      dyadicGrid s m (dyadicEndpointIndex T m) := by
    rw [dyadicGrid_endpoint]
    apply le_antisymm
    · calc
        s = dyadicGrid s m (dyadicEndpointIndex s m) := (dyadicGrid_endpoint s m).symm
        _ ≤ _ := dyadicGrid_monotone s m hind
    · exact min_le_right _ _
  change eulerChain b a (dyadicGrid s m) x₀ (dyadicEndpointIndex s m) ω = _
  rw [congrFun (eulerChain_eq_of_time_eq b a _ (dyadicGrid_monotone s m) x₀ _ _ hind ht) ω]
  rw [← hclip, eulerChain_clipped_eq_pathValue b a _ (dyadicGrid_monotone T m)]

/-- Joint nonvacuity of grid agreement and stopped-time hypotheses,
Section 4.3: a genuinely advancing grid stopped after its second step. -/
example : Monotone (fun j : ℕ => min (j : ℝ≥0) 2) ∧
    (∀ j ≤ 3, min (j : ℝ≥0) 2 = min (j : ℝ≥0) 2) ∧
    (2 : ℕ) ≤ 3 ∧ min (2 : ℝ≥0) 2 = min (3 : ℝ≥0) 2 ∧
    (1 : ℝ≥0) ≤ 2 := by
  refine ⟨fun i j hij => min_le_min (by exact_mod_cast hij) le_rfl,
    fun _ _ => rfl, by omega, by norm_num, by norm_num⟩

end Transformer.BatchSize
