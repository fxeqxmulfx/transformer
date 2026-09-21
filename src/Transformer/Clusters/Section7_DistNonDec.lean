/-
# The emergence of clusters in self-attention dynamics — particles never collide

§7 of arXiv:2305.05465v6: `l:distnondec`, the distances between the particles
of `e:Idnonresca` do not decrease, and `e:infdist`, the particles never
collide.

The drift of `e:Idnonresca` is the gradient of the log-sum-exp potential, and
its monotonicity `⟨v(x_i) - v(x_j), x_i - x_j⟩ ≥ 0` is proved here from the
softmax itself (`softmax_monotone`): the sum is a symmetrized Kullback–Leibler
divergence, nonnegative term by term.

Source: arXiv:2305.05465v6, `l:distnondec`, `e:infdist`.
-/

import Transformer.Clusters.Section7_LogSumExp
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- The softmax is a monotone map: `Σ_j (p_j - q_j)(s_j - u_j) ≥ 0` for
`p = softmax s`, `q = softmax u`.  With `log p_j = s_j - log Σ e^s`, the sum is
`Σ_j (p_j - q_j)(log p_j - log q_j)`, and every term is nonnegative. -/
theorem softmax_monotone (s u : Idx n → ℝ) :
    0 ≤ ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) * (s j - u j) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  set Zs := ∑ k : Idx n, Real.exp (s k)
  set Zu := ∑ k : Idx n, Real.exp (u k)
  have hZs : 0 < Zs := Perspective.softmaxPartition_pos hn s
  have hZu : 0 < Zu := Perspective.softmaxPartition_pos hn u
  have hsum : ∀ v : Idx n → ℝ, ∑ j, Perspective.softmaxWeight v j = 1 := fun v => by
    simp only [Perspective.softmaxWeight, ← Finset.sum_div]
    exact div_self (Perspective.softmaxPartition_pos hn v).ne'
  have key : ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) * (s j - u j) =
      ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) *
        ((s j - Real.log Zs) - (u j - Real.log Zu)) := by
    have : ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) *
        (Real.log Zs - Real.log Zu) = 0 := by
      rw [← Finset.sum_mul, Finset.sum_sub_distrib, hsum, hsum, sub_self, zero_mul]
    rw [← sub_eq_zero, ← Finset.sum_sub_distrib, ← this]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [key]
  refine Finset.sum_nonneg fun j _ => ?_
  have hp : Perspective.softmaxWeight s j = Real.exp (s j - Real.log Zs) := by
    rw [Real.exp_sub, Real.exp_log hZs]; rfl
  have hq : Perspective.softmaxWeight u j = Real.exp (u j - Real.log Zu) := by
    rw [Real.exp_sub, Real.exp_log hZu]; rfl
  rw [hp, hq]
  rcases le_total (s j - Real.log Zs) (u j - Real.log Zu) with h | h
  · exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.2 (Real.exp_le_exp.2 h)) (sub_nonpos.2 h)
  · exact mul_nonneg (sub_nonneg.2 (Real.exp_le_exp.2 h)) (sub_nonneg.2 h)

/-- The drift of `e:Idnonresca` is monotone:
`⟨v(a) - v(b), a - b⟩ ≥ 0` for `v(y) = Σ_j softmax(⟨y, x_l⟩)_j x_j`. -/
theorem inner_drift_sub_nonneg (X : Idx n → EucSpace d) (i j : Idx n) :
    0 ≤ inner (𝕜 := ℝ)
      (∑ k, attentionMatrix (1 : ParamMatrix d) 1 X i k • X k -
        ∑ k, attentionMatrix (1 : ParamMatrix d) 1 X j k • X k) (X i - X j) := by
  have := softmax_monotone (fun l => inner (𝕜 := ℝ) (X i) (X l))
    (fun l => inner (𝕜 := ℝ) (X j) (X l))
  convert this using 1
  simp only [attentionMatrix, one_apply_eq_self, ← Finset.sum_sub_distrib,
    ← sub_smul, sum_inner, inner_smul_left, RCLike.conj_to_real]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [inner_sub_right, real_inner_comm (X k) (X i), real_inner_comm (X k) (X j)]

/-- **Lemma (l:distnondec).**  Along a solution of `e:Idnonresca`, the map
`t ↦ ‖x_i(t) - x_j(t)‖` is non-decreasing for every pair `i, j ∈ [n]`.

The source's argument: `e:Idnonresca` is `ẋ_i = ∇f(x_i)` for the potential
`e:logsumexpfct`, and the gradient of a convex function is monotone, so
`⟨ẋ_i - ẋ_j, x_i - x_j⟩ ≥ 0`.  That monotonicity is `inner_drift_sub_nonneg`,
proved from the softmax directly rather than through `l:logsumexp`.

Source: arXiv:2305.05465v6, `l:distnondec`. -/
theorem norm_sub_monotone (X : ℝ → Idx n → EucSpace d) (hX : IdNonrescaledDynamics X)
    (i j : Idx n) : Monotone fun t => ‖X t i - X t j‖ := by
  have hg : ∀ t, HasDerivAt (fun t => ‖X t i - X t j‖ ^ 2)
      (2 * inner (𝕜 := ℝ) (X t i - X t j)
        (∑ k, attentionMatrix (1 : ParamMatrix d) 1 (X t) i k • X t k -
          ∑ k, attentionMatrix (1 : ParamMatrix d) 1 (X t) j k • X t k)) t :=
    fun t => ((hX t i).sub (hX t j)).norm_sq
  have hmono := monotone_of_hasDerivAt_nonneg hg fun t => by
    have := inner_drift_sub_nonneg (X t) i j
    rw [real_inner_comm] at this
    positivity
  intro a b hab
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).1 (hmono hab)

/-- The hypothesis of `norm_sub_monotone` is satisfiable. -/
example : IdNonrescaledDynamics (n := n) (fun _ _ => (0 : EucSpace d)) :=
  idNonrescaledDynamics_zero d n

/-- **Equation (e:infdist).**  Particles never collide: two tokens that start
apart stay apart, by at least their initial separation.

Stated from the conclusion of `l:distnondec`, which `norm_sub_monotone`
supplies along every solution of `e:Idnonresca`.

Source: arXiv:2305.05465v6, `e:infdist`. -/
theorem ne_of_norm_sub_monotone (X : ℝ → Idx n → EucSpace d)
    (hmono : ∀ i j : Idx n, Monotone fun t => ‖X t i - X t j‖)
    (i j : Idx n) (hij : X 0 i ≠ X 0 j) {t : ℝ} (ht : 0 ≤ t) : X t i ≠ X t j := by
  intro hcontra
  have h0 : ‖X 0 i - X 0 j‖ ≤ ‖X t i - X t j‖ := hmono i j ht
  rw [hcontra, sub_self, norm_zero] at h0
  exact hij (sub_eq_zero.mp (norm_le_zero_iff.mp h0))

/-- The hypotheses of `ne_of_norm_sub_monotone` are satisfiable: a
configuration whose tokens do not move has constant distances, and two of its
tokens may start apart. -/
example (z : EucSpace 1) (hz : z ≠ 0) :
    (∀ i j : Idx 2, Monotone fun t : ℝ => ‖(fun _ (k : Idx 2) => if k = 0 then z else 0) t i
        - (fun _ (k : Idx 2) => if k = 0 then z else 0) t j‖) ∧
      (if (0 : Idx 2) = 0 then z else 0) ≠ (if (1 : Idx 2) = 0 then z else 0) := by
  refine ⟨fun i j => monotone_const, ?_⟩
  simpa using hz

end Clusters
end Transformer
