/-
# Metastability — comparison for the within-cap minimum, without an ODE

Lemma `lem: collapsetime` and Steps 2–3 of the direct proof of arXiv:2410.06833v1 compare
the within-cap minimum `ρ_q(t) = min_{i,j ∈ I} ⟨x_i(t), x_j(t)⟩` with the solution of an
ODE.  Here the comparison is made with an explicit strict subsolution `B`:

  `B'(t) < (2/n) B(1 - B) e^{β(B - 1)} - 2 n e^{-(1-α)β}`,   `B ≥ 0`,   `B(a₀) ≤ ρ(a₀)`
    ⟹   `B(t) ≤ ⟨x_i(t), x_j(t)⟩`  for `t ∈ [a₀, b]`, `i, j ∈ I`,

as long as the tokens of `I` are `α`-separated from those outside `I` on `[a₀, b]`.

The proof is the finite-family barrier of `PropagationBarrier` applied to the `|I|²`
functions `⟨x_i, x_j⟩ - B`: at a time where one of them vanishes while the others are
nonnegative, `B(t)` is the minimum of the pair inner products, `pair_velocity_ge` bounds the
derivative of `⟨x_i, x_j⟩` below by the right-hand side above, and the strict inequality
makes `⟨x_i, x_j⟩ - B` strictly increasing there.  This avoids differentiating a minimum,
which has corners; `B` need only be continuous on `[a₀, b]` and differentiable on `[a₀, b)`.

It holds for every attention flow, so for `SA` and for `USA` alike.
-/

import Transformer.Metastability.AttnFlow
import Transformer.Metastability.AttnPair
import Transformer.Metastability.PropagationBarrier
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- The weights of an attention flow leak at most `e^{-(1-α)β}` to a token that is
`α`-separated: `a_{ik} ≤ e^{β(⟨x_i,x_k⟩ - 1)} ≤ e^{β(α - 1)}`. -/
theorem attn_far_le {β α : ℝ} (hβ : 0 ≤ β) {x : Idx n → EucSpace d} {a : Idx n → Idx n → ℝ}
    (ha : IsAttnWeights d n β x a) {i k : Idx n} (h : ⟪x i, x k⟫_ℝ ≤ α) :
    a i k ≤ Real.exp (-((1 - α) * β)) :=
  (ha.upper i k).trans (Real.exp_le_exp.2 (by nlinarith))

/-- **Comparison with a strict subsolution.**  Let `X` be an attention flow, `I` a set of
tokens that are `α`-separated from all others on `[a₀, b]`, and `B` a function that is
continuous on `[a₀, b]`, differentiable on `[a₀, b)`, nonnegative, and a strict subsolution

  `B' < (2/n) B (1 - B) e^{β(B-1)} - 2 n e^{-(1-α)β}`  on `[a₀, b)`.

If `B(a₀) ≤ ⟨x_i(a₀), x_j(a₀)⟩` for `i, j ∈ I`, then `B(t) ≤ ⟨x_i(t), x_j(t)⟩` for all
`t ∈ [a₀, b]` and `i, j ∈ I`.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation` with `eq: comparison`, and
`lem: collapsetime`; the ODE comparison of the source is replaced by a barrier argument. -/
theorem attn_comparison (β α : ℝ) (hβ : 0 ≤ β) {X : ℝ → SphereTuple d n}
    (hX : IsAttnFlow d n β X) (I : Finset (Idx n)) (a₀ b : ℝ) (hab : a₀ ≤ b) (B : ℝ → ℝ)
    (hBc : ContinuousOn B (Icc a₀ b))
    (hBd : ∀ t ∈ Ico a₀ b, DifferentiableAt ℝ B t)
    (hB0 : ∀ t ∈ Icc a₀ b, 0 ≤ B t)
    (hBsub : ∀ t ∈ Ico a₀ b, deriv B t <
        (2 / (n : ℝ)) * B t * (1 - B t) * Real.exp (β * (B t - 1))
          - 2 * (n : ℝ) * Real.exp (-((1 - α) * β)))
    (hfar : ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ k, k ∉ I →
        ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I,
        B a₀ ≤ ⟪(X a₀ i : EucSpace d), (X a₀ j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ j ∈ I,
      B t ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  obtain ⟨a, ha, hdiff⟩ := hX
  let ι := {p : Idx n × Idx n // p.1 ∈ I ∧ p.2 ∈ I}
  let f : ι → ℝ → ℝ := fun p t =>
    ⟪(X t p.1.1 : EucSpace d), (X t p.1.2 : EucSpace d)⟫_ℝ - B t
  have hderivP : ∀ (i j : Idx n) (t : ℝ), HasDerivAt
      (fun s => ⟪(X s i : EucSpace d), (X s j : EucSpace d)⟫_ℝ)
      (⟪(X t i : EucSpace d),
          proj d (X t j : EucSpace d) (∑ k, a t j k • (X t k : EucSpace d))⟫_ℝ
        + ⟪proj d (X t i : EucSpace d) (∑ k, a t i k • (X t k : EucSpace d)),
          (X t j : EucSpace d)⟫_ℝ) t :=
    fun i j t => (hdiff t i).inner ℝ (hdiff t j)
  have hcontP : ∀ i j : Idx n, Continuous
      (fun s => ⟪(X s i : EucSpace d), (X s j : EucSpace d)⟫_ℝ) := fun i j =>
    continuous_iff_continuousAt.2 fun t => (hderivP i j t).continuousAt
  have hcont : ∀ p : ι, ContinuousOn (f p) (Icc a₀ b) := fun p =>
    (hcontP _ _).continuousOn.sub hBc
  have hinit' : ∀ p : ι, 0 ≤ f p a₀ := fun p => sub_nonneg.2 (hinit _ p.2.1 _ p.2.2)
  have hboundary : ∀ t ∈ Ico a₀ b, (∀ p : ι, 0 ≤ f p t) →
      ∀ p : ι, f p t = 0 → 0 < deriv (f p) t := by
    intro t ht hall p hp
    obtain ⟨⟨i, j⟩, hi, hj⟩ := p
    have hρ : B t = ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
      dsimp [f] at hp; linarith
    have hmin : ∀ k ∈ I, ∀ l ∈ I,
        B t ≤ ⟪(X t k : EucSpace d), (X t l : EucSpace d)⟫_ℝ := fun k hk l hl => by
      have h := hall ⟨(k, l), hk, hl⟩
      dsimp [f] at h; linarith
    have hnorm : ∀ k : Idx n, ‖(X t k : EucSpace d)‖ = 1 := fun k =>
      mem_sphere_zero_iff_norm.mp (X t k).2
    have hwt := ha t
    have hE : (0 : ℝ) ≤ Real.exp (-((1 - α) * β)) := (Real.exp_pos _).le
    have hnn : ∀ i k : Idx n, 0 ≤ a t i k := fun i k =>
      le_trans (by have := Real.exp_pos (β * (⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ - 1))
                   positivity) (hwt.lower i k)
    have hm : (1 / (n : ℝ)) * Real.exp (β * (B t - 1)) ≤ a t i j := by
      rw [hρ]; exact hwt.lower i j
    have hm' : (1 / (n : ℝ)) * Real.exp (β * (B t - 1)) ≤ a t j i := by
      rw [hρ, real_inner_comm]; exact hwt.lower j i
    have h1 := pair_velocity_ge (fun k => (X t k : EucSpace d)) (a t i) I i j hi hj hnorm
      (Real.exp (-((1 - α) * β))) _ (B t) hE (hnn i)
      (fun k hk => attn_far_le hβ hwt (hfar t (Ico_subset_Icc_self ht) i hi k hk))
      hm (hB0 t (Ico_subset_Icc_self ht)) hρ hmin
    have h2 := pair_velocity_ge (fun k => (X t k : EucSpace d)) (a t j) I j i hj hi hnorm
      (Real.exp (-((1 - α) * β))) _ (B t) hE (hnn j)
      (fun k hk => attn_far_le hβ hwt (hfar t (Ico_subset_Icc_self ht) j hj k hk))
      hm' (hB0 t (Ico_subset_Icc_self ht)) (by rw [hρ, real_inner_comm])
      (fun k hk l hl => hmin k hk l hl)
    rw [real_inner_comm] at h2
    have hgain : (2 / (n : ℝ)) * B t * (1 - B t) * Real.exp (β * (B t - 1))
        ≤ 2 * ((1 / (n : ℝ)) * Real.exp (β * (B t - 1)) * (1 - B t ^ 2)) := by
      have hc : (0 : ℝ) ≤ 1 / (n : ℝ) := by positivity
      have hex : (0 : ℝ) < Real.exp (β * (B t - 1)) := Real.exp_pos _
      have hB1 : B t ≤ 1 := by
        rw [hρ]
        have h := abs_real_inner_le_norm (X t i : EucSpace d) (X t j : EucSpace d)
        rw [hnorm i, hnorm j, one_mul] at h
        exact (abs_le.mp h).2
      have hkey : B t * (1 - B t) ≤ 1 - B t ^ 2 := by
        nlinarith [hB0 t (Ico_subset_Icc_self ht)]
      have h2n : (2 : ℝ) / (n : ℝ) = 2 * (1 / (n : ℝ)) := by ring
      rw [h2n]
      nlinarith [mul_le_mul_of_nonneg_left hkey (mul_nonneg hc hex.le)]
    have hcardn : ((n : ℝ) - 1) * Real.exp (-((1 - α) * β))
        ≤ (n : ℝ) * Real.exp (-((1 - α) * β)) := by nlinarith
    have hlow : (2 / (n : ℝ)) * B t * (1 - B t) * Real.exp (β * (B t - 1))
          - 2 * (n : ℝ) * Real.exp (-((1 - α) * β))
        ≤ ⟪proj d (X t i : EucSpace d) (∑ k, a t i k • (X t k : EucSpace d)),
              (X t j : EucSpace d)⟫_ℝ
          + ⟪(X t i : EucSpace d),
              proj d (X t j : EucSpace d) (∑ k, a t j k • (X t k : EucSpace d))⟫_ℝ := by
      linarith
    have hd := ((hderivP i j t).fun_sub (hBd t ht).hasDerivAt).deriv
    show 0 < deriv (fun s => ⟪(X s i : EucSpace d), (X s j : EucSpace d)⟫_ℝ - B s) t
    rw [hd]
    linarith [hBsub t ht]
  have hall := finite_nonneg_of_boundary_deriv_on f a₀ b hcont ⟨hinit', hab⟩ hboundary
  intro t ht i hi j hj
  have h := hall t ht ⟨(i, j), hi, hj⟩
  dsimp [f] at h
  linarith

end Metastability
end Transformer
