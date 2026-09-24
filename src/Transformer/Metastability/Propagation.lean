/-
# Metastability — propagation of within-cap alignment for SA

The `SA` branch of `lem:collapsetime` from arXiv:2410.06833v1.
-/

import Transformer.Perspective.Section1_IPS
import Transformer.Metastability.PairVelocitySum
import Transformer.Metastability.PropagationBarrier
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

open Perspective
open scoped Classical

variable (d n : ℕ)

/-- **Lemma (lem:collapsetime) — *Propagation.*

Fix `β > 1`, and let `δ ∈ (0, 1)`, `α ∈ (-1, 1)` satisfy

  `(1/n) δ (1 - δ) e^{-δ β} > n e^{-(1-α) β}`.

If `⟨x_i(0), x_j(0)⟩ ≥ 1 - δ` for all `(i, j) ∈ I²` and
`⟨x_i(t), x_k(t)⟩ ≤ α` for all `i ∈ I`, `k ∈ I^c`, `t ∈ [0, T]`, then

  `⟨x_i(t), x_j(t)⟩ ≥ 1 - δ`  for all `(i, j) ∈ I²` and `t ∈ [0, T]`.

This is the `SA` branch of the source lemma; the `USA` branch remains open.
The proof uses a finite-family inward-pointing barrier, avoiding the paper's
informal differentiation of a minimum at a possible corner.

Source: arXiv:2410.06833v1, §2, `lem:collapsetime`, `eq: delta.alpha.cond`. -/
lemma propagation_sa
    (β : ℝ) (hβ : 1 < β) (δ α : ℝ)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (I : Finset (Idx n)) (T : ℝ) (hT : 0 ≤ T)
    (h_cond : (1 / (n : ℝ)) * δ * (1 - δ) * Real.exp (-(δ * β))
                > (n : ℝ) * Real.exp (-((1 - α) * β)))
    (h_init : ∀ i ∈ I, ∀ j ∈ I,
                1 - δ ≤ inner (𝕜 := ℝ)
                          ((X 0 i : EucSpace d)) ((X 0 j : EucSpace d)))
    (h_far  : ∀ i ∈ I, ∀ k ∈ Iᶜ, ∀ t : ℝ, 0 ≤ t → t ≤ T →
                inner (𝕜 := ℝ)
                  ((X t i : EucSpace d)) ((X t k : EucSpace d)) ≤ α) :
    ∀ i ∈ I, ∀ j ∈ I, ∀ t : ℝ, 0 ≤ t → t ≤ T →
      1 - δ ≤ inner (𝕜 := ℝ)
                  ((X t i : EucSpace d)) ((X t j : EucSpace d)) := by
  have hn : 0 < n := by
    by_contra h
    have hn0 : n = 0 := Nat.eq_zero_of_not_pos h
    subst n
    norm_num at h_cond
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hβ0 : 0 ≤ β := le_of_lt (by linarith : (0 : ℝ) < β)
  have hprod : 0 < δ * (1 - δ) := by
    have hright : 0 ≤ (n : ℝ) * Real.exp (-((1 - α) * β)) := by positivity
    have hleft := lt_of_le_of_lt hright h_cond
    have hc : 0 < (1 / (n : ℝ)) * Real.exp (-(δ * β)) := by positivity
    have heq : (1 / (n : ℝ)) * δ * (1 - δ) * Real.exp (-(δ * β)) =
        ((1 / (n : ℝ)) * Real.exp (-(δ * β))) * (δ * (1 - δ)) := by ring
    rw [heq] at hleft
    exact (mul_pos_iff_of_pos_left hc).mp hleft
  have hρ0 : 0 ≤ 1 - δ := by nlinarith [sq_nonneg δ]
  have hgain : 0 < (2 / (n : ℝ)) * (1 - δ) * δ *
      Real.exp (β * ((1 - δ) - 1)) -
      2 * (n : ℝ) * Real.exp (-((1 - α) * β)) := by
    have heq : (2 / (n : ℝ)) * (1 - δ) * δ *
        Real.exp (β * ((1 - δ) - 1)) -
        2 * (n : ℝ) * Real.exp (-((1 - α) * β)) =
        2 * ((1 / (n : ℝ)) * δ * (1 - δ) * Real.exp (-(δ * β)) -
          (n : ℝ) * Real.exp (-((1 - α) * β))) := by
      have hexp : β * ((1 - δ) - 1) = -(δ * β) := by ring
      rw [hexp]
      ring
    rw [heq]
    exact mul_pos (by norm_num) (sub_pos.mpr h_cond)
  let ι := {p : Idx n × Idx n // p.1 ∈ I ∧ p.2 ∈ I}
  let f : ι → ℝ → ℝ := fun p t =>
    inner (𝕜 := ℝ) ((X t p.1.1 : EucSpace d)) ((X t p.1.2 : EucSpace d)) - (1 - δ)
  have hderiv : ∀ p : ι, ∀ t : ℝ, HasDerivAt (f p)
      (inner (𝕜 := ℝ) (softmaxVel d n β (fun k => ((X t k : EucSpace d))) p.1.1)
          ((X t p.1.2 : EucSpace d)) +
        inner (𝕜 := ℝ) ((X t p.1.1 : EucSpace d))
          (softmaxVel d n β (fun k => ((X t k : EucSpace d))) p.1.2)) t := by
    intro p t
    have hvi : HasDerivAt (fun s => ((X s p.1.1 : EucSpace d)))
        (softmaxVel d n β (fun k => ((X t k : EucSpace d))) p.1.1) t := hX t p.1.1
    have hvj : HasDerivAt (fun s => ((X s p.1.2 : EucSpace d)))
        (softmaxVel d n β (fun k => ((X t k : EucSpace d))) p.1.2) t := hX t p.1.2
    have h := hvi.inner ℝ hvj
    convert h.sub_const (1 - δ) using 1; ring
  have hcont : ∀ p : ι, Continuous (f p) := fun p =>
    continuous_iff_continuousAt.mpr fun t => (hderiv p t).continuousAt
  have hinit : ∀ p : ι, 0 ≤ f p 0 := by
    intro p
    exact sub_nonneg.mpr (h_init p.1.1 p.2.1 p.1.2 p.2.2)
  have hboundary : ∀ (t : ℝ), t ∈ Set.Ico 0 T → (∀ p : ι, 0 ≤ f p t) →
      ∀ p : ι, f p t = 0 → 0 < deriv (f p) t := by
    intro t ht hall p hp
    have hnorm : ∀ k : Idx n, ‖((X t k : EucSpace d))‖ = 1 :=
      fun k => mem_sphere_zero_iff_norm.mp (X t k).2
    have hpair : (1 - δ) = inner (𝕜 := ℝ)
        ((X t p.1.1 : EucSpace d)) ((X t p.1.2 : EucSpace d)) := by
      dsimp [f] at hp
      linarith
    have hmin : ∀ k ∈ I, ∀ l ∈ I,
        1 - δ ≤ inner (𝕜 := ℝ) ((X t k : EucSpace d)) ((X t l : EucSpace d)) := by
      intro k hk l hl
      have h := hall (⟨(k, l), hk, hl⟩ : ι)
      dsimp [f] at h
      linarith
    have hfar' : ∀ k ∈ I, ∀ l : Idx n, l ∉ I →
        inner (𝕜 := ℝ) ((X t k : EucSpace d)) ((X t l : EucSpace d)) ≤ α := by
      intro k hk l hl
      exact h_far k hk l (Finset.mem_compl.mpr hl) t ht.1 ht.2.le
    have hvel := inner_proj_softmax_pair_sum d n β α (1 - δ) hβ0
      (fun k => ((X t k : EucSpace d))) I p.1.1 p.1.2 p.2.1 p.2.2
      hnorm hρ0 hpair hmin hfar'
    rw [(hderiv p t).deriv]
    linarith
  have hall := finite_nonneg_of_boundary_deriv f 0 T hcont ⟨hinit, hT⟩ hboundary
  intro i hi j hj t ht0 htT
  have h := hall t ⟨ht0, htT⟩ (⟨(i, j), hi, hj⟩ : ι)
  dsimp [f] at h
  linarith

/-- The hypotheses of the `SA` propagation lemma are jointly satisfiable:
one stationary particle, with its whole index set as the cap. -/
example :
    (1 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
      (1 / (1 : ℝ)) * (1 / 2) * (1 - 1 / 2) * Real.exp (-((1 / 2) * 2)) >
        (1 : ℝ) * Real.exp (-((1 - (-1 : ℝ)) * 2)) ∧
      Perspective.SA 1 1 2 (fun _ _ => basePoint 0) ∧
      (∀ i ∈ (Finset.univ : Finset (Idx 1)), ∀ j ∈ (Finset.univ : Finset (Idx 1)),
        1 - (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ)
          ((basePoint 0 : EucSpace 1)) ((basePoint 0 : EucSpace 1))) ∧
      (∀ i ∈ (Finset.univ : Finset (Idx 1)),
        ∀ k ∈ (Finset.univ : Finset (Idx 1))ᶜ, ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
          inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1))
            ((basePoint 0 : EucSpace 1)) ≤ -1) := by
  have h4 : (4 : ℝ) < Real.exp 3 := by
    have h := Real.add_one_lt_exp (x := (3 : ℝ)) (by norm_num)
    linarith
  have hsplit : Real.exp (-1 : ℝ) = Real.exp 3 * Real.exp (-4 : ℝ) := by
    rw [← Real.exp_add]
    norm_num
  have hcond : (1 / (1 : ℝ)) * (1 / 2) * (1 - 1 / 2) * Real.exp (-((1 / 2) * 2)) >
      (1 : ℝ) * Real.exp (-((1 - (-1 : ℝ)) * 2)) := by
    norm_num
    rw [hsplit]
    nlinarith [mul_pos (sub_pos.mpr h4) (Real.exp_pos (-4 : ℝ))]
  have hinner : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1))
      ((basePoint 0 : EucSpace 1)) = (1 : ℝ) := by
    have hx : ‖(basePoint 0 : EucSpace 1)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (basePoint 0).2
    rw [real_inner_self_eq_norm_mul_norm, hx]
    ring
  refine ⟨by norm_num, by norm_num, hcond,
    Perspective.SA_const_consensus 1 1 one_pos 2 (basePoint 0), ?_, ?_⟩
  · intro i _ j _
    rw [hinner]
    norm_num
  · intro i _ k hk
    exact absurd (Finset.mem_univ k) (Finset.mem_compl.mp hk)

end Metastability
end Transformer
