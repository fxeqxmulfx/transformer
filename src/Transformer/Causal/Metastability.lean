/-
# Causal attention — Meta-stable clustering and R'enyi parking (§5 of 2411.04990v2)

Equations and statements covered:

* *R'enyi centers* and *strong R'enyi centers* (Definition),
* `Lemma lemma:meta`               — strong R'enyi centers stay nearly fixed,
* `eq: renyi`, `eq: c`, `eq: time_bd`.

`Theorem thm:fixed_centers`, clustering to frozen tokens, is in
`Causal.FixedCenters`.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Causal.Packing
import Transformer.Causal.Section5_StrongGap

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

variable (d n : ℕ)

/-- A *R'enyi center subsequence* of `(x_j)_{j ≥ 1}`:

  `dist(x_{s_j}, x_{s_i}) > δ`  for all `i < j`. -/
def isRenyiCenters
    (x : ℕ → SSphere d) (s : ℕ → ℕ) (δ : ℝ) : Prop :=
  StrictMono s ∧
    ∀ i j : ℕ, i < j →
      δ < ‖((x (s j) : EucSpace d)) - ((x (s i) : EucSpace d))‖

/-- A *strong R'enyi center* subsequence:

  `dist(x_{s_j}, x_i) > δ`  for all `i < s_j`. -/
def isStrongRenyiCenters
    (x : ℕ → SSphere d) (s : ℕ → ℕ) (δ : ℝ) : Prop :=
  StrictMono s ∧
    ∀ j : ℕ, ∀ i : ℕ, i < s j →
      δ < ‖((x (s j) : EucSpace d)) - ((x i : EucSpace d))‖

/-- **Lemma (lemma: meta).**  *Strong R'enyi centers stay nearly fixed.*

At `d=2`, `Q=K=V=I₂`, initial predecessor separation
`c(1+2ε)β^{-1/2}` above the interaction peak and the time condition
`T_j s_j h(cβ^{-1/2}) < εcβ^{-1/2}` imply displacement below `εcβ^{-1/2}`.

The paper uses one-based labels and bounds its `s_j-1` interactions by
`s_j`. Here `s j` is zero-based and is the exact predecessor count; the
same proof gives this sharper time condition. Chord gaps are treated
directly, avoiding the source proof's changing closest angular coordinate.

Source: arXiv:2411.04990v2, §5, `lemma: meta`. -/
theorem lemma_meta
    (β c ε : ℝ) (hβ : 0 < β) (hc : 0 < c) (hε : 0 < ε)
    (X : ℝ → SphereTuple 2 n)
    (hX : Causal.CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
            (ContinuousLinearMap.id ℝ (EucSpace 2))
            (ContinuousLinearMap.id ℝ (EucSpace 2)) X)
    (s : ℕ → ℕ) (hs : ∀ j : ℕ, s j < n)
    (h_strong :
      ∀ (j : ℕ) (i : Idx n), (i : ℕ) < s j →
        c * (1 + 2 * ε) * β ^ (-(1/2 : ℝ))
          < ‖((X 0 ⟨s j, hs j⟩ : EucSpace 2)) - ((X 0 i : EucSpace 2))‖)
    (h_c : Real.sqrt β
              * Real.arccos ((-1 + Real.sqrt (4 * β^2 + 1)) / (2 * β))
            < c)
    (j : ℕ) (T_j : ℝ) (hT :
      T_j * (s j : ℝ) * Causal.h_pot β (c * β ^ (-(1/2 : ℝ)))
        < ε * c * β ^ (-(1/2 : ℝ))) :
    ∀ t : ℝ, 0 ≤ t → t ≤ T_j →
      ‖((X t ⟨s j, hs j⟩ : EucSpace 2))
          - ((X 0 ⟨s j, hs j⟩ : EucSpace 2))‖
        < ε * c * β ^ (-(1/2 : ℝ)) := by
  let r := c * β ^ (-(1 / 2 : ℝ))
  have hp : 0 < β ^ (-(1 / 2 : ℝ)) := Real.rpow_pos_of_pos hβ _
  have hr : 0 < r := mul_pos hc hp
  have hsqrt : Real.sqrt β * β ^ (-(1 / 2 : ℝ)) = 1 := by
    rw [Real.sqrt_eq_rpow, Real.rpow_neg hβ.le,
      mul_inv_cancel₀ (ne_of_gt (Real.rpow_pos_of_pos hβ _))]
  have hpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r := by
    apply le_of_lt
    calc
      _ = Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) *
          (Real.sqrt β * β ^ (-(1 / 2 : ℝ))) := by rw [hsqrt, mul_one]
      _ = (Real.sqrt β * Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β))) *
          β ^ (-(1 / 2 : ℝ)) := by ring
      _ < r := mul_lt_mul_of_pos_right h_c hp
  let k : Idx n := ⟨s j, hs j⟩
  have hinit (i : Idx n) (hi : i < k) :
      (1 + 2 * ε) * r < ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖ := by
    have h := h_strong j i (Fin.lt_def.mp hi)
    dsimp only [r, k]
    nlinarith [h]
  have htime : T_j * (k : ℝ) * h_pot β r < ε * r := by
    simpa only [k, r, mul_assoc] using hT
  intro t ht0 htT
  change ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ < _
  rw [mul_assoc]
  change ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ < ε * r
  by_cases hTpos : 0 < T_j
  · by_cases hk : (k : ℕ) = 0
    · have hd (u : ℝ) : HasDerivAt (fun v => (X v k : EucSpace 2)) 0 u := by
        have hempty : Finset.Iio k = ∅ := by
          ext i
          simp only [Finset.mem_Iio, Finset.notMem_empty, iff_false]
          change ¬ (i : ℕ) < (k : ℕ)
          omega
        have hz : causalDrift β (X u) k = 0 := by
          simp only [causalDrift, hempty, Finset.sum_empty, smul_zero]
        simpa only [hz] using hasDerivAt_causalDrift β X hX u k
      have hconst := is_const_of_deriv_eq_zero (fun u => (hd u).differentiableAt)
        (fun u => (hd u).deriv) t 0
      rw [hconst, sub_self, norm_zero]
      exact mul_pos hε hr
    · have hkpos : 0 < (k : ℕ) := by omega
      let i : Idx n := ⟨0, (hkpos.trans k.isLt)⟩
      have hi : i < k := hkpos
      have hsep := hinit i hi
      have hchord := norm_sub_le (X 0 k : EucSpace 2) (X 0 i : EucSpace 2)
      rw [mem_sphere_zero_iff_norm.mp (X 0 k).2,
        mem_sphere_zero_iff_norm.mp (X 0 i).2] at hchord
      have hrπ : r ≤ π := by nlinarith [mul_pos hε hr, Real.pi_gt_three]
      exact strong_center_displacement β hβ r ε T_j hr hε hTpos hrπ hpeak
        X hX k hinit htime t ⟨ht0, htT⟩
  · have ht : t = 0 := by linarith
    subst t
    rw [sub_self, norm_zero]
    exact mul_pos hε hr

/-- The hypotheses of `lemma_meta` are satisfiable: one token on the circle,
`s ≡ 0` — so that `h_strong` quantifies over an empty range and `hT` reads
`0 < ε c β^{-1/2}` — at `β = 1`, `c = 5` (a value above `π ≥ arccos`) and
`ε = 1`. -/
example (x₀ : SSphere 2) :
    ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      ‖((x₀ : EucSpace 2)) - ((x₀ : EucSpace 2))‖
        < 1 * 5 * (1 : ℝ) ^ (-(1/2 : ℝ)) :=
  lemma_meta 1 1 5 1 one_pos (by norm_num) one_pos (fun _ => fun _ => x₀)
    (csa_const_one 1 x₀) (fun _ => 0) (fun _ => one_pos)
    (fun _ i hi => absurd hi (Nat.not_lt_zero _))
    (by
      rw [Real.sqrt_one, one_mul]
      exact lt_of_le_of_lt (Real.arccos_le_pi _) (by linarith [Real.pi_le_four]))
    0 1
    (by
      rw [Real.one_rpow]
      norm_num)

/-- **Cardinality of the R'enyi centers.**

The number of (strong) R'enyi centers with separation `δ = c β^{-1/2}` is
`Θ(β^{(d-1)/2})` — the packing number of the sphere `𝕊^{d-1}` at that scale,
since the centers are exactly a `δ`-separated set of unit vectors.  The `Θ` is
spelled out as a pair of constants independent of `β`.

Proved, from the volume estimates of `Transformer.Causal.Packing`: the upper
half holds for *every* `δ`-separated set, and the lower half is witnessed by a
maximal one, which exists by `Packing.exists_maximalSeparated`.

What is proved is therefore the geometric half of the survey's claim — the
packing number of the sphere at scale `δ`.  The survey's centers are the
centers *of a given sequence*, and that they are as many as the packing number
allows needs in addition that they are a maximal separated set, which is a
property of the sequence and not of the geometry; the expected count, for an
i.i.d. sequence, is `Causal.strong_renyi_expected_count`.

Source: arXiv:2411.04990v2, §5 (the cardinality conjecture). -/
theorem renyi_count (d : ℕ) (c : ℝ) (hd : 2 ≤ d) (hc : 0 < c) :
    ∃ C₁ C₂ : ℝ, 0 < C₁ ∧ 0 < C₂ ∧ ∀ β : ℝ, 1 ≤ β →
      (∀ S : Finset (EucSpace d),
          Packing.SeparatedOnSphere d S (c * β ^ (-(1/2 : ℝ))) →
          (S.card : ℝ) ≤ C₂ * β ^ (((d : ℝ) - 1) / 2)) ∧
      ∃ S : Finset (EucSpace d),
        Packing.SeparatedOnSphere d S (c * β ^ (-(1/2 : ℝ))) ∧
        C₁ * β ^ (((d : ℝ) - 1) / 2) ≤ (S.card : ℝ) := by
  have hdpos : (0 : ℝ) < (d : ℝ) := by
    have : 0 < d := by omega
    exact_mod_cast this
  refine ⟨min ((d : ℝ) * (1 / (4 * c)) ^ (d - 1)) ((1 / (2 * c)) ^ (d - 1)),
    2 * d * max (4 / c) 2 ^ (d - 1),
    lt_min (mul_pos hdpos (by positivity)) (by positivity),
    mul_pos (by linarith) (by positivity), fun β hβ =>
      ⟨fun S hS => Packing.card_le_at_renyi_scale d (by omega) c hc β hβ S hS,
        Packing.exists_card_ge_at_renyi_scale d (by omega) c hc β hβ⟩⟩

/-- The hypotheses of `renyi_count` are satisfiable: `d = 2`, `c = 1`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 := ⟨le_rfl, one_pos⟩

end Causal
end Transformer
