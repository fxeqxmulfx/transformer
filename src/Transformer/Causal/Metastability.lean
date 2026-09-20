/-
# Causal attention — Meta-stable clustering and R'enyi parking (§5 of 2411.04990v2)

Equations and statements covered:

* *R'enyi centers* and *strong R'enyi centers* (Definition),
* `Lemma lemma:meta`               — strong R'enyi centers stay nearly fixed,
* `eq: renyi`, `eq: c`, `eq: time_bd`,
* `Theorem thm:fixed_centers`     — convergence of all tokens to the
                                    vicinity of strong R'enyi centers.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Causal.Packing

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

For `d = 2`, `Q = K = V = I_2`, suppose the strong R'enyi centers
`x_{s_1},…,x_{s_m}` satisfy

  `min_{i < s_j} |x_{s_j} - x_i| > c (1 + 2 ε) β^{-1/2}`

with `c > β^{1/2} arccos((-1 + √(4 β² + 1)) / (2 β))`.  For any `T_j` with

  `T_j · s_j · h(c β^{-1/2}) < ε c β^{-1/2}`,

the displacement of each center is bounded:

  `max_{t ∈ [0, T_j]} |x_{s_j}(t) - x_{s_j}(0)| < ε c β^{-1/2}`.

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
  sorry

/-- A single token on the circle is a stationary solution of `CSA`: its own
attention average is itself, and `Proj_x x = 0`. -/
theorem csa_const_one (β : ℝ) (x₀ : SSphere 2) :
    Causal.CSA 2 1 β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) (fun _ => fun _ => x₀) := by
  intro t k
  have hx : ‖(x₀ : EucSpace 2)‖ = 1 := by
    exact mem_sphere_zero_iff_norm.mp x₀.2
  have hE : (Real.exp (β * inner (𝕜 := ℝ) ((x₀ : EucSpace 2)) ((x₀ : EucSpace 2))))⁻¹
      • (Real.exp (β * inner (𝕜 := ℝ) ((x₀ : EucSpace 2)) ((x₀ : EucSpace 2)))
          • (x₀ : EucSpace 2)) = (x₀ : EucSpace 2) := by
    rw [smul_smul, inv_mul_cancel₀ (Real.exp_ne_zero _), one_smul]
  have hproj : proj 2 ((x₀ : EucSpace 2)) ((x₀ : EucSpace 2)) = 0 := by
    rw [proj, real_inner_self_eq_norm_mul_norm, hx, one_mul, one_smul, sub_self]
  simpa [Subsingleton.elim k (0 : Idx 1), hE, hproj] using
    (hasDerivAt_const t ((x₀ : EucSpace 2)))

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

/-- **Theorem (thm: fixed_centers).**  *Convergence to strong R'enyi centers.*

For an arbitrary set of stationary (strong R'enyi) tokens, all other tokens
converge to the vicinity of one of them as `t → ∞`: every token eventually
stays within the separation scale `δ` of some center.

Not proved here: the paper's proof runs through `lemma_meta`, which is itself a
`sorry`, and the "vicinity" of the statement is taken to be the separation
scale `δ` the centers are defined by.

Source: arXiv:2411.04990v2, §5, `thm: fixed_centers`. -/
theorem fixed_centers_convergence (n : ℕ) (β δ : ℝ) (hβ : 0 < β) (hδ : 0 < δ) :
  ∀ X : ℝ → SphereTuple 2 n,
    Causal.CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X →
    ∀ (m : ℕ) (s : ℕ → ℕ) (hs : ∀ j : ℕ, s j < n),
      (∀ (j : ℕ), j < m → ∀ i : Idx n, (i : ℕ) < s j →
        δ < ‖((X 0 ⟨s j, hs j⟩ : EucSpace 2)) - ((X 0 i : EucSpace 2))‖) →
      ∀ i : Idx n, ∃ j : ℕ, j < m ∧
        ∀ᶠ t : ℝ in Filter.atTop,
          ‖((X t i : EucSpace 2)) - ((X t ⟨s j, hs j⟩ : EucSpace 2))‖ < δ := by
  sorry

/-- The hypotheses of `fixed_centers_convergence` are satisfiable: `β = δ = 1`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := ⟨one_pos, one_pos⟩

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
