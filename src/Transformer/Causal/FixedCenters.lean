/-
# Causal attention — Clustering to frozen tokens (§5 of 2411.04990v2)

`Theorem thm:fixed_centers`: `n` particles on `𝕊^1` running the causal
dynamics `eqn:csa-2d`, with `m` frozen, well-separated tokens `θ_j` entering
with weights `a_j ≥ 1`, converge almost surely to an asymptotically stable
critical point with every particle `ε β^{-1/2}`-close to some `θ_j`.

The angles are lifted to `ℝ`; the geodesic distance on `𝕊^1` is
`|toReal (x : Real.Angle)|`.  Stated and not proved: the paper's proof goes
through `lemma:convergence` (`Causal.sequentialFlow_converges`), itself a
`sorry`.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Causal.Interaction
import Transformer.Causal.InteractionWindow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Angle
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (n m : ℕ)

/-- The geodesic distance on `𝕊^1` between angles whose difference is `x`. -/
noncomputable def circDist (x : ℝ) : ℝ := |(x : Real.Angle).toReal|

/-- **The normalization `Z_k`** of `thm:fixed_centers`:

  `Z_k = ∑_{j ≤ k} e^{β(cos(φ_k - φ_j) - 1)} + ∑_{j ≤ m} a_j e^{β(cos(φ_k - θ_j) - 1)}`.

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
noncomputable def frozenZ (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ) (k : Idx n) : ℝ :=
  (∑ j ∈ Finset.univ.filter (fun j : Idx n => j ≤ k),
      Real.exp (β * (Real.cos (φ k - φ j) - 1)))
    + ∑ j : Idx m, a j * Real.exp (β * (Real.cos (φ k - θ j) - 1))

/-- **The causal dynamics with frozen tokens**, the right-hand side of
`thm:fixed_centers`:

  `φ̇_k = Z_k⁻¹ (∑_{j ≤ k} e^{β(cos(φ_k - φ_j) - 1)} sin(φ_j - φ_k)
                + ∑_{j ≤ m} a_j e^{β(cos(φ_k - θ_j) - 1)} sin(θ_j - φ_k))`.

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
noncomputable def frozenVelocity (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ) :
    Idx n → ℝ := fun k =>
  (frozenZ n m β θ a φ k)⁻¹ *
    ((∑ j ∈ Finset.univ.filter (fun j : Idx n => j ≤ k),
        Real.exp (β * (Real.cos (φ k - φ j) - 1)) * Real.sin (φ j - φ k))
      + ∑ j : Idx m, a j * Real.exp (β * (Real.cos (φ k - θ j) - 1))
          * Real.sin (θ j - φ k))

/-- **Asymptotic stability** of a point `p` for the autonomous equation
`ẋ = F(x)`: solutions starting near `p` stay near it (Lyapunov stability), and
those starting near enough converge to it. -/
def IsAsymptoticallyStable (F : (Idx n → ℝ) → Idx n → ℝ) (p : Idx n → ℝ) : Prop :=
  (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ ∀ x : ℝ → Idx n → ℝ,
      (∀ t : ℝ, HasDerivAt x (F (x t)) t) → ‖x 0 - p‖ < δ →
        ∀ t : ℝ, 0 ≤ t → ‖x t - p‖ < ε) ∧
    ∃ η : ℝ, 0 < η ∧ ∀ x : ℝ → Idx n → ℝ,
      (∀ t : ℝ, HasDerivAt x (F (x t)) t) → ‖x 0 - p‖ < η →
        Filter.Tendsto x Filter.atTop (nhds p)

/-- **Theorem (thm:fixed_centers).** *Clustering to frozen tokens for
`K = Q = V = I_2`.*

Let `θ_1, …, θ_m` be `c β^{-1/2}`-separated frozen tokens with weights
`a_j ≥ 1`, `N = n + ∑ a_j`, and let `N, β, ε > 0`, `c > 2 + 2ε` satisfy

  `N h((c - 1 - 2ε) β^{-1/2}) < h(ε β^{-1/2})`,
  `-N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`.

If `φ(0) ∼ μ₀` for an absolutely continuous probability measure `μ₀`, then with
probability one `φ(t)` converges to an asymptotically stable critical point
`φ*` with every `φ*_k` within `ε β^{-1/2}` of some `θ_j`.

**What the source says and what is changed here.**

* *The sign in the condition on `g`.*  The theorem prints
  `N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`.  Its left side is negative in the
  regime the theorem is meant for (`g` is negative between the peak
  `τ_β^* < β^{-1/2}` of `h` and `π`), where the printed condition holds for
  free, while the proof ends on
  `g(ε β^{-1/2}) < -N g((c - 2ε) β^{-1/2})` "contradicts the parameters
  choice" — a contradiction only with the minus sign.  `lem:interaction`,
  which supplies these conditions, has the minus sign too.  It is restored.
* *`m ≥ 1`.*  At `m = 0` and `n ≥ 1` the statement is false: with no frozen
  token the conclusion `∃ j ∈ [m]` fails for every `φ*`, and one particle
  alone does not move.  The proof's induction starts from a centre; `0 < m`
  is taken as a hypothesis.
* The angles are lifted to `ℝ`, with `|θ_i - θ_j|` read as the geodesic
  distance on `𝕊^1` (`circDist`); `μ₀` is a probability measure on the
  lifted angles, absolutely continuous with respect to Lebesgue measure.
  Solutions are taken on all of `ℝ`: the field is smooth and `2π`-periodic,
  so every solution is global.

Not proved here: the paper derives it from `lemma:convergence`, itself a
`sorry` (`Causal.sequentialFlow_converges`).

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
theorem fixed_centers (hm : 0 < m) (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hh : ((n : ℝ) + ∑ j, a j) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
      < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(((n : ℝ) + ∑ j, a j) * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))))
      < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (μ₀ : Measure (Idx n → ℝ)) [IsProbabilityMeasure μ₀] (hμ₀ : μ₀ ≪ volume) :
    ∀ᵐ φ₀ ∂μ₀, ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ →
      (∀ t : ℝ, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) →
      ∃ φstar : Idx n → ℝ,
        Filter.Tendsto φ Filter.atTop (nhds φstar) ∧
        frozenVelocity n m β θ a φstar = 0 ∧
        IsAsymptoticallyStable n (frozenVelocity n m β θ a) φstar ∧
        ∀ k : Idx n, ∃ j : Idx m, circDist (φstar k - θ j) < ε * β ^ (-(1 / 2 : ℝ)) := by
  sorry

/-- The hypotheses of `fixed_centers` are satisfiable, and not vacuously so:
one particle, the two antipodal frozen tokens `θ = (0, π)` with unit weights,
`N = 3`, `β = 11`, `ε = 0.09`, `c = 5.68`, and `μ₀` uniform on `[0, 1)`.  The
conditions on `h` and `g` are `lem:interaction`
(`Causal.interaction_inequalities`), whose bound on `N` is
`3 < e^{3·4.5²/8} · 0.09/4.68`. -/
example :
    (0 : ℕ) < 2 ∧ (0 : ℝ) < 11 ∧ (0 : ℝ) < 9 / 100 ∧ 2 + 2 * (9 / 100 : ℝ) < 142 / 25 ∧
      (∀ j : Idx 2, (1 : ℝ) ≤ (fun _ => 1 : Idx 2 → ℝ) j) ∧
      (∀ i j : Idx 2, i ≠ j → (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ))
        < circDist ((![0, π] : Idx 2 → ℝ) i - (![0, π] : Idx 2 → ℝ) j)) ∧
      (((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j)
          * h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))
        < h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
      -((((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j)
          * g_pot 11 ((142 / 25 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))))
        < g_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
      IsProbabilityMeasure (volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1))) ∧
      volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1)) ≪ volume := by
  have hN : ((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ => 1 : Idx 2 → ℝ) j = 3 := by
    simp; norm_num
  have hex : (156 : ℝ) < Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) := by
    have h1 := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 81 / 32 by norm_num)
    have h3 : Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) = Real.exp (81 / 32) ^ 3 := by
      rw [← Real.exp_nat_mul]; norm_num
    rw [h3]
    calc (156 : ℝ) < (1 + 81 / 32 + (81 / 32) ^ 2 / 2) ^ 3 := by norm_num
      _ ≤ Real.exp (81 / 32) ^ 3 := by gcongr
  have hwin := interaction_inequalities 3 (142 / 25) (9 / 100) 11 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by nlinarith)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, fun _ => le_rfl, ?_,
    by rw [hN]; exact hwin.1, by rw [hN]; exact hwin.2, ?_,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self⟩
  · have hs : (11 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 1 / 3 := by
      rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow, one_div]
      have : (3 : ℝ) ≤ Real.sqrt 11 := by
        rw [show (3 : ℝ) = Real.sqrt 9 by
          rw [show (9 : ℝ) = 3 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
        exact Real.sqrt_le_sqrt (by norm_num)
      exact inv_anti₀ (by norm_num) this
    have hpi : circDist π = π := Real.Angle.abs_toReal_coe_eq_self_iff.mpr ⟨pi_pos.le, le_rfl⟩
    have hpi' : circDist (-π) = π := by
      rw [circDist, Real.Angle.coe_neg, Real.Angle.neg_coe_pi]; exact hpi
    have hlt : (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) < π := by
      nlinarith [pi_gt_three]
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all
  · constructor
    rw [Measure.restrict_apply_univ, Real.volume_pi_Ico]
    simp
end Causal
end Transformer
