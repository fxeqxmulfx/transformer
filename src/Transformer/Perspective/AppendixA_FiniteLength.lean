/-
# Appendix A — the trajectories of `e:gradfl` have finite length

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, proof of `p:beta0`.

`‖s‖²`, `s = Σ_j x_j`, increases along `e:gradfl` to its supremum `Q_∞`.  If it
reaches `Q_∞`, its derivative `(4/n) Σ_i ‖Proj_{x_i} s‖²` vanishes there, the
configuration is critical and the trajectory stands still
(`isCriticalE0_of_eq_iSup`).  Otherwise the gradient inequality of
`Perspective.AppendixA_Lojasiewicz`, the squares `m²`, `|m| ≤ n`, being
finitely many, becomes the Łojasiewicz inequality
`Q_∞ - ‖s‖² ≤ C Σ_i ‖Proj_{x_i} s‖²` from some time on
(`eventually_iSup_sub_le`).  Then `φ = √(Q_∞ - ‖s‖²)` decreases at rate at
least `‖ẋ_i‖ / K`, so each particle travels at most `K φ(t)` after time `t`
(`exists_norm_sub_le`).
-/

import Transformer.Perspective.AppendixA_Flow
import Transformer.Perspective.AppendixA_Lojasiewicz

open Filter Set
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The Łojasiewicz inequality along `e:gradfl`.**  Once `‖s‖²` is positive,
its gap to the supremum `Q_∞` is eventually at most `C Σ_i ‖Proj_{x_i} s‖²`:
`‖s‖²` is within `C Σ_i ‖Proj_{x_i} s‖²` of a square `m²`, `|m| ≤ n`
(`exists_int_mul_abs_sub_le`), which is `Q_∞` itself once `‖s‖²` is closer to
`Q_∞` than half the distance from `Q_∞` to the other squares. -/
theorem eventually_iSup_sub_le {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X)
    {t₀ : ℝ} (ht₀ : 0 < ‖∑ j : Idx n, (X t₀ j : EucSpace d)‖ ^ 2) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ t in atTop, (⨆ s, ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)
      - ‖∑ j : Idx n, (X t j : EucSpace d)‖ ^ 2 ≤ C * tangentSq (X t) := by
  set Q : ℝ → ℝ := fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2 with hQ
  obtain ⟨γ, hγ, hgap⟩ := exists_pos_le_abs_sq_sub n (⨆ s, Q s)
  have hlim := tendsto_atTop_ciSup (monotone_norm_sum_sq X hX) (bddAbove_norm_sum_sq X)
  refine ⟨2 * n / Q t₀, by positivity, ?_⟩
  filter_upwards [eventually_ge_atTop t₀,
    hlim.eventually (lt_mem_nhds (sub_lt_self _ (half_pos hγ)))] with t ht hQt
  have hle : Q t ≤ ⨆ s, Q s := le_ciSup (bddAbove_norm_sum_sq X) t
  obtain ⟨m, hm, hineq⟩ := exists_int_mul_abs_sub_le (X t)
  have h1 : |Q t - (m : ℝ) ^ 2| ≤ 2 * n / Q t₀ * tangentSq (X t) := by
    rw [div_mul_eq_mul_div, le_div_iff₀ ht₀]
    calc |Q t - (m : ℝ) ^ 2| * Q t₀ ≤ |Q t - (m : ℝ) ^ 2| * Q t :=
          mul_le_mul_of_nonneg_left (monotone_norm_sum_sq X hX ht) (abs_nonneg _)
      _ ≤ 2 * n * tangentSq (X t) := by rw [mul_comm]; exact hineq
  by_cases hm2 : (m : ℝ) ^ 2 = ⨆ s, Q s
  · rw [← hm2]
    exact (le_abs_self _).trans ((abs_sub_comm _ _).le.trans h1)
  · have hγm := hgap m hm hm2
    have htri := abs_sub_le ((m : ℝ) ^ 2) (Q t) (⨆ s, Q s)
    rw [abs_sub_comm (Q t), abs_of_nonneg (sub_nonneg.mpr hle),
      abs_sub_comm ((m : ℝ) ^ 2) (Q t)] at htri
    linarith

/-- The hypotheses of `eventually_iSup_sub_le` hold for a consensus of three
particles on the circle, a stationary solution with `‖s‖² = 9`. -/
example (x : SSphere 2) : E0GradientAscent 2 3 (fun _ => signedTuple x ∅) ∧
    0 < ‖∑ j : Idx 3, (signedTuple x ∅ j : EucSpace 2)‖ ^ 2 :=
  ⟨E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅), by simp [sum_signedTuple, norm_smul]⟩

/-- **A trajectory reaching the supremum of `‖s‖²` stands still at a critical
point:** the derivative `(4/n) Σ_i ‖Proj_{x_i} s‖²` vanishes at a maximum, and
the constant solution at a critical point is the only one through it. -/
theorem isCriticalE0_of_eq_iSup (hn : 0 < n) {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) {t₁ : ℝ} (h : ‖∑ j : Idx n, (X t₁ j : EucSpace d)‖ ^ 2
      = ⨆ s, ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2) :
    IsCriticalE0 d n (X t₁) ∧ ∀ t, X t = X t₁ := by
  have hmax : IsLocalMax (fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2) t₁ :=
    Eventually.of_forall fun s => (le_ciSup (bddAbove_norm_sum_sq X) s).trans h.ge
  have h0 := hmax.hasDerivAt_eq_zero (hasDerivAt_norm_sum_sq hn hX t₁)
  have hcrit : IsCriticalE0 d n (X t₁) := isCriticalE0_of_tangentSq_eq_zero
    ((mul_eq_zero.mp h0).resolve_left (by positivity))
  refine ⟨hcrit, fun t => ?_⟩
  have h1 := eq_e0Flow (fun t i => (hX (t₁ + t) i).comp_const_add t₁ t) (t - t₁)
  have h2 := eq_e0Flow (E0GradientAscent_const (X t₁) hcrit) (t - t₁)
  simp only [add_zero, add_sub_cancel] at h1
  exact h1.trans h2.symm

/-- The hypotheses of `isCriticalE0_of_eq_iSup` hold for a consensus of three
particles on the circle, along which `‖s‖²` is constant. -/
example (x : SSphere 2) : 0 < 3 ∧ E0GradientAscent 2 3 (fun _ => signedTuple x ∅) ∧
    ‖∑ j : Idx 3, (signedTuple x ∅ j : EucSpace 2)‖ ^ 2
      = ⨆ _ : ℝ, ‖∑ j : Idx 3, (signedTuple x ∅ j : EucSpace 2)‖ ^ 2 :=
  ⟨by norm_num, E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅), ciSup_const.symm⟩

/-- **Finite length.**  From some time `T` on, each particle of a trajectory of
`e:gradfl` travels at most `K √(Q_∞ - ‖s(t)‖²)` after time `t`, `Q_∞` the
supremum of `‖s‖²`.  A trajectory reaching `Q_∞` stands still
(`isCriticalE0_of_eq_iSup`); otherwise `φ = √(Q_∞ - ‖s‖²)` has
`φ' = -(2/n) Σ_k ‖Proj_{x_k} s‖² / φ` while `‖ẋ_i‖ = (2/n) ‖Proj_{x_i} s‖`, and
`‖Proj_{x_i} s‖ φ ≤ K Σ_k ‖Proj_{x_k} s‖²` by the Łojasiewicz inequality
(`eventually_iSup_sub_le`), so `‖ẋ_i‖ ≤ -K φ'`. -/
theorem exists_norm_sub_le (hn : 0 < n) {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) :
    ∃ K T : ℝ, ∀ i : Idx n, ∀ t₁, T ≤ t₁ → ∀ t₂, t₁ ≤ t₂ →
      ‖(X t₂ i : EucSpace d) - X t₁ i‖ ≤ K * √((⨆ s, ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2)
        - ‖∑ j : Idx n, (X t₁ j : EucSpace d)‖ ^ 2) := by
  set Q : ℝ → ℝ := fun s => ‖∑ j : Idx n, (X s j : EucSpace d)‖ ^ 2 with hQ
  by_cases hA : ∃ t₁, Q t₁ = ⨆ s, Q s
  · obtain ⟨t₁, h⟩ := hA
    have heq := (isCriticalE0_of_eq_iSup hn hX h).2
    exact ⟨0, 0, fun i t₁ _ t₂ _ => by simp [heq]⟩
  have hpos : ∀ t, 0 < (⨆ s, Q s) - Q t := fun t =>
    sub_pos.mpr (lt_of_le_of_ne (le_ciSup (bddAbove_norm_sum_sq X) t) fun h => hA ⟨t, h⟩)
  obtain ⟨t₀, ht₀⟩ := exists_lt_of_lt_ciSup (sub_pos.mp (hpos 0))
  obtain ⟨C, hC, hev⟩ := eventually_iSup_sub_le hX (lt_of_le_of_lt (sq_nonneg _) ht₀)
  obtain ⟨T, hT⟩ := eventually_atTop.mp hev
  refine ⟨C + 1, T, fun i t₁ ht₁ t₂ ht₂ => ?_⟩
  have hφ : ∀ t, HasDerivAt (fun s => √((⨆ s, Q s) - Q s))
      ((0 - 4 * (n : ℝ)⁻¹ * tangentSq (X t)) / (2 * √((⨆ s, Q s) - Q t))) t := fun t =>
    ((hasDerivAt_const t _).sub (hasDerivAt_norm_sum_sq hn hX t)).sqrt (hpos t).ne'
  have hbd : ∀ t ∈ Ico t₁ t₂,
      ‖proj d (X t i) ((2 * (n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d))‖ ≤ (C + 1) *
        (0 - (0 - 4 * (n : ℝ)⁻¹ * tangentSq (X t)) / (2 * √((⨆ s, Q s) - Q t))) := by
    intro t ht
    set G := tangentSq (X t) with hG
    set φ := √((⨆ s, Q s) - Q t) with hφdef
    set p := proj d (X t i) (∑ j : Idx n, (X t j : EucSpace d)) with hp
    have hφ0 : 0 < φ := Real.sqrt_pos.mpr (hpos t)
    have hG0 : 0 ≤ G := tangentSq_nonneg _
    have hp2 : ‖p‖ ^ 2 ≤ G :=
      Finset.single_le_sum (f := fun k => ‖proj d (X t k) (∑ j : Idx n, (X t j : EucSpace d))‖ ^ 2)
        (fun k _ => sq_nonneg _) (Finset.mem_univ i)
    have hφ2 : φ ^ 2 ≤ C * G := by
      rw [hφdef, Real.sq_sqrt (hpos t).le]
      exact hT t (ht₁.trans ht.1)
    have hpφ : ‖p‖ * φ ≤ (C + 1) * G := by
      rw [← sq_le_sq₀ (by positivity) (by positivity)]
      calc (‖p‖ * φ) ^ 2 = ‖p‖ ^ 2 * φ ^ 2 := by ring
        _ ≤ G * (C * G) := mul_le_mul hp2 hφ2 (sq_nonneg _) hG0
        _ ≤ ((C + 1) * G) ^ 2 := by
          nlinarith [sq_nonneg G, mul_nonneg hC (sq_nonneg G), sq_nonneg (C * G)]
    rw [proj_smul, norm_smul, Real.norm_of_nonneg (by positivity),
      show (C + 1) * (0 - (0 - 4 * (n : ℝ)⁻¹ * G) / (2 * φ))
        = 2 * (n : ℝ)⁻¹ * ((C + 1) * G) / φ by field_simp; ring, le_div_iff₀ hφ0]
    calc 2 * (n : ℝ)⁻¹ * ‖p‖ * φ = 2 * (n : ℝ)⁻¹ * (‖p‖ * φ) := by ring
      _ ≤ 2 * (n : ℝ)⁻¹ * ((C + 1) * G) := by gcongr
  have hB := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (f := fun t => (X t i : EucSpace d) - X t₁ i) (a := t₁) (b := t₂)
    (fun t _ => ((hX t i).sub_const _).continuousAt.continuousWithinAt)
    (fun t _ => ((hX t i).sub_const _).hasDerivWithinAt)
    (B := fun t => (C + 1) * (√((⨆ s, Q s) - Q t₁) - √((⨆ s, Q s) - Q t)))
    (by simp) (fun t => ((hasDerivAt_const t _).sub (hφ t)).const_mul (C + 1)) hbd
    ⟨ht₂, le_rfl⟩
  refine hB.trans ?_
  have := Real.sqrt_nonneg ((⨆ s, Q s) - Q t₂)
  nlinarith

/-- The hypotheses of `exists_norm_sub_le` hold for a consensus of three
particles on the circle. -/
example (x : SSphere 2) : 0 < 3 ∧ E0GradientAscent 2 3 (fun _ => signedTuple x ∅) :=
  ⟨by norm_num, E0GradientAscent_const _ (isCriticalE0_signedTuple x ∅)⟩

end Perspective
end Transformer
