/-
# Metastability — the three phases of the collapse, as explicit subsolutions

Steps 2–3 of the direct proof of `thm: metastability` (arXiv:2410.06833v1, §2) follow the
within-cap minimum `ρ = 1 - s` of a cap `I` while `s` falls from `8ε` to `δ = e^{-λβ}`.  With
`attn_comparison` the ODE of the source is replaced by three explicit profiles `s(t)`, each
a strict subsolution of `ρ̇ ≥ (2/n) F_β(s) - 2 n E` (`E = e^{-(1-α)β}`):

* `phase_linear` — `s(t) = r - c (t - a₀)`, for a constant `c ≤ F_β(s)/n` on the range;
* `phase_exp` — `s(t) = σ e^{-c₂ (t - a₀)}`, for `c₂ s ≤ F_β(s)/n` on the range;
* `phase_const` — `s(t) = δ` on a further interval (the propagation of smallness).

Each is an instance of `comparison_of_rate`: a profile `s` with `-s' ≤ F_β(s)/n` and `F_β(s)`
above the leakage level `2 n² E` while `s` is in the range of the phase.
-/

import Transformer.Metastability.AttnComparison
import Transformer.Metastability.CollapseScalar
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- **Comparison in terms of the distance `s` to `1`.**  If `s` is differentiable with
`-s' ≤ F_β(s)/n` and `F_β(s) > 2 n² e^{-(1-α)β}` on `[a₀, b)`, `s ≤ 1` on `[a₀, b]`, the
tokens of `I` are `α`-separated from the others on `[a₀, b]`, and `1 - s(a₀) ≤ ⟨x_i, x_j⟩` at
`a₀` for `i, j ∈ I`, then `1 - s(t) ≤ ⟨x_i(t), x_j(t)⟩` for all `t ∈ [a₀, b]`. -/
theorem comparison_of_rate {β α : ℝ} (hβ : 0 ≤ β) (hn : 1 ≤ n) {X : ℝ → SphereTuple d n}
    (hX : IsAttnFlow d n β X) (I : Finset (Idx n)) (a₀ b : ℝ) (hab : a₀ ≤ b)
    (s s' : ℝ → ℝ) (hsd : ∀ t, HasDerivAt s (s' t) t) (hs1 : ∀ t ∈ Icc a₀ b, s t ≤ 1)
    (hrate : ∀ t ∈ Ico a₀ b, -s' t ≤ collapseRate β (s t) / n)
    (hthr : ∀ t ∈ Ico a₀ b,
      2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β (s t))
    (hfar : ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ k, k ∉ I →
      ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I,
      1 - s a₀ ≤ ⟪(X a₀ i : EucSpace d), (X a₀ j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ j ∈ I,
      1 - s t ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  have hd : ∀ t, HasDerivAt (fun t => 1 - s t) (-s' t) t := fun t => (hsd t).const_sub 1
  refine attn_comparison β α hβ hX I a₀ b hab (fun t => 1 - s t)
    (continuous_iff_continuousAt.2 fun t => (hd t).continuousAt).continuousOn
    (fun t _ => (hd t).differentiableAt) (fun t ht => sub_nonneg.2 (hs1 t ht))
    (fun t ht => ?_) hfar hinit
  rw [(hd t).deriv]
  exact sub_of_rate hn (hthr t ht) (hrate t ht)

/-- **Phase 1: a linear subsolution.**  If `c > 0` and `c ≤ F_β(s)/n`, `F_β(s) > 2 n² E` for
all `s ∈ (σ, r]`, `σ ≤ r ≤ 1`, then on `[a₀, a₀ + (r - σ)/c]` the cap satisfies
`⟨x_i, x_j⟩ ≥ 1 - r + c (t - a₀)`, provided it does at `a₀` (`⟨x_i, x_j⟩ ≥ 1 - r`). -/
theorem phase_linear {β α : ℝ} (hβ : 0 ≤ β) (hn : 1 ≤ n) {X : ℝ → SphereTuple d n}
    (hX : IsAttnFlow d n β X) (I : Finset (Idx n)) (a₀ r σ c : ℝ) (hc : 0 < c)
    (hσr : σ ≤ r) (hr1 : r ≤ 1)
    (hcF : ∀ s ∈ Ioc σ r, c ≤ collapseRate β s / n)
    (hthr : ∀ s ∈ Ioc σ r, 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β s)
    (hfar : ∀ t ∈ Icc a₀ (a₀ + (r - σ) / c), ∀ i ∈ I, ∀ k, k ∉ I →
      ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I,
      1 - r ≤ ⟪(X a₀ i : EucSpace d), (X a₀ j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc a₀ (a₀ + (r - σ) / c), ∀ i ∈ I, ∀ j ∈ I,
      1 - (r - c * (t - a₀)) ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  have hb : a₀ ≤ a₀ + (r - σ) / c := by
    have : 0 ≤ (r - σ) / c := div_nonneg (by linarith) hc.le
    linarith
  have hmem : ∀ t ∈ Ico a₀ (a₀ + (r - σ) / c), r - c * (t - a₀) ∈ Ioc σ r := by
    intro t ht
    have h1 : t - a₀ < (r - σ) / c := by linarith [ht.2]
    rw [lt_div_iff₀ hc] at h1
    exact ⟨by nlinarith [ht.1], by nlinarith [ht.1]⟩
  refine comparison_of_rate hβ hn hX I a₀ (a₀ + (r - σ) / c) hb
    (fun t => r - c * (t - a₀)) (fun _ => -c) ?_ ?_ ?_ ?_ hfar ?_
  · intro t
    simpa using ((hasDerivAt_id t).sub_const a₀).const_mul c |>.const_sub r
  · intro t ht
    have : 0 ≤ c * (t - a₀) := mul_nonneg hc.le (by linarith [ht.1])
    show r - c * (t - a₀) ≤ 1
    linarith
  · intro t ht
    simpa using hcF _ (hmem t ht)
  · intro t ht
    exact hthr _ (hmem t ht)
  · intro i hi j hj
    simpa using hinit i hi j hj

/-- **Phase 2: an exponential subsolution.**  If `c₂ > 0`, `0 < σ₂ ≤ σ₀ ≤ 1`, and
`c₂ s ≤ F_β(s)/n`, `F_β(s) > 2 n² E` for all `s ∈ (σ₂, σ₀]`, then on
`[a₀, a₀ + log(σ₀/σ₂)/c₂]` the cap satisfies `⟨x_i, x_j⟩ ≥ 1 - σ₀ e^{-c₂(t - a₀)}`, provided
`⟨x_i, x_j⟩ ≥ 1 - σ₀` at `a₀`.  At the right end `σ₀ e^{-c₂(t - a₀)} = σ₂`. -/
theorem phase_exp {β α : ℝ} (hβ : 0 ≤ β) (hn : 1 ≤ n) {X : ℝ → SphereTuple d n}
    (hX : IsAttnFlow d n β X) (I : Finset (Idx n)) (a₀ σ₀ σ₂ c₂ : ℝ) (hc : 0 < c₂)
    (hσ₂ : 0 < σ₂) (hσ : σ₂ ≤ σ₀) (hσ1 : σ₀ ≤ 1)
    (hcF : ∀ s ∈ Ioc σ₂ σ₀, c₂ * s ≤ collapseRate β s / n)
    (hthr : ∀ s ∈ Ioc σ₂ σ₀, 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β s)
    (hfar : ∀ t ∈ Icc a₀ (a₀ + Real.log (σ₀ / σ₂) / c₂), ∀ i ∈ I, ∀ k, k ∉ I →
      ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I,
      1 - σ₀ ≤ ⟪(X a₀ i : EucSpace d), (X a₀ j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc a₀ (a₀ + Real.log (σ₀ / σ₂) / c₂), ∀ i ∈ I, ∀ j ∈ I,
      1 - σ₀ * Real.exp (-(c₂ * (t - a₀))) ≤
        ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  have hratio : 1 ≤ σ₀ / σ₂ := by rw [le_div_iff₀ hσ₂]; linarith
  have hlog : 0 ≤ Real.log (σ₀ / σ₂) := Real.log_nonneg hratio
  have hb : a₀ ≤ a₀ + Real.log (σ₀ / σ₂) / c₂ := by
    have : 0 ≤ Real.log (σ₀ / σ₂) / c₂ := div_nonneg hlog hc.le
    linarith
  have hσ₀ : 0 < σ₀ := lt_of_lt_of_le hσ₂ hσ
  have hbexp : Real.exp (-(c₂ * (a₀ + Real.log (σ₀ / σ₂) / c₂ - a₀))) = σ₂ / σ₀ := by
    have : c₂ * (a₀ + Real.log (σ₀ / σ₂) / c₂ - a₀) = Real.log (σ₀ / σ₂) := by
      field_simp; ring
    rw [this, Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hmem : ∀ t ∈ Ico a₀ (a₀ + Real.log (σ₀ / σ₂) / c₂),
      σ₀ * Real.exp (-(c₂ * (t - a₀))) ∈ Ioc σ₂ σ₀ := by
    intro t ht
    have hle : Real.exp (-(c₂ * (t - a₀))) ≤ 1 := Real.exp_le_one_iff.2 (by
      have : 0 ≤ c₂ * (t - a₀) := mul_nonneg hc.le (by linarith [ht.1]); linarith)
    refine ⟨?_, by nlinarith [Real.exp_pos (-(c₂ * (t - a₀)))]⟩
    have hlt : Real.exp (-(c₂ * (a₀ + Real.log (σ₀ / σ₂) / c₂ - a₀))) <
        Real.exp (-(c₂ * (t - a₀))) := Real.exp_lt_exp.2 (by
      have := mul_lt_mul_of_pos_left ht.2 hc
      nlinarith [mul_lt_mul_of_pos_left (sub_lt_sub_right ht.2 a₀) hc])
    rw [hbexp] at hlt
    calc σ₂ = σ₀ * (σ₂ / σ₀) := by field_simp
      _ < σ₀ * Real.exp (-(c₂ * (t - a₀))) := mul_lt_mul_of_pos_left hlt hσ₀
  refine comparison_of_rate hβ hn hX I a₀ (a₀ + Real.log (σ₀ / σ₂) / c₂) hb
    (fun t => σ₀ * Real.exp (-(c₂ * (t - a₀))))
    (fun t => -(c₂ * (σ₀ * Real.exp (-(c₂ * (t - a₀)))))) ?_ ?_ ?_ ?_ hfar ?_
  · intro t
    have h := ((((hasDerivAt_id t).sub_const a₀).const_mul c₂).fun_neg.exp).const_mul σ₀
    simp only [id] at h
    convert h using 1
    ring
  · intro t ht
    have hle : Real.exp (-(c₂ * (t - a₀))) ≤ 1 := Real.exp_le_one_iff.2 (by
      have : 0 ≤ c₂ * (t - a₀) := mul_nonneg hc.le (by linarith [ht.1]); linarith)
    show σ₀ * Real.exp (-(c₂ * (t - a₀))) ≤ 1
    nlinarith [Real.exp_pos (-(c₂ * (t - a₀)))]
  · intro t ht
    simpa using hcF _ (hmem t ht)
  · intro t ht
    exact hthr _ (hmem t ht)
  · intro i hi j hj
    simpa using hinit i hi j hj

/-- **Phase 3: propagation of smallness.**  If `F_β(δ) > 2 n² E`, the profile `s ≡ δ` is a
subsolution: if `⟨x_i, x_j⟩ ≥ 1 - δ` at `a₀` for `i, j ∈ I`, it stays so on `[a₀, b]` as long
as `I` is `α`-separated from the other tokens. -/
theorem phase_const {β α : ℝ} (hβ : 0 ≤ β) (hn : 1 ≤ n) {X : ℝ → SphereTuple d n}
    (hX : IsAttnFlow d n β X) (I : Finset (Idx n)) (a₀ b δ : ℝ) (hab : a₀ ≤ b) (hδ1 : δ ≤ 1)
    (hthr : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β δ)
    (hfar : ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ k, k ∉ I →
      ⟪(X t i : EucSpace d), (X t k : EucSpace d)⟫_ℝ ≤ α)
    (hinit : ∀ i ∈ I, ∀ j ∈ I,
      1 - δ ≤ ⟪(X a₀ i : EucSpace d), (X a₀ j : EucSpace d)⟫_ℝ) :
    ∀ t ∈ Icc a₀ b, ∀ i ∈ I, ∀ j ∈ I,
      1 - δ ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hF0 : 0 ≤ collapseRate β δ := by
    have : 0 ≤ 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) := by positivity
    linarith
  exact comparison_of_rate hβ hn hX I a₀ b hab (fun _ => δ) (fun _ => 0)
    (fun t => hasDerivAt_const t δ) (fun _ _ => hδ1)
    (fun t _ => by simpa using div_nonneg hF0 hn'.le) (fun _ _ => hthr) hfar hinit

end Metastability
end Transformer
