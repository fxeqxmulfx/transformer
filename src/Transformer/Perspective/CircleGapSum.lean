/-
# `φ` at the positions, and the sum over the gaps

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3, proof of `lemma:taumax`.

For a configuration `x ∈ 𝕋^n` with positions `θ_k`, gaps `Δ_k` and
multiplicities `N_k`, let `χ = Σ_j f(· - x_j)` and `φ = χ' = Σ_j f'(· - x_j)`.

* Stationarity (C1) puts `χ` at `0` at every position
  (`sum_comp_sub_circleSite_eq_zero`).
* Cut-stability (C2) for the particles at one position gives
  `φ(θ_k) ≥ N_k f'(0)` (`circleMult_mul_le_sum_comp_sub_circleSite`).
* With the gap inequality (`add_le_mul_integral_max_zero`), each gap gives
  `4 (N_k + N_{k+1}) f'(0) ≤ Δ_k ∫_{θ_k}^{θ_{k+1}} |φ''|₊`
  (`circleMult_add_mul_le_integral`).
* The gaps go once around the circle, and `|φ''|₊ ≤ Σ_j |f'''(· - x_j)|₊`,
  so the integrals over the gaps add up to at most `n ∫_{-π}^{π} |f'''|₊`
  (`sum_integral_circleSite_le`).
-/

import Transformer.Perspective.CircleMult
import Transformer.Perspective.CircleGapBounds

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- A `2π`-periodic `g` does not tell two angles at one point of `𝕋` apart. -/
theorem periodic_sub_eq_of_circlePos_eq {g : ℝ → ℝ} (hg : Function.Periodic g (2 * π))
    {a b : ℝ} (hab : circlePos a = circlePos b) (c : ℝ) : g (a - c) = g (b - c) := by
  rw [← periodic_circlePos_sub hg a c, hab, periodic_circlePos_sub hg b c]

/-- Every position is occupied. -/
theorem exists_circlePos_eq_circlePos_circleSite (x : Angles n)
    (h : 0 < (circleSites x).card) (k : ℕ) : ∃ i, circlePos (x i) = circlePos (circleSite x k) := by
  obtain ⟨i, hi⟩ := exists_circlePos_eq x (Nat.mod_lt k h)
  exact ⟨i, by rw [hi, circlePos_circleSite x h]⟩

/-- **Stationarity (C1) puts `χ = Σ_j f(· - x_j)` at `0` at every position.**

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`: `χ(θ_i) = 0`. -/
theorem sum_comp_sub_circleSite_eq_zero {f : ℝ → ℝ} (hf : Function.Periodic f (2 * π))
    {x : Angles n} (hC1 : ∀ i, ∑ j, f (x i - x j) = 0) (h : 0 < (circleSites x).card)
    (k : ℕ) : ∑ j, f (circleSite x k - x j) = 0 := by
  obtain ⟨i, hi⟩ := exists_circlePos_eq_circlePos_circleSite x h k
  rw [← hC1 i]
  exact Finset.sum_congr rfl fun j _ => periodic_sub_eq_of_circlePos_eq hf hi.symm (x j)

/-- **Equation (eq:varphival_general).**  Cut-stability (C2) for the particles
at one position gives `φ(θ_k) = Σ_j g(θ_k - x_j) ≥ N_k g(0)`, where `g` stands
for `f'`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`. -/
theorem circleMult_mul_le_sum_comp_sub_circleSite {g : ℝ → ℝ}
    (hg : Function.Periodic g (2 * π)) {x : Angles n}
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, g (x i - x j))
    (h : 0 < (circleSites x).card) (k : ℕ) :
    (circleMult x k : ℝ) * g 0 ≤ ∑ j, g (circleSite x k - x j) := by
  set S := Finset.univ.filter fun j => circlePos (x j) = circlePos (circleSite x k)
  obtain ⟨i₀, hi₀⟩ := exists_circlePos_eq_circlePos_circleSite x h k
  have hcard : (0 : ℝ) < S.card :=
    Nat.cast_pos.2 (Finset.card_pos.2 ⟨i₀, Finset.mem_filter.2 ⟨Finset.mem_univ _, hi₀⟩⟩)
  have hin : ∑ i ∈ S, ∑ j ∈ Sᶜ, g (x i - x j) = ∑ i ∈ S, ∑ j ∈ Sᶜ, g (circleSite x k - x j) :=
    Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j _ =>
      periodic_sub_eq_of_circlePos_eq hg (Finset.mem_filter.1 hi).2 (x j)
  have hout := hC2 S
  rw [hin, Finset.sum_const, nsmul_eq_mul] at hout
  have hself : ∑ j ∈ S, g (circleSite x k - x j) = S.card * g 0 := by
    rw [Finset.sum_congr rfl fun j hj => by
      rw [← periodic_circlePos_sub hg, (Finset.mem_filter.1 hj).2, sub_self],
      Finset.sum_const, nsmul_eq_mul]
  rw [← Finset.sum_add_sum_compl S, hself]
  exact le_add_of_nonneg_right (nonneg_of_mul_nonneg_right hout hcard)

/-- **One gap of `lemma:taumax`.**  For a stationary (C1), cut-stable (C2)
configuration and `f` with `f' = f₁`, `f'' = f₂`, `f''' = f₃` continuous,

  `4 (N_k + N_{k+1}) f'(0) ≤ Δ_k ∫_{θ_k}^{θ_{k+1}} |φ''|₊`,   `φ'' = Σ_j f'''(· - x_j)`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`, the bound before
(eq:prodineq_general) on one gap, with `a_i = 1`. -/
theorem circleMult_add_mul_le_integral {f f₁ f₂ f₃ : ℝ → ℝ}
    (hf : Function.Periodic f (2 * π)) (h₁ : ∀ y, HasDerivAt f (f₁ y) y)
    (h₂ : ∀ y, HasDerivAt f₁ (f₂ y) y) (h₃ : ∀ y, HasDerivAt f₂ (f₃ y) y)
    (hc₃ : Continuous f₃) {x : Angles n} (hC1 : ∀ i, ∑ j, f (x i - x j) = 0)
    (hC2 : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, f₁ (x i - x j))
    (h : 0 < (circleSites x).card) (k : ℕ) :
    4 * (((circleMult x k : ℝ) + circleMult x (k + 1)) * f₁ 0) ≤
      circleGap x k * ∫ y in circleSite x k..circleSite x (k + 1), max (∑ j, f₃ (y - x j)) 0 := by
  have hp₁ := periodic_of_hasDerivAt hf h₁
  have hgap := add_le_mul_integral_max_zero (strictMono_circleSite x h (Nat.lt_succ_self k))
    (hasDerivAt_sum_comp_sub Finset.univ x h₁) (hasDerivAt_sum_comp_sub Finset.univ x h₂)
    (hasDerivAt_sum_comp_sub Finset.univ x h₃)
    (continuous_finsetSum _ fun j _ => hc₃.comp (continuous_id.sub continuous_const))
    ((sum_comp_sub_circleSite_eq_zero hf hC1 h k).trans
      (sum_comp_sub_circleSite_eq_zero hf hC1 h (k + 1)).symm)
  have hk := circleMult_mul_le_sum_comp_sub_circleSite hp₁ hC2 h k
  have hk1 := circleMult_mul_le_sum_comp_sub_circleSite hp₁ hC2 h (k + 1)
  simp only [circleGap]
  linarith

/-- **The gaps go once around the circle.**  The integrals of
`|φ''|₊ = |Σ_j f'''(· - x_j)|₊` over the gaps add up to at most
`n ∫_{-π}^{π} |f'''|₊`.

Source: arXiv:2507.22857v1, §3, proof of `lemma:taumax`, from
`Σ_j |f'''(x - x_j)|₊ ≥ |φ''|₊` to `n ⟨1, |f'''|₊⟩`. -/
theorem sum_integral_circleSite_le {f₃ : ℝ → ℝ} (hp₃ : Function.Periodic f₃ (2 * π))
    (hc₃ : Continuous f₃) (x : Angles n) (h : 0 < (circleSites x).card) :
    ∑ k ∈ Finset.range (circleSites x).card,
        ∫ y in circleSite x k..circleSite x (k + 1), max (∑ j, f₃ (y - x j)) 0 ≤
      n * ∫ u in -π..π, max (f₃ u) 0 := by
  have hc : ∀ j : Idx n, Continuous fun y => max (f₃ (y - x j)) 0 := fun j =>
    (hc₃.comp (continuous_id.sub continuous_const)).max continuous_const
  have hF : Continuous fun y => max (∑ j, f₃ (y - x j)) 0 :=
    (continuous_finsetSum _ fun j _ => hc₃.comp (continuous_id.sub continuous_const)).max
      continuous_const
  have hp : Function.Periodic (fun u => max (f₃ u) 0) (2 * π) := fun u => by
    simp only [hp₃ u]
  rw [intervalIntegral.sum_integral_adjacent_intervals (a := circleSite x)
      fun k _ => hF.intervalIntegrable _ _,
    show circleSite x (circleSites x).card = circleSite x 0 + 2 * π by
      simpa using circleSite_add_card x h 0]
  calc ∫ y in circleSite x 0..circleSite x 0 + 2 * π, max (∑ j, f₃ (y - x j)) 0
      ≤ ∫ y in circleSite x 0..circleSite x 0 + 2 * π, ∑ j, max (f₃ (y - x j)) 0 :=
        intervalIntegral.integral_mono_on (by linarith [pi_pos]) (hF.intervalIntegrable _ _)
          ((continuous_finsetSum _ fun j _ => hc j).intervalIntegrable _ _)
          fun y _ => max_sum_le_sum_max _ _
    _ = ∑ _j : Idx n, ∫ u in -π..π, max (f₃ u) 0 := by
        rw [intervalIntegral.integral_finsetSum fun j _ => (hc j).intervalIntegrable _ _]
        exact Finset.sum_congr rfl fun j _ => integral_comp_sub_eq_of_periodic hp _ _
    _ = n * ∫ u in -π..π, max (f₃ u) 0 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- The hypotheses above hold together for two antipodal particles,
`x = (0, π)`, and the kernel `f = sin 2·`, with `f' = 2 cos 2·`,
`f'' = -4 sin 2·` and `f''' = -8 cos 2·`: every difference is a multiple of
`π`, where `sin 2·` vanishes and `2 cos 2· = 2`; `f'` and `f'''` are
`2π`-periodic with `f`.  The particles fill two
positions, and `0` and `2π` are one point of `𝕋`. -/
example :
    let x : Angles 2 := fun i => (i : ℕ) * π
    Function.Periodic (fun y => sin (2 * y)) (2 * π) ∧
      Function.Periodic (fun y => 2 * cos (2 * y)) (2 * π) ∧
      Function.Periodic (fun y => -(8 * cos (2 * y))) (2 * π) ∧
      (∀ y, HasDerivAt (fun y => sin (2 * y)) (2 * cos (2 * y)) y) ∧
      (∀ y, HasDerivAt (fun y => 2 * cos (2 * y)) (-(4 * sin (2 * y))) y) ∧
      (∀ y, HasDerivAt (fun y => -(4 * sin (2 * y))) (-(8 * cos (2 * y))) y) ∧
      Continuous (fun y => -(8 * cos (2 * y))) ∧ (∀ i, ∑ j, sin (2 * (x i - x j)) = 0) ∧
      (∀ S : Finset (Idx 2), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, 2 * cos (2 * (x i - x j))) ∧
      0 < (circleSites x).card ∧ circlePos 0 = circlePos (2 * π) := by
  intro x
  have hx : ∀ i j, ∃ k : ℤ, x i - x j = k * π := fun i j =>
    ⟨(i : ℕ) - (j : ℕ), by simp only [x]; push_cast; ring⟩
  have h2 : ∀ y : ℝ, HasDerivAt (fun y => 2 * y) 2 y := fun y =>
    ((hasDerivAt_id' y).const_mul 2).congr_deriv (mul_one 2)
  have hcos : ∀ y, cos (2 * (y + 2 * π)) = cos (2 * y) := fun y => by
    rw [show 2 * (y + 2 * π) = 2 * y + 2 * π + 2 * π by ring, cos_add_two_pi, cos_add_two_pi]
  refine ⟨fun y => ?_, fun y => by simp only [hcos], fun y => by simp only [hcos],
    fun y => (h2 y).sin.congr_deriv (by ring),
    fun y => ((h2 y).cos.const_mul 2).congr_deriv (by ring),
    fun y => ((h2 y).sin.const_mul 4).fun_neg.congr_deriv (by ring), by fun_prop,
    fun i => Finset.sum_eq_zero fun j _ => ?_,
    fun S => Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_,
    Finset.card_pos.2 ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ 0)⟩, ?_⟩
  · show sin (2 * (y + 2 * π)) = sin (2 * y)
    rw [show 2 * (y + 2 * π) = 2 * y + 2 * π + 2 * π by ring, sin_add_two_pi, sin_add_two_pi]
  · obtain ⟨k, hk⟩ := hx i j
    rw [hk, show 2 * (k * π) = ((2 * k : ℤ) : ℝ) * π by push_cast; ring]
    exact sin_int_mul_pi _
  · obtain ⟨k, hk⟩ := hx i j
    rw [hk, show 2 * (k * π) = (k : ℝ) * (2 * π) by ring, cos_int_mul_two_pi]
    norm_num
  · show toIcoMod two_pi_pos 0 0 = toIcoMod two_pi_pos 0 (2 * π)
    rw [← toIcoMod_add_right two_pi_pos 0 0, zero_add]

end Perspective
end Transformer
