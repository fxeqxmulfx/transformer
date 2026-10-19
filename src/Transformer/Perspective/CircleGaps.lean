module

public import Transformer.Perspective.CircleSites

/-!
# Gaps of a configuration on the circle

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3.

The gaps `Δ_j = θ_{j+1} - θ_j` between consecutive positions of a
configuration on `𝕋` (`Perspective.CircleSites`), counted from `0`, as
`circleGap x k`. They are positive and add up to `2π`.

`lemma:twoconsecutive_general`, that under cut-stability (C2) at most one gap
exceeds `τ`, is `not_two_circleGap_gt`. Nothing here is specific to the
kernel `f_β`: the function `g` plays the role of `f'`.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- **The gaps** `Δ_k = θ_{k+1} - θ_k` of a configuration; the last one,
`k = K - 1`, is `2π + θ_0 - θ_{K-1}`.

Source: arXiv:2507.22857v1, §3, before `lemma:twoconsecutive_general`. -/
noncomputable def circleGap (x : Angles n) (k : ℕ) : ℝ :=
  circleSite x (k + 1) - circleSite x k

/-- The gaps are positive. -/
theorem circleGap_pos (x : Angles n) (h : 0 < (circleSites x).card) (k : ℕ) :
    0 < circleGap x k :=
  sub_pos.mpr (strictMono_circleSite x h (Nat.lt_succ_self k))

/-- The gaps go once around the circle: `Σ_{k < K} Δ_k = 2π`. -/
theorem sum_circleGap (x : Angles n) (h : 0 < (circleSites x).card) :
    ∑ k ∈ Finset.range (circleSites x).card, circleGap x k = 2 * π := by
  simp only [circleGap]
  rw [Finset.sum_range_sub (circleSite x), ← zero_add (circleSites x).card,
    circleSite_add_card x h]
  ring

/-- **Lemma (lemma:twoconsecutive_general).** *At most one gap exceeds `τ`.*

Let `g` be `2π`-periodic and negative off `[-τ, τ]` on `𝕋`, that is on
`(τ, 2π - τ)`: `g` plays the role of `f'` in `thm:main`. If the configuration
`x` is cut-stable, `Σ_{i ∈ S, j ∉ S} g(x_i - x_j) ≥ 0` for every `S ⊆ [n]`
(C2), then no two distinct gaps exceed `τ`.

The source's proof: the particles on the arc between the two gaps form `S`.
Every difference across the cut is farther than `τ` from `0` on `𝕋`, so every
term of (C2) is negative, and so is their sum.

Source: arXiv:2507.22857v1, §3, `lemma:twoconsecutive_general`. Changed: the
source states (C2) only for sets `S` of coincident particles, yet its proof
applies (C2) to the particles of an arc; here (C2) is assumed for every `S`.
A critical point of `𝖤_β` whose Hessian is non-positive satisfies it
(`taylor3_inequality`; arXiv:2507.22857v1, §4, `remark:negative_semidef`).
The source's `τ ∈ (0, π]` is not needed. -/
theorem not_two_circleGap_gt (g : ℝ → ℝ) (hg : Function.Periodic g (2 * π)) (τ : ℝ)
    (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), g u < 0) (x : Angles n)
    (hcut : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, g (x i - x j)) :
    ¬ ∃ k l, k < (circleSites x).card ∧ l < (circleSites x).card ∧ k ≠ l ∧
      τ < circleGap x k ∧ τ < circleGap x l := by
  rintro ⟨k, l, hk, hl, hkl, hτk, hτl⟩
  wlog hlt : k < l generalizing k l
  · exact this l k hl hk hkl.symm hτl hτk (lt_of_le_of_ne (not_lt.mp hlt) hkl.symm)
  have hK : 0 < (circleSites x).card := lt_of_le_of_lt (Nat.zero_le k) hk
  have hmono := strictMono_circleSite x hK
  have hlast : circleSite x (circleSites x).card = circleSite x 0 + 2 * π := by
    simpa using circleSite_add_card x hK 0
  unfold circleGap at hτk hτl
  -- `S`: the particles on the arc `(θ_k, θ_l]`, that is at `θ_{k+1}, …, θ_l`
  set S := Finset.univ.filter fun i : Idx n =>
    circleSite x k < circlePos (x i) ∧ circlePos (x i) ≤ circleSite x l with hS
  have hpair : ∀ i ∈ S, ∀ j ∈ Sᶜ, g (x i - x j) < 0 := by
    intro i hi j hj
    rw [← periodic_circlePos_sub hg]
    simp only [hS, Finset.mem_compl, Finset.mem_filter, Finset.mem_univ, true_and,
      not_and, not_le] at hi hj
    obtain ⟨a, ha, hai⟩ := exists_circleSite_eq x i
    obtain ⟨b, hb, hbj⟩ := exists_circleSite_eq x j
    rw [← hai] at hi
    have hia : circleSite x (k + 1) ≤ circlePos (x i) :=
      hai ▸ hmono.monotone (Nat.succ_le_of_lt (hmono.lt_iff_lt.mp hi.1))
    have h0k : circleSite x 0 ≤ circleSite x k := hmono.monotone (Nat.zero_le k)
    by_cases hjk : circlePos (x j) ≤ circleSite x k
    · -- `x_j` before the arc: `x_i - x_j ∈ (Δ_k, 2π - Δ_l)`
      have h0j : circleSite x 0 ≤ circlePos (x j) := hbj ▸ hmono.monotone (Nat.zero_le b)
      have hlK : circleSite x (l + 1) ≤ circleSite x (circleSites x).card :=
        hmono.monotone (Nat.succ_le_of_lt hl)
      exact hneg _ ⟨by linarith, by linarith [hi.2]⟩
    · -- `x_j` after the arc: `x_i - x_j + 2π ∈ (Δ_k, 2π - Δ_l)`
      have hlj := hj (lt_of_not_ge hjk)
      rw [← hbj] at hlj
      have hjb : circleSite x (l + 1) ≤ circlePos (x j) :=
        hbj ▸ hmono.monotone (Nat.succ_le_of_lt (hmono.lt_iff_lt.mp hlj))
      have hjK : circlePos (x j) < circleSite x (circleSites x).card := hbj ▸ hmono hb
      rw [← hg]
      exact hneg _ ⟨by linarith, by linarith [hi.2]⟩
  obtain ⟨i₀, hi₀⟩ := exists_circlePos_eq x (show k + 1 < (circleSites x).card by omega)
  obtain ⟨j₀, hj₀⟩ := exists_circlePos_eq x hk
  have hi₀S : i₀ ∈ S := by
    simp only [hS, Finset.mem_filter, Finset.mem_univ, true_and, hi₀]
    exact ⟨hmono (Nat.lt_succ_self k), hmono.monotone (Nat.succ_le_of_lt hlt)⟩
  have hj₀S : j₀ ∈ Sᶜ := by
    simp [hS, hj₀]
  have hsum : ∑ i ∈ S, ∑ j ∈ Sᶜ, g (x i - x j) < 0 :=
    Finset.sum_neg (fun i hi => Finset.sum_neg (hpair i hi) ⟨j₀, hj₀S⟩) ⟨i₀, hi₀S⟩
  exact absurd (hcut S) (not_le.mpr hsum)

/-- The form `lemma:taumax` uses: once one gap exceeds `τ`, every other gap is
at most `τ`. -/
theorem circleGap_le_of_lt_circleGap (g : ℝ → ℝ) (hg : Function.Periodic g (2 * π)) (τ : ℝ)
    (hneg : ∀ u ∈ Set.Ioo τ (2 * π - τ), g u < 0) (x : Angles n)
    (hcut : ∀ S : Finset (Idx n), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, g (x i - x j))
    {k l : ℕ} (hk : k < (circleSites x).card) (hl : l < (circleSites x).card) (hkl : k ≠ l)
    (hτ : τ < circleGap x l) : circleGap x k ≤ τ :=
  le_of_not_gt fun hτk => not_two_circleGap_gt g hg τ hneg x hcut ⟨k, l, hk, hl, hkl, hτk, hτ⟩

/-- The hypotheses of `not_two_circleGap_gt` and `circleGap_le_of_lt_circleGap`
are satisfiable together, by a configuration that is not synchronized:
`g = cos`, which is negative on `(π/2, 3π/2)`, and two particles at `0` and
`π/3`. Every difference lies in `[-π/3, π/3]`, where `cos ≥ 0`, so (C2) holds
for every `S`. The two gaps add up to `2π`, so one of them exceeds `π/2`. -/
example :
    let x : Angles 2 := fun i => ((i : ℕ) : ℝ) * (π / 3)
    Function.Periodic Real.cos (2 * π) ∧
      (∀ u ∈ Set.Ioo (π / 2) (2 * π - π / 2), Real.cos u < 0) ∧
      (∀ S : Finset (Idx 2), 0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ, Real.cos (x i - x j)) ∧
      ∃ k l, k < (circleSites x).card ∧ l < (circleSites x).card ∧ k ≠ l ∧
        π / 2 < circleGap x l := by
  intro x
  have hx : ∀ i, 0 ≤ x i ∧ x i ≤ π / 3 := fun i => by
    have hi : ((i : ℕ) : ℝ) ≤ 1 := by exact_mod_cast Nat.le_of_lt_succ i.2
    have hpi := Real.pi_pos
    show 0 ≤ ((i : ℕ) : ℝ) * (π / 3) ∧ ((i : ℕ) : ℝ) * (π / 3) ≤ π / 3
    exact ⟨by positivity, by nlinarith⟩
  have hpos : ∀ i, circlePos (x i) = x i := fun i =>
    (toIcoMod_eq_self _).mpr ⟨(hx i).1, by linarith [(hx i).2, Real.pi_pos]⟩
  have hcard : (circleSites x).card = 2 := by
    rw [circleSites, Finset.card_image_of_injective, Finset.card_univ, Fintype.card_fin]
    intro i j hij
    simp only [hpos, x] at hij
    exact Fin.ext (by exact_mod_cast mul_right_cancel₀ (by positivity) hij)
  refine ⟨Real.cos_periodic, fun u hu => Real.cos_neg_of_pi_div_two_lt_of_lt hu.1 (by
    linarith [hu.2]), fun S => Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ =>
    Real.cos_nonneg_of_mem_Icc ⟨by linarith [hx i, hx j], by linarith [hx i, hx j]⟩, ?_⟩
  have hsum := sum_circleGap x (by rw [hcard]; norm_num)
  rw [hcard, Finset.sum_range_succ, Finset.sum_range_one] at hsum
  rw [hcard]
  by_cases h1 : π / 2 < circleGap x 1
  · exact ⟨0, 1, by norm_num, by norm_num, by norm_num, h1⟩
  · exact ⟨1, 0, by norm_num, by norm_num, by norm_num, by linarith [Real.pi_pos]⟩

end Perspective
end Transformer
