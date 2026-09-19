/-
# Appendix B — the second-order expansion of `𝖤_β` at a critical point

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The computation behind `e:Hessianincoord`.  Along a curve `t ↦ P t` on
`(𝕊^{d-1})^n` through a critical point `x` with velocity `v`,

  `Σ_{i,j} e^{β ⟨P_i, P_j⟩} = Σ_{i,j} e^{β ⟨x_i, x_j⟩}
      + t² [ β (Σ_{i,j} K_{ij} ⟨v_i, v_j⟩ - Σ_i λ_i ‖v_i‖²)
             + Σ_{i,j} K_{ij} β² (⟨x_i, v_j⟩ + ⟨v_i, x_j⟩)² / 2 ] + o(t²)`,

where `K_{ij} = e^{β ⟨x_i, x_j⟩}` and `λ_i` is the Lagrange multiplier
`⟨x_i, Σ_j K_{ij} x_j⟩`.  The point of the statement is the shape of the
coefficient: it is written in `v` alone, not in the curve.

Two facts of the sphere make the first-order term disappear.  Writing
`w_i(t) = P_i(t) - x_i`, the constraint `‖P_i‖ = 1` gives
`⟨x_i, w_i⟩ = -‖w_i‖²/2` — a *second*-order quantity — and criticality gives
`Σ_j K_{ij} x_j = λ_i x_i`, which is what turns the two cross terms into
`-Σ_i λ_i ‖w_i‖²`.

Everything is stated for abstract `x, v, P, K, λ, m` so that no unfolding of
`Perspective.selfEnergy` happens inside the proof; the instantiation is
`Perspective.isLittleO_selfEnergy_secondOrder`, in
`Perspective.AppendixB_Intrinsic`.
-/

import Transformer.Perspective.AppendixB_EBeta
import Transformer.Perspective.InnerAsymptotics

open scoped BigOperators
open Real Asymptotics Filter

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **The second-order expansion of `Σ_{i,j} e^{β ⟨P_i, P_j⟩}` at a critical
point**, with a Peano remainder.

`x` is the base configuration, `v` the velocity, `P` the curve; `K`, `lam` and
`m` are named so that the quadratic coefficient can be read off:
`K i j = e^{β⟨x_i,x_j⟩}`, `lam i = ⟨x_i, Σ_j K_{ij} x_j⟩`,
`m i j = ⟨x_i, v_j⟩ + ⟨v_i, x_j⟩`.

Source: arXiv:2312.10794v5, Appendix B, `e:Hessianincoord`. -/
theorem selfEnergy_expansion_aux (β : ℝ)
    (x v : Idx n → EucSpace d) (P : ℝ → Idx n → EucSpace d)
    (K : Idx n → Idx n → ℝ) (lam : Idx n → ℝ) (m : Idx n → Idx n → ℝ)
    (hK : ∀ i j : Idx n, K i j = Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)))
    (hlam : ∀ i : Idx n, lam i = inner (𝕜 := ℝ) (x i) (∑ j : Idx n, K i j • x j))
    (hm : ∀ i j : Idx n,
      m i j = inner (𝕜 := ℝ) (x i) (v j) + inner (𝕜 := ℝ) (v i) (x j))
    (hxnorm : ∀ i : Idx n, ‖x i‖ = 1) (hPnorm : ∀ (t : ℝ) (i : Idx n), ‖P t i‖ = 1)
    (hP0 : ∀ i : Idx n, P 0 i = x i)
    (hderiv : ∀ i : Idx n, HasDerivAt (fun s : ℝ => P s i) (v i) 0)
    (hcrit : ∀ i : Idx n, proj d (x i) (∑ j : Idx n, K i j • x j) = 0) :
    (fun t : ℝ => (∑ i : Idx n, ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (P t i) (P t j)))
        - (∑ i : Idx n, ∑ j : Idx n, K i j)
        - (β * ((∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (v i) (v j))
                  - ∑ i : Idx n, lam i * ‖v i‖ ^ 2)
            + ∑ i : Idx n, ∑ j : Idx n, K i j * (β ^ 2 * m i j ^ 2 / 2)) * t ^ 2)
      =o[nhds 0] fun t : ℝ => t ^ 2 := by
  classical
  set w : ℝ → Idx n → EucSpace d := fun t i => P t i - x i with hwdef
  set q : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    inner (𝕜 := ℝ) (x i) (w t j) + inner (𝕜 := ℝ) (w t i) (x j)
      + inner (𝕜 := ℝ) (w t i) (w t j) with hqdef
  set ρ : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    Real.exp (β * inner (𝕜 := ℝ) (P t i) (P t j)) - K i j - K i j * (β * q i j t)
      - K i j * (β ^ 2 * q i j t ^ 2 / 2) with hρdef
  -- the sphere constraint: `⟨x_i, w_i⟩` is second order
  have hPw : ∀ (t : ℝ) (i : Idx n), P t i = x i + w t i := by
    intro t i; simp [hwdef]
  have hxw : ∀ (t : ℝ) (i : Idx n),
      inner (𝕜 := ℝ) (x i) (w t i) = -(‖w t i‖ ^ 2) / 2 := by
    intro t i
    have h2 : ‖x i + w t i‖ ^ 2 = 1 := by rw [← hPw, hPnorm]; norm_num
    rw [norm_add_sq_real, hxnorm] at h2
    linarith
  have hKsymm : ∀ i j : Idx n, K i j = K j i := by
    intro i j; rw [hK, hK, real_inner_comm (x i) (x j)]
  have hKpos : ∀ i j : Idx n, 0 < K i j := by
    intro i j; rw [hK]; exact Real.exp_pos _
  -- criticality: the interaction field is a multiple of the token
  have hcrit' : ∀ i : Idx n, (∑ j : Idx n, K i j • x j) = lam i • x i := by
    intro i
    have h := hcrit i
    rw [proj, sub_eq_zero] at h
    rw [hlam i]; exact h
  -- the displacement is `t v + o(t)`
  have hwo : ∀ i : Idx n, (fun t : ℝ => w t i - t • v i) =o[nhds 0] fun t : ℝ => t := by
    intro i
    refine (hasDerivAt_iff_isLittleO.mp (hderiv i)).congr' ?_ ?_
    · filter_upwards with t; simp [hwdef, hP0 i]
    · filter_upwards with t; simp
  have hwO : ∀ i : Idx n, (fun t : ℝ => w t i) =O[nhds 0] fun t : ℝ => t := by
    intro i
    refine (((hwo i).isBigO).add (isBigO_smul_const d (v i))).congr' ?_ (by rfl)
    filter_upwards with t; abel
  -- the quadratic form in the displacement
  have hww : ∀ i j : Idx n,
      (fun t : ℝ => inner (𝕜 := ℝ) (w t i) (w t j) - t ^ 2 * inner (𝕜 := ℝ) (v i) (v j))
        =o[nhds 0] fun t : ℝ => t ^ 2 := by
    intro i j
    have h1 : (fun t : ℝ => inner (𝕜 := ℝ) (w t i - t • v i) (w t j))
        =o[nhds 0] fun t : ℝ => t * t := isLittleO_inner d (hwo i) (hwO j)
    have h2 : (fun t : ℝ => inner (𝕜 := ℝ) (t • v i) (w t j - t • v j))
        =o[nhds 0] fun t : ℝ => t * t := by
      refine (isLittleO_inner d (hwo j) (isBigO_smul_const d (v i))).congr' ?_ (by rfl)
      filter_upwards with t; exact real_inner_comm _ _
    refine (h1.add h2).congr' ?_ ?_
    · filter_upwards with t
      simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right]
      ring
    · filter_upwards with t; ring
  have hnormsq : ∀ i : Idx n,
      (fun t : ℝ => ‖w t i‖ ^ 2 - t ^ 2 * ‖v i‖ ^ 2) =o[nhds 0] fun t : ℝ => t ^ 2 := by
    intro i
    refine (hww i i).congr' ?_ (by rfl)
    filter_upwards with t
    rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq]
  have hA3lo : (fun t : ℝ => (∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (w t i) (w t j))
      - t ^ 2 * ∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (v i) (v j))
        =o[nhds 0] fun t : ℝ => t ^ 2 := by
    have hpair : ∀ i j : Idx n,
        (fun t : ℝ => K i j * inner (𝕜 := ℝ) (w t i) (w t j)
            - t ^ 2 * (K i j * inner (𝕜 := ℝ) (v i) (v j))) =o[nhds 0] fun t : ℝ => t ^ 2 := by
      intro i j
      refine ((hww i j).const_mul_left (K i j)).congr' ?_ (by rfl)
      filter_upwards with t; ring
    refine (Asymptotics.IsLittleO.fun_sum (fun i (_ : i ∈ Finset.univ) =>
      Asymptotics.IsLittleO.fun_sum (fun j (_ : j ∈ Finset.univ) =>
        hpair i j))).congr' ?_ (by rfl)
    filter_upwards with t
    simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hLlo : (fun t : ℝ => (∑ i : Idx n, lam i * ‖w t i‖ ^ 2)
      - t ^ 2 * ∑ i : Idx n, lam i * ‖v i‖ ^ 2) =o[nhds 0] fun t : ℝ => t ^ 2 := by
    have hpair : ∀ i : Idx n,
        (fun t : ℝ => lam i * ‖w t i‖ ^ 2 - t ^ 2 * (lam i * ‖v i‖ ^ 2))
          =o[nhds 0] fun t : ℝ => t ^ 2 := by
      intro i
      refine ((hnormsq i).const_mul_left (lam i)).congr' ?_ (by rfl)
      filter_upwards with t; ring
    refine (Asymptotics.IsLittleO.fun_sum
      (fun i (_ : i ∈ Finset.univ) => hpair i)).congr' ?_ (by rfl)
    filter_upwards with t
    simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  -- the linear term collapses onto the constraint
  have hA2 : ∀ (t : ℝ) (i : Idx n),
      (∑ j : Idx n, K i j * inner (𝕜 := ℝ) (w t i) (x j))
        = lam i * (-(‖w t i‖ ^ 2) / 2) := by
    intro t i
    have h1 : (∑ j : Idx n, K i j * inner (𝕜 := ℝ) (w t i) (x j))
        = inner (𝕜 := ℝ) (w t i) (∑ j : Idx n, K i j • x j) := by
      rw [inner_sum]
      exact Finset.sum_congr rfl fun j _ => (real_inner_smul_right _ _ _).symm
    rw [h1, hcrit' i, real_inner_smul_right, ← real_inner_comm (w t i) (x i), hxw]
  have hA1 : ∀ t : ℝ, (∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (x i) (w t j))
      = ∑ i : Idx n, lam i * (-(‖w t i‖ ^ 2) / 2) := by
    intro t
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    have h1 : (∑ i : Idx n, K i j * inner (𝕜 := ℝ) (x i) (w t j))
        = inner (𝕜 := ℝ) (∑ i : Idx n, K j i • x i) (w t j) := by
      rw [sum_inner]
      exact Finset.sum_congr rfl fun i _ => by rw [real_inner_smul_left, hKsymm j i]
    rw [h1, hcrit' j, real_inner_smul_left, hxw]
  have hT1 : ∀ t : ℝ, (∑ i : Idx n, ∑ j : Idx n, K i j * (β * q i j t))
      = β * ((∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (w t i) (w t j))
              - ∑ i : Idx n, lam i * ‖w t i‖ ^ 2) := by
    intro t
    have hexp : ∀ i j : Idx n, K i j * (β * q i j t)
        = β * (K i j * inner (𝕜 := ℝ) (x i) (w t j))
          + β * (K i j * inner (𝕜 := ℝ) (w t i) (x j))
          + β * (K i j * inner (𝕜 := ℝ) (w t i) (w t j)) := by
      intro i j; simp only [hqdef]; ring
    have hA2' : (∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (w t i) (x j))
        = ∑ i : Idx n, lam i * (-(‖w t i‖ ^ 2) / 2) :=
      Finset.sum_congr rfl fun i _ => hA2 t i
    have hcomb : (∑ i : Idx n, lam i * (-(‖w t i‖ ^ 2) / 2))
        + ∑ i : Idx n, lam i * (-(‖w t i‖ ^ 2) / 2)
        = -∑ i : Idx n, lam i * ‖w t i‖ ^ 2 := by
      rw [← Finset.sum_add_distrib, ← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    simp only [hexp, Finset.sum_add_distrib, ← Finset.mul_sum]
    rw [hA1 t, hA2']
    linear_combination β * hcomb
  have hP2 : (fun t : ℝ => (∑ i : Idx n, ∑ j : Idx n, K i j * (β * q i j t))
      - t ^ 2 * (β * ((∑ i : Idx n, ∑ j : Idx n, K i j * inner (𝕜 := ℝ) (v i) (v j))
          - ∑ i : Idx n, lam i * ‖v i‖ ^ 2))) =o[nhds 0] fun t : ℝ => t ^ 2 := by
    refine ((hA3lo.sub hLlo).const_mul_left β).congr' ?_ (by rfl)
    filter_upwards with t
    rw [hT1 t]; ring
  -- the quadratic term
  have hqO : ∀ i j : Idx n, (fun t : ℝ => q i j t) =O[nhds 0] fun t : ℝ => t := by
    intro i j
    have h3 : (fun t : ℝ => inner (𝕜 := ℝ) (w t i) (w t j)) =O[nhds 0] fun t : ℝ => t := by
      refine (isBigO_inner d (hwO i) (hwO j)).trans ?_
      refine isLittleO_sq_id.isBigO.congr' ?_ (by rfl)
      filter_upwards with t; ring
    simp only [hqdef]
    exact ((isBigO_inner_const_right d (x i) (hwO j)).add
      (isBigO_inner_const_left d (x j) (hwO i))).add h3
  have hqm : ∀ i j : Idx n, (fun t : ℝ => q i j t - t * m i j) =o[nhds 0] fun t : ℝ => t := by
    intro i j
    have h3 : (fun t : ℝ => inner (𝕜 := ℝ) (w t i) (w t j)) =o[nhds 0] fun t : ℝ => t := by
      refine (isBigO_inner d (hwO i) (hwO j)).trans_isLittleO ?_
      refine isLittleO_sq_id.congr' ?_ (by rfl)
      filter_upwards with t; ring
    refine (((isLittleO_inner_const_right d (x i) (hwo j)).add
      (isLittleO_inner_const_left d (x j) (hwo i))).add h3).congr' ?_ (by rfl)
    filter_upwards with t
    simp only [hqdef, hm, inner_sub_right, inner_sub_left, real_inner_smul_right,
      real_inner_smul_left]
    ring
  have hqsq : ∀ i j : Idx n,
      (fun t : ℝ => q i j t ^ 2 - t ^ 2 * m i j ^ 2) =o[nhds 0] fun t : ℝ => t ^ 2 := by
    intro i j
    have h2 : (fun t : ℝ => q i j t + t * m i j) =O[nhds 0] fun t : ℝ => t :=
      (hqO i j).add (isBigO_mul_const (m i j))
    refine ((hqm i j).mul_isBigO h2).congr' ?_ ?_
    · filter_upwards with t; ring
    · filter_upwards with t; ring
  have hP3 : (fun t : ℝ => (∑ i : Idx n, ∑ j : Idx n, K i j * (β ^ 2 * q i j t ^ 2 / 2))
      - t ^ 2 * ∑ i : Idx n, ∑ j : Idx n, K i j * (β ^ 2 * m i j ^ 2 / 2))
        =o[nhds 0] fun t : ℝ => t ^ 2 := by
    have hpair : ∀ i j : Idx n,
        (fun t : ℝ => K i j * (β ^ 2 * q i j t ^ 2 / 2)
            - t ^ 2 * (K i j * (β ^ 2 * m i j ^ 2 / 2))) =o[nhds 0] fun t : ℝ => t ^ 2 := by
      intro i j
      refine ((hqsq i j).const_mul_left (K i j * (β ^ 2 / 2))).congr' ?_ (by rfl)
      filter_upwards with t; ring
    refine (Asymptotics.IsLittleO.fun_sum (fun i (_ : i ∈ Finset.univ) =>
      Asymptotics.IsLittleO.fun_sum (fun j (_ : j ∈ Finset.univ) =>
        hpair i j))).congr' ?_ (by rfl)
    filter_upwards with t
    simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  -- the remainder of the exponential
  have hρeq : ∀ (i j : Idx n) (t : ℝ), ρ i j t
      = K i j * (Real.exp (β * q i j t) - 1 - β * q i j t - (β * q i j t) ^ 2 / 2) := by
    intro i j t
    have hpq : inner (𝕜 := ℝ) (P t i) (P t j) = inner (𝕜 := ℝ) (x i) (x j) + q i j t := by
      rw [hPw t i, hPw t j]
      simp only [hqdef, inner_add_left, inner_add_right]
      ring
    have hexp : Real.exp (β * inner (𝕜 := ℝ) (P t i) (P t j))
        = K i j * Real.exp (β * q i j t) := by
      rw [hpq, mul_add, Real.exp_add, hK]
    simp only [hρdef, hexp]
    ring
  have hρbound : ∀ (i j : Idx n) (t : ℝ), |β * q i j t| ≤ 1 →
      |ρ i j t| ≤ K i j * (2 / 9) * |β * q i j t| ^ 3 := by
    intro i j t ha
    have hb := Real.exp_bound ha (n := 3) (by norm_num)
    have hsum : ∑ mm ∈ Finset.range 3, (β * q i j t) ^ mm / (Nat.factorial mm)
        = 1 + β * q i j t + (β * q i j t) ^ 2 / 2 := by
      simp [Finset.sum_range_succ, Nat.factorial]
    rw [hsum] at hb
    have hb' : |Real.exp (β * q i j t) - 1 - β * q i j t - (β * q i j t) ^ 2 / 2|
        ≤ |β * q i j t| ^ 3 * (2 / 9) := by
      have he : Real.exp (β * q i j t) - (1 + β * q i j t + (β * q i j t) ^ 2 / 2)
          = Real.exp (β * q i j t) - 1 - β * q i j t - (β * q i j t) ^ 2 / 2 := by ring
      rw [he] at hb
      refine hb.trans (le_of_eq ?_)
      norm_num [Nat.factorial]
    rw [hρeq, abs_mul, abs_of_pos (hKpos i j)]
    calc K i j * |Real.exp (β * q i j t) - 1 - β * q i j t - (β * q i j t) ^ 2 / 2|
        ≤ K i j * (|β * q i j t| ^ 3 * (2 / 9)) :=
          mul_le_mul_of_nonneg_left hb' (hKpos i j).le
      _ = K i j * (2 / 9) * |β * q i j t| ^ 3 := by ring
  have hρlo : ∀ i j : Idx n, (fun t : ℝ => ρ i j t) =o[nhds 0] fun t : ℝ => t ^ 2 := by
    intro i j
    have htend : Filter.Tendsto (fun t : ℝ => β * q i j t) (nhds 0) (nhds 0) := by
      have h0 : Filter.Tendsto (fun t : ℝ => q i j t) (nhds 0) (nhds 0) :=
        (hqO i j).trans_tendsto tendsto_id
      simpa using h0.const_mul β
    have hev : ∀ᶠ t in nhds 0, |β * q i j t| ≤ 1 := by
      filter_upwards [htend (Metric.closedBall_mem_nhds (0 : ℝ) one_pos)] with t ht
      simpa [Real.dist_eq] using ht
    have hbig : (fun t : ℝ => ρ i j t) =O[nhds 0] fun t : ℝ => t ^ 3 := by
      refine Asymptotics.IsBigO.trans ?_ ((hqO i j).pow 3)
      refine Asymptotics.isBigO_iff.mpr ⟨K i j * (2 / 9) * |β| ^ 3, ?_⟩
      filter_upwards [hev] with t ht
      have h1 := hρbound i j t ht
      rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_pow]
      calc |ρ i j t| ≤ K i j * (2 / 9) * |β * q i j t| ^ 3 := h1
        _ = K i j * (2 / 9) * |β| ^ 3 * |q i j t| ^ 3 := by
            rw [abs_mul, mul_pow]; ring
    exact hbig.trans_isLittleO isLittleO_cube_sq
  have hRsum : (fun t : ℝ => ∑ i : Idx n, ∑ j : Idx n, ρ i j t)
      =o[nhds 0] fun t : ℝ => t ^ 2 :=
    Asymptotics.IsLittleO.fun_sum (fun i (_ : i ∈ Finset.univ) =>
      Asymptotics.IsLittleO.fun_sum (fun j (_ : j ∈ Finset.univ) => hρlo i j))
  -- assembling the three pieces
  have hsplit : ∀ t : ℝ,
      (∑ i : Idx n, ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (P t i) (P t j)))
        - (∑ i : Idx n, ∑ j : Idx n, K i j)
      = (∑ i : Idx n, ∑ j : Idx n, ρ i j t)
        + (∑ i : Idx n, ∑ j : Idx n, K i j * (β * q i j t))
        + ∑ i : Idx n, ∑ j : Idx n, K i j * (β ^ 2 * q i j t ^ 2 / 2) := by
    intro t
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by simp only [hρdef]; ring
  refine ((hRsum.add hP2).add hP3).congr' ?_ (by rfl)
  filter_upwards with t
  linear_combination -(hsplit t)

/-- The hypotheses of `selfEnergy_expansion_aux` are satisfiable: one token
standing still on `𝕊^0`, with velocity `0`. -/
example :
    (∀ i : Idx 1, ‖((singleToken i : SSphere 1) : EucSpace 1)‖ = 1) ∧
      (∀ i : Idx 1,
        HasDerivAt (fun _ : ℝ => ((singleToken i : SSphere 1) : EucSpace 1))
          ((fun _ : Idx 1 => (0 : EucSpace 1)) i) 0) ∧
      IsCriticalEBeta 1 1 1 singleToken :=
  ⟨fun i => mem_sphere_zero_iff_norm.mp (singleToken i).2,
    fun _ => hasDerivAt_const _ _,
    singleToken_isSkew_critical_hessianNonPos.2.1⟩

end Perspective
end Transformer
