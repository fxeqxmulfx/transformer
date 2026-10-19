module

public import Transformer.Landscape.SaddleAvoidance.Lojasiewicz
public import Mathlib.Analysis.Asymptotics.Lemmas
public import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Saddle avoidance — sequences that drop by a power of themselves

A bound `u k ≤ C g k` from some index on holds for every index, with a larger
constant, when `g` is positive (`exists_forall_le_mul_of_eventually`). A
nonnegative sequence with `r (k + 1) ≤ q r k` from some index on, `0 < q`, is
therefore at most `C q ^ k` (`exists_le_mul_pow_of_le_mul`). A positive one
with `r (k + 1) ≤ r k - c r k ^ (1 + β)`, `c ≥ 0` and `0 < β ≤ 1`, gains at
least `β c` in `r ^ (-β)` at each step (`rpow_neg_add_le_of_le_sub`, by the
concavity of `s ↦ s ^ β`, `mul_sub_le_rpow_mul_sub`), so for `c > 0` a
nonnegative one is at most `C (k + 1) ^ (-1/β)`
(`exists_le_mul_rpow_of_le_sub`). A sequence bounded from some index on by
`K r k ^ p` is then at most `C g k ^ p` when `r ≤ C₁ g`
(`exists_le_mul_and_le_mul_rpow`).

These are the recursions of the proof of `prop:rates` of arXiv:1602.04915,
there for the length `e_k` of the rest of an orbit of gradient descent, here
for `f x_k - f x*` (`Landscape.SaddleAvoidance.Rates`).
-/

@[expose] public section

open Filter Asymptotics

namespace Transformer.Landscape.SaddleAvoidance

/-- **A bound from some index on holds for every index**, with a larger
constant, when the bounding sequence is positive. -/
theorem exists_forall_le_mul_of_eventually {u g : ℕ → ℝ} (hu : ∀ k, 0 ≤ u k)
    (hg : ∀ k, 0 < g k) {C : ℝ} (h : ∀ᶠ k in atTop, u k ≤ C * g k) :
    ∃ C', ∀ k, u k ≤ C' * g k := by
  have hO : u =O[cofinite] g := by
    rw [Nat.cofinite_eq_atTop]
    refine IsBigO.of_bound C (h.mono fun k hk => ?_)
    rwa [Real.norm_of_nonneg (hu k), Real.norm_of_nonneg (hg k).le]
  obtain ⟨C', hC'⟩ := (isBigO_cofinite_iff fun k hk => absurd hk (hg k).ne').1 hO
  refine ⟨C', fun k => ?_⟩
  have h := hC' k
  rwa [Real.norm_of_nonneg (hu k), Real.norm_of_nonneg (hg k).le] at h

/-- **Geometric decay.**  A nonnegative sequence with `r (k + 1) ≤ q r k`
from some index on, `0 < q`, is at most `C q ^ k` for every `k`. -/
theorem exists_le_mul_pow_of_le_mul {r : ℕ → ℝ} {q : ℝ} (hq : 0 < q) (hr : ∀ k, 0 ≤ r k)
    (h : ∀ᶠ k in atTop, r (k + 1) ≤ q * r k) : ∃ C, ∀ k, r k ≤ C * q ^ k := by
  obtain ⟨K, hK⟩ := eventually_atTop.1 h
  refine exists_forall_le_mul_of_eventually hr (fun k => pow_pos hq k) (C := r K / q ^ K)
    ((eventually_ge_atTop K).mono fun k hk => ?_)
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hk
  have hgeom := le_geom (u := fun j => r (K + j)) hq.le j fun i _ => hK (K + i) le_self_add
  have hqK : q ^ K ≠ 0 := pow_ne_zero K hq.ne'
  calc r (K + j) ≤ q ^ j * r K := by simpa using hgeom
    _ = r K / q ^ K * q ^ (K + j) := by
      rw [pow_add]
      field_simp

/-- **A drop by a power gains a constant in a negative power.**  If
`0 < t ≤ s` and `t ≤ s - c s ^ (1 + β)`, with `c ≥ 0` and `0 < β ≤ 1`, then
`s ^ (-β) + β c ≤ t ^ (-β)`. By the concavity of `s ↦ s ^ β`,
`β (s - t) ≤ s ^ (1 - β) (s ^ β - t ^ β)`. -/
theorem rpow_neg_add_le_of_le_sub {β c s t : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1) (hc : 0 ≤ c)
    (ht : 0 < t) (hts : t ≤ s) (h : t ≤ s - c * s ^ (1 + β)) :
    s ^ (-β) + β * c ≤ t ^ (-β) := by
  have hs : 0 < s := ht.trans_le hts
  have hconc := mul_sub_le_rpow_mul_sub (a := 1 - β) (by linarith) (by linarith) hs.le ht.le
  rw [sub_sub_cancel] at hconc
  have hA : 0 < s ^ β := Real.rpow_pos_of_pos hs β
  have hB : 0 < t ^ β := Real.rpow_pos_of_pos ht β
  have hBA : t ^ β ≤ s ^ β := Real.rpow_le_rpow ht.le hts hβ.le
  have h1 : s ^ (1 - β) * s ^ β = s := by
    rw [← Real.rpow_add hs, sub_add_cancel, Real.rpow_one]
  have h2 : s ^ (1 + β) = s * s ^ β := by rw [Real.rpow_add hs, Real.rpow_one]
  have h3 : β * (s - t) * s ^ β ≤ s * (s ^ β - t ^ β) := by
    calc β * (s - t) * s ^ β ≤ s ^ (1 - β) * (s ^ β - t ^ β) * s ^ β :=
          mul_le_mul_of_nonneg_right hconc hA.le
      _ = s ^ (1 - β) * s ^ β * (s ^ β - t ^ β) := by ring
      _ = s * (s ^ β - t ^ β) := by rw [h1]
  have hdrop : c * (s * s ^ β) ≤ s - t := by
    rw [h2] at h
    linarith
  have h4 : s * (β * c * (s ^ β) ^ 2) ≤ s * (s ^ β - t ^ β) := by
    nlinarith [mul_le_mul_of_nonneg_left hdrop (mul_nonneg hβ.le hA.le)]
  have hkey := le_of_mul_le_mul_left h4 hs
  rw [Real.rpow_neg hs.le, Real.rpow_neg ht.le, ← sub_nonneg]
  have hA' := hA.ne'
  have hB' := hB.ne'
  have hdiff : (t ^ β)⁻¹ - ((s ^ β)⁻¹ + β * c) =
      (s ^ β - t ^ β - β * c * s ^ β * t ^ β) / (s ^ β * t ^ β) := by
    field_simp
    ring
  rw [hdiff]
  refine div_nonneg ?_ (mul_pos hA hB).le
  nlinarith [mul_le_mul_of_nonneg_left hBA (mul_nonneg (mul_nonneg hβ.le hc) hA.le)]

/-- **Inverting a lower bound on a negative power**: if `0 < y ≤ x ^ (-β)`
with `x, β > 0`, then `x ≤ y ^ (-1/β)`. -/
theorem le_rpow_neg_one_div_of_le_rpow_neg {x y β : ℝ} (hβ : 0 < β) (hx : 0 < x) (hy : 0 < y)
    (h : y ≤ x ^ (-β)) : x ≤ y ^ (-(1 / β)) := by
  calc x = (x ^ (-β)) ^ (-(1 / β)) := by
        rw [← Real.rpow_mul hx.le, neg_mul_neg, mul_one_div_cancel hβ.ne', Real.rpow_one]
    _ ≤ y ^ (-(1 / β)) :=
        Real.rpow_le_rpow_of_nonpos hy h (neg_nonpos.2 (one_div_pos.2 hβ).le)

/-- **Polynomial decay.**  A nonnegative sequence with
`r (k + 1) ≤ r k - c r k ^ (1 + β)` from some index on, `c > 0` and
`0 < β ≤ 1`, is at most `C (k + 1) ^ (-1/β)` for every `k`. -/
theorem exists_le_mul_rpow_of_le_sub {r : ℕ → ℝ} {β c : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1)
    (hc : 0 < c) (hr : ∀ k, 0 ≤ r k)
    (h : ∀ᶠ k in atTop, r (k + 1) ≤ r k - c * r k ^ (1 + β)) :
    ∃ C, ∀ k : ℕ, r k ≤ C * ((k : ℝ) + 1) ^ (-(1 / β)) := by
  obtain ⟨K, hK⟩ := eventually_atTop.1 h
  have hg : ∀ k : ℕ, 0 < ((k : ℝ) + 1) ^ (-(1 / β)) := fun k =>
    Real.rpow_pos_of_pos (by positivity) _
  have hdec : ∀ k, K ≤ k → r (k + 1) ≤ r k := fun k hk => by
    nlinarith [hK k hk, Real.rpow_nonneg (hr k) (1 + β)]
  by_cases hzero : ∃ k₁, K ≤ k₁ ∧ r k₁ = 0
  · obtain ⟨k₁, hk₁, h0⟩ := hzero
    refine exists_forall_le_mul_of_eventually hr hg (C := 0)
      ((eventually_ge_atTop k₁).mono fun k hk => ?_)
    rw [zero_mul]
    induction k, hk using Nat.le_induction with
    | base => exact h0.le
    | succ k hk ih => exact (hdec k (hk₁.trans hk)).trans ih
  push Not at hzero
  have hpos : ∀ k, K ≤ k → 0 < r k := fun k hk => (hr k).lt_of_ne (hzero k hk).symm
  have hgain : ∀ j : ℕ, r K ^ (-β) + β * c * j ≤ r (K + j) ^ (-β) := by
    intro j
    induction j with
    | zero => simp
    | succ j ih =>
      have hstep := rpow_neg_add_le_of_le_sub hβ hβ1 hc.le (hpos (K + j + 1) (by omega))
        (hdec (K + j) le_self_add) (hK (K + j) le_self_add)
      push_cast
      rw [← add_assoc]
      linarith
  refine exists_forall_le_mul_of_eventually hr hg (C := (β * c / 2) ^ (-(1 / β)))
    ((eventually_ge_atTop (2 * K + 1)).mono fun k hk => ?_)
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le (show K ≤ k by omega)
  have hj : ((K + j : ℕ) : ℝ) + 1 ≤ 2 * j := by
    have : ((2 * K + 1 : ℕ) : ℝ) ≤ ((K + j : ℕ) : ℝ) := by exact_mod_cast hk
    push_cast at this ⊢
    linarith
  have hlow : β * c / 2 * (((K + j : ℕ) : ℝ) + 1) ≤ r (K + j) ^ (-β) := by
    have h0 : 0 ≤ r K ^ (-β) := Real.rpow_nonneg (hr K) _
    have hbc : 0 ≤ β * c / 2 := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hj hbc, hgain j]
  calc r (K + j) ≤ (β * c / 2 * (((K + j : ℕ) : ℝ) + 1)) ^ (-(1 / β)) :=
        le_rpow_neg_one_div_of_le_rpow_neg hβ (hpos _ le_self_add) (by positivity) hlow
    _ = (β * c / 2) ^ (-(1 / β)) * (((K + j : ℕ) : ℝ) + 1) ^ (-(1 / β)) :=
        Real.mul_rpow (by positivity) (by positivity)

/-- **Bounds for every index from a bound on `r` and an eventual bound of
`d` by a power of `r`.**  If `r k ≤ C₁ g k` for every `k` and, from some index
on, `d k ≤ K r k ^ p`, with `r, d ≥ 0`, `g > 0` and `K, p ≥ 0`, then for some
`C ≥ 0`, `r k ≤ C g k` and `d k ≤ C g k ^ p` for every `k`. -/
theorem exists_le_mul_and_le_mul_rpow {r d g : ℕ → ℝ} {C₁ K p : ℝ} (hr : ∀ k, 0 ≤ r k)
    (hd : ∀ k, 0 ≤ d k) (hg : ∀ k, 0 < g k) (hK : 0 ≤ K) (hp : 0 ≤ p)
    (hC₁ : ∀ k, r k ≤ C₁ * g k) (h : ∀ᶠ k in atTop, d k ≤ K * r k ^ p) :
    ∃ C, 0 ≤ C ∧ ∀ k, r k ≤ C * g k ∧ d k ≤ C * g k ^ p := by
  have hC₁0 : 0 ≤ C₁ := nonneg_of_mul_nonneg_left ((hr 0).trans (hC₁ 0)) (hg 0)
  have hgp : ∀ k, 0 < g k ^ p := fun k => Real.rpow_pos_of_pos (hg k) p
  obtain ⟨C₂, hC₂⟩ := exists_forall_le_mul_of_eventually hd hgp (C := K * C₁ ^ p)
    (h.mono fun k hk => by
      calc d k ≤ K * r k ^ p := hk
        _ ≤ K * (C₁ * g k) ^ p :=
          mul_le_mul_of_nonneg_left (Real.rpow_le_rpow (hr k) (hC₁ k) hp) hK
        _ = K * C₁ ^ p * g k ^ p := by rw [Real.mul_rpow hC₁0 (hg k).le, mul_assoc])
  refine ⟨max C₁ C₂, le_max_of_le_left hC₁0, fun k => ⟨?_, ?_⟩⟩
  · exact (hC₁ k).trans (mul_le_mul_of_nonneg_right (le_max_left _ _) (hg k).le)
  · exact (hC₂ k).trans (mul_le_mul_of_nonneg_right (le_max_right _ _) (hgp k).le)

/-- The hypotheses of the theorems above hold for `u = 0`, `g = 1`, `C = 0`;
for `r k = 2⁻ᵏ` with `q = 1/2`; for `β = 1`, `c = 0`, `s = 1`, `t = 1/2`; for
`x = 1`, `y = 1`, `β = 1`; for the zero sequence with `β = 1`, `c = 1`; and
for `r = d = 0`, `g = 1`, `C₁ = K = 0`, `p = 1`. -/
example : ((∀ _ : ℕ, (0 : ℝ) ≤ 0) ∧ (∀ _ : ℕ, (0 : ℝ) < 1) ∧
      (∀ᶠ _ : ℕ in atTop, (0 : ℝ) ≤ 0 * 1)) ∧
    ((0 : ℝ) < 1 / 2 ∧ (∀ k : ℕ, (0 : ℝ) ≤ (1 / 2) ^ k) ∧
      ∀ᶠ k : ℕ in atTop, ((1 : ℝ) / 2) ^ (k + 1) ≤ 1 / 2 * (1 / 2) ^ k) ∧
    ((0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
      (1 / 2 : ℝ) ≤ 1 - 0 * 1 ^ (1 + 1 : ℝ)) ∧
    ((0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ^ (-1 : ℝ)) ∧
    (∀ᶠ _ : ℕ in atTop, (0 : ℝ) ≤ 0 - 1 * 0 ^ (1 + 1 : ℝ)) ∧
    ((0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 0 * 1) ∧
      ∀ᶠ _ : ℕ in atTop, (0 : ℝ) ≤ 0 * 0 ^ (1 : ℝ)) := by
  refine ⟨⟨fun _ => le_rfl, fun _ => one_pos, by simp⟩,
    ⟨by norm_num, fun k => by positivity, Eventually.of_forall fun k => le_of_eq (by ring)⟩,
    ⟨one_pos, le_rfl, le_rfl, by norm_num, by norm_num, by norm_num⟩,
    ⟨one_pos, one_pos, one_pos, by simp⟩, Eventually.of_forall fun _ => by norm_num,
    ⟨le_rfl, zero_le_one, fun _ => by norm_num, Eventually.of_forall fun _ => by norm_num⟩⟩

end Transformer.Landscape.SaddleAvoidance
