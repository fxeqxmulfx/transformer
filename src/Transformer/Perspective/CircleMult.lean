/-
# Multiplicities and the largest gap

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §3, before and in the proof of `lemma:taumax`.

For a configuration `x ∈ 𝕋^n` with sorted positions `θ_k` (`circleSite`), the
source writes `N_k` for the number of particles at `θ_k` (`circleMult`, read
on `𝕋`, so that `N_{k+K} = N_k`) and `τ_max` for the largest gap
(`circleGapMax`).

* The multiplicities add up to `n`, also shifted by one position
  (`sum_circleMult`, `sum_circleMult_succ`), and two consecutive ones to at
  most `n` once there are two positions (`circleMult_add_circleMult_succ_le`).
* A configuration that is not synchronized has two positions at least
  (`two_le_card_circleSites`), and then every gap is below `2π`
  (`circleGap_lt_two_pi`).
-/

import Transformer.Perspective.CircleGaps

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {n : ℕ}

/-- The point of `𝕋` of the `k`-th position is the position `k mod K`. -/
theorem circlePos_circleSite (x : Angles n) (h : 0 < (circleSites x).card) (k : ℕ) :
    circlePos (circleSite x k) = circleSite x (k % (circleSites x).card) := by
  rw [circleSite_of_lt (Nat.mod_lt k h)]
  simp only [circleSite, h, ↓reduceDIte, circlePos]
  rw [show 2 * π * ((k / (circleSites x).card : ℕ) : ℝ) =
      (k / (circleSites x).card) • (2 * π) by rw [nsmul_eq_mul]; ring, toIcoMod_add_nsmul,
    toIcoMod_eq_self, zero_add]
  exact mem_Ico_of_mem_circleSites (Finset.orderEmbOfFin_mem _ _ _)

/-- **`N_k`**, the number of particles at the `k`-th position, read on `𝕋`.

Source: arXiv:2507.22857v1, §3, before `lemma:taumax`, the multiplicity `N_i`
of `θ_i`. -/
noncomputable def circleMult (x : Angles n) (k : ℕ) : ℕ :=
  (Finset.univ.filter fun j => circlePos (x j) = circlePos (circleSite x k)).card

theorem injOn_circleSite (x : Angles n) :
    Set.InjOn (circleSite x) (Finset.range (circleSites x).card) := fun a ha _ _ hab =>
  (strictMono_circleSite x (lt_of_le_of_lt (Nat.zero_le a) (Finset.mem_range.1 ha))).injective hab

/-- The positions in order are the positions. -/
theorem image_circleSite (x : Angles n) :
    (Finset.range (circleSites x).card).image (circleSite x) = circleSites x := by
  refine Finset.eq_of_subset_of_card_le (fun s hs => ?_) ?_
  · obtain ⟨k, hk, rfl⟩ := Finset.mem_image.1 hs
    exact circleSite_mem (Finset.mem_range.1 hk)
  · rw [Finset.card_image_of_injOn (injOn_circleSite x), Finset.card_range]

/-- **The multiplicities add up to `n`.** -/
theorem sum_circleMult (x : Angles n) :
    ∑ k ∈ Finset.range (circleSites x).card, (circleMult x k : ℝ) = n := by
  have hfib := Finset.card_eq_sum_card_fiberwise (f := fun j => circlePos (x j))
    (s := Finset.univ) (t := circleSites x) fun j _ =>
      Finset.mem_image_of_mem _ (Finset.mem_univ j)
  rw [← image_circleSite x, Finset.sum_image (injOn_circleSite x)] at hfib
  have key : ∑ k ∈ Finset.range (circleSites x).card, circleMult x k = n := by
    calc ∑ k ∈ Finset.range (circleSites x).card, circleMult x k
        = ∑ k ∈ Finset.range (circleSites x).card,
            (Finset.univ.filter fun j => circlePos (x j) = circleSite x k).card :=
          Finset.sum_congr rfl fun k hk => by
            have hK := lt_of_le_of_lt (Nat.zero_le k) (Finset.mem_range.1 hk)
            simp only [circleMult, circlePos_circleSite x hK,
              Nat.mod_eq_of_lt (Finset.mem_range.1 hk)]
      _ = (Finset.univ : Finset (Idx n)).card := hfib.symm
      _ = n := Finset.card_fin n
  exact_mod_cast key

/-- **`N_K = N_0`**, so the multiplicities shifted by one position add up to
the same. -/
theorem sum_circleMult_succ (x : Angles n) :
    ∑ k ∈ Finset.range (circleSites x).card, (circleMult x (k + 1) : ℝ) =
      ∑ k ∈ Finset.range (circleSites x).card, (circleMult x k : ℝ) := by
  rcases Nat.eq_zero_or_pos (circleSites x).card with h0 | h
  · simp [h0]
  have hK : circleMult x (circleSites x).card = circleMult x 0 := by
    simp only [circleMult, circlePos_circleSite x h, Nat.mod_self, Nat.zero_mod]
  have h1 := Finset.sum_range_succ' (fun k => (circleMult x k : ℝ)) (circleSites x).card
  have h2 := Finset.sum_range_succ (fun k => (circleMult x k : ℝ)) (circleSites x).card
  rw [hK] at h2
  linarith

/-- With two positions at least, two consecutive multiplicities add up to at
most `n`. -/
theorem circleMult_add_circleMult_succ_le (x : Angles n) (h : 2 ≤ (circleSites x).card)
    {l : ℕ} (hl : l < (circleSites x).card) :
    (circleMult x l : ℝ) + circleMult x (l + 1) ≤ n := by
  have h0 : 0 < (circleSites x).card := by omega
  have hmod : circleMult x (l + 1) = circleMult x ((l + 1) % (circleSites x).card) := by
    simp only [circleMult, circlePos_circleSite x h0, Nat.mod_mod]
  have hne : l ≠ (l + 1) % (circleSites x).card := by
    rcases Nat.lt_or_ge (l + 1) (circleSites x).card with hlt | hge
    · rw [Nat.mod_eq_of_lt hlt]; omega
    · rw [show l + 1 = (circleSites x).card by omega, Nat.mod_self]; omega
  have hsub : {l, (l + 1) % (circleSites x).card} ⊆ Finset.range (circleSites x).card := by
    intro k hk
    rcases Finset.mem_insert.1 hk with rfl | hk
    · exact Finset.mem_range.2 hl
    · rw [Finset.mem_singleton.1 hk]; exact Finset.mem_range.2 (Nat.mod_lt _ h0)
  calc (circleMult x l : ℝ) + circleMult x (l + 1)
      = ∑ k ∈ {l, (l + 1) % (circleSites x).card}, (circleMult x k : ℝ) := by
        rw [Finset.sum_pair hne, hmod]
    _ ≤ ∑ k ∈ Finset.range (circleSites x).card, (circleMult x k : ℝ) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Nat.cast_nonneg _
    _ = n := sum_circleMult x

/-- **A configuration that is not synchronized has two positions at least.** -/
theorem two_le_card_circleSites (x : Angles n)
    (hx : ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) : 2 ≤ (circleSites x).card := by
  by_contra hlt
  refine hx fun i j => ⟨toIcoDiv two_pi_pos 0 (x i) - toIcoDiv two_pi_pos 0 (x j), ?_⟩
  have hle : (circleSites x).card ≤ 1 := by omega
  have hij : circlePos (x i) = circlePos (x j) :=
    Finset.card_le_one.1 hle _ (Finset.mem_image_of_mem _ (Finset.mem_univ i)) _
      (Finset.mem_image_of_mem _ (Finset.mem_univ j))
  have hi := self_sub_toIcoMod two_pi_pos 0 (x i)
  have hj := self_sub_toIcoMod two_pi_pos 0 (x j)
  simp only [circlePos] at hij
  rw [zsmul_eq_mul] at hi hj
  push_cast
  linarith

/-- With two positions at least, every gap is below `2π`. -/
theorem circleGap_lt_two_pi (x : Angles n) (h : 2 ≤ (circleSites x).card) {k : ℕ}
    (hk : k < (circleSites x).card) : circleGap x k < 2 * π := by
  have h0 : 0 < (circleSites x).card := by omega
  rw [← sum_circleGap x h0]
  refine Finset.single_lt_sum (j := if k = 0 then 1 else 0) ?_ (Finset.mem_range.2 hk) ?_
    (circleGap_pos x h0 _) fun l _ _ => (circleGap_pos x h0 l).le
  · split_ifs <;> omega
  · rw [Finset.mem_range]; split_ifs <;> omega

/-- **`τ_max`**, the largest gap.

Source: arXiv:2507.22857v1, §3, before `lemma:twoconsecutive_general`. -/
noncomputable def circleGapMax (x : Angles n) : ℝ :=
  ⨆ k : Fin (circleSites x).card, circleGap x k

theorem circleGap_le_circleGapMax (x : Angles n) {k : ℕ} (hk : k < (circleSites x).card) :
    circleGap x k ≤ circleGapMax x :=
  le_ciSup (f := fun k : Fin (circleSites x).card => circleGap x k) (Finite.bddAbove_range _)
    ⟨k, hk⟩

/-- The largest gap is one of the gaps. -/
theorem exists_circleGap_eq_circleGapMax (x : Angles n) (h : 0 < (circleSites x).card) :
    ∃ l < (circleSites x).card, circleGap x l = circleGapMax x := by
  have : Nonempty (Fin (circleSites x).card) := ⟨⟨0, h⟩⟩
  obtain ⟨l, hl⟩ :=
    exists_eq_ciSup_of_finite (f := fun k : Fin (circleSites x).card => circleGap x k)
  exact ⟨l, l.2, hl⟩

theorem circleGapMax_pos (x : Angles n) (h : 0 < (circleSites x).card) :
    0 < circleGapMax x :=
  (circleGap_pos x h 0).trans_le (circleGap_le_circleGapMax x h)

/-- With two positions at least, the largest gap is below `2π`. -/
theorem circleGapMax_lt_two_pi (x : Angles n) (h : 2 ≤ (circleSites x).card) :
    circleGapMax x < 2 * π := by
  obtain ⟨l, hl, hmax⟩ := exists_circleGap_eq_circleGapMax x (by omega)
  exact hmax ▸ circleGap_lt_two_pi x h hl

/-- The hypotheses above hold for two antipodal particles, `x = (0, π)`: they
are not synchronized, since `π` is not a multiple of `2π`, so there are two
positions, and the indices `k = 0` and `l = 0` are below their number. -/
example :
    let x : Angles 2 := fun i => (i : ℕ) * π
    (¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π)) ∧ 2 ≤ (circleSites x).card := by
  intro x
  have hsync : ¬ ∀ i j, ∃ k : ℤ, x i - x j = k * (2 * π) := fun h => by
    obtain ⟨k, hk⟩ := h 1 0
    have h1 : π * 1 = π * (2 * k) := by simp [x] at hk; linarith
    have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast mul_left_cancel₀ pi_pos.ne' h1
    omega
  exact ⟨hsync, two_le_card_circleSites x hsync⟩

end Perspective
end Transformer
