module

public import Transformer.PhasorV2.SSMax
public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

/-!
# Phasor v2: SSMax at every length

The two items of arXiv:2501.19399v1, §2.2, follow from Eqs. (8) and (9) as `n` grows. In the
source's words, SSMax "ensures that attention is focused on elements whose values exceed others by
approximately `1/s`, while distributing attention when all values are within a range of
approximately `1/s`". Here the gap is one `g` for all `n`, as in the source's Figs. 1 and 3:

* `exists_forall_focus`: for `s g > 1` there is a length beyond which every key that leads all
  other scores by `g` has weight at least `1 − ε`, whatever the scores;
* `exists_forall_fade`: for `s g < 1` there is a length beyond which, if all scores lie within `g`
  of one another, every weight is at most `ε`.

Plain softmax fades for every gap (`exists_forall_softmax_fade`, from Eq. (6)). The leader of
Fig. 1, `5` above the other scores, loses its softmax weight as `n` grows, and keeps its SSMax
weight at `s = 0.43`. This is what SSMax gives the model: one gap focuses a head at every length.
-/

@[expose] public section

open Filter Topology
open Transformer.AttentionResiduals

namespace Transformer.PhasorV2

/-- **§2.2, first item: SSMax focuses at every length.** For `s ≥ 0` and a gap `g` with
`s g > 1`, there is a length beyond which every key that leads all other scores by `g` has weight
at least `1 − ε`. The source: "if `z_max − z_2nd > 1/s`, the maximum value approaches 1". -/
theorem exists_forall_focus {s g : ℝ} (hs : 0 ≤ s) (hg : 1 < s * g) {ε : ℝ} (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ (z : Fin n → ℝ) (j : Fin n), (∀ k ≠ j, z k + g ≤ z j) →
      1 - ε ≤ ssmaxWeight s z j := by
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (-(s * g - 1))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ :=
    eventually_atTop.1 ((hlim.eventually (gt_mem_nhds hε)).and (eventually_ge_atTop 1))
  refine ⟨N, fun n hn z j hz => ?_⟩
  have hsmall : (n : ℝ) ^ (-(s * g - 1)) < ε := (hN n hn).1
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (hN n hn).2
  have hpos : (0 : ℝ) < n := by linarith
  have h := le_ssmaxWeight hs z j (M := z j - g) fun k hk => by linarith [hz k hk]
  rw [Fintype.card_fin, sub_sub_cancel] at h
  have hx0 : 0 ≤ ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) :=
    div_nonneg (by linarith) (Real.rpow_nonneg hpos.le _)
  have hxε : ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) ≤ ε :=
    calc ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) ≤ (n : ℝ) / (n : ℝ) ^ (s * g) :=
          div_le_div_of_nonneg_right (by linarith) (Real.rpow_nonneg hpos.le _)
      _ = (n : ℝ) ^ (-(s * g - 1)) := by
          rw [show -(s * g - 1) = 1 - s * g by ring, Real.rpow_sub hpos, Real.rpow_one]
      _ ≤ ε := hsmall.le
  refine le_trans ?_ h
  rw [le_div_iff₀ (by linarith)]
  nlinarith [mul_nonneg hε.le hx0]

/-- Fig. 1: at `s = 0.43` a lead of `5` keeps 99% of the weight at every large length. -/
example := exists_forall_focus (s := 0.43) (g := 5) (by norm_num) (by norm_num)
  (ε := 1 / 100) (by norm_num)

/-- **§2.2, second item: below the threshold SSMax fades.** For `s ≥ 0` and a gap `g` with
`s g < 1`, there is a length beyond which, if all scores lie within `g` of one another, every
weight is at most `ε`. The source: "if `z_max − z_min < 1/s`, the maximum value approaches 0". -/
theorem exists_forall_fade {s g : ℝ} (hs : 0 ≤ s) (hg : s * g < 1) {ε : ℝ} (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ z : Fin n → ℝ, (∀ j k, z j ≤ z k + g) →
      ∀ j, ssmaxWeight s z j ≤ ε := by
  have hlim : Tendsto (fun n : ℕ => (n : ℝ) ^ (1 - s * g)) atTop atTop :=
    (tendsto_rpow_atTop (by linarith)).comp tendsto_natCast_atTop_atTop
  obtain ⟨N, hN⟩ :=
    eventually_atTop.1 ((hlim.eventually_ge_atTop (2 / ε)).and (eventually_ge_atTop 2))
  refine ⟨N, fun n hn z hz j => ?_⟩
  have hbig : 2 / ε ≤ (n : ℝ) ^ (1 - s * g) := (hN n hn).1
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast (hN n hn).2
  have hpos : (0 : ℝ) < n := by linarith
  have hp : 0 < (n : ℝ) ^ (s * g) := Real.rpow_pos_of_pos hpos _
  have h := ssmaxWeight_le hs z j (m := z j - g) fun k => by linarith [hz j k]
  rw [Fintype.card_fin, sub_sub_cancel] at h
  have hy : 1 / ε ≤ ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) :=
    calc 1 / ε = 2 / ε / 2 := by ring
      _ ≤ (n : ℝ) ^ (1 - s * g) / 2 := div_le_div_of_nonneg_right hbig (by norm_num)
      _ = (n : ℝ) / 2 / (n : ℝ) ^ (s * g) := by
          rw [Real.rpow_sub hpos, Real.rpow_one, div_right_comm]
      _ ≤ ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) := div_le_div_of_nonneg_right (by linarith) hp.le
  have hy0 : 0 < ((n : ℝ) - 1) / (n : ℝ) ^ (s * g) := lt_of_lt_of_le (by positivity) hy
  refine le_trans h ?_
  rw [div_le_iff₀ (by linarith)]
  nlinarith [(div_le_iff₀ hε).1 hy]

/-- At `s = 0.43` scores within `2` of one another spread their weight at every large length. -/
example := exists_forall_fade (s := 0.43) (g := 2) (by norm_num) (by norm_num)
  (ε := 1 / 100) (by norm_num)

/-- **Eq. (6): softmax fades for every gap.** There is a length beyond which, if all scores lie
within `g` of one another, every softmax weight is at most `ε`, however large `g` is. The source:
"the maximum element of the output vector produced by Softmax approaches zero as the input vector
size `n` increases". -/
theorem exists_forall_softmax_fade (g : ℝ) {ε : ℝ} (hε : 0 < ε) :
    ∃ N : ℕ, ∀ n ≥ N, ∀ z : Fin n → ℝ, (∀ j k, z j ≤ z k + g) →
      ∀ j, depthWeight z j ≤ ε := by
  obtain ⟨N, hN⟩ := exists_nat_ge (Real.exp g / ε + 1)
  refine ⟨N, fun n hn z hz j => ?_⟩
  have hn' : Real.exp g / ε + 1 ≤ n := hN.trans (by exact_mod_cast hn)
  have h := depthWeight_le_of_forall_le z j (m := z j - g) fun k => by linarith [hz j k]
  rw [Fintype.card_fin, sub_sub_cancel] at h
  have hy : 1 / ε ≤ ((n : ℝ) - 1) / Real.exp g := by
    rw [le_div_iff₀ (Real.exp_pos g)]
    calc 1 / ε * Real.exp g = Real.exp g / ε := by ring
      _ ≤ (n : ℝ) - 1 := by linarith
  have hy0 : 0 < ((n : ℝ) - 1) / Real.exp g := lt_of_lt_of_le (by positivity) hy
  refine le_trans h ?_
  rw [div_le_iff₀ (by linarith)]
  nlinarith [(div_le_iff₀ hε).1 hy]

/-- Fig. 1 under softmax: the lead of `5` fades. -/
example := exists_forall_softmax_fade 5 (ε := 1 / 100) (by norm_num)

end Transformer.PhasorV2
