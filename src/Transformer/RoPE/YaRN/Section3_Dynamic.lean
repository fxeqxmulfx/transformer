module

public import Transformer.RoPE.YaRN.Section2_Background

/-!
# YaRN §3.4: Dynamic Scaling and the key–value cache

Peng et al., arXiv:2309.00071v3, §3.4. Dynamic Scaling sets `s = max(1, l'/L)` from the current
sequence length `l'` (`contextScale`), so the embedding moves while a sequence grows; with the
NTK-aware base of Eq. (19) it is "Dynamic NTK" interpolation.

* `rotary_ne_of_phase`, `dynamic_phase_step`, `dynamic_pi_rotary_ne`, `dynamic_ntk_rotary_ne`:
  "the RoPE of every token changes when `s` changes": a rotation moves when the phase of one of
  its pairs moves by a nonzero amount less than a turn, and each step past `L` moves the rotation
  of every token at `0 < n ≤ l'`, under PI through any pair of frequency at most `1`, under
  Dynamic NTK through the last pair. The token at position `0` keeps the identity at every scale
  (`RoFormer.rotary_zero`), so "every token" leaves out the first;
* `stale_cache_score`: "The correct implementation should cache the kv-embeddings before applying
  rotary position embeddings": a key cached after RoPE at the old scale gives the query a score
  that a fresh pass does not. A key cached before RoPE is rotated at the current scale and gives
  the fresh score by definition.

Both settings of §3.4 agree with the unextended model up to the pretrained length
(`context_scale_unextended`, `full_score_unextended`). The "performance discount", the "abrupt
degradation" of setting (1) and the results of Appendix B.7 are measurements.
-/

@[expose] public section

open Matrix

namespace Transformer.RoPE.YaRN

open RoFormer

variable {p : ℕ}

/-- §3.4: if the phase of the pair `l` at position `n` moves by a nonzero amount less than a full
turn, the rotation at `n` changes. -/
theorem rotary_ne_of_phase {θ θ' : Fin p → ℝ} {n : ℝ} (l : Fin p)
    (h0 : 0 < |n * θ l - n * θ' l|) (h1 : |n * θ l - n * θ' l| < 2 * Real.pi) :
    rotary θ n ≠ rotary θ' n := by
  intro h
  have hc : Real.cos (n * θ' l) = Real.cos (n * θ l) := by
    simpa [rotary, blockDiagonal_apply, rotation] using
      (congr_fun (congr_fun h (0, l)) (0, l)).symm
  have hs : Real.sin (n * θ' l) = Real.sin (n * θ l) := by
    simpa [rotary, blockDiagonal_apply, rotation] using
      (congr_fun (congr_fun h (1, l)) (0, l)).symm
  obtain ⟨k, hk⟩ := Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)
  rw [abs_sub_comm, hk, abs_mul, abs_of_pos Real.two_pi_pos] at h0 h1
  have hk1 : |(k : ℝ)| < 1 := by nlinarith [Real.pi_pos]
  have hk0 : k = 0 := by
    have : |k| < 1 := by exact_mod_cast hk1
    rw [abs_lt] at this
    omega
  rw [hk0, Int.cast_zero, abs_zero, mul_zero] at h0
  exact lt_irrefl 0 h0

example : rotary (fun _ : Fin 1 => (1 : ℝ)) 1 ≠ rotary (fun _ : Fin 1 => (1 : ℝ) / 2) 1 :=
  rotary_ne_of_phase 0 (by norm_num) (by norm_num; linarith [Real.pi_gt_d2])

/-- §3.4: the step from the length `l'` to `l' + 1` past `L` moves the scale from `l'/L` to
`(l' + 1)/L`, and the phase `nω/s` of a pair of frequency `0 < ω ≤ 1` at the position
`0 < n ≤ l'` by `nωL/(l'(l' + 1))`, which lies in `(0, 1)`. -/
theorem dynamic_phase_step {ω : ℝ} (hω0 : 0 < ω) (hω1 : ω ≤ 1) {L l' n : ℕ} (hL : 0 < L)
    (hl' : L ≤ l') (hn : 0 < n) (hnl : n ≤ l') :
    0 < |n * (ω / contextScale L l') - n * (ω / contextScale L (l' + 1))| ∧
      |n * (ω / contextScale L l') - n * (ω / contextScale L (l' + 1))| < 2 * Real.pi := by
  have hL' : (0 : ℝ) < L := by exact_mod_cast hL
  have hl'' : (L : ℝ) ≤ l' := by exact_mod_cast hl'
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hnl' : (n : ℝ) ≤ l' := by exact_mod_cast hnl
  have hpos : (0 : ℝ) < l' := by linarith
  rw [context_scale_extension hL' hl'', context_scale_extension hL' (by linarith)]
  have hdiff : (n : ℝ) * (ω / (l' / L)) - n * (ω / ((l' + 1) / L)) =
      n * ω * L / (l' * (l' + 1)) := by
    field_simp
    ring
  have hD : 0 < (n : ℝ) * ω * L / (l' * (l' + 1)) := by positivity
  have hD1 : (n : ℝ) * ω * L / (l' * (l' + 1)) < 1 := by
    rw [div_lt_one (by positivity)]
    nlinarith [mul_le_mul hnl' hl'' hL'.le hpos.le,
      mul_nonneg (sub_nonneg.mpr hω1) (mul_pos hn' hL').le]
  rw [hdiff, abs_of_pos hD]
  exact ⟨hD, by linarith [Real.pi_gt_d2]⟩

/-- §3.4, Dynamic Scaling of PI: each step past `L` moves the rotation of every token at
`0 < n ≤ l'`, through any pair of frequency in `(0, 1]`, such as RoPE's first pair. -/
theorem dynamic_pi_rotary_ne {θ : Fin p → ℝ} {l : Fin p} (hl0 : 0 < θ l) (hl1 : θ l ≤ 1)
    {L l' n : ℕ} (hL : 0 < L) (hl' : L ≤ l') (hn : 0 < n) (hnl : n ≤ l') :
    rotary (fun i => θ i / contextScale L l') n ≠
      rotary (fun i => θ i / contextScale L (l' + 1)) n :=
  rotary_ne_of_phase l (dynamic_phase_step hl0 hl1 hL hl' hn hnl).1
    (dynamic_phase_step hl0 hl1 hL hl' hn hnl).2

example : rotary (fun _ : Fin 1 => (1 : ℝ) / contextScale ((2048 : ℕ) : ℝ) ((4096 : ℕ) : ℝ))
      ((100 : ℕ) : ℝ) ≠
    rotary (fun _ : Fin 1 => (1 : ℝ) / contextScale ((2048 : ℕ) : ℝ) (((4096 : ℕ) : ℝ) + 1))
      ((100 : ℕ) : ℝ) :=
  dynamic_pi_rotary_ne (θ := fun _ => 1) (l := 0) one_pos le_rfl (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

/-- §3.4, "Dynamic NTK": Dynamic Scaling of the NTK-aware base `b·s^{|D|/(|D|−2)}` of Eq. (19),
`|D| = 2p`, `b ≥ 1`. Each step past `L` moves the rotation of every token at `0 < n ≤ l'`, through
the last pair, whose frequency the base divides by `s` (Eq. (22)); the first pair stays
(`highest_frequency_preserved`). -/
theorem dynamic_ntk_rotary_ne {base : ℝ} (hbase : 1 ≤ base) {L l' n : ℕ} (hp : 1 < p)
    (hL : 0 < L) (hl' : L ≤ l') (hn : 0 < n) (hnl : n ≤ l') :
    rotary (fun l : Fin p =>
        frequency (adjustedBase base (contextScale L l') (2 * p : ℕ)) (2 * p) l) n ≠
      rotary (fun l : Fin p =>
        frequency (adjustedBase base (contextScale L (l' + 1)) (2 * p : ℕ)) (2 * p) l) n := by
  have hp' : (1 : ℝ) < p := by exact_mod_cast hp
  have hlast : ∀ s : ℝ, 0 < s → frequency (adjustedBase base s (2 * p : ℕ)) (2 * p) (p - 1) =
      frequency base (2 * p) (p - 1) / s := by
    intro s hs
    rw [coordinate_frequency_adjusted (by linarith) hs (by omega)]
    have he : 2 * ((p - 1 : ℕ) : ℝ) / (((2 * p : ℕ) : ℝ) - 2) = 1 := by
      rw [Nat.cast_sub (by omega),
        div_eq_one_iff_eq (show (0 : ℝ) < ((2 * p : ℕ) : ℝ) - 2 by push_cast; linarith).ne']
      push_cast
      ring
    rw [he, Real.rpow_one]
  have h0 : frequency base (2 * p) 0 = 1 := by simp [frequency]
  have hω0 : 0 < frequency base (2 * p) (p - 1) := Real.rpow_pos_of_pos (by linarith) _
  have hω1 : frequency base (2 * p) (p - 1) ≤ 1 :=
    h0 ▸ frequency_antitone hbase (2 * p) (Nat.zero_le (p - 1))
  have hs : ∀ t : ℝ, 0 < contextScale L t := fun t => lt_of_lt_of_le one_pos (le_max_left _ _)
  have h := dynamic_phase_step hω0 hω1 hL hl' hn hnl
  rw [← hlast _ (hs _), ← hlast _ (hs _)] at h
  exact rotary_ne_of_phase ⟨p - 1, by omega⟩ h.1 h.2

example : rotary (fun l : Fin 64 => frequency (adjustedBase 10000
        (contextScale ((4096 : ℕ) : ℝ) ((8192 : ℕ) : ℝ)) ((2 * 64 : ℕ) : ℝ)) (2 * 64) l)
      ((100 : ℕ) : ℝ) ≠
    rotary (fun l : Fin 64 => frequency (adjustedBase 10000
        (contextScale ((4096 : ℕ) : ℝ) (((8192 : ℕ) : ℝ) + 1)) ((2 * 64 : ℕ) : ℝ)) (2 * 64) l)
      ((100 : ℕ) : ℝ) :=
  dynamic_ntk_rotary_ne (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- `(R_θ^b e₁)ᵀ(R_{θ'}^d e₁) = cos(bθ − dθ')` for one pair. -/
theorem single_pair_score (a b c d : ℝ) :
    (rotary (fun _ : Fin 1 => a) b *ᵥ Pi.single (0, 0) 1) ⬝ᵥ
        (rotary (fun _ : Fin 1 => c) d *ᵥ Pi.single (0, 0) 1) = Real.cos (b * a - d * c) := by
  simp [rotary, blockDiagonal_apply, rotation, dotProduct, Fintype.sum_prod_type, Real.cos_sub]

/-- §3.4: a key cached after RoPE at the old scale gives a score that a fresh pass does not. One
pair of frequency `1`, PI from `s = 1` to `s = 2`, the key `e₁` at position `1` and the query
`e₁` at position `2`: the stale score is `cos 0 = 1`, the fresh one `cos(1/2)`. -/
theorem stale_cache_score :
    (rotary (fun _ : Fin 1 => (1 : ℝ) / 2) 2 *ᵥ Pi.single (0, 0) 1) ⬝ᵥ
        (rotary (fun _ : Fin 1 => (1 : ℝ) / 1) 1 *ᵥ Pi.single (0, 0) 1) ≠
      (rotary (fun _ : Fin 1 => (1 : ℝ) / 2) 2 *ᵥ Pi.single (0, 0) 1) ⬝ᵥ
        (rotary (fun _ : Fin 1 => (1 : ℝ) / 2) 1 *ᵥ Pi.single (0, 0) 1) := by
  rw [single_pair_score, single_pair_score,
    show (2 : ℝ) * (1 / 2) - 1 * (1 / 1) = 0 by norm_num,
    show (2 : ℝ) * (1 / 2) - 1 * (1 / 2) = 1 / 2 by norm_num, Real.cos_zero]
  intro h
  have h0 := (Real.cos_eq_one_iff_of_lt_of_lt (by linarith [Real.pi_pos])
    (by linarith [Real.pi_gt_d2])).mp h.symm
  norm_num at h0

end Transformer.RoPE.YaRN
