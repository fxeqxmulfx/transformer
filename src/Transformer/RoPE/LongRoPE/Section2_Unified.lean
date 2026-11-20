module

public import Transformer.RoPE.YaRN.Section3_Pressure

/-!
# LongRoPE §2.1: PI, NTK and YaRN as rescale factors of one formula

Ding et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*, arXiv:2402.13753v1,
§2.1, Eqs. (1)–(2). The pair `i` of a head of dimension `d` rotates at `θ_i = θ^{−2i/d}`
(`frequency θ d i`); Eq. (2) writes the rescaled angle at the position `n` as `n/(λ_i β^i)`, where
"`β = θ^{2/d}`, and `λ` denote[s] the actual rescale factor related to `s`".

* `beta_pow_eq`, `unifiedAngle_eq`: `β^i = 1/θ_i`, so Eq. (2) is Eq. (1) with `θ_i` divided by
  `λ_i`, and with every `λ_i = 1` it is RoPE's angle `nθ_i`;
* `pi_unifiedAngle`: PI, "the rotation angles of all positions are linearly reduced by `λ = s`
  across all RoPE dimensions", is RoPE at the position `n/s`;
* `ntk_unifiedAngle`: the NTK-aware base `θ·s^{d/(d−2)}` is Eq. (2) with `λ_i = s^{2i/(d−2)}`,
  which "scales lower (high frequency) dimensions less and higher (low frequency) dimensions
  more" (`YaRN.ntk_factor_monotone`), from `λ_0 = 1` to `λ_{d/2−1} = s`;
* `printed_ntk_factor`: §2.1 prints the NTK factor as `λ = s^i`. That factor is the base
  `θ·s^{d/2}`; for `d > 4`, `s > 1` and `i ≥ 1` it exceeds the NTK-aware `s^{2i/(d−2)}`, and at
  the last pair it is `s^{d/2−1}` instead of `s`;
* `yarn_unifiedAngle`, `yarnFactor_bounds`, `yarnFactor_monotone`: YaRN's frequencies (its
  Eq. (13)) are Eq. (2) with `λ_i = 1/((1 − γ_i)/s + γ_i)`, `γ_i` the ramp of the rotations of the
  pair `i`: "High frequency dimensions undergo extrapolation (`λ=1`)" where `γ_i = 1`, "low
  frequency dimensions use linear interpolation (PI)" where `γ_i = 0`. In between, where §2.1 says
  YaRN "employs the NTK", the frequency is the blend `(1 − γ_i)θ_i/s + γ_iθ_i` of the PI and the
  native frequencies. Every `λ_i` lies in `[1, s]` and they grow with `i`.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE

/-- Eq. (2): the angle `n/(λ_i β^i)` of the pair `i` at the position `n`, `β = θ^{2/d}`. -/
noncomputable def unifiedAngle (base : ℝ) (d : ℕ) (factor : ℕ → ℝ) (n : ℝ) (i : ℕ) : ℝ :=
  n / (factor i * (base ^ (2 / (d : ℝ))) ^ i)

/-- §2.1: `β = θ^{2/d}` gives `β^i = θ^{2i/d} = 1/θ_i`. -/
theorem beta_pow_eq {base : ℝ} (hbase : 0 < base) (d i : ℕ) :
    (base ^ (2 / (d : ℝ))) ^ i = (frequency base d i)⁻¹ := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul hbase.le, frequency, neg_div,
    Real.rpow_neg hbase.le, inv_inv]
  congr 1
  ring

example : ((10000 : ℝ) ^ (2 / ((128 : ℕ) : ℝ))) ^ 3 = (frequency 10000 128 3)⁻¹ :=
  beta_pow_eq (by norm_num) 128 3

/-- Eqs. (1)–(2): `n/(λ_iβ^i) = nθ_i/λ_i`, and with every `λ_i = 1` the angle is `nθ_i`. -/
theorem unifiedAngle_eq {base : ℝ} (hbase : 0 < base) (d : ℕ) (factor : ℕ → ℝ) (n : ℝ)
    (i : ℕ) : unifiedAngle base d factor n i = n * frequency base d i / factor i ∧
      unifiedAngle base d (fun _ => 1) n i = n * frequency base d i := by
  have h : ∀ f : ℕ → ℝ, unifiedAngle base d f n i = n * frequency base d i / f i := fun f => by
    rw [unifiedAngle, beta_pow_eq hbase, div_mul_eq_div_div, div_inv_eq_mul, div_mul_eq_mul_div]
  exact ⟨h factor, by rw [h, div_one]⟩

example : unifiedAngle 10000 128 (fun _ => 4) 100 3 = 100 * frequency 10000 128 3 / 4 ∧
    unifiedAngle 10000 128 (fun _ => 1) 100 3 = 100 * frequency 10000 128 3 :=
  unifiedAngle_eq (by norm_num) 128 _ 100 3

/-- §2.1, PI: "the rotation angles of all positions are linearly reduced by `λ = s` across all
RoPE dimensions": Eq. (2) with every `λ_i = s` is RoPE at the position `n/s`. -/
theorem pi_unifiedAngle {base : ℝ} (hbase : 0 < base) (d : ℕ) (s n : ℝ) (i : ℕ) :
    unifiedAngle base d (fun _ => s) n i = n / s * frequency base d i := by
  rw [(unifiedAngle_eq hbase d _ n i).1, div_mul_eq_mul_div]

example : unifiedAngle 10000 128 (fun _ => 8) 100 3 = 100 / 8 * frequency 10000 128 3 :=
  pi_unifiedAngle (by norm_num) 128 8 100 3

/-- The NTK-aware factor `s^{2i/(d−2)}` of the pair `i` (YaRN, Appendix A.2). -/
noncomputable def ntkFactor (s : ℝ) (d i : ℕ) : ℝ := s ^ (2 * (i : ℝ) / ((d : ℝ) - 2))

/-- §2.1, NTK: the NTK-aware base `θ·s^{d/(d−2)}` is Eq. (2) with `λ_i = s^{2i/(d−2)}`. -/
theorem ntk_unifiedAngle {base s : ℝ} (hbase : 0 < base) (hs : 0 < s) {d : ℕ} (hd : 2 < d)
    (n : ℝ) (i : ℕ) :
    n * frequency (YaRN.adjustedBase base s d) d i = unifiedAngle base d (ntkFactor s d) n i := by
  rw [YaRN.coordinate_frequency_adjusted hbase hs hd, (unifiedAngle_eq hbase d _ n i).1,
    ntkFactor, ← mul_div_assoc]

example : 100 * frequency (YaRN.adjustedBase 10000 8 (128 : ℕ)) 128 3 =
    unifiedAngle 10000 128 (ntkFactor 8 128) 100 3 :=
  ntk_unifiedAngle (by norm_num) (by norm_num) (by norm_num) 100 3

/-- §2.1 prints the NTK factor as `λ = s^i`. That factor is the base `θ·s^{d/2}`, here with
`d = 2p`; for `d > 4`, `s > 1` and `i ≥ 1` it exceeds the NTK-aware factor `s^{2i/(d−2)}`, which
at the last pair `p − 1` is `s`. -/
theorem printed_ntk_factor {base s : ℝ} (hbase : 0 < base) (hs : 1 < s) {p : ℕ} (hp : 2 < p)
    (n : ℝ) {i : ℕ} (hi : 0 < i) :
    n * frequency (base * s ^ p) (2 * p) i = unifiedAngle base (2 * p) (fun i => s ^ i) n i ∧
      ntkFactor s (2 * p) i < s ^ i ∧ ntkFactor s (2 * p) (p - 1) = s := by
  have hs0 : 0 < s := by linarith
  have hp' : (2 : ℝ) < p := by exact_mod_cast hp
  have hi' : (0 : ℝ) < i := by exact_mod_cast hi
  refine ⟨?_, ?_, ?_⟩
  · have key : frequency (base * s ^ p) (2 * p) i = frequency base (2 * p) i / s ^ i := by
      unfold frequency
      rw [Real.mul_rpow hbase.le (pow_pos hs0 p).le, ← Real.rpow_natCast s p,
        ← Real.rpow_mul hs0.le,
        show (p : ℝ) * (-(2 * (i : ℝ)) / ((2 * p : ℕ) : ℝ)) = -(i : ℝ) by
          push_cast
          field_simp,
        Real.rpow_neg hs0.le, Real.rpow_natCast, ← div_eq_mul_inv]
    rw [(unifiedAngle_eq hbase (2 * p) _ n i).1, key, mul_div_assoc]
  · have hlt : 2 * (i : ℝ) / (((2 * p : ℕ) : ℝ) - 2) < (i : ℝ) := by
      rw [div_lt_iff₀ (by push_cast; linarith)]
      push_cast
      nlinarith
    calc ntkFactor s (2 * p) i = s ^ (2 * (i : ℝ) / (((2 * p : ℕ) : ℝ) - 2)) := rfl
      _ < s ^ (i : ℝ) := Real.rpow_lt_rpow_of_exponent_lt hs hlt
      _ = s ^ i := Real.rpow_natCast s i
  · rw [ntkFactor, Nat.cast_sub (by omega : 1 ≤ p),
      show 2 * ((p : ℝ) - ((1 : ℕ) : ℝ)) / (((2 * p : ℕ) : ℝ) - 2) = 1 by
        push_cast
        rw [div_eq_one_iff_eq (by linarith)]
        ring,
      Real.rpow_one]

example : 100 * frequency (10000 * 8 ^ 64) (2 * 64) 63 =
      unifiedAngle 10000 (2 * 64) (fun i => (8 : ℝ) ^ i) 100 63 ∧
    ntkFactor 8 (2 * 64) 63 < 8 ^ 63 ∧ ntkFactor 8 (2 * 64) (64 - 1) = 8 :=
  printed_ntk_factor (by norm_num) (by norm_num) (by norm_num) 100 (by norm_num)

/-- YaRN's weight `γ_i` of the pair `i`: its ramp, `α = 1`, `β = 32`, of the rotations of the pair
over the original context `L`. -/
noncomputable def yarnWeight (base : ℝ) (d : ℕ) (L : ℝ) (i : ℕ) : ℝ :=
  YaRN.ramp 1 32 (YaRN.rotations L (frequency base d i))

/-- YaRN's factor of the pair `i`, `1/((1 − γ_i)/s + γ_i)`. -/
noncomputable def yarnFactor (base : ℝ) (d : ℕ) (s L : ℝ) (i : ℕ) : ℝ :=
  1 / ((1 - yarnWeight base d L i) / s + yarnWeight base d L i)

/-- §2.1, YaRN: the YaRN frequency of the pair `i` is Eq. (2) with `λ_i = yarnFactor`. -/
theorem yarn_unifiedAngle {base : ℝ} (hbase : 0 < base) (d : ℕ) (s L n : ℝ) (i : ℕ) :
    n * YaRN.yarnFrequency (frequency base d i) s L =
      unifiedAngle base d (yarnFactor base d s L) n i := by
  rw [(unifiedAngle_eq hbase d _ n i).1, yarnFactor, YaRN.yarnFrequency, YaRN.blendedFrequency,
    yarnWeight, div_div_eq_mul_div, div_one]
  ring

example : 100 * YaRN.yarnFrequency (frequency 10000 128 3) 8 4096 =
    unifiedAngle 10000 128 (yarnFactor 10000 128 8 4096) 100 3 :=
  yarn_unifiedAngle (by norm_num) 128 8 4096 100 3

/-- §2.1, YaRN: `λ_i = 1` where `γ_i = 1`, `λ_i = s` where `γ_i = 0`, and `1 ≤ λ_i ≤ s`. -/
theorem yarnFactor_bounds {base s : ℝ} (hs : 1 ≤ s) (d : ℕ) (L : ℝ) (i : ℕ) :
    (yarnWeight base d L i = 1 → yarnFactor base d s L i = 1) ∧
      (yarnWeight base d L i = 0 → yarnFactor base d s L i = s) ∧
      yarnFactor base d s L i ∈ Set.Icc 1 s := by
  have hγ := YaRN.ramp_bounds 1 32 (YaRN.rotations L (frequency base d i))
  rw [← yarnWeight] at hγ
  have hs0 : 0 < s := by linarith
  refine ⟨fun h => by simp [yarnFactor, h], fun h => by simp [yarnFactor, h], ?_⟩
  set γ := yarnWeight base d L i
  have hu : (1 - γ) / s * s = 1 - γ := div_mul_cancel₀ _ hs0.ne'
  have hX1 : (1 - γ) / s + γ ≤ 1 := by
    linarith [div_le_self (by linarith [hγ.2] : (0 : ℝ) ≤ 1 - γ) hs]
  have hXs : 1 ≤ s * ((1 - γ) / s + γ) := by nlinarith [hγ.1]
  have hX0 : 0 < (1 - γ) / s + γ := by nlinarith
  exact ⟨one_le_one_div hX0 hX1, (div_le_iff₀ hX0).2 hXs⟩

example : (yarnWeight 10000 128 4096 3 = 1 → yarnFactor 10000 128 8 4096 3 = 1) ∧
    (yarnWeight 10000 128 4096 3 = 0 → yarnFactor 10000 128 8 4096 3 = 8) ∧
    yarnFactor 10000 128 8 4096 3 ∈ Set.Icc 1 8 :=
  yarnFactor_bounds (by norm_num) 128 4096 3

/-- §2.1, YaRN: the factors grow with `i`, since the rotations and their ramp fall with `i`. -/
theorem yarnFactor_monotone {base s : ℝ} (hbase : 1 ≤ base) (hs : 1 ≤ s) (d : ℕ) {L : ℝ}
    (hL : 0 ≤ L) : Monotone (yarnFactor base d s L) := by
  intro i j hij
  have hs0 : 0 < s := by linarith
  have hrot : YaRN.rotations L (frequency base d j) ≤ YaRN.rotations L (frequency base d i) :=
    div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_left (frequency_antitone hbase d hij) hL) (by positivity)
  have hγ : yarnWeight base d L j ≤ yarnWeight base d L i :=
    max_le_max le_rfl (min_le_min le_rfl (div_le_div_of_nonneg_right (by linarith) (by norm_num)))
  have hγj := YaRN.ramp_bounds 1 32 (YaRN.rotations L (frequency base d j))
  rw [← yarnWeight] at hγj
  have hu : ∀ k, (1 - yarnWeight base d L k) / s = (1 - yarnWeight base d L k) * (1 / s) :=
    fun k => by ring
  have h1s : 1 / s ≤ 1 := by rw [div_le_one hs0]; exact hs
  apply one_div_le_one_div_of_le
  · rw [hu]
    nlinarith [one_div_pos.mpr hs0, mul_nonneg hγj.1 (sub_nonneg.mpr h1s)]
  · rw [hu, hu]
    nlinarith

example : Monotone (yarnFactor 10000 128 8 4096) :=
  yarnFactor_monotone (by norm_num) (by norm_num) 128 (by norm_num)

end Transformer.RoPE.LongRoPE
