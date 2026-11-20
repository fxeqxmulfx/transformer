module

public import Transformer.RoPE.LongRoPE.Section2_Unified
public import Mathlib.Analysis.SpecificLimits.Normed

/-!
# LongRoPE §§3.1–3.2: Eq. (3) and its search space

Ding et al., arXiv:2402.13753v1, §3.1, Eq. (3), §3.2 and Table 4. Eq. (3) multiplies the angle
`n/β^i` of the pair `i` at the position `n` by `𝕀(λ̂_i, n̂)`, which is `1` for `n < n̂` and
`1/λ_i` for `n ≥ n̂`.

* `longAngle_eq`: below the threshold `n̂` the angle is RoPE's `nθ_i`, from it on Eq. (2)'s
  `n/(λ_iβ^i)`;
* `longAngle_zero`: "When `n̂ = 0`, all token positions use the searched rescale factors", and with
  PI's and NTK's factors "it reverts to the original PI and NTK" (§2.2);
* `unscaled_positions`: "For initial `n̂-1` token positions, the rescale factor `λ̂_i` will not take
  effect": for `λ_i ≠ 1` the factor is `1` at exactly the `n̂` positions `0, …, n̂ − 1`, the
  "number of initial token positions that are retained" of §3.2; `n̂ − 1` is the last of them;
* `longAngle_threshold_one`: the thresholds `n̂ = 0` and `n̂ = 1` of Table 4 give the same angles,
  since the position `0` has the angle `0` under every factor;
* `search_space_count`: "a `s=4×` extension leads to `400^{128/2}×14=4×10^{167}` choices": Table 4
  lists `14` thresholds, and `400^{64}·14` lies between `4.76·10^{167}` and `4.77·10^{167}`; the
  grid from `1.0` to `5.0` in steps of `0.01` has `401` values, which give more than
  `5.5·10^{167}`;
* `searchCount_subexponential`: "With larger extension ratio, the search space expands
  exponentially": at an integer ratio `s` each of the `p` pairs takes the `125s − 99` values
  `1.00, …, 1.25s` (`searchCount`), and the count divided by `c^s` tends to `0` for every `c > 1`;
* `initial_population_admissible`: the PI, NTK and YaRN factors put into the initial population
  satisfy "a non-decreasing monotonicity constraint …: `λ_i ≤ λ_{i+1}`" and lie in Table 4's range
  `[1, 1.25s]`;
* `printed_ntk_inadmissible`: the factors `λ_i = s^i` that §2.1 prints for NTK leave that range at
  `i = 2` for `s ≥ 2`;
* `ntk_factor_off_grid`: the NTK-aware factor `4^{1/63}` of the pair `1` at `s = 4`, `d = 128` lies
  strictly between the grid values `1.02` and `1.03`.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.LongRoPE

/-- Eq. (3): `𝕀(λ̂_i, n̂)`, `1` below the threshold `n̂` and `1/λ_i` from it on. -/
noncomputable def thresholdScale (factor : ℝ) (threshold n : ℕ) : ℝ :=
  if n < threshold then 1 else 1 / factor

/-- Eq. (3): the angle `𝕀(λ̂_i, n̂)·n/β^i` of the pair `i` at the position `n`. -/
noncomputable def longAngle (base : ℝ) (d : ℕ) (factor : ℕ → ℝ) (threshold n i : ℕ) : ℝ :=
  thresholdScale (factor i) threshold n * ((n : ℝ) / (base ^ (2 / (d : ℝ))) ^ i)

/-- Eq. (3): RoPE's angle `nθ_i` below the threshold, Eq. (2)'s `n/(λ_iβ^i)` from it on. -/
theorem longAngle_eq {base : ℝ} (hbase : 0 < base) (d : ℕ) (factor : ℕ → ℝ)
    (threshold n i : ℕ) :
    (n < threshold → longAngle base d factor threshold n i = n * frequency base d i) ∧
      (threshold ≤ n → longAngle base d factor threshold n i = unifiedAngle base d factor n i) := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rw [longAngle, thresholdScale, ite_eq_left h, one_mul, beta_pow_eq hbase, div_inv_eq_mul]
  · rw [longAngle, thresholdScale, ite_eq_right (not_lt.mpr h), unifiedAngle]
    ring

example : (3 < 4 → longAngle 10000 128 (fun _ => 8) 4 3 5 = (3 : ℕ) * frequency 10000 128 5) ∧
    (4 ≤ 3 → longAngle 10000 128 (fun _ => 8) 4 3 5 = unifiedAngle 10000 128 (fun _ => 8) 3 5) :=
  longAngle_eq (by norm_num) 128 _ 4 3 5

/-- §3.2, "When `n̂=0`, all token positions use the searched rescale factors"; §2.2, with PI's and
NTK's factors "it reverts to the original PI and NTK". -/
theorem longAngle_zero {base s : ℝ} (hbase : 0 < base) (hs : 0 < s) {d : ℕ} (hd : 2 < d)
    (factor : ℕ → ℝ) (n i : ℕ) :
    longAngle base d factor 0 n i = unifiedAngle base d factor n i ∧
      longAngle base d (fun _ => s) 0 n i = n / s * frequency base d i ∧
      longAngle base d (ntkFactor s d) 0 n i = n * frequency (YaRN.adjustedBase base s d) d i := by
  have h := fun f => (longAngle_eq hbase d f 0 n i).2 (Nat.zero_le n)
  exact ⟨h factor, by rw [h, pi_unifiedAngle hbase], by rw [h, ntk_unifiedAngle hbase hs hd]⟩

example : longAngle 10000 128 (fun _ => 3) 0 7 5 = unifiedAngle 10000 128 (fun _ => 3) 7 5 ∧
    longAngle 10000 128 (fun _ => 8) 0 7 5 = (7 : ℕ) / 8 * frequency 10000 128 5 ∧
    longAngle 10000 128 (ntkFactor 8 128) 0 7 5 =
      (7 : ℕ) * frequency (YaRN.adjustedBase 10000 8 (128 : ℕ)) 128 5 :=
  longAngle_zero (by norm_num) (by norm_num) (by norm_num) _ 7 5

/-- §3.1: for `λ_i ≠ 1` the factor does not take effect at exactly the `n̂` positions
`0, …, n̂ − 1` of a window `[0, N)`, `n̂ ≤ N`. -/
theorem unscaled_positions {factor : ℝ} (hfactor : factor ≠ 1) {threshold N : ℕ}
    (h : threshold ≤ N) :
    (Finset.range N).filter (fun n => thresholdScale factor threshold n = 1) =
        Finset.range threshold ∧
      ((Finset.range N).filter (fun n => thresholdScale factor threshold n = 1)).card =
        threshold := by
  have key : (Finset.range N).filter (fun n => thresholdScale factor threshold n = 1) =
      Finset.range threshold := by
    ext n
    rw [Finset.mem_filter, Finset.mem_range, Finset.mem_range]
    by_cases hn : n < threshold
    · rw [thresholdScale, ite_eq_left hn]
      exact ⟨fun _ => hn, fun _ => ⟨by omega, rfl⟩⟩
    · rw [thresholdScale, ite_eq_right hn, one_div, inv_eq_one]
      exact ⟨fun h' => absurd h'.2 hfactor, fun h' => absurd h' hn⟩
  exact ⟨key, by rw [key, Finset.card_range]⟩

example : (Finset.range 4096).filter (fun n => thresholdScale 8 4 n = 1) = Finset.range 4 ∧
    ((Finset.range 4096).filter (fun n => thresholdScale 8 4 n = 1)).card = 4 :=
  unscaled_positions (by norm_num) (by norm_num)

/-- Table 4: the thresholds `n̂ = 0` and `n̂ = 1` give the same angle at every position. -/
theorem longAngle_threshold_one (base : ℝ) (d : ℕ) (factor : ℕ → ℝ) (n i : ℕ) :
    longAngle base d factor 1 n i = longAngle base d factor 0 n i := by
  unfold longAngle thresholdScale
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  · rw [ite_eq_right (by omega), ite_eq_right (by omega)]

/-- Table 4: the thresholds `n̂`. -/
def thresholds : Finset ℕ := {0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256}

/-- §3.2: "a `s=4×` extension leads to `400^{128/2}×14=4×10^{167}` choices". -/
theorem search_space_count :
    thresholds.card = 14 ∧ (476 : ℝ) * 10 ^ 165 < 400 ^ (128 / 2) * 14 ∧
      (400 : ℝ) ^ (128 / 2) * 14 < 477 * 10 ^ 165 ∧ (Finset.Icc 100 500).card = 401 ∧
      (55 : ℝ) * 10 ^ 166 < 401 ^ (128 / 2) * 14 :=
  ⟨rfl, by norm_num, by norm_num, by simp, by norm_num⟩

/-- Table 4's choices at an integer ratio `s`: a threshold and, for each of `p` pairs, a factor
`k/100` with `100 ≤ k ≤ 125s`. -/
def searchCount (p s : ℕ) : ℕ := (Finset.Icc 100 (125 * s)).card ^ p * thresholds.card

/-- §3.2, "With larger extension ratio, the search space expands exponentially": the count is
bounded by `14·125^p·s^p`, a polynomial in `s`, and is `o(c^s)` for every `c > 1`. -/
theorem searchCount_subexponential (p : ℕ) {c : ℝ} (hc : 1 < c) :
    Tendsto (fun s : ℕ => (searchCount p s : ℝ) / c ^ s) atTop (𝓝 0) := by
  have hbound : ∀ s : ℕ, (searchCount p s : ℝ) / c ^ s ≤ 14 * 125 ^ p * ((s : ℝ) ^ p / c ^ s) := by
    intro s
    have hcard : ((Finset.Icc 100 (125 * s)).card : ℝ) ≤ 125 * s := by
      rw [Nat.card_Icc]
      exact_mod_cast (by omega : 125 * s + 1 - 100 ≤ 125 * s)
    have hc0 : 0 ≤ c ^ s := pow_nonneg (by linarith) s
    calc (searchCount p s : ℝ) / c ^ s
        = ((Finset.Icc 100 (125 * s)).card : ℝ) ^ p * 14 / c ^ s := by
          rw [searchCount, show thresholds.card = 14 from rfl]
          push_cast
          rfl
      _ ≤ (125 * (s : ℝ)) ^ p * 14 / c ^ s := by gcongr
      _ = 14 * 125 ^ p * ((s : ℝ) ^ p / c ^ s) := by ring
  have hlim := (tendsto_pow_const_div_const_pow_of_one_lt p hc).const_mul (14 * 125 ^ p)
  rw [mul_zero] at hlim
  exact squeeze_zero (fun s => div_nonneg (Nat.cast_nonneg _) (pow_nonneg (by linarith) s))
    hbound hlim

example : Tendsto (fun s : ℕ => (searchCount 64 s : ℝ) / 2 ^ s) atTop (𝓝 0) :=
  searchCount_subexponential 64 (by norm_num)

/-- §3.2: `p` factors in Table 4's range `[1, 1.25s]` under the constraint `λ_i ≤ λ_{i+1}`. -/
def Admissible (s : ℝ) (p : ℕ) (factor : ℕ → ℝ) : Prop :=
  (∀ i j, i ≤ j → j < p → factor i ≤ factor j) ∧ ∀ i < p, factor i ∈ Set.Icc 1 (1.25 * s)

/-- §3.2: the PI, NTK and YaRN factors of the initial population of `p + 1` pairs are admissible. -/
theorem initial_population_admissible {base s : ℝ} (hbase : 1 ≤ base) (hs : 1 ≤ s) {p : ℕ}
    (hp : 0 < p) {L : ℝ} (hL : 0 ≤ L) :
    Admissible s (p + 1) (fun _ => s) ∧ Admissible s (p + 1) (ntkFactor s (2 * (p + 1))) ∧
      Admissible s (p + 1) (yarnFactor base (2 * (p + 1)) s L) := by
  have hs125 : s ≤ 1.25 * s := by linarith
  refine ⟨⟨fun _ _ _ _ => le_rfl, fun _ _ => ⟨hs, hs125⟩⟩, ⟨fun i j hij _ => ?_, fun i hi => ?_⟩,
    ⟨fun i j hij _ => yarnFactor_monotone hbase hs _ hL hij, fun i _ => ?_⟩⟩
  · exact YaRN.ntk_factor_monotone hs (by omega) hij
  · have h := YaRN.ntk_factor_bounds hs hp (by omega : i ≤ p)
    exact ⟨h.1, h.2.trans hs125⟩
  · have h := (yarnFactor_bounds (base := base) hs (2 * (p + 1)) L i).2.2
    exact ⟨h.1, h.2.trans hs125⟩

example : Admissible 4 (63 + 1) (fun _ => 4) ∧ Admissible 4 (63 + 1) (ntkFactor 4 (2 * (63 + 1))) ∧
    Admissible 4 (63 + 1) (yarnFactor 10000 (2 * (63 + 1)) 4 4096) :=
  initial_population_admissible (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- §2.1's NTK factors `λ_i = s^i` leave Table 4's range at `i = 2` for `s ≥ 2`. -/
theorem printed_ntk_inadmissible {s : ℝ} (hs : 2 ≤ s) {p : ℕ} (hp : 2 < p) :
    ¬ Admissible s p (fun i => s ^ i) := fun h => by
  have h2 : s ^ 2 ≤ 1.25 * s := (h.2 2 hp).2
  nlinarith

example : ¬ Admissible 4 64 (fun i => (4 : ℝ) ^ i) :=
  printed_ntk_inadmissible (by norm_num) (by norm_num)

/-- §3.2: the NTK-aware factor `4^{1/63}` of the pair `1` at `s = 4`, `d = 128` is not on Table 4's
grid of step `0.01`. -/
theorem ntk_factor_off_grid : 1.02 < ntkFactor 4 128 1 ∧ ntkFactor 4 128 1 < 1.03 := by
  rw [ntkFactor, show 2 * ((1 : ℕ) : ℝ) / (((128 : ℕ) : ℝ) - 2) = ((1 : ℕ) : ℝ) / ((63 : ℕ) : ℝ) by
    norm_num]
  exact ⟨YaRN.lt_rpow_div (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    YaRN.rpow_div_lt (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.RoPE.LongRoPE
