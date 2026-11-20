module

public import Transformer.RoPE.LongRoPE2.Section3_Candidates

/-!
# LongRoPE2 §3.2, Algorithms 1–2: the factor vector of the search

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, §3.2: "we restrict the search to dimensions
`i ≥ d_rcd`, while applying NTK-aware scaling to lower dimensions (`i < d_rcd`) using the adjusted
base value derived from `d_rcd`"; the higher group "performs mutation with probability `p` under
the monotonic non-decreasing constraint: `λ_i ≤ λ_{i+1}`"; "rescaling factors above `L/L_train`
are randomly sampled for dimension `i ≥ d_rcd`", from Algorithm 2's search space
`SP = [L/L_train, 2×L/L_train]`.

* `adjustedBase`, `ntk_lower_group`: Algorithm 2, line 7,
  `θ_drcd = 1/(λ_right[0]×θbase^{2×drcd/d})`: the base `θbase·s^{d/(2c)}` divides the frequency
  of the pair `c` by `s` and gives every pair `i` the factor `s^{i/c}`;
* `searchedFactor`, `searchedFactor_monotone`: the upper group from the pair `c = d_rcd` on and
  the lower group's NTK factors anchored at the upper group's first factor: a non-decreasing upper
  group from a first factor `≥ 1` gives a non-decreasing vector;
* `searched_meets_eq10`: when `d_rcd ≤ d_tcd` and the upper factors are at least `L/L_train`,
  Eq. (10) holds at every pair whose period is at least `L_train`;
* `printedInitFactor`, `printed_init_not_monotone`: Algorithm 1, lines 5–7 as printed, set
  `λ[d_rcd : d/2 − 1] = s` and anchor the lower group's NTK at `θ_{d10_tcd}`, so the pair `d10_tcd`
  gets `s`: for `d_rcd ≥ d10_tcd + 2` the pair `d10_tcd + 1` gets `s^{(d10_tcd+1)/d10_tcd} > s`, the
  factor of `d_rcd`, and the initial vector breaks the constraint. Anchored at `d_rcd`, as in
  Algorithm 2 and the text, it is `searchedFactor` with a constant upper group, non-decreasing.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

open ScalingLaws

/-- Algorithm 2, line 7: the NTK base `θbase·s^{d/(2c)}` derived from the pair `c`. -/
noncomputable def adjustedBase (base s : ℝ) (d c : ℕ) : ℝ := base * s ^ ((d : ℝ) / (2 * c))

/-- Algorithm 2, line 7: the adjusted base divides the frequency of the pair `c` by `s` and
gives the pair `i` the factor `s^{i/c}`. -/
theorem ntk_lower_group {base s : ℝ} {d c : ℕ} (hbase : 0 < base) (hs : 0 < s) (hd : 0 < d)
    (hc : 0 < c) :
    frequency (adjustedBase base s d c) d c = frequency base d c / s ∧
      ∀ i : ℕ, frequency base d i / frequency (adjustedBase base s d c) d i =
        s ^ ((i : ℝ) / c) := by
  have hd' : (d : ℝ) ≠ 0 := by positivity
  have hc' : (c : ℝ) ≠ 0 := by positivity
  have key : ∀ i : ℕ, frequency (adjustedBase base s d c) d i =
      frequency base d i * (s ^ ((i : ℝ) / c))⁻¹ := by
    intro i
    rw [frequency, frequency, adjustedBase, Real.mul_rpow hbase.le (Real.rpow_nonneg hs.le _),
      ← Real.rpow_mul hs.le, ← Real.rpow_neg hs.le]
    congr 2
    field_simp
  refine ⟨by rw [key c, div_self hc', Real.rpow_one, div_eq_mul_inv], fun i => ?_⟩
  have hθ : 0 < frequency base d i := Real.rpow_pos_of_pos hbase _
  rw [key i, div_mul_eq_div_div, div_self hθ.ne', one_div, inv_inv]

example : frequency (adjustedBase 10000 64 96 25) 96 25 = frequency 10000 96 25 / 64 ∧
    ∀ i : ℕ, frequency 10000 96 i / frequency (adjustedBase 10000 64 96 25) 96 i =
      (64 : ℝ) ^ ((i : ℝ) / (25 : ℕ)) :=
  ntk_lower_group (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Algorithms 1–2: the factor `λ_i` with the upper group `upper` from the pair `c = d_rcd` on
and the NTK factors `upper(c)^{i/c}` below it. -/
noncomputable def searchedFactor (c : ℕ) (upper : ℕ → ℝ) (i : ℕ) : ℝ :=
  if i < c then upper c ^ ((i : ℝ) / c) else upper i

/-- §3.2: a non-decreasing upper group from a first factor `≥ 1` gives a non-decreasing vector. -/
theorem searchedFactor_monotone {c : ℕ} {upper : ℕ → ℝ} (h1 : 1 ≤ upper c)
    (hmono : ∀ i j, c ≤ i → i ≤ j → upper i ≤ upper j) : Monotone (searchedFactor c upper) := by
  intro i j hij
  by_cases hj : j < c
  · rw [searchedFactor, searchedFactor, ite_eq_left (lt_of_le_of_lt hij hj), ite_eq_left hj]
    exact Real.rpow_le_rpow_of_exponent_le h1
      (div_le_div_of_nonneg_right (by exact_mod_cast hij) (Nat.cast_nonneg c))
  · rw [searchedFactor, searchedFactor, ite_eq_right hj]
    by_cases hi : i < c
    · rw [ite_eq_left hi]
      have hc : (0 : ℝ) < c := by exact_mod_cast (show 0 < c by omega)
      calc upper c ^ ((i : ℝ) / c) ≤ upper c ^ (1 : ℝ) :=
            Real.rpow_le_rpow_of_exponent_le h1 ((div_le_one hc).mpr (by exact_mod_cast hi.le))
        _ = upper c := Real.rpow_one _
        _ ≤ upper j := hmono c j le_rfl (not_lt.mp hj)
    · rw [ite_eq_right hi]
      exact hmono i j (not_lt.mp hi) hij

example : Monotone (searchedFactor 25 fun _ => 64) :=
  searchedFactor_monotone (by norm_num) fun _ _ _ _ => le_rfl

/-- §3.2: with `d_rcd ≤ d_tcd` and upper factors at least `L/L_train`, Eq. (10) holds at every
pair whose period is at least `L_train`. -/
theorem searched_meets_eq10 {base train target : ℝ} {d c : ℕ} {upper : ℕ → ℝ}
    (hbase : 1 < base) (htrain : 0 < train) (hd : 0 < d)
    (hc : (c : ℤ) ≤ criticalPair base d train) (hupper : ∀ i, c ≤ i → target / train ≤ upper i)
    {i : ℕ} (hi : train ≤ period (frequency base d i)) :
    target / train ≤ searchedFactor c upper i := by
  have hci : c ≤ i := by
    have h := mt (period_lt_window_iff_critical hbase htrain hd i).2 (not_lt.mpr hi)
    omega
  rw [searchedFactor, ite_eq_right (not_lt.mpr hci)]
  exact hupper i hci

example : (131072 : ℝ) / 2048 ≤ searchedFactor 25 (fun _ => 64) 40 :=
  searched_meets_eq10 (base := 10000) (d := 96) (by norm_num) (by norm_num) (by norm_num)
    (by rw [phi3_critical.1]; norm_num) (fun _ _ => by norm_num)
    (not_lt.mp fun h => absurd ((phi3_critical.2.2 40).1 h) (by omega))

/-- Algorithm 1, lines 5–7 as printed: `s` from the pair `c = d_rcd` on and the NTK factors
anchored at the pair `a = d10_tcd` below it. -/
noncomputable def printedInitFactor (s : ℝ) (a c i : ℕ) : ℝ :=
  if i < c then s ^ ((i : ℝ) / a) else s

/-- Algorithm 1, line 6 as printed: for `c ≥ a + 2` the pair `a + 1` gets a larger factor than
the pair `c`, so the initial vector is not non-decreasing. -/
theorem printed_init_not_monotone {s : ℝ} {a c : ℕ} (hs : 1 < s) (ha : 0 < a)
    (hac : a + 1 < c) :
    printedInitFactor s a c c < printedInitFactor s a c (a + 1) ∧
      ¬ Monotone (printedInitFactor s a c) := by
  have hlt : printedInitFactor s a c c < printedInitFactor s a c (a + 1) := by
    rw [printedInitFactor, printedInitFactor, ite_eq_right (lt_irrefl c), ite_eq_left hac]
    have ha' : (0 : ℝ) < a := by exact_mod_cast ha
    have hexp : (1 : ℝ) < ((a + 1 : ℕ) : ℝ) / a := by
      rw [one_lt_div ha']
      push_cast
      linarith
    calc s = s ^ (1 : ℝ) := (Real.rpow_one s).symm
      _ < s ^ (((a + 1 : ℕ) : ℝ) / a) := Real.rpow_lt_rpow_of_exponent_lt hs hexp
  exact ⟨hlt, fun hmono => absurd (hmono hac.le) (not_le.mpr hlt)⟩

example : printedInitFactor 64 19 25 25 < printedInitFactor 64 19 25 20 ∧
    ¬ Monotone (printedInitFactor 64 19 25) :=
  printed_init_not_monotone (by norm_num) (by norm_num) (by norm_num)

end Transformer.RoPE.LongRoPE2
