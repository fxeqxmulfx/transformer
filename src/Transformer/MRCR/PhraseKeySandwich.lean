module

public import Transformer.MRCR.PhraseKeyLogits
public import Transformer.MRCR.Returns
public import Transformer.Sandwich.Section6_Sandwich

/-!
# Sandwich has no window: a far key beats the token before

In no paper. Sandwich (arXiv:2212.10356v2, §6.1, Eq. (6)) adds `(p_m^⊤ p_n − d̄/2)/h =
(Σ_i cos((m − n)θ_i) − d̄/2)/h` to the logit "after the scaled softmax is done in Eq. (1)", that is
to the dot product divided by `√(d/H)` (`sandwichLogit`). The bias is `0` at the distance `0` and
below `0` at every other integer distance (`Transformer.Sandwich.sandwich_eq_zero_iff`), but it
does not decay. Enough positions for `exists_returns` leave a return `t ≥ N` that brings every
phase `t θ_i` near a multiple of `2π` (`exists_return_ge`); there the bias is near `0`, above its
value at the distance `1` (`exists_sum_cos_lt`, `exists_sandwich_lt`).

So for every query `i ≥ t` the key at the distance `t` gets a larger logit than the same key at
the distance `1` (`sandwichLogit_lt`), under SSMax as well (`ssmax_sandwichLogit_lt`). The weight
`round(exp ·)` is monotone in the logit, so no head of a rounded transformer with Sandwich attends
to the token before alone, and `ReadsPrev` fails for every divisor, compression ratio and dot
product (`not_readsPrev_sandwich`, `not_readsPrev_ssmax_sandwich`). The five layers of
`Transformer.MRCR.PhraseKeyLogits` therefore have no Sandwich version; whether several layers
with Sandwich read the token before by other means is open.

The inequality compares two distances, so it survives any shift of the bias, such as Appendix E's
reference implementation without `− d̄/2`, and `exists_sum_cos_lt` holds for any frequencies with
some `cos θ_i < 1`.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.CRASP Transformer.CRASP.Grams

variable {K : ℕ}

/-- **Returns as far as wanted**: for every `η > 0` some distance `d ≥ N` brings every phase
`θ_i d` within `η` of a multiple of `2π`, since `exists_returns` over enough positions finds more
returns than there are distances below `N`. -/
theorem exists_return_ge (θ : Fin K → ℝ) {η : ℝ} (hη : 0 < η) (N : ℕ) :
    ∃ d : ℕ, N ≤ d ∧ ∀ i, ∃ m : ℤ, |θ i * d - m * (2 * Real.pi)| ≤ η := by
  obtain ⟨M, hM⟩ : ∃ M : ℕ, 2 * Real.pi / ((M : ℝ) + 1) ≤ η := by
    refine ⟨⌈2 * Real.pi / η⌉₊, ?_⟩
    have h1 := Nat.le_ceil (2 * Real.pi / η)
    rw [div_le_iff₀ hη] at h1
    rw [div_le_iff₀ (by positivity)]
    nlinarith
  have hP : (0 : ℝ) < ((M : ℝ) + 1) ^ K := by positivity
  obtain ⟨R, hcard, hR⟩ := exists_returns θ M (H := (N + 2) * (M + 1) ^ K) (by positivity)
  have e : (((N + 2) * (M + 1) ^ K : ℕ) : ℝ) / ((M : ℝ) + 1) ^ K - 1 = N + 1 := by
    push_cast
    rw [mul_div_cancel_right₀ _ hP.ne']
    ring
  rw [e] at hcard
  by_contra hcon
  push Not at hcon
  have hsub : R ⊆ Ico 1 N := fun d hd => by
    obtain ⟨h1, -, hph⟩ := hR d hd
    refine mem_Ico.2 ⟨h1, not_le.1 fun hdN => ?_⟩
    obtain ⟨i, hi⟩ := hcon d hdN
    obtain ⟨m, hm⟩ := hph i
    exact absurd (hm.trans hM) (not_le.2 (hi m))
  have hle := card_le_card hsub
  rw [Nat.card_Ico] at hle
  have : (R.card : ℝ) ≤ N := by exact_mod_cast hle.trans (Nat.sub_le N 1)
  linarith

/-- One frequency `1`, accuracy `1`, past `5`. -/
example := exists_return_ge (fun _ : Fin 1 => (1 : ℝ)) one_pos 5

/-- **`Σ cos(t θ_i)` comes back above `Σ cos θ_i`** at distances `t` as large as wanted, when some
`cos θ_i < 1`. -/
theorem exists_sum_cos_lt (θ : Fin K → ℝ) (hθ : ∃ i, Real.cos (θ i) < 1) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ ∑ i, Real.cos (θ i) < ∑ i, Real.cos (t * θ i) := by
  obtain ⟨i₀, hi₀⟩ := hθ
  have hK : (0 : ℝ) < K := by exact_mod_cast i₀.pos
  have hg : 0 < ∑ i, (1 - Real.cos (θ i)) :=
    sum_pos' (fun i _ => sub_nonneg.2 (Real.cos_le_one _)) ⟨i₀, mem_univ _, sub_pos.2 hi₀⟩
  obtain ⟨t, htN, ht⟩ := exists_return_ge θ (div_pos hg (mul_pos two_pos hK)) N
  refine ⟨t, htN, ?_⟩
  have h := abs_ropeScore_add_sub_le (a := fun _ => 1) (b := fun _ => 0) ht 0
  have e1 : ropeScore θ (fun _ => 1) (fun _ => 0) (0 + t) = ∑ i, Real.cos (t * θ i) := by
    simp [ropeScore, mul_comm]
  have e2 : ropeScore θ (fun _ => 1) (fun _ => 0) 0 = K := by simp [ropeScore]
  have e3 : ropeNorm (fun _ : Fin K => (1 : ℝ)) (fun _ => 0) = K := by simp [ropeNorm]
  have e5 : (∑ i, (1 - Real.cos (θ i))) / (2 * K) * K = (∑ i, (1 - Real.cos (θ i))) / 2 := by
    field_simp
  rw [e1, e2, e3, e5] at h
  have e4 : ∑ i, Real.cos (θ i) = K - ∑ i, (1 - Real.cos (θ i)) := by simp [sum_sub_distrib]
  have := (abs_le.1 h).1
  rw [e4]
  linarith

/-- The hypothesis of `exists_sum_cos_lt` holds for the angle `π`. -/
example := exists_sum_cos_lt (fun _ : Fin 1 => Real.pi) ⟨0, by rw [Real.cos_pi]; norm_num⟩ 5

/-- **Sandwich's bias at the distance `1` is exceeded arbitrarily far away** (Eq. (6) with
`d̄ = 2D ≥ 2` and `h > 0`). -/
theorem exists_sandwich_lt {D : ℕ} (hD : 1 ≤ D) {h : ℝ} (hh : 0 < h) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ Sandwich.sandwich D h 1 < Sandwich.sandwich D h t := by
  have hlast : Real.cos (Sandwich.freq D ⟨D - 1, by omega⟩) < 1 := by
    rw [Sandwich.freq_last hD]
    refine lt_of_le_of_ne (Real.cos_le_one _) fun h1 => ?_
    have := (Sandwich.cos_div_eq_one_iff 1).1 (by simpa using h1)
    norm_num at this
  obtain ⟨t, htN, ht⟩ := exists_sum_cos_lt (Sandwich.freq D) ⟨_, hlast⟩ N
  refine ⟨t, htN, ?_⟩
  simp only [Sandwich.sandwich, one_mul]
  exact div_lt_div_of_pos_right (by linarith) hh

/-- The hypotheses of `exists_sandwich_lt` hold for `d̄ = 128` and `h = 8`. -/
example := exists_sandwich_lt (D := 64) (by norm_num) (h := 8) (by norm_num) 8192

/-- Sandwich's logit: the dot product divided by `c` (`√(d/H)` in Eq. (1)) plus Eq. (6) with
`d̄ = 2D` and the compression ratio `h` at the distance `i − j`. -/
noncomputable def sandwichLogit (c : ℝ) (D : ℕ) (h : ℝ) (i j : ℕ) (x : ℝ) : ℝ :=
  x / c + Sandwich.sandwich D h ((i : ℝ) - j)

/-- **A far key beats the token before under Sandwich**: a distance `t ≥ 2`, as large as wanted,
gives every dot product `x` a larger logit at the key `i − t` than at the key `i − 1`, for every
query `i ≥ t`. -/
theorem sandwichLogit_lt (c : ℝ) {D : ℕ} (hD : 1 ≤ D) {h : ℝ} (hh : 0 < h) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ 2 ≤ t ∧ ∀ i, t ≤ i → ∀ x,
      sandwichLogit c D h i (i - 1) x < sandwichLogit c D h i (i - t) x := by
  obtain ⟨t, ht, hlt⟩ := exists_sandwich_lt hD hh (max N 2)
  have ht2 : 2 ≤ t := le_of_max_le_right ht
  refine ⟨t, le_of_max_le_left ht, ht2, fun i hi x => ?_⟩
  have h1 : (i : ℝ) - ((i - 1 : ℕ) : ℝ) = 1 := by
    rw [Nat.cast_sub (by omega), Nat.cast_one]; ring
  have h2 : (i : ℝ) - ((i - t : ℕ) : ℝ) = t := by rw [Nat.cast_sub hi]; ring
  simp only [sandwichLogit, h1, h2]
  linarith

/-- The hypotheses of `sandwichLogit_lt` hold for `d̄ = 128`, `h = 8` and the divisor `8`. -/
example := sandwichLogit_lt 8 (D := 64) (by norm_num) (h := 8) (by norm_num) 512

/-- **The same under SSMax** (`eq:ssmax`), which multiplies both logits by `s log(i + 1) > 0`. -/
theorem ssmax_sandwichLogit_lt {s : ℝ} (hs : 0 < s) (c : ℝ) {D : ℕ} (hD : 1 ≤ D) {h : ℝ}
    (hh : 0 < h) (N : ℕ) :
    ∃ t : ℕ, N ≤ t ∧ 2 ≤ t ∧ ∀ i, t ≤ i → ∀ x,
      ssmaxLogit s (sandwichLogit c D h) i (i - 1) x <
        ssmaxLogit s (sandwichLogit c D h) i (i - t) x := by
  obtain ⟨t, htN, ht2, hlt⟩ := sandwichLogit_lt c hD hh N
  refine ⟨t, htN, ht2, fun i hi x => ?_⟩
  have hi2 : (2 : ℝ) ≤ i := by exact_mod_cast ht2.trans hi
  have hlog : 0 < Real.log ((i : ℝ) + 1) := Real.log_pos (by linarith)
  exact mul_lt_mul_of_pos_left (hlt i hi x) (mul_pos hs hlog)

/-- The hypotheses of `ssmax_sandwichLogit_lt` hold for `s = 1`, `d̄ = 128` and `h = 8`. -/
example := ssmax_sandwichLogit_lt one_pos 8 (D := 64) (by norm_num) (h := 8) (by norm_num) 512

/-- `ReadsPrev L b` fails once a far key gets a logit at least the previous key's. -/
theorem not_readsPrev_of_le {L : ℕ → ℕ → ℝ → ℝ} {b i t : ℕ} (ht : 2 ≤ t) (hti : t ≤ i)
    (hle : L i (i - 1) b ≤ L i (i - t) b) : ¬ ReadsPrev L b := fun hL => by
  have h0 := hL.prev i (by omega)
  have h1 := (hL.far i (i - t) (by omega)).2
  linarith

/-- The hypotheses of `not_readsPrev_of_le` hold for the logit `0`. -/
example : ¬ ReadsPrev (fun _ _ _ => 0) 1 := not_readsPrev_of_le (i := 2) le_rfl le_rfl le_rfl

/-- **Sandwich does not meet `ReadsPrev`**, whatever the divisor, the compression ratio `h > 0`
and the dot product `b`. -/
theorem not_readsPrev_sandwich (c : ℝ) {D : ℕ} (hD : 1 ≤ D) {h : ℝ} (hh : 0 < h) (b : ℕ) :
    ¬ ReadsPrev (sandwichLogit c D h) b := by
  obtain ⟨t, -, ht2, hlt⟩ := sandwichLogit_lt c hD hh 0
  exact not_readsPrev_of_le ht2 le_rfl (hlt t le_rfl b).le

/-- The hypotheses of `not_readsPrev_sandwich` hold for `d̄ = 128`, `h = 8` and the divisor `8`. -/
example := not_readsPrev_sandwich 8 (D := 64) (by norm_num) (h := 8) (by norm_num) 1

/-- **Nor does SSMax of Sandwich.** -/
theorem not_readsPrev_ssmax_sandwich {s : ℝ} (hs : 0 < s) (c : ℝ) {D : ℕ} (hD : 1 ≤ D) {h : ℝ}
    (hh : 0 < h) (b : ℕ) : ¬ ReadsPrev (ssmaxLogit s (sandwichLogit c D h)) b := by
  obtain ⟨t, -, ht2, hlt⟩ := ssmax_sandwichLogit_lt hs c hD hh 0
  exact not_readsPrev_of_le ht2 le_rfl (hlt t le_rfl b).le

/-- The hypotheses of `not_readsPrev_ssmax_sandwich` hold for `s = 1`, `d̄ = 128` and `h = 8`. -/
example := not_readsPrev_ssmax_sandwich one_pos 8 (D := 64) (by norm_num) (h := 8) (by norm_num) 1

end Transformer.MRCR
