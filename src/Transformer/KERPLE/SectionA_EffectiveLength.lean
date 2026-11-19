module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Transformer.KERPLE.Section4_Attention

/-!
# Appendix A.5: effective lengths

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.5 (Additional Analyses): the
*effective length* of a head is `eff = min_{k̃(0,|m−n|) < −2} |m − n|`, "the relative positional
difference `|m − n|` such that `k̃(m, n) = k̃(0, |m − n|)` just becomes smaller than `−2`. Note
`k̃(0, |m − n|)` strictly decreases in `|m − n|`, so there is only one possible value."

* `effLength`: the least distance `d ∈ ℕ` with `k̃(0, d) < −2`. For a kernel decreasing in the
  distance and reaching below `−2`, `k̃(0, d) < −2` exactly when `d ≥ eff` (`effLength_le_iff`):
  below `eff` the bias is at least `−2`, from `eff` on it is below `−2`.
* The power and logarithmic variants strictly decrease in the distance (`strictAnti_powerKernel`,
  `strictAnti_logKernel`), and their effective lengths are `⌊(2/r₁)^{1/r₂}⌋ + 1` and
  `⌊(e^{2/r₁} − 1)/r₂⌋ + 1` (`effLength_powerKernel`, `effLength_logKernel`). ALiBi's slope
  `2^{−8h/H}` at head `h` gives `⌊2^{1+8h/H}⌋ + 1`, whatever the dataset (`effLength_alibi`).
* Interpretation of Curves: the number of heads with `eff ≤ x` is the number of heads whose bias is
  below `−2` at every distance `d ≥ x` (`card_effLength_le`), and it grows with `x`
  (`monotone_card_effLength_le`).
-/

@[expose] public section

namespace Transformer.KERPLE

/-- The *effective length* of a kernel `k̃` on positions (A.5): the least distance `d ∈ ℕ` with
`k̃(0, d) < −2`. -/
noncomputable def effLength (kt : ℝ → ℝ → ℝ) : ℕ := sInf {d : ℕ | kt 0 d < -2}

/-- **The effective length is the one crossing of `−2`** (A.5): for a kernel decreasing in the
distance that reaches below `−2`, `k̃(0, d) < −2` exactly when `d ≥ eff`. -/
theorem effLength_le_iff {kt : ℝ → ℝ → ℝ} (hanti : Antitone fun d : ℕ => kt 0 d)
    (hne : ∃ d : ℕ, kt 0 d < -2) (d : ℕ) : effLength kt ≤ d ↔ kt 0 d < -2 :=
  ⟨fun h => lt_of_le_of_lt (hanti h) (Nat.sInf_mem (s := {d : ℕ | kt 0 d < -2}) hne),
    fun h => Nat.sInf_le h⟩

/-- **The power variant strictly decreases in the distance** (A.5) for `r₁, r₂ > 0`. -/
theorem strictAnti_powerKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    StrictAnti fun d : ℕ => powerKernel r₁ r₂ 0 d := fun a b hab => by
  simp only [powerKernel, zero_sub, abs_neg, Nat.abs_cast, neg_mul, neg_lt_neg_iff]
  exact mul_lt_mul_of_pos_left
    (Real.rpow_lt_rpow (Nat.cast_nonneg a) (by exact_mod_cast hab) hr₂) hr₁

/-- The hypotheses of `strictAnti_powerKernel` hold for ALiBi's `r₂ = 1`. -/
example : StrictAnti fun d : ℕ => powerKernel 1 1 0 d :=
  strictAnti_powerKernel one_pos one_pos

/-- **The logarithmic variant strictly decreases in the distance** (A.5) for `r₁, r₂ > 0`. -/
theorem strictAnti_logKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    StrictAnti fun d : ℕ => logKernel r₁ r₂ 0 d := fun a b hab => by
  simp only [logKernel, zero_sub, abs_neg, Nat.abs_cast, neg_mul, neg_lt_neg_iff]
  have hab' : (a : ℝ) < b := by exact_mod_cast hab
  exact mul_lt_mul_of_pos_left (Real.log_lt_log (by positivity) (by nlinarith)) hr₁

/-- The hypotheses of `strictAnti_logKernel` hold for `r₁ = r₂ = 1`. -/
example : StrictAnti fun d : ℕ => logKernel 1 1 0 d :=
  strictAnti_logKernel one_pos one_pos

/-- The hypotheses of `effLength_le_iff` hold for ALiBi's kernel `−|m − n|`. -/
example (d : ℕ) : effLength (powerKernel 1 1) ≤ d ↔ powerKernel 1 1 0 d < -2 :=
  effLength_le_iff (strictAnti_powerKernel one_pos one_pos).antitone
    ⟨3, by norm_num [powerKernel]⟩ d

/-- The power variant's bias is below `−2` exactly from the distance `⌊(2/r₁)^{1/r₂}⌋ + 1` on. -/
theorem powerKernel_lt_neg_two_iff {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (d : ℕ) :
    powerKernel r₁ r₂ 0 d < -2 ↔ ⌊(2 / r₁) ^ r₂⁻¹⌋₊ + 1 ≤ d := by
  rw [← Nat.lt_iff_add_one_le, Nat.floor_lt (by positivity),
    Real.rpow_inv_lt_iff_of_pos (by positivity) (Nat.cast_nonneg d) hr₂, div_lt_iff₀ hr₁]
  simp only [powerKernel, zero_sub, abs_neg, Nat.abs_cast]
  constructor <;> intro h <;> linarith

/-- **The power variant's effective length** is `⌊(2/r₁)^{1/r₂}⌋ + 1` for `r₁, r₂ > 0`. -/
theorem effLength_powerKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    effLength (powerKernel r₁ r₂) = ⌊(2 / r₁) ^ r₂⁻¹⌋₊ + 1 := by
  have h : {d : ℕ | powerKernel r₁ r₂ 0 d < -2} = Set.Ici (⌊(2 / r₁) ^ r₂⁻¹⌋₊ + 1) :=
    Set.ext fun d => powerKernel_lt_neg_two_iff hr₁ hr₂ d
  rw [effLength, h, csInf_Ici]

/-- The hypotheses of `effLength_powerKernel` hold for ALiBi's slope `1/2`: the effective length
is `5`. -/
example : effLength (powerKernel (1 / 2) 1) = 5 := by
  rw [effLength_powerKernel (by norm_num) one_pos]
  norm_num

/-- The logarithmic variant's bias is below `−2` exactly from the distance
`⌊(e^{2/r₁} − 1)/r₂⌋ + 1` on. -/
theorem logKernel_lt_neg_two_iff {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (d : ℕ) :
    logKernel r₁ r₂ 0 d < -2 ↔ ⌊(Real.exp (2 / r₁) - 1) / r₂⌋₊ + 1 ≤ d := by
  have he : 1 ≤ Real.exp (2 / r₁) := Real.one_le_exp (by positivity)
  have hpos : 0 < 1 + r₂ * d := by positivity
  rw [← Nat.lt_iff_add_one_le, Nat.floor_lt (div_nonneg (by linarith) hr₂.le), div_lt_iff₀ hr₂,
    sub_lt_iff_lt_add', mul_comm, ← Real.lt_log_iff_exp_lt hpos, div_lt_iff₀ hr₁]
  simp only [logKernel, zero_sub, abs_neg, Nat.abs_cast]
  constructor <;> intro h <;> linarith

/-- **The logarithmic variant's effective length** is `⌊(e^{2/r₁} − 1)/r₂⌋ + 1` for
`r₁, r₂ > 0`. -/
theorem effLength_logKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    effLength (logKernel r₁ r₂) = ⌊(Real.exp (2 / r₁) - 1) / r₂⌋₊ + 1 := by
  have h : {d : ℕ | logKernel r₁ r₂ 0 d < -2} =
      Set.Ici (⌊(Real.exp (2 / r₁) - 1) / r₂⌋₊ + 1) :=
    Set.ext fun d => logKernel_lt_neg_two_iff hr₁ hr₂ d
  rw [effLength, h, csInf_Ici]

/-- The hypotheses of `effLength_logKernel` hold for `r₁ = r₂ = 1`: the effective length is
`⌊e² − 1⌋ + 1 = 7`. -/
example : effLength (logKernel 1 1) = 7 := by
  rw [effLength_logKernel one_pos one_pos, div_one, div_one]
  have he1 := Real.exp_one_gt_d9
  have he2 := Real.exp_one_lt_d9
  have h2 : Real.exp 2 = Real.exp 1 ^ 2 := by rw [sq, ← Real.exp_add]; norm_num
  have hfl : ⌊Real.exp 2 - 1⌋₊ = 6 :=
    (Nat.floor_eq_iff (by nlinarith)).mpr ⟨by push_cast; nlinarith, by push_cast; nlinarith⟩
  rw [hfl]

/-- **ALiBi's effective lengths** (A.5, The Advantage of Learnable Parameters): ALiBi's head `h`
of `H` has the slope `r = 2^{−8h/H}`, and its effective length `⌊2^{1+8h/H}⌋ + 1` depends on `h`
and `H` alone: "ALiBi produces the same curve no matter which dataset is used". -/
theorem effLength_alibi (h H : ℕ) :
    effLength (powerKernel (2 ^ (-(8 * h / H : ℝ))) 1) = ⌊(2 : ℝ) ^ (1 + 8 * h / H : ℝ)⌋₊ + 1 := by
  rw [effLength_powerKernel (by positivity) one_pos, inv_one, Real.rpow_one,
    Real.rpow_neg zero_le_two, div_inv_eq_mul, Real.rpow_add two_pos, Real.rpow_one]

/-- ALiBi's last head of `12` has the slope `2^{−8}` and the effective length `513`. -/
example : effLength (powerKernel (2 ^ (-(8 * (12 : ℕ) / (12 : ℕ) : ℝ))) 1) = 513 := by
  rw [effLength_alibi]
  norm_num

variable {H : ℕ}

open scoped Classical in
/-- **Interpretation of Curves** (A.5): the number of heads with `eff^{(h)} ≤ x` is the number of
heads whose bias is below `−2` at every distance `d ≥ x`, "`y` heads with at least `−2` bias when
the token distance `|m − n|` is greater than `x`". -/
theorem card_effLength_le {kt : Fin H → ℝ → ℝ → ℝ} (hanti : ∀ h, Antitone fun d : ℕ => kt h 0 d)
    (hne : ∀ h, ∃ d : ℕ, kt h 0 d < -2) (x : ℕ) :
    (Finset.univ.filter fun h => effLength (kt h) ≤ x).card =
      (Finset.univ.filter fun h => ∀ d : ℕ, x ≤ d → kt h 0 d < -2).card := by
  refine congrArg Finset.card (Finset.filter_congr fun h _ => ?_)
  rw [effLength_le_iff (hanti h) (hne h)]
  exact ⟨fun hx d hd => lt_of_le_of_lt (hanti h hd) hx, fun hx => hx x le_rfl⟩

open scoped Classical in
/-- The hypotheses of `card_effLength_le` hold for two heads of slopes `1` and `1/2`. -/
example (x : ℕ) :
    (Finset.univ.filter fun h : Fin 2 => effLength (powerKernel (1 / (h + 1)) 1) ≤ x).card =
      (Finset.univ.filter fun h : Fin 2 =>
        ∀ d : ℕ, x ≤ d → powerKernel (1 / (h + 1)) 1 0 d < -2).card :=
  card_effLength_le (fun h => (strictAnti_powerKernel (by positivity) one_pos).antitone)
    (fun h => ⟨_, (powerKernel_lt_neg_two_iff (by positivity) one_pos _).mpr le_rfl⟩) x

/-- **The curves are cumulative** (A.5): the number of heads with `eff^{(h)} ≤ x` grows with `x`. -/
theorem monotone_card_effLength_le (kt : Fin H → ℝ → ℝ → ℝ) :
    Monotone fun x : ℕ => (Finset.univ.filter fun h => effLength (kt h) ≤ x).card :=
  fun _ _ hxy => Finset.card_le_card
    (Finset.monotone_filter_right _ fun _ _ hh => le_trans hh hxy)

end Transformer.KERPLE
