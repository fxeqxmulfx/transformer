module

public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Data.Nat.Choose.Basic

/-!
# PT5: the parts that miss a line

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT5. Of the `m`-subsets of
`N` cells, `C(N − c, m)` miss a given set of `c` cells (`card_disjoint_powersetCard`). Their share is
at most `(1 − c/N)^m ≤ e^{−m c / N}` (`choose_mul_pow_le`, `choose_sub_le_exp`). Summed over all
`m`-subsets, a family of lines of `c` cells is missed `|J| C(N − c, m)` times
(`sum_card_disjoint`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset

/-- Drawing `m` cells from `a ≤ N` of them: `C(a, m) N^m ≤ a^m C(N, m)`, the share
`C(a, m) / C(N, m)` is at most `(a / N)^m`. -/
lemma choose_mul_pow_le {a N : ℕ} (h : a ≤ N) (m : ℕ) :
    a.choose m * N ^ m ≤ a ^ m * N.choose m := by
  induction m with
  | zero => simp
  | succ m ih =>
    have hstep : (a - m) * N ≤ a * (N - m) := by
      rcases le_or_gt m a with hm | hm
      · obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le hm
        obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le h
        have h1 : m + x - m = x := by omega
        have h2 : m + x + y - m = x + y := by omega
        rw [h1, h2]
        nlinarith [Nat.zero_le (m * y)]
      · simp [Nat.sub_eq_zero_of_le hm.le]
    refine Nat.le_of_mul_le_mul_right ?_ (Nat.succ_pos m)
    calc a.choose (m + 1) * N ^ (m + 1) * (m + 1)
        = (a.choose (m + 1) * (m + 1)) * N ^ (m + 1) := by ring
      _ = (a.choose m * N ^ m) * ((a - m) * N) := by rw [Nat.choose_succ_right_eq]; ring
      _ ≤ (a ^ m * N.choose m) * (a * (N - m)) := Nat.mul_le_mul ih hstep
      _ = a ^ (m + 1) * (N.choose m * (N - m)) := by ring
      _ = a ^ (m + 1) * N.choose (m + 1) * (m + 1) := by
        rw [← Nat.choose_succ_right_eq]; ring

example : (2 : ℕ).choose 1 * 4 ^ 1 ≤ 2 ^ 1 * (4 : ℕ).choose 1 := choose_mul_pow_le (by norm_num) 1

/-- **The share of the parts that miss `c` of `N` cells** (`review/flat_spectrum.md`, PT5):
`C(N − c, m) ≤ e^{−m c / N} C(N, m)`. -/
lemma choose_sub_le_exp {N c : ℕ} (hN : 0 < N) (hc : c ≤ N) (m : ℕ) :
    ((N - c).choose m : ℝ) ≤ Real.exp (-(m * ((c : ℝ) / N))) * N.choose m := by
  have hN' : (0 : ℝ) < N := Nat.cast_pos.2 hN
  have hx : (0 : ℝ) ≤ 1 - (c : ℝ) / N := by
    rw [sub_nonneg, div_le_one hN']
    exact_mod_cast hc
  have hreal : ((N - c).choose m : ℝ) * N ^ m ≤ (N * (1 - (c : ℝ) / N)) ^ m * N.choose m := by
    have h : ((N - c : ℕ) : ℝ) = N * (1 - (c : ℝ) / N) := by
      rw [Nat.cast_sub hc]
      field_simp
    rw [← h]
    exact_mod_cast choose_mul_pow_le (Nat.sub_le N c) m
  have hpow : (1 - (c : ℝ) / N) ^ m ≤ Real.exp (-(m * ((c : ℝ) / N))) := by
    calc (1 - (c : ℝ) / N) ^ m ≤ Real.exp (-((c : ℝ) / N)) ^ m :=
          pow_le_pow_left₀ hx (Real.one_sub_le_exp_neg _) m
      _ = Real.exp (-(m * ((c : ℝ) / N))) := by
        rw [← Real.exp_nat_mul]
        ring_nf
  have hle : ((N - c).choose m : ℝ) ≤ (1 - (c : ℝ) / N) ^ m * N.choose m := by
    refine le_of_mul_le_mul_right ?_ (pow_pos hN' m)
    calc ((N - c).choose m : ℝ) * N ^ m ≤ (N * (1 - (c : ℝ) / N)) ^ m * N.choose m := hreal
      _ = (1 - (c : ℝ) / N) ^ m * N.choose m * N ^ m := by rw [mul_pow]; ring
  exact hle.trans (mul_le_mul_of_nonneg_right hpow (Nat.cast_nonneg _))

example := choose_sub_le_exp (N := 4) (c := 2) (by norm_num) (by norm_num) 1

/-- The first-moment bound for two families of lines, in a table of `a × b` cells: `a` lines of `b`
cells and `b` lines of `a` cells. -/
lemma mul_choose_add_le {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (m : ℕ) :
    ((a * (a * b - b).choose m + b * (a * b - a).choose m : ℕ) : ℝ) ≤
      ((a : ℝ) * Real.exp (-((m : ℝ) / a)) + (b : ℝ) * Real.exp (-((m : ℝ) / b))) *
        ((a * b).choose m : ℝ) := by
  have hab : 0 < a * b := Nat.mul_pos ha hb
  have h1 := choose_sub_le_exp hab (Nat.le_mul_of_pos_left b ha) m
  have h2 := choose_sub_le_exp hab (Nat.le_mul_of_pos_right a hb) m
  have ha' : (a : ℝ) ≠ 0 := Nat.cast_ne_zero.2 ha.ne'
  have hb' : (b : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hb.ne'
  have e1 : (m : ℝ) * ((b : ℝ) / ((a * b : ℕ) : ℝ)) = (m : ℝ) / a := by
    push_cast
    field_simp
  have e2 : (m : ℝ) * ((a : ℝ) / ((a * b : ℕ) : ℝ)) = (m : ℝ) / b := by
    push_cast
    field_simp
  rw [e1] at h1
  rw [e2] at h2
  push_cast
  calc (a : ℝ) * ((a * b - b).choose m : ℝ) + (b : ℝ) * ((a * b - a).choose m : ℝ)
      ≤ (a : ℝ) * (Real.exp (-((m : ℝ) / a)) * ((a * b).choose m : ℝ)) +
          (b : ℝ) * (Real.exp (-((m : ℝ) / b)) * ((a * b).choose m : ℝ)) :=
        add_le_add (mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg a))
          (mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg b))
    _ = _ := by ring

example := mul_choose_add_le (a := 2) (b := 3) (by norm_num) (by norm_num) 1

variable {B J : Type*} [Fintype B] [DecidableEq B] [Fintype J]

/-- Of the `m`-subsets of the cells, `C(N − c, m)` miss a given set of `c` cells. -/
lemma card_disjoint_powersetCard (L : Finset B) (m : ℕ) :
    #{Ω ∈ powersetCard m (univ : Finset B) | Disjoint Ω L} =
      (Fintype.card B - #L).choose m := by
  have h : {Ω ∈ powersetCard m (univ : Finset B) | Disjoint Ω L} = powersetCard m Lᶜ := by
    ext Ω
    simp [mem_powersetCard, subset_compl_iff_disjoint_right, and_comm]
  rw [h, card_powersetCard, card_compl]

/-- **Counting the missed lines over all parts** (`review/flat_spectrum.md`, PT5): a family of lines
of `c` cells is missed `|J| C(N − c, m)` times in total. -/
lemma sum_card_disjoint (line : J → Finset B) {c : ℕ} (hc : ∀ j, #(line j) = c) (m : ℕ) :
    ∑ Ω ∈ powersetCard m (univ : Finset B), #{j | Disjoint Ω (line j)} =
      Fintype.card J * (Fintype.card B - c).choose m := by
  simp only [card_filter]
  rw [sum_comm]
  simp only [← card_filter, card_disjoint_powersetCard, hc, sum_const, card_univ, smul_eq_mul]

example := sum_card_disjoint (B := Fin 2) (J := Fin 2) (fun j => {j}) (c := 1)
  (fun _ => card_singleton _) 1

end Transformer.Grokking.FlatSpectrum.TablePart
