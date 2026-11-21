module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatThreshold
public import Transformer.Grokking.FlatSpectrum.TablePart.RandomCount

/-!
# PT5: a random part of the flat table

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT5. A part with an empty
row or column does not determine the flat class (`not_determines_of_emptyLines`). Over all parts of
`m` cells the empty lines number exactly `q^k C(N − q^l, m) + q^l C(N − q^k, m)` (`sum_emptyLines`),
so on average at most `q^k e^{−m/q^k} + q^l e^{−m/q^l}` (`sum_emptyLines_le`). At `p = 97` this is
`194 e^{−m/97}`, which is below 1 exactly from `m = 511` (`mul_exp_lt_one_iff`), the first integer
above `97 ln 194` (`mul_log_194_mem`). The source writes "about" where this is an upper bound. The
table of random parts in the source stays a measurement.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset

variable {F ι κ : Type*} [Fintype F] [DecidableEq F]
  [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

/-- The cells of the row `u`. -/
def rowCells (u : ι → F) : Finset ((ι → F) × (κ → F)) := univ.image fun v => (u, v)

/-- The cells of the column `v`. -/
def colCells (v : κ → F) : Finset ((ι → F) × (κ → F)) := univ.image fun u => (u, v)

omit [DecidableEq ι] in
lemma card_rowCells (u : ι → F) :
    #(rowCells (κ := κ) u) = Fintype.card F ^ Fintype.card κ := by
  rw [rowCells, card_image_of_injective _ fun _ _ h => (Prod.ext_iff.1 h).2, card_univ,
    Fintype.card_fun]

omit [DecidableEq κ] in
lemma card_colCells (v : κ → F) :
    #(colCells (ι := ι) v) = Fintype.card F ^ Fintype.card ι := by
  rw [colCells, card_image_of_injective _ fun _ _ h => (Prod.ext_iff.1 h).1, card_univ,
    Fintype.card_fun]

/-- The number of empty rows and columns of a part (`review/flat_spectrum.md`, PT5). -/
def emptyLines (Ω : Finset ((ι → F) × (κ → F))) : ℕ :=
  #{u | Disjoint Ω (rowCells u)} + #{v | Disjoint Ω (colCells v)}

/-- **A part with an empty line does not determine the flat class** (`review/flat_spectrum.md`,
PT5). -/
theorem not_determines_of_emptyLines [Field F] {Ω : Finset ((ι → F) × (κ → F))}
    (h : emptyLines Ω ≠ 0) :
    ¬ Determines (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) Ω := by
  intro hd
  rcases Nat.eq_zero_or_pos #{u | Disjoint Ω (rowCells (κ := κ) u)} with h0 | hpos
  · have hcol : 0 < #{v | Disjoint Ω (colCells (ι := ι) v)} := by
      unfold emptyLines at h
      omega
    obtain ⟨v, hv⟩ := card_pos.1 hcol
    refine not_determines_off_col (ι := ι) v (hd.mono fun c hc hcv => ?_)
    exact disjoint_left.1 (mem_filter.1 hv).2 (mem_coe.1 hc)
      (mem_image.2 ⟨c.1, mem_univ _, Prod.ext rfl hcv.symm⟩)
  · obtain ⟨u, hu⟩ := card_pos.1 hpos
    refine not_determines_off_row (κ := κ) u (hd.mono fun c hc hcu => ?_)
    exact disjoint_left.1 (mem_filter.1 hu).2 (mem_coe.1 hc)
      (mem_image.2 ⟨c.2, mem_univ _, Prod.ext hcu.symm rfl⟩)

example : ¬ Determines (flatClass (ZMod 2) (Fin 1) (Fin 1) :
    Set ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2) → ZMod 2))
    (∅ : Finset ((Fin 1 → ZMod 2) × (Fin 1 → ZMod 2))) :=
  not_determines_of_emptyLines (by simp [emptyLines])

/-- **The empty lines over all parts of `m` cells**, counted exactly (`review/flat_spectrum.md`,
PT5). -/
theorem sum_emptyLines (m : ℕ) :
    ∑ Ω ∈ powersetCard m (univ : Finset ((ι → F) × (κ → F))), emptyLines Ω =
      Fintype.card F ^ Fintype.card ι *
          (Fintype.card F ^ Fintype.card ι * Fintype.card F ^ Fintype.card κ -
            Fintype.card F ^ Fintype.card κ).choose m +
        Fintype.card F ^ Fintype.card κ *
          (Fintype.card F ^ Fintype.card ι * Fintype.card F ^ Fintype.card κ -
            Fintype.card F ^ Fintype.card ι).choose m := by
  simp only [emptyLines, sum_add_distrib]
  rw [sum_card_disjoint _ (card_rowCells (κ := κ)), sum_card_disjoint _ (card_colCells (ι := ι))]
  simp only [Fintype.card_prod, Fintype.card_fun]

/-- **The first-moment bound** (`review/flat_spectrum.md`, PT5): the average number of empty lines
of a part of `m` cells is at most `q^k e^{−m/q^k} + q^l e^{−m/q^l}`. -/
theorem sum_emptyLines_le [Nonempty F] (m : ℕ) :
    ((∑ Ω ∈ powersetCard m (univ : Finset ((ι → F) × (κ → F))), emptyLines Ω : ℕ) : ℝ) ≤
      (((Fintype.card F ^ Fintype.card ι : ℕ) : ℝ) *
            Real.exp (-((m : ℝ) / (Fintype.card F ^ Fintype.card ι : ℕ))) +
          ((Fintype.card F ^ Fintype.card κ : ℕ) : ℝ) *
            Real.exp (-((m : ℝ) / (Fintype.card F ^ Fintype.card κ : ℕ)))) *
        ((Fintype.card F ^ Fintype.card ι * Fintype.card F ^ Fintype.card κ).choose m : ℝ) := by
  rw [sum_emptyLines]
  exact mul_choose_add_le (pow_pos Fintype.card_pos _) (pow_pos Fintype.card_pos _) m

/-- The parts with an empty line are at most the empty lines counted over all parts. -/
lemma card_emptyLine_le (m : ℕ) :
    #{Ω ∈ powersetCard m (univ : Finset ((ι → F) × (κ → F))) | emptyLines Ω ≠ 0} ≤
      ∑ Ω ∈ powersetCard m (univ : Finset ((ι → F) × (κ → F))), emptyLines Ω := by
  rw [card_filter]
  refine sum_le_sum fun Ω _ => ?_
  split_ifs with h
  · exact Nat.one_le_iff_ne_zero.2 h
  · exact Nat.zero_le _

/-- At `m = p ln 2p` the bound `2 p e^{−m/p}` equals 1 (`review/flat_spectrum.md`, PT5). -/
lemma two_mul_exp_crossing {p : ℝ} (hp : 0 < p) :
    2 * p * Real.exp (-(p * Real.log (2 * p) / p)) = 1 := by
  rw [mul_div_cancel_left₀ _ hp.ne', Real.exp_neg, Real.exp_log (by positivity)]
  field_simp

example : 2 * 97 * Real.exp (-(97 * Real.log (2 * 97) / 97)) = (1 : ℝ) :=
  two_mul_exp_crossing (by norm_num)

set_option exponentiation.threshold 1000 in
/-- `e^{510/97} < 194`. -/
lemma exp_510_div_97_lt : Real.exp (510 / 97) < 194 := by
  have h : Real.exp (510 / 97) ^ 97 = Real.exp 1 ^ 510 := by
    rw [← Real.exp_nat_mul, ← Real.exp_nat_mul]
    norm_num
  refine lt_of_pow_lt_pow_left₀ 97 (by norm_num) ?_
  rw [h]
  calc Real.exp 1 ^ 510 < 2.7182818286 ^ 510 :=
        pow_lt_pow_left₀ Real.exp_one_lt_d9 (Real.exp_pos 1).le (by norm_num)
    _ < 194 ^ 97 := by norm_num

set_option exponentiation.threshold 1000 in
/-- `194 < e^{511/97}`. -/
lemma lt_exp_511_div_97 : 194 < Real.exp (511 / 97) := by
  have h : Real.exp (511 / 97) ^ 97 = Real.exp 1 ^ 511 := by
    rw [← Real.exp_nat_mul, ← Real.exp_nat_mul]
    norm_num
  refine lt_of_pow_lt_pow_left₀ 97 (Real.exp_pos _).le ?_
  rw [h]
  calc (194 : ℝ) ^ 97 < 2.7182818283 ^ 511 := by norm_num
    _ < Real.exp 1 ^ 511 :=
        pow_lt_pow_left₀ Real.exp_one_gt_d9 (by norm_num) (by norm_num)

/-- **`p ln 2p ≈ 511` at `p = 97`** (`review/flat_spectrum.md`, PT5). -/
theorem mul_log_194_mem : 510 < 97 * Real.log 194 ∧ 97 * Real.log 194 < 511 := by
  constructor
  · have h := (Real.lt_log_iff_exp_lt (by norm_num)).2 exp_510_div_97_lt
    linarith
  · have h := (Real.log_lt_iff_lt_exp (by norm_num)).2 lt_exp_511_div_97
    linarith

/-- **The bound falls below 1 exactly from `m = 511`** (`review/flat_spectrum.md`, PT5). -/
theorem mul_exp_lt_one_iff (m : ℕ) : 194 * Real.exp (-((m : ℝ) / 97)) < 1 ↔ 511 ≤ m := by
  have hE (x : ℝ) : 194 * Real.exp (-x) < 1 ↔ 194 < Real.exp x := by
    rw [Real.exp_neg, ← div_eq_mul_inv, div_lt_one (Real.exp_pos x)]
  rw [hE]
  constructor
  · intro h
    by_contra hm
    have hm' : (m : ℝ) ≤ 510 := by exact_mod_cast Nat.lt_succ_iff.1 (not_le.1 hm)
    have hle : Real.exp ((m : ℝ) / 97) ≤ Real.exp (510 / 97) :=
      Real.exp_le_exp.2 (by linarith)
    linarith [exp_510_div_97_lt]
  · intro hm
    have hm' : (511 : ℝ) ≤ m := by exact_mod_cast hm
    have hle : Real.exp (511 / 97) ≤ Real.exp ((m : ℝ) / 97) :=
      Real.exp_le_exp.2 (by linarith)
    linarith [lt_exp_511_div_97]

section Numbers

attribute [local instance] fact_prime_97

/-- **The first moment at `p = 97`** (`review/flat_spectrum.md`, PT5): the parts of `m` cells with
an empty line are at most a share `194 e^{−m/97}` of all of them. -/
theorem card_emptyLine_le_97 (m : ℕ) :
    ((#{Ω ∈ powersetCard m (univ : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))) |
        emptyLines Ω ≠ 0} : ℕ) : ℝ) ≤ 194 * Real.exp (-((m : ℝ) / 97)) * ((9409).choose m : ℝ) := by
  have h := sum_emptyLines_le (F := ZMod 97) (ι := Fin 1) (κ := Fin 1) m
  simp only [ZMod.card, Fintype.card_fin, pow_one] at h
  have e : (97 * 97 : ℕ) = 9409 := rfl
  rw [e] at h
  push_cast at h
  have h2 : ((∑ Ω ∈ powersetCard m (univ : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))),
      emptyLines Ω : ℕ) : ℝ) ≤ 194 * Real.exp (-((m : ℝ) / 97)) * ((9409).choose m : ℝ) := by
    push_cast
    calc _ ≤ (97 * Real.exp (-((m : ℝ) / 97)) + 97 * Real.exp (-((m : ℝ) / 97))) *
          ((9409).choose m : ℝ) := h
      _ = _ := by ring
  exact (Nat.cast_le.2 (card_emptyLine_le m)).trans h2

end Numbers

end Transformer.Grokking.FlatSpectrum.TablePart
