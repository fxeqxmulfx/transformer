module

public import Transformer.CRASP.Rotary.Logic

/-!
# Rounding cells expressed by position-weighted integer comparisons

Extension of arXiv:2506.16055v3, Appendix B.1/B.2 rounding cells;
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U1.
Division and saturated rounding are eliminated into explicit linear integer
comparisons. They are not extra primitives of the counting language.
-/

@[expose] public section

namespace Transformer.CRASP.Rotary

universe u v
variable {σ : Type u} {κ : Type v} {p s d : ℕ}

/-- Zero testing by the two possible signs. -/
def zeroTest (D : Term σ κ) : Form σ κ :=
  .and (.neg (.lt D (.const 0))) (.neg (.lt (.scale (-1) D) (.const 0)))

/-- Exact lower and upper rounding endpoints for a positive denominator. -/
def positiveCell (N D : Term σ κ) (y : Fx p s) : Form σ κ :=
  .and
    (if -2 ^ (p - 1) < y.m then
      .neg (.lt (.scale (2 ^ s) N) (.scale y.m D)) else .truth true)
    (if y.m < 2 ^ (p - 1) - 1 then
      .lt (.scale (2 ^ s) N) (.scale (y.m + 1) D) else .truth true)

/-- Totalized division uses zero when its denominator is zero. -/
def cell (N D : Term σ κ) (y : Fx p s) : Form σ κ :=
  .or (.and (zeroTest D) (.truth (decide ((0 : Fx p s) = y))))
    (.and (.lt (.const 0) D) (positiveCell N D y))

/-- Rounding endpoints do not add a counting level (U1). -/
theorem positiveCell_depth (N D : Term σ κ) (hN : N.depth ≤ d) (hD : D.depth ≤ d)
    (y : Fx p s) : (positiveCell N D y).depth ≤ d := by
  unfold positiveCell
  split_ifs <;> simp only [Form.depth, Term.depth]
  · exact max_le (max_le hN hD) (max_le hN hD)
  · exact max_le (max_le hN hD) (Nat.zero_le _)
  · exact max_le (Nat.zero_le _) (max_le hN hD)
  · exact Nat.zero_le _

/-- Testing the whole quotient has only the depth of its counts (U1). -/
theorem cell_depth (N D : Term σ κ) (hN : N.depth ≤ d) (hD : D.depth ≤ d)
    (y : Fx p s) : (cell N D y).depth ≤ d := by
  change max (max (zeroTest D).depth 0)
    (max (max 0 D.depth) (positiveCell N D y).depth) ≤ d
  have hz : (zeroTest D).depth ≤ d := by
    simpa only [zeroTest, Form.depth, Term.depth, max_zero, max_self] using hD
  exact max_le (max_le hz (Nat.zero_le _))
    (max_le (max_le (Nat.zero_le _) hD) (positiveCell_depth N D hN hD y))

variable [DecidableEq σ]

/-- The zero formula tests the integer value, including signed counts (U1). -/
theorem zeroTest_sat (D : Term σ κ) (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) :
    (zeroTest D).sat K w i = decide (D.val K w i = 0) := by
  rw [Bool.eq_iff_iff]
  simp only [zeroTest, Form.sat, Bool.and_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not, decide_eq_true_eq]
  simp only [Term.val, neg_one_mul]
  omega

/-- Cross multiplication proves exact saturated rounding, without adding
rounding as a syntax constructor (Appendix B.1/B.2 extension). -/
theorem positiveCell_sat (N D : Term σ κ) (K : κ → ℕ → ℕ → ℤ)
    (w : List σ) (i : ℕ) (hD : 0 < D.val K w i) (y : Fx p s) :
    (positiveCell N D y).sat K w i = true ↔
      Fx.round p s ((N.val K w i : ℝ) / (D.val K w i : ℝ)) = y := by
  have hd : (0 : ℝ) < D.val K w i := by exact_mod_cast hD
  have he : (N.val K w i : ℝ) / (D.val K w i : ℝ) * 2 ^ s =
      ((2 ^ s * N.val K w i : ℤ) : ℝ) / (D.val K w i : ℝ) := by
    push_cast
    ring
  have hlo : ¬ (2 ^ s * N.val K w i < y.m * D.val K w i) ↔
      (y.m : ℝ) ≤ ((2 ^ s * N.val K w i : ℤ) : ℝ) / (D.val K w i : ℝ) := by
    rw [le_div_iff₀ hd, not_lt]
    exact_mod_cast Iff.rfl
  have hhi : 2 ^ s * N.val K w i < (y.m + 1) * D.val K w i ↔
      ((2 ^ s * N.val K w i : ℤ) : ℝ) / (D.val K w i : ℝ) < (y.m + 1 : ℤ) := by
    rw [div_lt_iff₀ hd]
    exact_mod_cast Iff.rfl
  rw [Fx.round_eq_iff, he]
  unfold positiveCell
  split_ifs <;> simp only [Form.sat, Bool.and_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not, decide_eq_true_eq]
  all_goals simp_all [Term.val]

/-- Nonnegative denominator counts suffice, including complete underflow to
zero; negative numerator sums remain allowed (U1). -/
theorem cell_sat (N D : Term σ κ) (K : κ → ℕ → ℕ → ℤ)
    (w : List σ) (i : ℕ) (hD : 0 ≤ D.val K w i) (y : Fx p s) :
    (cell N D y).sat K w i = true ↔
      Fx.round p s ((N.val K w i : ℝ) / (D.val K w i : ℝ)) = y := by
  simp only [cell, Form.sat, Bool.or_eq_true, Bool.and_eq_true, zeroTest_sat,
    decide_eq_true_eq]
  simp only [Term.val]
  by_cases hz : D.val K w i = 0
  · simp [hz]
  · have hp : 0 < D.val K w i := by omega
    simp only [hz, false_and, false_or, hp, true_and]
    exact positiveCell_sat N D K w i hp y

/-- Positive and zero-denominator premises occur for concrete finite terms. -/
example : (Term.const 2 : Term Bool Unit).depth ≤ 0 ∧
    (Term.const 1 : Term Bool Unit).depth ≤ 0 ∧
    0 < (Term.const 1 : Term Bool Unit).val (fun _ _ _ => 1) [true] 1 ∧
    0 ≤ (Term.const 0 : Term Bool Unit).val (fun _ _ _ => 1) [true] 1 := by decide

universe w
variable {Q : Type w} [Fintype Q]

/-- Sum the independently positioned contributions of every possible source
state. Kernel names contain the fixed query state, not a query-word oracle. -/
noncomputable def stateSum (key : Q → κ) (ψ : Q → Form σ κ) : Term σ κ :=
  Term.sum (Finset.univ.toList.map fun a : Q => .count (key a) (ψ a))

omit [DecidableEq σ] in
/-- The weighted state sum adds exactly one available counting level (U1). -/
theorem stateSum_depth (key : Q → κ) (ψ : Q → Form σ κ)
    (hψ : ∀ a, (ψ a).depth ≤ d) : (stateSum key ψ).depth ≤ d + 1 := by
  apply Term.depth_sum
  rintro A hA
  obtain ⟨a, _, rfl⟩ := List.mem_map.mp hA
  exact Nat.add_le_add_right (hψ a) 1

/-- Reordering finite sums removes the state enumeration and recovers each
source's actual position-dependent coefficient, without a periodic code (U1). -/
theorem stateSum_val [DecidableEq Q] (key : Q → κ) (ψ : Q → Form σ κ)
    (K : κ → ℕ → ℕ → ℤ) (q : ℕ → Q) (w : List σ) (i : ℕ)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat K w j = decide (q j = a)) :
    (stateSum key ψ).val K w i = ∑ j ∈ Finset.Icc 1 i, K (key (q j)) i j := by
  classical
  rw [stateSum, Term.val_sum, List.map_map, Finset.sum_map_toList]
  simp only [Function.comp_def, Term.val]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [hψ j hj, decide_eq_true_eq]
  exact Fintype.sum_ite_eq (q j) (fun a => K (key a) i j)

/-- State-depth and prefix agreement assumptions have a binary token witness. -/
example : (∀ a : Bool, (Form.sym a : Form Bool Unit).depth ≤ 0) ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Bool,
      (Form.sym a : Form Bool Unit).sat (fun _ _ _ => 1) [true] j = decide (true = a)) := by
  refine ⟨fun _ => le_rfl, ?_⟩
  intro j hj a
  have : j = 1 := by have := Finset.mem_Icc.mp hj; omega
  subst j
  simp [Form.sat]

end Transformer.CRASP.Rotary
