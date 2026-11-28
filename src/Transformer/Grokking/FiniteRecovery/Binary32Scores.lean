module

public import Mathlib.Algebra.Field.ZMod
public import Mathlib.Tactic

/-! Check a recorded FP32 logit table with exact integer arithmetic. Each
finite binary32 value is an integer multiple of 2⁻¹⁴⁹. A row expresses
these integers in bit planes shared across the 239 vocabulary tokens. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Binary32Scores

/-- Unsigned value of column j in a list of bit planes, most significant first. -/
def column : List Nat → Nat → Nat
  | [], _ => 0
  | p::ps, j => (if p.testBit j then 2^ps.length else 0) + column ps j

theorem column_lt (ps : List Nat) (j : Nat) : column ps j < 2^ps.length := by
  induction ps with
  | nil => exact Nat.zero_lt_one
  | cons p ps ih =>
    simp only [column,List.length_cons,pow_succ]
    split <;> omega

/-- Test all equally competitive tokens in one bitwise AND per plane. After
the last plane the winner must be the sole survivor. -/
def sweep (winner : Nat) : List Nat → Nat → Bool
  | [], active => active == 2^winner
  | p::ps, active =>
    let ones := active &&& p
    if p.testBit winner then sweep winner ps ones
    else ones == 0 && sweep winner ps active

theorem sweep_sound (winner : Nat) (ps : List Nat) (active j : Nat)
    (h : sweep winner ps active = true) (hj : active.testBit j = true) (hne : j ≠ winner) :
    column ps j < column ps winner := by
  induction ps generalizing active with
  | nil =>
    have he : active=2^winner := by simpa [sweep] using h
    simp [he,Ne.symm hne] at hj
  | cons p ps ih =>
    by_cases hw : p.testBit winner = true
    · simp only [sweep,hw,ite_true] at h
      by_cases hp : p.testBit j = true
      · have hm : (active &&& p).testBit j = true := by simp [Nat.testBit_and,hj,hp]
        simpa [column,hw,hp] using ih _ h hm
      · have hp' : p.testBit j = false := Bool.eq_false_iff.mpr hp
        simp only [column,hp',hw]
        simp only [Bool.false_eq_true,ite_false,ite_true,zero_add]
        exact (column_lt ps j).trans_le (Nat.le_add_right _ _)
    · have hw' : p.testBit winner = false := Bool.eq_false_iff.mpr hw
      simp only [sweep,hw',Bool.false_eq_true,ite_false,Bool.and_eq_true,beq_iff_eq] at h
      have hp : p.testBit j = false := by
        by_contra! hn
        have hm : (active &&& p).testBit j = true := by simp [Nat.testBit_and,hj,hn]
        simp [h.1] at hm
      simpa [column,hw',hp] using ih _ h.2 hj

/-- A shared power of two and a common offset encode the exact signed
numerators of the row. The negative-zero mask preserves the FP32 bit image. -/
structure Row where
  quotient : Nat
  shift : Nat
  negativeZero : Nat
  planes : List Nat
  deriving Inhabited

/-- Exact signed logit in units of 2⁻¹⁴⁹. -/
def rowNumerator (r : Row) (j : Nat) : Int :=
  ((column r.planes j : Int) - 2^(r.planes.length-1)) * 2^r.shift

def goodMaskedBlock (a observed : Nat) (rows : Array Row) : Bool :=
  rows.size == 96 && (List.range 96).all fun j =>
    let r := rows[j]!
    observed.testBit j || (decide (r.quotient < 97) && ((j+1)*r.quotient % 97 == a) &&
      sweep (22+r.quotient) r.planes (2^239-1))

abbrev goodBlock (a : Nat) := goodMaskedBlock a 0
abbrev MaskedBlock (a observed : Nat) := {rows : Array Row // goodMaskedBlock a observed rows = true}
abbrev Block (a : Nat) := MaskedBlock a 0

/-- An input outside the observed mask has the modular quotient as its
strict winner over all 239 logits in the recorded row. -/
theorem masked_row_correct (a : Fin 97) (b : Fin 96) (observed : Nat)
    (block : MaskedBlock a.val observed) (hidden : observed.testBit b.val = false) :
    let r := block.val[b.val]!
    r.quotient < 97 ∧ (b.val+1)*r.quotient % 97 = a.val ∧
      ∀ j : Fin 239, j.val ≠ 22+r.quotient → rowNumerator r j.val < rowNumerator r (22+r.quotient) := by
  intro r
  have hb := block.property
  simp only [goodMaskedBlock,Bool.and_eq_true,List.all_eq_true,List.mem_range,Bool.or_eq_true,beq_iff_eq,decide_eq_true_eq] at hb
  obtain ⟨⟨hlt,heq⟩,hrow⟩ := (hb.2 b.val b.isLt).resolve_left (by simp [hidden])
  refine ⟨hlt,heq,fun j hne => ?_⟩
  have hbit : (2^239-1 : Nat).testBit j.val = true :=
    (Nat.testBit_two_pow_sub_one 239 j.val).trans (decide_eq_true j.isLt)
  have h := sweep_sound (22+r.quotient) r.planes (2^239-1) j.val hrow hbit hne
  unfold rowNumerator
  exact mul_lt_mul_of_pos_right (sub_lt_sub_right (by exact_mod_cast h) _) (by positivity)

/-- Checking every row corresponds to an empty observed mask. -/
theorem division_table_correct (blocks : (a : Fin 97) → Block a.val) (a : Fin 97) (b : Fin 96) :
    let r := (blocks a).val[b.val]!
    r.quotient < 97 ∧ (b.val+1)*r.quotient % 97 = a.val ∧
      ∀ j : Fin 239, j.val ≠ 22+r.quotient → rowNumerator r j.val < rowNumerator r (22+r.quotient) :=
  masked_row_correct a b 0 (blocks a) (by simp)

example : sweep 1 [2,5] 7 = true ∧ column [2,5] 1 = 2 ∧ sweep 0 [3] 3 = false := by decide

local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

/-- Canonical residue of a/(b+1) in the field with 97 elements. -/
def fieldQuotient (a : Fin 97) (b : Fin 96) : Nat :=
  ((a.val : ZMod 97) / ((b.val+1 : ℕ) : ZMod 97)).val

/-- At the answer position the token for field division strictly beats
every other vocabulary entry, including nonnumeric tokens. -/
theorem masked_quotient_wins (observed : Fin 97 → Nat) (blocks : (a : Fin 97) → MaskedBlock a.val (observed a))
    (a : Fin 97) (b : Fin 96) (hidden : (observed a).testBit b.val = false) :
    let r := (blocks a).val[b.val]!
    let q := fieldQuotient a b
    ∀ j : Fin 239, j.val ≠ 22+q → rowNumerator r j.val < rowNumerator r (22+q) := by
  intro r q
  obtain ⟨hq,heq,hwin⟩ := masked_row_correct a b _ (blocks a) hidden
  have hb : ((b.val+1 : ℕ) : ZMod 97) ≠ 0 := fun hz =>
    Nat.not_dvd_of_pos_of_lt (by omega : 0 < b.val+1) (by omega : b.val+1 < 97)
      ((ZMod.natCast_eq_zero_iff _ _).mp hz)
  have hv : (r.quotient : ZMod 97) = (a.val : ZMod 97) / ((b.val+1 : ℕ) : ZMod 97) := by
    apply (eq_div_iff hb).2
    rw [mul_comm,← Nat.cast_mul,← ZMod.natCast_mod _ 97,heq]
  have hval : r.quotient = q := (ZMod.val_natCast_of_lt hq).symm.trans (congrArg ZMod.val hv)
  rw [← hval]
  exact hwin

theorem quotient_wins (blocks : (a : Fin 97) → Block a.val) (a : Fin 97) (b : Fin 96) :
    let r := (blocks a).val[b.val]!
    let q := fieldQuotient a b
    ∀ j : Fin 239, j.val ≠ 22+q → rowNumerator r j.val < rowNumerator r (22+q) :=
  masked_quotient_wins (fun _ => 0) blocks a b (by simp)

end Transformer.Grokking.FiniteRecovery.Binary32Scores
