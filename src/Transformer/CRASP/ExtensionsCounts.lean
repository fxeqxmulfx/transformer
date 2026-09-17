/-
# Eliminating the sugar of Appendix A.3: the counting operators

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A.3, `thm:strict`: the rewriting rules
`◁#_<[φ] ≡ ◁#[φ] − (φ ? 1 : 0)`, `▷#_>[φ] ≡ ▷#[φ] − (φ ? 1 : 0)` and
`#[φ] ≡ ◁#[φ] + ▷#_>[φ]`, as guarded pieces (`Covers`).

Each lemma takes a plain formula `e` agreeing with the counted predicate `g` up
to the length of the word, and the counts of the extended operator over `g`.
At position `0` the first rule fails, since both counts are `0` there; its
pieces guard against it with `Form.pos`.
-/

import Transformer.CRASP.ExtensionsPieces

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- Counting a range inside `[0, n]` sees a function only up to `n`. -/
theorem filter_range'_congr {f g : ℕ → Bool} {n : ℕ} (h : ∀ j ≤ n, f j = g j) {s m : ℕ}
    (hm : s + m ≤ n + 1) : (List.range' s m).filter f = (List.range' s m).filter g :=
  List.filter_congr fun j hj => h j (by rw [List.mem_range'_1] at hj; omega)

/-- `◁#[φ] = ◁#_<[φ] + (φ ? 1 : 0)` away from position `0` (Appendix A.3). -/
theorem length_filter_range'_strictL (f : ℕ → Bool) (i : ℕ) :
    ((List.range' 1 (i + 1)).filter f).length =
      ((List.range' 1 i).filter f).length + if f (i + 1) then 1 else 0 := by
  rw [List.range'_concat, List.filter_append, List.length_append, Nat.one_mul, Nat.add_comm 1 i]
  cases h : f (i + 1) <;> simp [h]

/-- `▷#[φ] = ▷#_>[φ] + (φ ? 1 : 0)` (Appendix A.3). -/
theorem length_filter_range'_strictR (f : ℕ → Bool) {i n : ℕ} (hi : i ≤ n) :
    ((List.range' i (n + 1 - i)).filter f).length =
      ((List.range' (i + 1) (n - i)).filter f).length + if f i then 1 else 0 := by
  rw [show n + 1 - i = n - i + 1 by omega, List.range'_succ, List.filter_cons]
  cases f i <;> simp [Nat.add_comm]

/-- `#[φ] = ◁#[φ] + ▷#_>[φ]` (Appendix A.3). -/
theorem length_filter_range'_all (f : ℕ → Bool) {i n : ℕ} (hi : i ≤ n) :
    ((List.range' 1 i).filter f).length + ((List.range' (i + 1) (n - i)).filter f).length =
      ((List.range' 1 n).filter f).length := by
  rw [← List.length_append, ← List.filter_append, Nat.add_comm i 1, List.range'_append_1,
    Nat.add_sub_cancel' hi]

/-- The hypothesis of the lemmas below is satisfiable: `Q_a` agrees with the
letter test up to the length of the word. -/
example : ∀ j ≤ [true].length, (Form.sym true : Form Bool).sat [true] j =
    decide ([true][j - 1]? = some true) := fun _ _ => rfl

/-- `◁#[φ]` is one piece (§2.2). -/
theorem covers_countL {e : Form σ} {g : ℕ → Bool} {w : List σ} {i : ℕ}
    (h : ∀ j ≤ w.length, e.sat w j = g j) (hi : i ≤ w.length) :
    Covers [(Form.topAt 0, .countL e, 0)] w i ((List.range' 1 i).filter g).length := by
  refine ⟨⟨_, List.mem_singleton_self _, Form.sat_topAt ..⟩, fun x hx _ => ?_⟩
  rw [List.mem_singleton] at hx
  subst hx
  simp only [Term.val, filter_range'_congr h (s := 1) (m := i) (by omega), Nat.add_zero]

/-- `▷#[φ]` is one piece (§2.2). -/
theorem covers_countR {e : Form σ} {g : ℕ → Bool} {w : List σ} {i : ℕ}
    (h : ∀ j ≤ w.length, e.sat w j = g j) (hi : i ≤ w.length) :
    Covers [(Form.topAt 0, .countR e, 0)] w i
      ((List.range' i (w.length + 1 - i)).filter g).length := by
  refine ⟨⟨_, List.mem_singleton_self _, Form.sat_topAt ..⟩, fun x hx _ => ?_⟩
  rw [List.mem_singleton] at hx
  subst hx
  simp only [Term.val, filter_range'_congr h (s := i) (m := w.length + 1 - i) (by omega),
    Nat.add_zero]

/-- Two pieces guarded by `¬e` and `e` cover every position. -/
theorem exists_sat_neg_or {e : Form σ} {w : List σ} {i : ℕ} {p q : Term σ} {k l : ℕ}
    (L : List (Form σ × Term σ × ℕ)) :
    ∃ x ∈ (Form.neg e, p, k) :: (e, q, l) :: L, x.1.sat w i = true := by
  cases hc : e.sat w i
  · exact ⟨_, List.mem_cons_self .., by simp [Form.sat, hc]⟩
  · exact ⟨_, List.mem_cons_of_mem _ (List.mem_cons_self ..), hc⟩

/-- `#[φ] + (φ ? 1 : 0) = ◁#[φ] + ▷#[φ]` (Appendix A.3). -/
theorem covers_countAll {e : Form σ} {g : ℕ → Bool} {w : List σ} {i : ℕ}
    (h : ∀ j ≤ w.length, e.sat w j = g j) (hi : i ≤ w.length) :
    Covers [(.neg e, .add (.countL e) (.countR e), 0), (e, .add (.countL e) (.countR e), 1)] w i
      ((List.range' 1 w.length).filter g).length := by
  have hS := length_filter_range'_strictR g hi
  have hA := length_filter_range'_all g hi
  refine ⟨exists_sat_neg_or [], ?_⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false]
  rintro x (rfl | rfl) hx <;>
    simp only [Term.val, filter_range'_congr h (s := 1) (m := i) (by omega),
      filter_range'_congr h (s := i) (m := w.length + 1 - i) (by omega)]
  · simp only [Form.sat, h i hi, Bool.not_eq_true'] at hx
    simp only [hx] at hS
    simp at hS
    omega
  · rw [h i hi] at hx
    simp only [hx] at hS
    simp at hS
    omega

/-- `◁#_<[φ] + (φ ? 1 : 0) = ◁#[φ]` away from position `0`, and `◁#_<[φ] = 0`
at it (Appendix A.3). -/
theorem covers_countLStrict {e : Form σ} {g : ℕ → Bool} {w : List σ} {i : ℕ}
    (h : ∀ j ≤ w.length, e.sat w j = g j) (hi : i ≤ w.length) :
    Covers [(.neg e, .countL e, 0), (.and e Form.pos, .countL e, 1),
      (.and e (.neg Form.pos), .one, 1)] w i ((List.range' 1 (i - 1)).filter g).length := by
  have eL := filter_range'_congr h (s := 1) (m := i) (by omega)
  refine ⟨?_, ?_⟩
  · cases hc : e.sat w i
    · exact ⟨_, List.mem_cons_self .., by simp [Form.sat, hc]⟩
    · rcases Nat.eq_zero_or_pos i with rfl | hp
      · exact ⟨(.and e (.neg Form.pos), .one, 1), by simp, by simp [Form.sat, hc]⟩
      · exact ⟨_, List.mem_cons_of_mem _ (List.mem_cons_self ..), by simp [Form.sat, hc, hp]⟩
  · simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro x (rfl | rfl | rfl) hx <;> simp only [Term.val, eL]
    · cases i with
      | zero => rfl
      | succ i =>
          have hL := length_filter_range'_strictL g i
          simp only [Form.sat, h _ hi, Bool.not_eq_true'] at hx
          simp only [hx] at hL
          simp at hL
          simp [hL]
    · simp only [Form.sat, Form.sat_pos, h i hi, Bool.and_eq_true, decide_eq_true_eq] at hx
      obtain ⟨i, rfl⟩ := Nat.exists_eq_add_of_lt hx.2
      have hL := length_filter_range'_strictL g (0 + i)
      simp only [hx.1] at hL
      simp at hL ⊢
      omega
    · simp only [Form.sat, Form.sat_pos, Bool.and_eq_true, Bool.not_eq_true',
        decide_eq_false_iff_not, Nat.not_lt, Nat.le_zero] at hx
      simp [hx.2]

/-- `▷#_>[φ] + (φ ? 1 : 0) = ▷#[φ]` (Appendix A.3). -/
theorem covers_countRStrict {e : Form σ} {g : ℕ → Bool} {w : List σ} {i : ℕ}
    (h : ∀ j ≤ w.length, e.sat w j = g j) (hi : i ≤ w.length) :
    Covers [(.neg e, .countR e, 0), (e, .countR e, 1)] w i
      ((List.range' (i + 1) (w.length - i)).filter g).length := by
  have hS := length_filter_range'_strictR g hi
  refine ⟨exists_sat_neg_or [], ?_⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false]
  rintro x (rfl | rfl) hx <;>
    simp only [Term.val, filter_range'_congr h (s := i) (m := w.length + 1 - i) (by omega)]
  · simp only [Form.sat, h i hi, Bool.not_eq_true'] at hx
    simp only [hx] at hS
    simpa using hS.symm
  · rw [h i hi] at hx
    simp only [hx] at hS
    simpa using hS.symm

end CRASP
end Transformer
