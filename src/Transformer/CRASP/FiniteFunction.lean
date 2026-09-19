/-
# `lem:finite_function`: postcomposing a definable map with any `g : 𝔽 → 𝔽`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix B.2, `lem:finite_function` (Chiang, Cholak & Pillay).

If every bit of a length-preserving map `F : Σ* → 𝔽*` is defined by a formula,
so is every bit of `g ∘ F`, for an arbitrary `g : 𝔽 → 𝔽`, and at no extra
depth.  The construction is the disjunctive normal form over the bit patterns:
`[g(F(w)_i)]_b = 1` exactly when the bits of `F(w)_i` form one of the finitely
many patterns `v` carried by some `x` with `[g x]_b = 1`, and each pattern is
a conjunction of the given formulas and their negations.  `Fx.ext_of_bit` is
what makes the disjunction exact — a pattern determines its number — and the
guard `Form.onStr` is what keeps the formula false off the string, where all
the given formulas are false and the pattern of all-zero bits would otherwise
fire.

Two departures from the statement of the paper, which leaves the depth clause
implicit:

* the bound is the supremum over `c < p + 2`, not over `c < p`.  Bits `0` and
  `p + 1` are along for the ride, but bit `p` is not: the bits below `p` leave
  the two's-complement sign undetermined, so a bound over `c < p` is false —
  at `p = 2` the mantissas `0` and `-2` share the bits `0, 1`, and a `g`
  separating them would have to be defined by a formula of the depth of the
  sign bit, which the bound does not mention;
* the alphabet is finite, as it is in the paper, so that `Form.onStr` can be
  written at depth `0`.
-/

import Transformer.CRASP.Conjunctions
import Transformer.CRASP.FixedBits
import Transformer.CRASP.Transformers

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ] {p s : ℕ}

omit [DecidableEq σ] in
/-- The depth of a literal is the depth of the formula under it. -/
theorem depth_ite_neg (φ : Form σ) (t : Bool) :
    (if t then φ else φ.neg).depth = φ.depth := by
  cases t <;> rfl

omit [DecidableEq σ] in
theorem past_ite_neg (φ : Form σ) (t : Bool) : (if t then φ else φ.neg).past = φ.past := by
  cases t <;> rfl

omit [DecidableEq σ] in
theorem pnpFree_ite_neg (φ : Form σ) (t : Bool) :
    (if t then φ else φ.neg).pnpFree = φ.pnpFree := by
  cases t <;> rfl

/-- A literal `φ` or `¬φ`, selected by a bit `t`, holds exactly when `φ` has
truth value `t`. -/
theorem sat_ite_neg (φ : Form σ) (t : Bool) (w : List σ) (i : ℕ) :
    (if t then φ else φ.neg).sat w i = true ↔ φ.sat w i = t := by
  cases t with
  | false => simp [Form.sat]
  | true => simp

variable [Fintype σ]

open scoped Classical in
/-- The formula for bit `b` of `g ∘ F`, out of formulas `ψ` for the bits of
`F`: the disjunction, over the bit patterns `v` of the numbers `x` with
`[g x]_b = 1`, of "this position carries a letter, and its bits are `v`". -/
noncomputable def bitsFormula (g : Fx p s → Fx p s) (ψ : ℕ → Form σ) (b : ℕ) : Form σ :=
  Form.any (((Finset.univ : Finset (Fin (p + 2) → Bool)).filter fun v =>
      ∃ x : Fx p s, (∀ c : Fin (p + 2), x.bit (c : ℕ) = v c) ∧ (g x).bit b = true).toList.map
    fun v => .and Form.onStr
      (Form.all ((List.finRange (p + 2)).map fun c =>
        if v c then ψ (c : ℕ) else .neg (ψ (c : ℕ)))))

/-- **It defines the bits of `g ∘ F`.** -/
theorem sat_bitsFormula {F : List σ → List (Fx p s)} (hF : ∀ w, (F w).length = w.length)
    (g : Fx p s → Fx p s) {ψ : ℕ → Form σ} (hψ : ∀ b w i, (ψ b).sat w i = bitAt F w i b)
    (b : ℕ) (w : List σ) (i : ℕ) :
    (bitsFormula g ψ b).sat w i = bitAt (fun w => (F w).map g) w i b := by
  classical
  rw [Bool.eq_iff_iff]
  unfold bitsFormula
  rw [Form.sat_any]
  rcases lt_or_ge (i - 1) w.length with hi | hi
  · have hlen : i - 1 < (F w).length := by rw [hF]; exact hi
    have hsome : (F w)[i - 1]? = some ((F w)[i - 1]'hlen) := List.getElem?_eq_getElem hlen
    set x₀ := (F w)[i - 1]'hlen with hx₀
    have hbit : ∀ c : ℕ, (ψ c).sat w i = x₀.bit c := by
      intro c
      rw [hψ, bitAt, hsome]
      rfl
    have hrhs : bitAt (fun w => (F w).map g) w i b = (g x₀).bit b := by
      rw [bitAt, List.getElem?_map, hsome]
      rfl
    rw [hrhs]
    constructor
    · rintro ⟨φ, hmem, hsat⟩
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
      obtain ⟨x, hxb, hxg⟩ := (Finset.mem_filter.mp (Finset.mem_toList.mp hv)).2
      simp only [Form.sat, Bool.and_eq_true] at hsat
      have hvx : ∀ c : Fin (p + 2), x₀.bit (c : ℕ) = v c := by
        intro c
        have hc := (Form.sat_all w i _).mp hsat.2 _
          (List.mem_map_of_mem (List.mem_finRange c))
        rw [← hbit]
        exact (sat_ite_neg _ _ w i).mp hc
      have hxx : x = x₀ := Fx.ext_of_bit fun c hc => by
        rw [hxb ⟨c, hc⟩, ← hvx ⟨c, hc⟩]
      rwa [hxx] at hxg
    · intro hgb
      refine ⟨_, List.mem_map_of_mem (a := fun c : Fin (p + 2) => x₀.bit (c : ℕ))
        (Finset.mem_toList.mpr (Finset.mem_filter.mpr
          ⟨Finset.mem_univ _, ⟨x₀, fun _ => rfl, hgb⟩⟩)), ?_⟩
      have honstr : (Form.onStr : Form σ).sat w i = true := by
        rw [Form.sat_onStr]
        simpa using hi
      have hall : (Form.all ((List.finRange (p + 2)).map fun c : Fin (p + 2) =>
          if x₀.bit (c : ℕ) then ψ (c : ℕ) else Form.neg (ψ (c : ℕ)))).sat w i = true := by
        refine (Form.sat_all w i _).mpr ?_
        rintro χ hχ
        obtain ⟨c, -, rfl⟩ := List.mem_map.mp hχ
        exact (sat_ite_neg _ _ w i).mpr (hbit (c : ℕ))
      simp only [Form.sat, Bool.and_eq_true]
      exact ⟨honstr, hall⟩
  · have hnone : (F w)[i - 1]? = none := List.getElem?_eq_none (by rw [hF]; exact hi)
    have hrhs : bitAt (fun w => (F w).map g) w i b = false := by
      rw [bitAt, List.getElem?_map, hnone]
      rfl
    rw [hrhs]
    simp only [Bool.false_eq_true, iff_false, not_exists, not_and]
    intro φ hmem hsat
    obtain ⟨v, -, rfl⟩ := List.mem_map.mp hmem
    simp only [Form.sat, Bool.and_eq_true] at hsat
    have honstr := hsat.1
    rw [Form.sat_onStr] at honstr
    simp only [decide_eq_true_eq] at honstr
    omega

omit [DecidableEq σ] in
/-- **And it adds no depth**: every formula it is built from is one of the
given ones, negated or not. -/
theorem depth_bitsFormula_le (g : Fx p s → Fx p s) (ψ : ℕ → Form σ) (b : ℕ) :
    (bitsFormula g ψ b).depth ≤ (Finset.range (p + 2)).sup fun c => (ψ c).depth := by
  classical
  unfold bitsFormula
  refine Form.depth_any_le _ ?_
  rintro φ hφ
  obtain ⟨v, -, rfl⟩ := List.mem_map.mp hφ
  simp only [Form.depth, max_le_iff]
  refine ⟨by simp, Form.depth_all_le _ ?_⟩
  rintro χ hχ
  obtain ⟨c, -, rfl⟩ := List.mem_map.mp hχ
  rw [depth_ite_neg]
  exact Finset.le_sup (f := fun c => (ψ c).depth) (Finset.mem_range.mpr c.isLt)

omit [DecidableEq σ] in
/-- It stays in the past-only fragment when the given formulas do. -/
theorem past_bitsFormula (g : Fx p s → Fx p s) (ψ : ℕ → Form σ)
    (hpast : ∀ c, (ψ c).past = true) (b : ℕ) : (bitsFormula g ψ b).past = true := by
  classical
  unfold bitsFormula
  refine Form.past_any _ ?_
  rintro φ hφ
  obtain ⟨v, -, rfl⟩ := List.mem_map.mp hφ
  simp only [Form.past, Bool.and_eq_true]
  refine ⟨Form.past_onStr, Form.past_all _ ?_⟩
  rintro χ hχ
  obtain ⟨c, -, rfl⟩ := List.mem_map.mp hχ
  rw [past_ite_neg]
  exact hpast (c : ℕ)

omit [DecidableEq σ] in
/-- And it uses no Parikh numerical predicate when the given formulas do not. -/
theorem pnpFree_bitsFormula (g : Fx p s → Fx p s) (ψ : ℕ → Form σ)
    (hfree : ∀ c, (ψ c).pnpFree = true) (b : ℕ) : (bitsFormula g ψ b).pnpFree = true := by
  classical
  unfold bitsFormula
  refine Form.pnpFree_any _ ?_
  rintro φ hφ
  obtain ⟨v, -, rfl⟩ := List.mem_map.mp hφ
  simp only [Form.pnpFree, Bool.and_eq_true]
  refine ⟨Form.pnpFree_onStr, Form.pnpFree_all _ ?_⟩
  rintro χ hχ
  obtain ⟨c, -, rfl⟩ := List.mem_map.mp hχ
  rw [pnpFree_ite_neg]
  exact hfree (c : ℕ)

/-- **Lemma `lem:finite_function`** (Chiang, Cholak & Pillay).  A
length-preserving fixed-precision map whose bits are definable stays definable
after any postcomposition `g : 𝔽 → 𝔽`, at no extra depth. -/
theorem finite_function (F : List σ → List (Fx p s))
    (hF : ∀ w, (F w).length = w.length) (g : Fx p s → Fx p s) (ψ : ℕ → Form σ)
    (hψ : ∀ b w i, (ψ b).sat w i = bitAt F w i b) :
    ∃ ψ' : ℕ → Form σ, (∀ b w i, (ψ' b).sat w i = bitAt (fun w => (F w).map g) w i b) ∧
      ∀ b, (ψ' b).depth ≤ (Finset.range (p + 2)).sup fun c => (ψ c).depth :=
  ⟨bitsFormula g ψ, fun b w i => sat_bitsFormula hF g hψ b w i,
    fun b => depth_bitsFormula_le g ψ b⟩

/-- The hypotheses are satisfiable: the map sending every position to `0` is
length-preserving, and `⊥` — written `1 < 1` — defines all of its bits, since
every bit of `0` is `0`. -/
example :
    (∀ w : List σ, ((w.map fun _ => (0 : Fx p s))).length = w.length) ∧
      ∀ b (w : List σ) i,
        (Form.lt .one .one : Form σ).sat w i = bitAt (fun w => w.map fun _ => (0 : Fx p s)) w i b := by
  refine ⟨fun w => by simp, fun b w i => ?_⟩
  have hbit : ∀ b, (0 : Fx p s).bit b = false := by
    intro b
    simp [Fx.bit, Fx.val_zero]
  rw [Form.sat, bitAt, List.getElem?_map]
  cases w[i - 1]? with
  | none => simp [Term.val]
  | some _ => simp [Term.val, hbit]

end CRASP
end Transformer
