/-
# `MAJ²`: majority quantification over two variables

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E: `def:MAJtwo`, `def:depth_MAJtwo`, and the elimination
of `∀`/`∃` in favour of `MAJ`.

`MAJ²` has atoms `Q_σ(x)`, `Q_σ(y)`, `x < y`, `y < x`, the Boolean
connectives, and `MAJ_v⟨φ₁, …, φ_m⟩`, which holds when more than half of the
`|w|·m` pairs `(i, t)` satisfy `φ_t` with `v ↦ i`.  Its depth counts nested
majority quantifiers.  Ordinary quantification is definable without cost:
`∃v[φ] ≡ MAJ_v⟨φ, ⊤⟩` and `∀v[φ] ≡ ¬MAJ_v⟨¬φ, ⊤⟩`, because the `⊤` disjunct
contributes exactly half the mass.

The translations between `MAJ²` and `TL[◁#, ▷#]` are in
`Transformer.CRASP.MajTwoEquiv`.

Every atom of `MAJ²` names a variable, so a closed formula must bind it; the
cheapest closed tautology is therefore `¬(∃x[⊤] ∧ ¬∃x[⊤])`, of depth `1`, and
not `¬(x < x)`.
-/

import Transformer.CRASP.TLCDepth

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- The two variables of `MAJ²` (Definition `def:MAJtwo`). -/
inductive Var : Type
  /-- The variable `x`. -/
  | x : Var
  /-- The variable `y`. -/
  | y : Var
  deriving DecidableEq

/-- The syntax of `MAJ²` (Definition `def:MAJtwo`).  A majority quantifier
carries `m + 1` formulas, the paper's `m ≥ 1`. -/
inductive Maj2 (σ : Type u) : Type u
  /-- `Q_σ(v)`. -/
  | sym : σ → Var → Maj2 σ
  /-- `v < u`. -/
  | lt : Var → Var → Maj2 σ
  /-- `¬φ`. -/
  | neg : Maj2 σ → Maj2 σ
  /-- `φ₁ ∧ φ₂`. -/
  | and : Maj2 σ → Maj2 σ → Maj2 σ
  /-- `MAJ_v⟨φ₀, …, φ_m⟩`. -/
  | maj : Var → (m : ℕ) → (Fin (m + 1) → Maj2 σ) → Maj2 σ

namespace Maj2

variable [DecidableEq σ]

/-- Satisfaction `w, ξ ⊨ φ`, with `ξ` a total assignment of positions to the
two variables; an unused variable may be sent anywhere (Definition
`def:MAJtwo`).  A majority quantifier compares twice the number of satisfied
pairs against `|w|·m`, which is the paper's `> |w|m/2` over the integers. -/
def sat (w : List σ) : (Var → ℕ) → Maj2 σ → Bool
  | ξ, .sym a v => w[ξ v - 1]? = some a
  | ξ, .lt v u => decide (ξ v < ξ u)
  | ξ, .neg φ => !φ.sat w ξ
  | ξ, .and φ₁ φ₂ => φ₁.sat w ξ && φ₂.sat w ξ
  | ξ, .maj v m φ =>
      decide (w.length * (m + 1) <
        2 * ∑ i ∈ Finset.Icc 1 w.length, ∑ t : Fin (m + 1),
          (if (φ t).sat w (Function.update ξ v i) then 1 else 0))

/-- `w ⊨ φ` means `w, ∅ ⊨ φ`; the empty assignment is written as `0`
(Definition `def:MAJtwo`). -/
def models (w : List σ) (φ : Maj2 σ) : Prop := φ.sat w (fun _ => 0) = true

/-- The language a closed formula defines (Definition `def:MAJtwo`). -/
def lang (φ : Maj2 σ) : Set (List σ) := {w | φ.models w}

/-- Quantifier depth (Definition `def:depth_MAJtwo`). -/
def depth : Maj2 σ → ℕ
  | .sym _ _ => 0
  | .lt _ _ => 0
  | .neg φ => φ.depth
  | .and φ₁ φ₂ => max φ₁.depth φ₂.depth
  | .maj _ m φ => 1 + Finset.univ.sup fun t : Fin (m + 1) => (φ t).depth

/-- Whether a variable occurs free. -/
def freeIn : Maj2 σ → Var → Bool
  | .sym _ u, v => u = v
  | .lt u₁ u₂, v => (u₁ = v) || (u₂ = v)
  | .neg φ, v => φ.freeIn v
  | .and φ₁ φ₂, v => φ₁.freeIn v || φ₂.freeIn v
  | .maj u m φ, v => if u = v then false else decide (∃ t : Fin (m + 1), (φ t).freeIn v)

/-- A formula is closed when neither variable occurs free. -/
def Closed (φ : Maj2 σ) : Prop := ∀ v, φ.freeIn v = false

/-- `⊤`, written `¬(x < x)`. -/
def top : Maj2 σ := .neg (.lt .x .x)

@[simp] theorem sat_top (w : List σ) (ξ : Var → ℕ) : (top : Maj2 σ).sat w ξ = true := by
  simp [top, sat]

/-- `∃v[φ]`, written `MAJ_v⟨φ, ⊤⟩` (Appendix E, elimination of quantifiers). -/
def ex (v : Var) (φ : Maj2 σ) : Maj2 σ := .maj v 1 ![φ, top]

/-- `∀v[φ]`, written `¬MAJ_v⟨¬φ, ⊤⟩` (Appendix E). -/
def all (v : Var) (φ : Maj2 σ) : Maj2 σ := .neg (.maj v 1 ![.neg φ, top])

/-- **The `∃` half of the quantifier-elimination lemma.**  `MAJ_v⟨φ, ⊤⟩` holds
exactly when some position satisfies `φ`, since the `⊤` disjunct contributes
exactly half the mass. -/
theorem sat_ex (w : List σ) (ξ : Var → ℕ) (v : Var) (φ : Maj2 σ) :
    (ex v φ).sat w ξ = true ↔
      ∃ i ∈ Finset.Icc 1 w.length, φ.sat w (Function.update ξ v i) = true := by
  have hinner : ∀ i : ℕ,
      (∑ t : Fin 2, (if (![φ, (top : Maj2 σ)] t).sat w (Function.update ξ v i) then 1 else 0))
        = (if φ.sat w (Function.update ξ v i) then 1 else 0) + 1 := by
    intro i
    rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, sat_top]
    norm_num
    congr 1
  simp only [ex, sat, decide_eq_true_eq, hinner, Finset.sum_add_distrib, Finset.sum_const,
    Nat.card_Icc, smul_eq_mul, mul_one, Nat.add_sub_cancel]
  constructor
  · intro h
    by_contra hc
    push Not at hc
    have hz : (∑ i ∈ Finset.Icc 1 w.length,
        (if φ.sat w (Function.update ξ v i) then 1 else 0)) = 0 := by
      refine Finset.sum_eq_zero fun i hi => ?_
      rw [ite_eq_right]
      simpa using hc i hi
    omega
  · rintro ⟨i, hi, hsat⟩
    have hpos : 0 < ∑ i ∈ Finset.Icc 1 w.length,
        (if φ.sat w (Function.update ξ v i) then 1 else 0) := by
      refine Finset.sum_pos' (fun _ _ => Nat.zero_le _) ⟨i, hi, ?_⟩
      rw [ite_eq_left hsat]
      omega
    omega

/-- **The `∀` half.**  `¬MAJ_v⟨¬φ, ⊤⟩` holds exactly when every position
satisfies `φ`. -/
theorem sat_all (w : List σ) (ξ : Var → ℕ) (v : Var) (φ : Maj2 σ) :
    (all v φ).sat w ξ = true ↔
      ∀ i ∈ Finset.Icc 1 w.length, φ.sat w (Function.update ξ v i) = true := by
  have h := sat_ex w ξ v (.neg φ)
  rw [show (ex v (Maj2.neg φ)).sat w ξ = (maj v 1 ![Maj2.neg φ, top] : Maj2 σ).sat w ξ from rfl]
    at h
  rw [all, sat, Bool.not_eq_true']
  constructor
  · intro hf i hi
    by_contra hc
    have : (maj v 1 ![Maj2.neg φ, top] : Maj2 σ).sat w ξ = true :=
      h.2 ⟨i, hi, by simp [sat, hc]⟩
    rw [hf] at this
    exact Bool.noConfusion this
  · intro hf
    by_contra hc
    obtain ⟨i, hi, hsat⟩ := h.1 (by simpa using hc)
    rw [sat, hf i hi] at hsat
    exact Bool.noConfusion hsat

omit [DecidableEq σ] in
/-- The quantifiers keep the depth of the body, as the lemma requires. -/
theorem depth_ex (v : Var) (φ : Maj2 σ) : (ex v φ).depth = 1 + φ.depth := by
  have hsup : (Finset.univ.sup fun t : Fin 2 => (![φ, (top : Maj2 σ)] t).depth) = φ.depth := by
    refine le_antisymm (Finset.sup_le fun t _ => ?_) ?_
    · fin_cases t <;> simp [top, depth]
    · simpa using
        Finset.le_sup (f := fun t : Fin 2 => (![φ, (top : Maj2 σ)] t).depth) (Finset.mem_univ 0)
  rw [ex, depth, hsup]

/-- A closed tautology, `¬(∃x[⊤] ∧ ¬∃x[⊤])`.  Every atom of `MAJ²` names a
variable, so the only closed formulas are those in which a majority quantifier
binds it; this is the cheapest such tautology. -/
def closedTop : Maj2 σ := .neg (.and (ex .x top) (.neg (ex .x top)))

omit [DecidableEq σ] in
theorem closed_closedTop : (closedTop : Maj2 σ).Closed := by
  intro v
  cases v <;> simp [closedTop, freeIn, ex, top, Fin.exists_fin_two]

omit [DecidableEq σ] in
@[simp] theorem depth_closedTop : (closedTop : Maj2 σ).depth = 1 := by
  simp [closedTop, depth, depth_ex, top]

@[simp] theorem lang_closedTop : (closedTop : Maj2 σ).lang = Set.univ := by
  ext w
  simp [lang, models, closedTop, sat]

omit [DecidableEq σ] in
/-- **No closed formula has depth `0`.**  Every atom names a variable, and
only a majority quantifier binds one; this is why `closedTop` has depth `1`,
and why `MAJ²_0` defines no language at all. -/
theorem not_closed_of_depth_eq_zero : ∀ φ : Maj2 σ, φ.depth = 0 → ¬ φ.Closed
  | .sym _ v, _, h => by simpa [freeIn] using h v
  | .lt v _, _, h => by simpa [freeIn] using h v
  | .neg φ, hd, h => not_closed_of_depth_eq_zero φ hd fun v => by simpa [freeIn] using h v
  | .and φ₁ φ₂, hd, h => by
      simp only [depth, Nat.max_eq_zero_iff] at hd
      exact not_closed_of_depth_eq_zero φ₁ hd.1 fun v => by
        simpa [freeIn] using (Bool.or_eq_false_iff.1 (h v)).1
  | .maj _ _ _, hd, _ => by simp [depth] at hd

end Maj2

/-- `MAJ²_k`, the formulas of depth at most `k` (Definition
`def:depth_MAJtwo`). -/
def MajTwo (σ : Type u) (k : ℕ) : Set (Maj2 σ) := {φ | φ.depth ≤ k}

end CRASP
end Transformer
