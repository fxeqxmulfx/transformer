/-
# `TL[◁#]^pos` reduces to `TL[◁#]`: the `Y`-atoms

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, the proof of `lem:tlclpos_reduction`.

The reduction reads a string `w` over `Σ` through
`f(w) = e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` block by block: position `i` of `w`
stands for the positions `r·i + 1, …, r·i + r` of `f(w)`, and the translation
`T_ρ⟦φ⟧` of `Transformer.CRASP.PositionalReduction` simulates `φ` at
`r·i + ρ`.  A `Y`-atom `Y^c α` there looks at position `r·i + ρ − c`, which
carries `wᵢ` when `ρ − c = 1` and `e` otherwise, and which is congruent to
`r + ρ − c` modulo every modulus dividing `r`.  So `FormP.atomTr`, its
translation, is `Q_{wᵢ}` or a constant, and costs no depth.

**The block size.**  The paper takes `r = M(Y + 1)`, for `M` the least common
multiple of the moduli and `Y` the `Y`-depth, which gives `r ≥ c + 1`.  The
translation also reads the first block `e^r` at its end `ρ = r`, where
`Y^c Q_e` holds, and `T_r⟦Y^c Q_e⟧ = ⊥` exactly when `r − c = 1`: that takes
`r ≥ c + 2`.  `FormP.reach` bounds `c + 2` over the `Y`-atoms, and `s + c + 2`
over `Y^c MOD_0^s`, which compares with `s` rather than reducing modulo `0`;
`FormP.period` is a common multiple of the positive moduli.
-/

import Transformer.CRASP.Positional

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

namespace Form

/-- The constant formula: `¬(1 < 1)` for `true`, `1 < 1` for `false`. -/
def ofBool (b : Bool) : Form σ := if b then .neg (.lt .one .one) else .lt .one .one

/-- A constant formula lies in `TL[◁#]_d` for every `d`. -/
theorem ofBool_mem_TLCl (b : Bool) (d : ℕ) : (ofBool b : Form σ) ∈ TLCl σ d := by
  cases b <;> exact ⟨rfl, rfl, Nat.zero_le d⟩

/-- A constant formula has its constant value. -/
@[simp] theorem sat_ofBool [DecidableEq σ] (w : List σ) (i : ℕ) (b : Bool) :
    (ofBool b : Form σ).sat w i = b := by
  cases b <;> simp [ofBool, Form.sat, Term.val]

end Form

mutual

/-- Two more than the number of `Y`s around any atom, and `s + 2` more around
`MOD_m^s`: the translation is exact once `reach ≤ r` (Appendix E, the bound
`r > Y` of `lem:tlclpos_reduction`, raised by one). -/
def FormP.reach : FormP σ → ℕ
  | .sym _ => 2
  | .mod _ s => s + 2
  | .prev φ => φ.reach + 1
  | .lt t₁ t₂ => max t₁.reach t₂.reach
  | .neg φ => φ.reach
  | .and φ₁ φ₂ => max φ₁.reach φ₂.reach

/-- The reach of a term. -/
def TermP.reach : TermP σ → ℕ
  | .countL φ => φ.reach
  | .add t₁ t₂ => max t₁.reach t₂.reach
  | .one => 0

end

mutual

/-- A common multiple of the positive moduli (Appendix E, the `M` of
`lem:tlclpos_reduction`, as a product rather than a least common multiple). -/
def FormP.period : FormP σ → ℕ
  | .sym _ => 1
  | .mod m _ => max m 1
  | .prev φ => φ.period
  | .lt t₁ t₂ => t₁.period * t₂.period
  | .neg φ => φ.period
  | .and φ₁ φ₂ => φ₁.period * φ₂.period

/-- The period of a term. -/
def TermP.period : TermP σ → ℕ
  | .countL φ => φ.period
  | .add t₁ t₂ => t₁.period * t₂.period
  | .one => 1

end

mutual

/-- The period of a formula is positive. -/
theorem FormP.period_pos : ∀ φ : FormP σ, 0 < φ.period
  | .sym _ => by rw [FormP.period]; exact Nat.one_pos
  | .mod _ _ => by rw [FormP.period]; exact lt_max_of_lt_right Nat.one_pos
  | .prev φ => by rw [FormP.period]; exact φ.period_pos
  | .lt t₁ t₂ => by rw [FormP.period]; exact Nat.mul_pos t₁.period_pos t₂.period_pos
  | .neg φ => by rw [FormP.period]; exact φ.period_pos
  | .and φ₁ φ₂ => by rw [FormP.period]; exact Nat.mul_pos φ₁.period_pos φ₂.period_pos

/-- The period of a term is positive. -/
theorem TermP.period_pos : ∀ t : TermP σ, 0 < t.period
  | .countL φ => by rw [TermP.period]; exact φ.period_pos
  | .add t₁ t₂ => by rw [TermP.period]; exact Nat.mul_pos t₁.period_pos t₂.period_pos
  | .one => by rw [TermP.period]; exact Nat.one_pos

end

/-- **The translation of a `Y`-atom** `T_ρ⟦Y^c α⟧` (Appendix E, proof of
`lem:tlclpos_reduction`): position `r·i + ρ − c` of `f(w)` carries `wᵢ` when
`ρ − c = 1` and `e` otherwise, and is congruent to `r + ρ − c` modulo every
modulus dividing `r`.  `atomTr r ρ c φ` translates `Y^c φ`, so a `Y` is peeled
off by raising `c`; what is not a `Y`-atom goes to `⊥`. -/
def FormP.atomTr (r ρ : ℕ) : ℕ → FormP (Option σ) → Form σ
  | c, .sym none => Form.ofBool (decide (ρ ≠ c + 1))
  | c, .sym (some b) => if ρ = c + 1 then .sym b else Form.ofBool false
  | c, .mod m s => Form.ofBool (decide ((r + ρ - c) % m = s % m))
  | c, .prev φ => FormP.atomTr r ρ (c + 1) φ
  | _, .lt _ _ => Form.ofBool false
  | _, .neg _ => Form.ofBool false
  | _, .and _ _ => Form.ofBool false

/-- The translation of a `Y`-atom has depth `0`, so lies in every `TL[◁#]_d`. -/
theorem FormP.atomTr_mem_TLCl (r ρ d : ℕ) :
    ∀ (c : ℕ) (φ : FormP (Option σ)), FormP.atomTr r ρ c φ ∈ TLCl σ d
  | c, .sym none => by rw [FormP.atomTr]; exact Form.ofBool_mem_TLCl _ d
  | c, .sym (some b) => by
      rw [FormP.atomTr]
      split
      · exact ⟨rfl, rfl, Nat.zero_le d⟩
      · exact Form.ofBool_mem_TLCl false d
  | c, .mod m s => by rw [FormP.atomTr]; exact Form.ofBool_mem_TLCl _ d
  | c, .prev φ => by rw [FormP.atomTr]; exact FormP.atomTr_mem_TLCl r ρ d (c + 1) φ
  | c, .lt t₁ t₂ => by rw [FormP.atomTr]; exact Form.ofBool_mem_TLCl false d
  | c, .neg φ => by rw [FormP.atomTr]; exact Form.ofBool_mem_TLCl false d
  | c, .and φ₁ φ₂ => by rw [FormP.atomTr]; exact Form.ofBool_mem_TLCl false d

end CRASP
end Transformer
