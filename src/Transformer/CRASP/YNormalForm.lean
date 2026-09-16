/-
# The `Y`-normal form of `TL[◁#]^pos`: the transformation

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F (`app:tlclpos`, "Depth Hierarchy"): `thm:ynf`.

A formula is in `Y`-normal form when `Y` occurs only around atomic formulas.
The proof pushes the `Y`s inwards with a transformation `N^c⟦·⟧` that carries
the number `c` of `Y`s pushed so far and puts them back around the atoms:
`N^c⟦Q_σ⟧ = Y^c Q_σ`, `N^c⟦Y φ⟧ = N^{c+1}⟦φ⟧`, and `N^c` commutes with every
other constructor.  `FormP.pushY` is that transformation with one change: at
`¬` and `<` it keeps the pushed `Y`s behind as a guard `Y^c ⊤`, with
`⊤ = MOD_1^0`, which is a `Y`-atom and costs no depth.  Why the guard is
needed, and the equivalence it buys, are in `Transformer.CRASP.YNormalFormEquiv`;
this file has the syntax: the result has the depth of the input and is in
`Y`-normal form.
-/

import Transformer.CRASP.Positional

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- `Y`-atoms: `Y` applied to a symbol or a modular predicate, any number of
times (Appendix F, the `ψ` line of the `Y`-normal form grammar). -/
inductive YAtomic : FormP σ → Prop
  /-- `Q_σ` is a `Y`-atom. -/
  | sym (a : σ) : YAtomic (.sym a)
  /-- `MOD_m^r` is a `Y`-atom. -/
  | mod (m r : ℕ) : YAtomic (.mod m r)
  /-- `Y` of a `Y`-atom is a `Y`-atom. -/
  | prev {ψ : FormP σ} : YAtomic ψ → YAtomic (.prev ψ)

mutual

/-- **`Y`-normal form** (Appendix F): `Y` occurs only around atomic
formulas. -/
inductive YNormal : FormP σ → Prop
  /-- A comparison of two normal terms. -/
  | lt {t₁ t₂ : TermP σ} : YNormalT t₁ → YNormalT t₂ → YNormal (.lt t₁ t₂)
  /-- A negation. -/
  | neg {φ : FormP σ} : YNormal φ → YNormal (.neg φ)
  /-- A conjunction. -/
  | and {φ₁ φ₂ : FormP σ} : YNormal φ₁ → YNormal φ₂ → YNormal (.and φ₁ φ₂)
  /-- A `Y`-atom. -/
  | atom {ψ : FormP σ} : YAtomic ψ → YNormal ψ

/-- `Y`-normal form for terms. -/
inductive YNormalT : TermP σ → Prop
  /-- A count of a normal formula. -/
  | countL {φ : FormP σ} : YNormal φ → YNormalT (.countL φ)
  /-- A sum. -/
  | add {t₁ t₂ : TermP σ} : YNormalT t₁ → YNormalT t₂ → YNormalT (.add t₁ t₂)
  /-- The constant `1`. -/
  | one : YNormalT .one

end

/-- `Y^c φ`, the operator `Y` applied `c` times (Appendix F, the shorthand
introduced before `thm:ynf`). -/
def FormP.prevN : ℕ → FormP σ → FormP σ
  | 0, φ => φ
  | c + 1, φ => .prev (prevN c φ)

/-- The outermost `Y` of `Y^{c+1} φ` can be taken as the innermost one. -/
theorem FormP.prevN_succ' (c : ℕ) (φ : FormP σ) :
    FormP.prevN (c + 1) φ = FormP.prevN c (.prev φ) := by
  induction c with
  | zero => rfl
  | succ c ih => rw [FormP.prevN, ih, FormP.prevN]

/-- `Y` adds no depth (Appendix F), so neither does `Y^c`. -/
@[simp] theorem FormP.depth_prevN (c : ℕ) (φ : FormP σ) : (FormP.prevN c φ).depth = φ.depth := by
  induction c with
  | zero => rfl
  | succ c ih => rw [FormP.prevN, FormP.depth, ih]

/-- `Y^c` of a `Y`-atom is a `Y`-atom. -/
theorem YAtomic.prevN {ψ : FormP σ} (h : YAtomic ψ) : ∀ c : ℕ, YAtomic (FormP.prevN c ψ)
  | 0 => h
  | c + 1 => .prev (h.prevN c)

mutual

/-- **The transformation `N^c⟦·⟧` of `thm:ynf`**, with the guard `Y^c MOD_1^0`
at `¬` and `<` that the paper leaves out. -/
def FormP.pushY (c : ℕ) : FormP σ → FormP σ
  | .sym a => FormP.prevN c (.sym a)
  | .mod m r => FormP.prevN c (.mod m r)
  | .prev φ => φ.pushY (c + 1)
  | .lt t₁ t₂ => .and (FormP.prevN c (.mod 1 0)) (.lt (t₁.pushY c) (t₂.pushY c))
  | .neg φ => .and (FormP.prevN c (.mod 1 0)) (.neg (φ.pushY c))
  | .and φ₁ φ₂ => .and (φ₁.pushY c) (φ₂.pushY c)

/-- `N^c⟦·⟧` on terms (Appendix F, proof of `thm:ynf`). -/
def TermP.pushY (c : ℕ) : TermP σ → TermP σ
  | .countL φ => .countL (φ.pushY c)
  | .add t₁ t₂ => .add (t₁.pushY c) (t₂.pushY c)
  | .one => .one

end

mutual

/-- `N^c⟦φ⟧` has the depth of `φ` (Appendix F, proof of `thm:ynf`). -/
@[simp] theorem FormP.depth_pushY : ∀ (φ : FormP σ) (c : ℕ), (φ.pushY c).depth = φ.depth
  | .sym _, c => FormP.depth_prevN c _
  | .mod _ _, c => FormP.depth_prevN c _
  | .prev φ, c => φ.depth_pushY (c + 1)
  | .lt t₁ t₂, c => by
      rw [FormP.pushY, FormP.depth, FormP.depth, FormP.depth_prevN, FormP.depth, FormP.depth,
        t₁.depth_pushY c, t₂.depth_pushY c, Nat.zero_max]
  | .neg φ, c => by
      rw [FormP.pushY, FormP.depth, FormP.depth, FormP.depth_prevN, FormP.depth, FormP.depth,
        φ.depth_pushY c, Nat.zero_max]
  | .and φ₁ φ₂, c => by
      rw [FormP.pushY, FormP.depth, FormP.depth, φ₁.depth_pushY c, φ₂.depth_pushY c]

/-- `N^c⟦t⟧` has the depth of `t` (Appendix F, proof of `thm:ynf`). -/
@[simp] theorem TermP.depth_pushY : ∀ (t : TermP σ) (c : ℕ), (t.pushY c).depth = t.depth
  | .countL φ, c => by rw [TermP.pushY, TermP.depth, TermP.depth, φ.depth_pushY c]
  | .add t₁ t₂, c => by
      rw [TermP.pushY, TermP.depth, TermP.depth, t₁.depth_pushY c, t₂.depth_pushY c]
  | .one, _ => rfl

end

mutual

/-- `N^c⟦φ⟧` is in `Y`-normal form (Appendix F, proof of `thm:ynf`). -/
theorem FormP.yNormal_pushY : ∀ (φ : FormP σ) (c : ℕ), YNormal (φ.pushY c)
  | .sym a, c => .atom ((YAtomic.sym a).prevN c)
  | .mod m r, c => .atom ((YAtomic.mod m r).prevN c)
  | .prev φ, c => φ.yNormal_pushY (c + 1)
  | .lt t₁ t₂, c =>
      .and (.atom ((YAtomic.mod 1 0).prevN c)) (.lt (t₁.yNormalT_pushY c) (t₂.yNormalT_pushY c))
  | .neg φ, c => .and (.atom ((YAtomic.mod 1 0).prevN c)) (.neg (φ.yNormal_pushY c))
  | .and φ₁ φ₂, c => .and (φ₁.yNormal_pushY c) (φ₂.yNormal_pushY c)

/-- `N^c⟦t⟧` is in `Y`-normal form (Appendix F, proof of `thm:ynf`). -/
theorem TermP.yNormalT_pushY : ∀ (t : TermP σ) (c : ℕ), YNormalT (t.pushY c)
  | .countL φ, c => .countL (φ.yNormal_pushY c)
  | .add t₁ t₂, c => .add (t₁.yNormalT_pushY c) (t₂.yNormalT_pushY c)
  | .one, _ => .one

end

end CRASP
end Transformer
