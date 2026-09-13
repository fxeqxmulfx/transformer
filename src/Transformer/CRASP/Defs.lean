/-
# Temporal logic with counting: syntax, semantics, and depth

Huang, Yang, Chiang, Cotterell (as posted) — arXiv:2506.16055v3, "Knee-Deep
in C-RASP: A Transformer Depth Hierarchy" (COLM 2025), §2.2 and
Appendix A.1 (Definitions `def:TLC_semantics` and `def:TLC_depth`).

*Past temporal logic with counting*, `TL[◁#]`, has Boolean-valued formulas
and integer-valued terms:

    φ ::= Q_σ | t₁ < t₂ | ¬φ₁ | φ₁ ∧ φ₂
    t ::= ◁#[φ₁] | t₁ + t₂ | 1

and `TL[◁#, ▷#]` adds the future-counting term `▷#[φ]`.  One inductive pair
carries both: `Form.past` and `Term.past` cut out the fragment that does not
use `▷#`, which is the one equivalent to C-RASP and to future-masked
transformers.

The semantics is a satisfaction relation `w, i ⊨ φ` at a position `i` of a
string `w`, with `◁#[φ]` counting the positions of `w[1:i]` that satisfy `φ`
and `▷#[φ]` those of `w[i:n]`.  Positions are numbered from `1`, as in the
paper.  Satisfaction is `Bool`-valued here because the counting terms filter
by it; the logic is decidable, so nothing is lost.

A string is judged at its *last* position — `w ⊨ φ` iff `w, |w| ⊨ φ` — which
the paper notes is not the usual convention for temporal logic but is the one
that "mimics the behavior of generative transformer decoders".

The syntax carries one further constructor: a *Parikh numerical predicate*
(Definition `def:PNP`), a predicate on the Parikh vector of the string and
the current position.  The paper writes `TL[◁#]^P` for the logic augmented
with these, and the depth-hierarchy proof lives in the augmented logic
throughout; `Form.pnpFree` cuts back to the plain logic.
-/

import Transformer.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- Boolean-valued formulas of `TL[◁#, ▷#]` (Definition `def:TLC_semantics`). -/
inductive Form (σ : Type u) : Type u
  /-- `Q_σ`: the symbol at the current position is `σ`. -/
  | sym : σ → Form σ
  /-- `t₁ < t₂`. -/
  | lt : Term σ → Term σ → Form σ
  /-- `¬φ`. -/
  | neg : Form σ → Form σ
  /-- `φ₁ ∧ φ₂`. -/
  | and : Form σ → Form σ → Form σ
  /-- A Parikh numerical predicate `Π`, given by its partial characteristic
  function of the Parikh vector of the string and the current position
  (Definition `def:PNP`). -/
  | pnp : ((σ → ℕ) → ℕ → Bool) → Form σ

/-- Integer-valued terms of `TL[◁#, ▷#]` (Definition `def:TLC_semantics`). -/
inductive Term (σ : Type u) : Type u
  /-- `◁#[φ]`: how many positions of the prefix `w[1:i]` satisfy `φ`. -/
  | countL : Form σ → Term σ
  /-- `▷#[φ]`: how many positions of the suffix `w[i:n]` satisfy `φ`. -/
  | countR : Form σ → Term σ
  /-- `t₁ + t₂`. -/
  | add : Term σ → Term σ → Term σ
  /-- The constant `1`. -/
  | one : Term σ

end

variable [DecidableEq σ]

/-- The Parikh vector of a string: how often each symbol occurs in it
(Definition `def:Parikh_map`). -/
def parikh (w : List σ) (a : σ) : ℕ := (w.filter (· = a)).length

mutual

/-- `w, i ⊨ φ`, with positions numbered from `1`. -/
def Form.sat (w : List σ) (i : ℕ) : Form σ → Bool
  | .sym a => w[i - 1]? = some a
  | .lt t₁ t₂ => decide (t₁.val w i < t₂.val w i)
  | .neg φ => !φ.sat w i
  | .and φ₁ φ₂ => φ₁.sat w i && φ₂.sat w i
  | .pnp π => π (parikh w) i

/-- `t^{w,i}`, the value of a term at a position. -/
def Term.val (w : List σ) (i : ℕ) : Term σ → ℕ
  | .countL φ => ((List.range' 1 i).filter (fun j => φ.sat w j)).length
  | .countR φ => ((List.range' i (w.length + 1 - i)).filter (fun j => φ.sat w j)).length
  | .add t₁ t₂ => t₁.val w i + t₂.val w i
  | .one => 1

end

/-- `w ⊨ φ`: the string is judged at its last position (Definition
`def:TLC_semantics`). -/
def Form.models (w : List σ) (φ : Form σ) : Prop := φ.sat w w.length = true

instance (w : List σ) (φ : Form σ) : Decidable (φ.models w) := by
  unfold Form.models; infer_instance

/-- `L(φ)`, the language the formula defines. -/
def Form.lang (φ : Form σ) : Set (List σ) := {w | φ.models w}

mutual

/-- The depth of a formula: how deeply counting operators nest
(Definition `def:TLC_depth`). -/
def Form.depth : Form σ → ℕ
  | .sym _ => 0
  | .lt t₁ t₂ => max t₁.depth t₂.depth
  | .neg φ => φ.depth
  | .and φ₁ φ₂ => max φ₁.depth φ₂.depth
  | .pnp _ => 0

/-- The depth of a term (Definition `def:TLC_depth`). -/
def Term.depth : Term σ → ℕ
  | .countL φ => φ.depth + 1
  | .countR φ => φ.depth + 1
  | .add t₁ t₂ => max t₁.depth t₂.depth
  | .one => 0

end

mutual

/-- Whether a formula stays inside the past-only fragment `TL[◁#]`: it uses
no future-counting term (§2.2). -/
def Form.past : Form σ → Bool
  | .sym _ => true
  | .lt t₁ t₂ => t₁.past && t₂.past
  | .neg φ => φ.past
  | .and φ₁ φ₂ => φ₁.past && φ₂.past
  | .pnp _ => true

/-- Whether a term stays inside `TL[◁#]`. -/
def Term.past : Term σ → Bool
  | .countL φ => φ.past
  | .countR _ => false
  | .add t₁ t₂ => t₁.past && t₂.past
  | .one => true

end

mutual

/-- Whether a formula uses no Parikh numerical predicate, i.e. lies in the
plain logic rather than in its augmentation `TL[◁#]^P` (Definition
`def:PNP`). -/
def Form.pnpFree : Form σ → Bool
  | .sym _ => true
  | .lt t₁ t₂ => t₁.pnpFree && t₂.pnpFree
  | .neg φ => φ.pnpFree
  | .and φ₁ φ₂ => φ₁.pnpFree && φ₂.pnpFree
  | .pnp _ => false

/-- Whether a term uses no Parikh numerical predicate. -/
def Term.pnpFree : Term σ → Bool
  | .countL φ => φ.pnpFree
  | .countR φ => φ.pnpFree
  | .add t₁ t₂ => t₁.pnpFree && t₂.pnpFree
  | .one => true

end

/-- `TL[◁#]^P_k`: past-only formulas of depth at most `k`, Parikh numerical
predicates allowed (§4.1). -/
def TLClP (σ : Type u) (k : ℕ) : Set (Form σ) := {φ | φ.past = true ∧ φ.depth ≤ k}

/-- `TL[◁#, ▷#]^P_k`: formulas of depth at most `k`, future counting and
Parikh numerical predicates allowed. -/
def TLCP (σ : Type u) (k : ℕ) : Set (Form σ) := {φ | φ.depth ≤ k}

/-- `TL[◁#]_k`: past-only formulas of depth at most `k` (§2.2). -/
def TLCl (σ : Type u) (k : ℕ) : Set (Form σ) :=
  {φ | φ.past = true ∧ φ.pnpFree = true ∧ φ.depth ≤ k}

/-- `TL[◁#, ▷#]_k`: formulas of depth at most `k`, future counting allowed. -/
def TLC (σ : Type u) (k : ℕ) : Set (Form σ) := {φ | φ.pnpFree = true ∧ φ.depth ≤ k}

/-- A language is definable at depth `k` in `TL[◁#]` when some formula of that
class defines it. -/
def DefinableL (L : Set (List σ)) (k : ℕ) : Prop := ∃ φ ∈ TLCl σ k, φ.lang = L

/-- Definability at depth `k` in `TL[◁#, ▷#]`. -/
def Definable (L : Set (List σ)) (k : ℕ) : Prop := ∃ φ ∈ TLC σ k, φ.lang = L

namespace Form

/-- `t₁ = t₂`, written out with the primitives. -/
def eq (t₁ t₂ : Term σ) : Form σ := .and (.neg (.lt t₁ t₂)) (.neg (.lt t₂ t₁))

@[simp] theorem sat_eq (w : List σ) (i : ℕ) (t₁ t₂ : Term σ) :
    (Form.eq t₁ t₂).sat w i = decide (t₁.val w i = t₂.val w i) := by
  rw [Bool.eq_iff_iff]
  simp only [Form.eq, Form.sat, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true,
    decide_eq_false_iff_not, Nat.not_lt, decide_eq_true_eq]
  omega

omit [DecidableEq σ] in
theorem depth_eq (t₁ t₂ : Term σ) : (Form.eq t₁ t₂).depth = max t₁.depth t₂.depth := by
  simp only [Form.eq, Form.depth]
  omega

end Form

end CRASP
end Transformer
