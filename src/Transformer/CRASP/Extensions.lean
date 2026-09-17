/-
# Syntactic sugar: the `?` operator and the unmasked/strict counting terms

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A.3 ("Extensions to `TL[◁#,▷#]`"): the conditional term
`φ ? t : t'`, the unmasked counting operator `#`, the strict counting
operators `◁#_<` and `▷#_>`, the elimination lemma of Yang & Chiang for `?`,
and `thm:strict` for the three counting variants.

None of the four adds expressive power or depth, and the two lemmas say so.
To state them one needs a syntax that *has* them, so `FormX`/`TermX` below is
`Form`/`Term` with the four extra constructors; `Form.toX` embeds the plain
syntax into it, and `exists_form_of_formX` (in `CRASP.ExtensionsElim`) is both
lemmas at once — every extended formula is matched by a plain one of the same
language and the same depth.

**A typo.**  Appendix A.3 defines the strict left-counting operator as
`◁#_<[φ]^{w,i} = |{j ∈ [i, |w|−1] | w,j ⊨ φ}|`.  That range is a *right*-hand
range; it contradicts both the operator's name and the rewriting rule
`◁#_<[φ] ≡ ◁#[φ] − (φ ? 1 : 0)` given a few lines below, which force
`[1, i−1]`.  The definition below takes `[1, i−1]`, and the proof of
`lem:cropping` in Appendix D uses `◁#_<` in exactly that sense.
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- Formulas of `TL[◁#,▷#]` extended with the sugar of Appendix A.3. -/
inductive FormX (σ : Type u) : Type u
  /-- `Q_σ`. -/
  | sym : σ → FormX σ
  /-- `t₁ < t₂`. -/
  | lt : TermX σ → TermX σ → FormX σ
  /-- `¬φ`. -/
  | neg : FormX σ → FormX σ
  /-- `φ₁ ∧ φ₂`. -/
  | and : FormX σ → FormX σ → FormX σ
  /-- A Parikh numerical predicate. -/
  | pnp : ((σ → ℕ) → ℕ → Bool) → FormX σ

/-- Counting terms extended with the sugar of Appendix A.3. -/
inductive TermX (σ : Type u) : Type u
  /-- `◁#[φ]`, counting `[1, i]`. -/
  | countL : FormX σ → TermX σ
  /-- `▷#[φ]`, counting `[i, |w|]`. -/
  | countR : FormX σ → TermX σ
  /-- `#[φ]`, the unmasked operator, counting `[1, |w|]`. -/
  | countAll : FormX σ → TermX σ
  /-- `◁#_<[φ]`, counting `[1, i−1]`. -/
  | countLStrict : FormX σ → TermX σ
  /-- `▷#_>[φ]`, counting `[i+1, |w|]`. -/
  | countRStrict : FormX σ → TermX σ
  /-- `φ ? t : t'`, the conditional term. -/
  | cond : FormX σ → TermX σ → TermX σ → TermX σ
  /-- `t₁ + t₂`. -/
  | add : TermX σ → TermX σ → TermX σ
  /-- The constant `1`. -/
  | one : TermX σ

end

mutual

/-- Counting depth for the extended syntax.  The `?` operator adds nothing
("does not increase its expressive power or affect the depth of formulas"). -/
def FormX.depth : FormX σ → ℕ
  | .sym _ => 0
  | .lt t₁ t₂ => max t₁.depth t₂.depth
  | .neg φ => φ.depth
  | .and φ₁ φ₂ => max φ₁.depth φ₂.depth
  | .pnp _ => 0

/-- Counting depth of an extended term. -/
def TermX.depth : TermX σ → ℕ
  | .countL φ | .countR φ | .countAll φ | .countLStrict φ | .countRStrict φ => φ.depth + 1
  | .cond φ t₁ t₂ => max φ.depth (max t₁.depth t₂.depth)
  | .add t₁ t₂ => max t₁.depth t₂.depth
  | .one => 0

end

mutual

/-- The plain syntax sits inside the extended one. -/
def Form.toX : Form σ → FormX σ
  | .sym a => .sym a
  | .lt t₁ t₂ => .lt t₁.toX t₂.toX
  | .neg φ => .neg φ.toX
  | .and φ₁ φ₂ => .and φ₁.toX φ₂.toX
  | .pnp π => .pnp π

/-- The plain counting terms sit inside the extended ones. -/
def Term.toX : Term σ → TermX σ
  | .countL φ => .countL φ.toX
  | .countR φ => .countR φ.toX
  | .add t₁ t₂ => .add t₁.toX t₂.toX
  | .one => .one

end

mutual

/-- The embedding is depth-preserving, so the plain formulas already realize
the easy half of `exists_form_of_formX`. -/
theorem Form.depth_toX : ∀ φ : Form σ, φ.toX.depth = φ.depth
  | .sym _ => rfl
  | .lt t₁ t₂ => by rw [Form.toX, FormX.depth, Form.depth, t₁.depth_toX, t₂.depth_toX]
  | .neg φ => by rw [Form.toX, FormX.depth, Form.depth, φ.depth_toX]
  | .and φ₁ φ₂ => by rw [Form.toX, FormX.depth, Form.depth, φ₁.depth_toX, φ₂.depth_toX]
  | .pnp _ => rfl

/-- The embedding is depth-preserving on terms. -/
theorem Term.depth_toX : ∀ t : Term σ, t.toX.depth = t.depth
  | .countL φ => by rw [Term.toX, TermX.depth, Term.depth, φ.depth_toX]
  | .countR φ => by rw [Term.toX, TermX.depth, Term.depth, φ.depth_toX]
  | .add t₁ t₂ => by rw [Term.toX, TermX.depth, Term.depth, t₁.depth_toX, t₂.depth_toX]
  | .one => rfl

end

variable [DecidableEq σ]

mutual

/-- Satisfaction for the extended syntax, extending `Form.sat`. -/
def FormX.sat (w : List σ) (i : ℕ) : FormX σ → Bool
  | .sym a => w[i - 1]? = some a
  | .lt t₁ t₂ => decide (t₁.val w i < t₂.val w i)
  | .neg φ => !φ.sat w i
  | .and φ₁ φ₂ => φ₁.sat w i && φ₂.sat w i
  | .pnp π => π (parikh w) i

/-- The value of an extended term, extending `Term.val`. -/
def TermX.val (w : List σ) (i : ℕ) : TermX σ → ℕ
  | .countL φ => ((List.range' 1 i).filter (fun j => φ.sat w j)).length
  | .countR φ => ((List.range' i (w.length + 1 - i)).filter (fun j => φ.sat w j)).length
  | .countAll φ => ((List.range' 1 w.length).filter (fun j => φ.sat w j)).length
  | .countLStrict φ => ((List.range' 1 (i - 1)).filter (fun j => φ.sat w j)).length
  | .countRStrict φ => ((List.range' (i + 1) (w.length - i)).filter (fun j => φ.sat w j)).length
  | .cond φ t₁ t₂ => if φ.sat w i then t₁.val w i else t₂.val w i
  | .add t₁ t₂ => t₁.val w i + t₂.val w i
  | .one => 1

end

/-- An extended formula is judged at the last position, as in `Form.models`. -/
def FormX.models (w : List σ) (φ : FormX σ) : Prop := φ.sat w w.length = true

instance (w : List σ) (φ : FormX σ) : Decidable (φ.models w) := by
  unfold FormX.models; infer_instance

/-- The language of an extended formula. -/
def FormX.lang (φ : FormX σ) : Set (List σ) := {w | φ.models w}

mutual

/-- The embedding preserves satisfaction. -/
theorem Form.sat_toX (w : List σ) (i : ℕ) : ∀ φ : Form σ, φ.toX.sat w i = φ.sat w i
  | .sym a => rfl
  | .lt t₁ t₂ => by rw [Form.toX, FormX.sat, Form.sat, t₁.val_toX w i, t₂.val_toX w i]
  | .neg φ => by rw [Form.toX, FormX.sat, Form.sat, φ.sat_toX w i]
  | .and φ₁ φ₂ => by
      rw [Form.toX, FormX.sat, Form.sat, φ₁.sat_toX w i, φ₂.sat_toX w i]
  | .pnp _ => rfl

/-- The embedding preserves values. -/
theorem Term.val_toX (w : List σ) (i : ℕ) : ∀ t : Term σ, t.toX.val w i = t.val w i
  | .countL φ => by
      rw [Term.toX, TermX.val, Term.val]
      exact congrArg List.length (List.filter_congr fun j _ => by rw [φ.sat_toX w j])
  | .countR φ => by
      rw [Term.toX, TermX.val, Term.val]
      exact congrArg List.length (List.filter_congr fun j _ => by rw [φ.sat_toX w j])
  | .add t₁ t₂ => by rw [Term.toX, TermX.val, Term.val, t₁.val_toX w i, t₂.val_toX w i]
  | .one => rfl

end

/-- The embedding preserves languages. -/
theorem Form.lang_toX (φ : Form σ) : φ.toX.lang = φ.lang := by
  ext w
  simp only [FormX.lang, Form.lang, FormX.models, Form.models, Set.mem_ofPred_eq,
    φ.sat_toX w w.length]

/-- The sugar is not vacuous: `◁#_<[Q_a] < ◁#[Q_a]` says that the current
position carries an `a`, which is the content of the rewriting rule
`◁#_<[φ] ≡ ◁#[φ] − (φ ? 1 : 0)`. -/
example : (FormX.lt (.countLStrict (.sym true)) (.countL (.sym true))).models [false, true] := by
  decide

/-- And it fails one position earlier. -/
example : ¬ (FormX.lt (.countLStrict (.sym true)) (.countL (.sym true))).models [false] := by
  decide

end CRASP
end Transformer
