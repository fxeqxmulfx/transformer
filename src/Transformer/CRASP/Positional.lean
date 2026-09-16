/-
# `TL[◁#]^pos`: position encodings on the logic side

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E (`app:pes`, `app:tlclpos`): the extension of `TL[◁#]`
by the predicates `MOD_m^r` and the operator `Y`, its `Y`-normal form
(`thm:ynf`), the reduction to plain `TL[◁#]` (`lem:tlclpos_reduction`) and the
depth hierarchy that follows (`thm:tlclpos_depth_hierarchy`).

`TL[◁#]^pos` is `TL[◁#]` — past-only counting, no Parikh numerical predicates
— with two additions: `MOD_m^r` holds at position `i` when `i ≡ r (mod m)`,
and `Y φ` holds at `i` when `i > 1` and `φ` holds at `i − 1`.  The paper notes
it is the logic `C-RASP[local, periodic]` of Huang et al. (2025).  Its two
sublogics `TL[◁#, MOD]` and `TL[◁#, Y]` are the `MOD`-only and `Y`-only
fragments, which are what the sinusoidal/RoPE and the ALiBi transformers of
Appendix E simulate; those transformers are not yet formalized.

The separating language is `E_k`, the language `altPlus` of
`Transformer.CRASP.PiecewiseTestable` with a neutral letter `e` allowed to be
inserted anywhere; the neutral letter is `none` of `Option σ`, and deleting it
is `List.reduceOption`.
-/

import Transformer.CRASP.TLCDepth

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- Formulas of `TL[◁#]^pos` (Appendix E, `app:tlclpos`). -/
inductive FormP (σ : Type u) : Type u
  /-- `Q_σ`. -/
  | sym : σ → FormP σ
  /-- `MOD_m^r`: the current position is congruent to `r` modulo `m`. -/
  | mod : ℕ → ℕ → FormP σ
  /-- `Y φ`: `φ` holds at the previous position, and there is one. -/
  | prev : FormP σ → FormP σ
  /-- `t₁ < t₂`. -/
  | lt : TermP σ → TermP σ → FormP σ
  /-- `¬φ`. -/
  | neg : FormP σ → FormP σ
  /-- `φ₁ ∧ φ₂`. -/
  | and : FormP σ → FormP σ → FormP σ

/-- Terms of `TL[◁#]^pos`: past counting only. -/
inductive TermP (σ : Type u) : Type u
  /-- `◁#[φ]`. -/
  | countL : FormP σ → TermP σ
  /-- `t₁ + t₂`. -/
  | add : TermP σ → TermP σ → TermP σ
  /-- The constant `1`. -/
  | one : TermP σ

end

variable [DecidableEq σ]

mutual

/-- `w, i ⊨ φ` for `TL[◁#]^pos` (Appendix E, the two new semantic rules). -/
def FormP.sat (w : List σ) (i : ℕ) : FormP σ → Bool
  | .sym a => w[i - 1]? = some a
  | .mod m r => decide (i % m = r % m)
  | .prev φ => decide (1 < i) && φ.sat w (i - 1)
  | .lt t₁ t₂ => decide (t₁.val w i < t₂.val w i)
  | .neg φ => !φ.sat w i
  | .and φ₁ φ₂ => φ₁.sat w i && φ₂.sat w i

/-- `t^{w,i}` for `TL[◁#]^pos`. -/
def TermP.val (w : List σ) (i : ℕ) : TermP σ → ℕ
  | .countL φ => ((List.range' 1 i).filter (fun j => φ.sat w j)).length
  | .add t₁ t₂ => t₁.val w i + t₂.val w i
  | .one => 1

end

/-- `w ⊨ φ`: the string is judged at its last position. -/
def FormP.models (w : List σ) (φ : FormP σ) : Prop := φ.sat w w.length = true

instance decidableModelsFormP (w : List σ) (φ : FormP σ) : Decidable (φ.models w) := by
  unfold FormP.models; infer_instance

/-- The language a formula defines. -/
def FormP.lang (φ : FormP σ) : Set (List σ) := {w | φ.models w}

mutual

/-- Counting depth; `Y` and `MOD` add none (Appendix E). -/
def FormP.depth : FormP σ → ℕ
  | .sym _ => 0
  | .mod _ _ => 0
  | .prev φ => φ.depth
  | .lt t₁ t₂ => max t₁.depth t₂.depth
  | .neg φ => φ.depth
  | .and φ₁ φ₂ => max φ₁.depth φ₂.depth

/-- The depth of a term. -/
def TermP.depth : TermP σ → ℕ
  | .countL φ => φ.depth + 1
  | .add t₁ t₂ => max t₁.depth t₂.depth
  | .one => 0

end

mutual

/-- Whether a formula uses no `MOD`, i.e. lies in `TL[◁#, Y]`. -/
def FormP.modFree : FormP σ → Bool
  | .sym _ => true
  | .mod _ _ => false
  | .prev φ => φ.modFree
  | .lt t₁ t₂ => t₁.modFree && t₂.modFree
  | .neg φ => φ.modFree
  | .and φ₁ φ₂ => φ₁.modFree && φ₂.modFree

/-- Whether a term uses no `MOD`. -/
def TermP.modFree : TermP σ → Bool
  | .countL φ => φ.modFree
  | .add t₁ t₂ => t₁.modFree && t₂.modFree
  | .one => true

end

mutual

/-- Whether a formula uses no `Y`, i.e. lies in `TL[◁#, MOD]`. -/
def FormP.prevFree : FormP σ → Bool
  | .sym _ => true
  | .mod _ _ => true
  | .prev _ => false
  | .lt t₁ t₂ => t₁.prevFree && t₂.prevFree
  | .neg φ => φ.prevFree
  | .and φ₁ φ₂ => φ₁.prevFree && φ₂.prevFree

/-- Whether a term uses no `Y`. -/
def TermP.prevFree : TermP σ → Bool
  | .countL φ => φ.prevFree
  | .add t₁ t₂ => t₁.prevFree && t₂.prevFree
  | .one => true

end

/-- `TL[◁#]^pos_k` (Appendix E). -/
def TLClPos (σ : Type u) (k : ℕ) : Set (FormP σ) := {φ | φ.depth ≤ k}

/-- `TL[◁#, MOD]_k`, the `Y`-free fragment. -/
def TLClMod (σ : Type u) (k : ℕ) : Set (FormP σ) := {φ | φ.prevFree = true ∧ φ.depth ≤ k}

/-- `TL[◁#, Y]_k`, the `MOD`-free fragment. -/
def TLClY (σ : Type u) (k : ℕ) : Set (FormP σ) := {φ | φ.modFree = true ∧ φ.depth ≤ k}

/-- Definability at depth `k` in `TL[◁#]^pos`. -/
def DefinablePos (L : Set (List σ)) (k : ℕ) : Prop := ∃ φ ∈ TLClPos σ k, φ.lang = L

/-- Definability at depth `k` in `TL[◁#, MOD]`. -/
def DefinableMod (L : Set (List σ)) (k : ℕ) : Prop := ∃ φ ∈ TLClMod σ k, φ.lang = L

/-- Definability at depth `k` in `TL[◁#, Y]`. -/
def DefinableY (L : Set (List σ)) (k : ℕ) : Prop := ∃ φ ∈ TLClY σ k, φ.lang = L

/-- `TL[◁#, MOD]_k` is a fragment of `TL[◁#]^pos_k`: the two ask for the same
depth bound, and the full logic does not ask for `Y`-freeness. -/
theorem TLClMod_subset_TLClPos (σ : Type u) (k : ℕ) : TLClMod σ k ⊆ TLClPos σ k :=
  fun _ hφ => hφ.2

/-- `TL[◁#, Y]_k` is a fragment of `TL[◁#]^pos_k`, for the same reason as
`TLClMod_subset_TLClPos`. -/
theorem TLClY_subset_TLClPos (σ : Type u) (k : ℕ) : TLClY σ k ⊆ TLClPos σ k :=
  fun _ hφ => hφ.2

end CRASP
end Transformer
