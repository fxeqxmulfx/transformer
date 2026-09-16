/-
# `TL[◁#]^pos`: normal form, reduction, and the depth hierarchy

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E (`app:tlclpos`, "Depth Hierarchy"): `thm:ynf`,
`lem:tlclpos_reduction`, `thm:tlclpos_depth_hierarchy`.

A formula is in `Y`-normal form when `Y` occurs only around atomic formulas;
every formula has an equivalent one of the same depth in that form.  The
reduction then pulls a `TL[◁#]^pos_k` formula over `Σ ∪ {e}` back along the
string map `w ↦ e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` to a plain `TL[◁#]_k` formula
over `Σ`, which turns the hierarchy of `thm:TLCl_depth` into a hierarchy for
`TL[◁#]^pos` with the separating language `E_{k+1}`.
-/

import Mathlib.Data.List.ReduceOption
import Transformer.CRASP.Positional

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- `Y`-atoms: `Y` applied to a symbol or a modular predicate, any number of
times (Appendix E, the `ψ` line of the `Y`-normal form grammar). -/
inductive YAtomic : FormP σ → Prop
  /-- `Q_σ` is a `Y`-atom. -/
  | sym (a : σ) : YAtomic (.sym a)
  /-- `MOD_m^r` is a `Y`-atom. -/
  | mod (m r : ℕ) : YAtomic (.mod m r)
  /-- `Y` of a `Y`-atom is a `Y`-atom. -/
  | prev {ψ : FormP σ} : YAtomic ψ → YAtomic (.prev ψ)

mutual

/-- **`Y`-normal form** (Appendix E): `Y` occurs only around atomic
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

variable [DecidableEq σ]

/-- **Lemma `thm:ynf`.**  Every `TL[◁#]^pos_k` formula has an equivalent
`TL[◁#]^pos_k` formula in `Y`-normal form, of the same depth: the
transformation `N^c⟦·⟧` of the proof pushes `Y` inwards while remembering how
many have been pushed. -/
theorem exists_yNormal (k : ℕ) (φ : FormP σ) (hφ : φ ∈ TLClPos σ k) :
    ∃ φ' ∈ TLClPos σ k, YNormal φ' ∧ ∀ (w : List σ) (i : ℕ), φ.sat w i = φ'.sat w i :=
  sorry

/-- The hypothesis is satisfiable: `Q_a` lies in `TL[◁#]^pos_k` at every
depth. -/
example (a : σ) (k : ℕ) : (FormP.sym a : FormP σ) ∈ TLClPos σ k := Nat.zero_le k

/-- The string map `f` of `lem:tlclpos_reduction`:
`w₁ ⋯ wₙ ↦ e^r w₁ e^{r−1} w₂ e^{r−1} ⋯ wₙ e^{r−1}`, with the neutral letter
`e` written `none`. -/
def spread (r : ℕ) (w : List σ) : List (Option σ) :=
  List.replicate r none ++ w.flatMap fun a => some a :: List.replicate (r - 1) none

omit [DecidableEq σ] in
@[simp] theorem spread_nil (r : ℕ) : spread (σ := σ) r [] = List.replicate r none := by
  simp [spread]

omit [DecidableEq σ] in
/-- `spread` only inserts neutral letters, so deleting them gives the string
back: `spread r` is a section of `List.reduceOption`. -/
@[simp] theorem reduceOption_spread (r : ℕ) (w : List σ) : (spread r w).reduceOption = w := by
  rw [spread, List.reduceOption_append, List.reduceOption_replicate_none, List.nil_append]
  induction w with
  | nil => simp
  | cons a l ih =>
      rw [List.flatMap_cons, List.reduceOption_append, ih, List.reduceOption_cons_of_some,
        List.reduceOption_replicate_none]
      rfl

/-- **Lemma `lem:tlclpos_reduction`.**  A `TL[◁#]^pos_k` formula over
`Σ ∪ {e}` is pulled back along `spread r`, for a suitable `r ≥ 1`, to a plain
`TL[◁#]_k` formula over `Σ`.  The `r` of the proof is `M(Y+1)` for `M` the
lcm of the moduli of `φ` and `Y` its `Y`-depth, which makes `r` divisible by
every modulus and larger than every `Y`-nesting. -/
theorem exists_form_of_formP (k : ℕ) (φ : FormP (Option σ)) (hφ : φ ∈ TLClPos (Option σ) k) :
    ∃ r : ℕ, 1 ≤ r ∧ ∃ φ' ∈ TLCl σ k, ∀ w : List σ, FormP.models (spread r w) φ ↔ w ∈ φ'.lang :=
  sorry

/-- The hypothesis is satisfiable: `Q_{some a}` lies in `TL[◁#]^pos_k`. -/
example (a : σ) (k : ℕ) : (FormP.sym (some a) : FormP (Option σ)) ∈ TLClPos (Option σ) k :=
  Nat.zero_le k

/-- `E_k`, the language `altPlus` with a neutral letter: the strings over
`Σ ∪ {e}` that lie in `A_k` once every `e` is deleted (Appendix E,
`app:tlclpos`). -/
def altPlusNeutral (k : ℕ) : Set (List (Option Bool)) := List.reduceOption ⁻¹' altPlus false k

/-- **Theorem `thm:tlclpos_depth_hierarchy`.**  `E_{k+1}` is definable in
`TL[◁#]^pos_{k+1}` but not in `TL[◁#]^pos_k`: a depth-`k` definition would
reduce along `spread` to a `TL[◁#]_k` definition of `A_{k+1}`, contradicting
`thm:TLCl_depth`.

That is the half proved here, and it is proved in those words: `spread` is a
section of `List.reduceOption`, so pulling `E_{k+1}` back along it gives
`A_{k+1}` on the nose.  The positive half needs a `TL[◁#]^pos` formula and so
an embedding of the plain syntax into the positional one, which this
development does not have; it is left open. -/
theorem definablePos_altPlusNeutral (k : ℕ) (hk : 0 < k) :
    DefinablePos (altPlusNeutral (k + 1)) (k + 1) ∧
      ¬ DefinablePos (altPlusNeutral (k + 1)) k := by
  refine ⟨sorry, ?_⟩
  rintro ⟨φ, hφ, hlang⟩
  obtain ⟨r, -, φ', hφ', hiff⟩ := exists_form_of_formP (σ := Bool) k φ hφ
  refine (definableL_altPlus k hk).2 ⟨φ', hφ', ?_⟩
  ext w
  rw [← hiff w]
  have hmem : FormP.models (spread r w) φ ↔ spread r w ∈ altPlusNeutral (k + 1) := by
    rw [← hlang]
    exact Iff.rfl
  rw [hmem, altPlusNeutral, Set.mem_preimage, reduceOption_spread]

end CRASP
end Transformer
