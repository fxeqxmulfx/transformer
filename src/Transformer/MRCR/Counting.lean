module

public import Transformer.CRASP.BoundedExists
public import Transformer.CRASP.ExtensionsPieces
public import Transformer.CRASP.FormulaBounds

/-!
# Counts of a letter test are cardinalities

In no paper: tools for `Transformer.MRCR.Ordinal` in the logic `TL[◁#]` of arXiv:2506.16055v3,
§2.2. `holdsIn L p` is the disjunction of the letters of `L` that pass `p`: a test of depth zero
(`holdsIn_mem`) that tests `p` on the letter at the position once `L` lists every letter
(`sat_holdsIn`), as `Transformer.CRASP.holds` does over the triples. If a formula `ψ` holds at
the position `j + 1` exactly when `P j`, its count `#ψ` at a position `m` is the number of the
`j < m` with `P j` (`val_countL_eq_card`), and `#ψ = r + 1` has depth one when `ψ` has depth zero
(`eq_countL_ofPos_mem`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.CRASP

/-- `⋁ Q_a` over the letters `a` of `L` with `p a`: a test of depth `0`. -/
def holdsIn {σ : Type*} (L : List σ) (p : σ → Bool) : Form σ :=
  Form.any ((L.filter p).map Form.sym)

/-- Over a list `L` of every letter, `holdsIn L p` tests `p` on the letter at the position. -/
theorem sat_holdsIn {σ : Type*} [DecidableEq σ] {L : List σ} (hL : ∀ a, a ∈ L) (p : σ → Bool)
    (w : List σ) (i : ℕ) :
    (holdsIn L p).sat w i = true ↔ ∃ a, w[i - 1]? = some a ∧ p a = true := by
  simp only [holdsIn, Form.sat_any, List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨_, ⟨a, ⟨-, ha⟩, rfl⟩, hsat⟩
    exact ⟨a, by simpa [Form.sat] using hsat, ha⟩
  · rintro ⟨a, hw, ha⟩
    exact ⟨_, ⟨a, ⟨hL a, ha⟩, rfl⟩, by simp [Form.sat, hw]⟩

/-- The two letters of `Bool`, tested at the first position of `[true]`. -/
example := sat_holdsIn (L := [true, false]) (fun a => by cases a <;> simp) (fun a => a) [true] 1

theorem holdsIn_mem {σ : Type*} (L : List σ) (p : σ → Bool) : holdsIn L p ∈ TLCl σ 0 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨a, -, rfl⟩ := List.mem_map.mp hφ
    exact ⟨rfl, rfl, le_rfl⟩

/-- `#ψ = r + 1` has depth one for a test `ψ` of depth zero. -/
theorem eq_countL_ofPos_mem {σ : Type*} {ψ : Form σ} (hψ : ψ ∈ TLCl σ 0) (r : ℕ) :
    Form.eq (.countL ψ) (Term.ofPos r) ∈ TLCl σ 1 := by
  obtain ⟨h1, h2, h3⟩ := hψ
  refine ⟨?_, ?_, ?_⟩
  · simp [Form.eq, Form.past, Term.past, h1, Term.past_ofPos]
  · simp [Form.eq, Form.pnpFree, Term.pnpFree, h2, Term.pnpFree_ofPos]
  · rw [Form.depth_eq, Term.depth_ofPos, Term.depth]
    omega

/-- The count of `true` against `3`. -/
example := eq_countL_ofPos_mem (holdsIn_mem [true, false] fun a => a) 2

/-- **Counts are cardinalities**: if `ψ` holds at the position `j + 1` exactly when `P j`, for
`j < n`, then `#ψ` at a position `m ≤ n` counts the `j < m` with `P j`. -/
theorem val_countL_eq_card {σ : Type*} [DecidableEq σ] (w : List σ) (ψ : Form σ) {n : ℕ}
    (P : Fin n → Prop) [DecidablePred P] (hψ : ∀ j : Fin n, ψ.sat w (j.val + 1) = true ↔ P j) :
    ∀ m ≤ n, (Term.countL ψ).val w m = (univ.filter fun j : Fin n => j.val < m ∧ P j).card := by
  intro m
  induction m with
  | zero => intro _; simp [Term.val]
  | succ m ih =>
    intro hm
    have hm' : m < n := by omega
    have hsingle : (if P ⟨m, hm'⟩ then 1 else 0) =
        ∑ j : Fin n, if j = ⟨m, hm'⟩ then (if P j then 1 else 0) else 0 :=
      (Fintype.sum_ite_eq' _ fun j => if P j then 1 else 0).symm
    rw [val_countL_succ, ih hm'.le]
    simp only [show ψ.sat w (m + 1) = true ↔ P ⟨m, hm'⟩ from hψ ⟨m, hm'⟩]
    rw [hsingle, card_filter, card_filter, ← sum_add_distrib]
    refine sum_congr rfl fun j _ => ?_
    by_cases hj : j = ⟨m, hm'⟩
    · subst hj
      simp
    · have hj' : j.val ≠ m := fun h => hj (Fin.ext h)
      have hlt : j.val < m + 1 ↔ j.val < m := by omega
      rw [ite_eq_right hj, add_zero]
      simp only [hlt]

/-- Over `[true, false]` the count of `true` at the end is `1`. -/
example := val_countL_eq_card [true, false] (.sym true) (n := 2) (fun j => j = 0)
  (fun j => by fin_cases j <;> decide)
end Transformer.MRCR
