/-
# `TL[◁#, ▷#]` inside `MAJ²`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, proof of `thm:tlc_to_majtwo`.

A formula read at the position a variable `p` names becomes a `MAJ²` formula
whose only free variable is `p` (`Form.toMaj`).  A term becomes the list of
formulas it counts, over the other variable `u` (`Term.toMajs`): `◁#[ψ]` is
`ψ(u) ∧ u ≤ p`, `▷#[ψ]` is `ψ(u) ∧ p ≤ u`, `1` is `u = p`, and `+` appends,
so the mass of the list is the value of the term (`Term.mass_toMajs`).  A
comparison `t₁ < t₂` is one majority quantifier over `u` (`Maj2.cmpList`),
which is what keeps the depth: the formulas under it are one level below.

A comparison of depth `0` compares two constants.  The paper translates it
like any other, which costs a quantifier; here it is `⊤` or `⊥` outright, since
otherwise a depth-`0` formula would land at depth `1`.
-/

import Transformer.CRASP.MajTwoCount

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- The value of a counting-free term: the number of its `1`s. -/
def Term.ones : Term σ → ℕ
  | .countL _ => 0
  | .countR _ => 0
  | .add t₁ t₂ => t₁.ones + t₂.ones
  | .one => 1

mutual

/-- The translation of a formula read at the position `p` names (Appendix E,
proof of `thm:tlc_to_majtwo`).  Parikh numerical predicates are outside
`TL[◁#,▷#]` and go to `⊥`. -/
def Form.toMaj (p : Var) : Form σ → Maj2 σ
  | .sym a => .sym a p
  | .lt t₁ t₂ =>
      if max t₁.depth t₂.depth = 0 then
        (if t₁.ones < t₂.ones then Maj2.topv p else .lt p p)
      else Maj2.majList p.other (Maj2.cmpList p.other (t₁.toMajs p) (t₂.toMajs p))
  | .neg φ => .neg (φ.toMaj p)
  | .and φ₁ φ₂ => .and (φ₁.toMaj p) (φ₂.toMaj p)
  | .pnp _ => .lt p p

/-- The formulas a term counts, over the variable other than `p`. -/
def Term.toMajs (p : Var) : Term σ → List (Maj2 σ)
  | .countL φ => [.and (φ.toMaj p.other) (.neg (.lt p p.other))]
  | .countR φ => [.and (φ.toMaj p.other) (.neg (.lt p.other p))]
  | .add t₁ t₂ => t₁.toMajs p ++ t₂.toMajs p
  | .one => [.and (.neg (.lt p p.other)) (.neg (.lt p.other p))]

end

/-- **Only `p` is free in the translation.** -/
theorem Form.freeIn_toMaj (p : Var) : ∀ φ : Form σ, (φ.toMaj p).freeIn p.other = false
  | .sym _ => by simp [Form.toMaj, Maj2.freeIn]
  | .lt t₁ t₂ => by
      rw [Form.toMaj]
      split_ifs <;> simp [Maj2.topv, Maj2.majList, Maj2.freeIn]
  | .neg φ => by rw [Form.toMaj, Maj2.freeIn, φ.freeIn_toMaj p]
  | .and φ₁ φ₂ => by
      rw [Form.toMaj, Maj2.freeIn, φ₁.freeIn_toMaj p, φ₂.freeIn_toMaj p, Bool.or_false]
  | .pnp _ => by simp [Form.toMaj, Maj2.freeIn]

/-- The depth of a majority over a list is one more than its formulas'. -/
theorem Maj2.depth_majList_le (u : Var) (l : List (Maj2 σ)) (d : ℕ)
    (h : ∀ g ∈ l, g.depth ≤ d) : (majList u l).depth ≤ d + 1 := by
  rw [majList, depth, Nat.add_comm 1]
  refine Nat.add_le_add_right (Finset.sup_le fun t _ => ?_) 1
  rcases List.mem_cons.mp (List.get_mem (topv u :: l) t) with ht | ht
  · rw [ht]
    simp [topv, depth]
  · exact h _ ht

mutual

/-- **The translation keeps the depth** (Appendix E, `thm:tlc_to_majtwo`). -/
theorem Form.depth_toMaj_le (p : Var) : ∀ φ : Form σ, (φ.toMaj p).depth ≤ φ.depth
  | .sym _ => le_rfl
  | .lt t₁ t₂ => by
      rw [Form.toMaj, Form.depth]
      split_ifs with h
      · simp [Maj2.topv, Maj2.depth]
      · simp [Maj2.depth]
      · have h₁ := Term.depth_toMajs_le p t₁
        have h₂ := Term.depth_toMajs_le p t₂
        refine (Maj2.depth_majList_le _ _ (max t₁.depth t₂.depth - 1) fun g hg => ?_).trans
          (by omega)
        simp only [Maj2.cmpList, List.mem_append, List.mem_map, List.mem_replicate] at hg
        rcases hg with ((hg | ⟨g, hg, rfl⟩) | ⟨-, rfl⟩) | ⟨-, rfl⟩
        · exact (h₂ g hg).trans (by omega)
        · rw [Maj2.depth]
          exact (h₁ g hg).trans (by omega)
        · simp [Maj2.topv, Maj2.depth]
        · simp [Maj2.depth]
  | .neg φ => by rw [Form.toMaj, Maj2.depth, Form.depth]; exact φ.depth_toMaj_le p
  | .and φ₁ φ₂ => by
      rw [Form.toMaj, Maj2.depth, Form.depth]
      exact max_le_max (φ₁.depth_toMaj_le p) (φ₂.depth_toMaj_le p)
  | .pnp _ => by simp [Form.toMaj, Maj2.depth]

/-- The formulas a term counts are one level below it. -/
theorem Term.depth_toMajs_le (p : Var) :
    ∀ t : Term σ, ∀ g ∈ t.toMajs p, g.depth ≤ t.depth - 1
  | .countL φ, g, hg => by
      rw [Term.toMajs, List.mem_singleton] at hg
      subst hg
      have := φ.depth_toMaj_le p.other
      simp only [Maj2.depth, Term.depth]
      omega
  | .countR φ, g, hg => by
      rw [Term.toMajs, List.mem_singleton] at hg
      subst hg
      have := φ.depth_toMaj_le p.other
      simp only [Maj2.depth, Term.depth]
      omega
  | .add t₁ t₂, g, hg => by
      rw [Term.toMajs, List.mem_append] at hg
      rw [Term.depth]
      rcases hg with hg | hg
      · exact (t₁.depth_toMajs_le p g hg).trans (by omega)
      · exact (t₂.depth_toMajs_le p g hg).trans (by omega)
  | .one, g, hg => by
      rw [Term.toMajs, List.mem_singleton] at hg
      subst hg
      simp [Maj2.depth]

end

variable [DecidableEq σ]

/-- A counting-free term is the number of its `1`s. -/
theorem Term.val_eq_ones (w : List σ) (i : ℕ) : ∀ t : Term σ, t.depth = 0 → t.val w i = t.ones
  | .countL _, h => by rw [Term.depth] at h; omega
  | .countR _, h => by rw [Term.depth] at h; omega
  | .add t₁ t₂, h => by
      rw [Term.depth] at h
      rw [Term.val, Term.ones, t₁.val_eq_ones w i (by omega), t₂.val_eq_ones w i (by omega)]
  | .one, _ => rfl

mutual

/-- **The translation preserves meaning** at every position the variable `p`
names (Appendix E, `thm:tlc_to_majtwo`). -/
theorem Form.sat_toMaj (w : List σ) : ∀ (φ : Form σ) (p : Var) (ξ : Var → ℕ),
    φ.pnpFree = true → 1 ≤ ξ p → ξ p ≤ w.length → (φ.toMaj p).sat w ξ = φ.sat w (ξ p)
  | .sym _, _, _, _, _, _ => rfl
  | .lt t₁ t₂, p, ξ, hf, h₁, h₂ => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.toMaj, Form.sat]
      split_ifs with h h'
      · rw [t₁.val_eq_ones w _ (by omega), t₂.val_eq_ones w _ (by omega)]
        simp [h']
      · rw [t₁.val_eq_ones w _ (by omega), t₂.val_eq_ones w _ (by omega)]
        simp [Maj2.sat, h']
      · rw [Maj2.sat_cmpList, t₁.mass_toMajs w p ξ hf.1 h₁ h₂, t₂.mass_toMajs w p ξ hf.2 h₁ h₂]
  | .neg φ, p, ξ, hf, h₁, h₂ => by
      rw [Form.pnpFree] at hf
      rw [Form.toMaj, Maj2.sat, Form.sat, φ.sat_toMaj w p ξ hf h₁ h₂]
  | .and φ₁ φ₂, p, ξ, hf, h₁, h₂ => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.toMaj, Maj2.sat, Form.sat, φ₁.sat_toMaj w p ξ hf.1 h₁ h₂,
        φ₂.sat_toMaj w p ξ hf.2 h₁ h₂]
  | .pnp _, _, _, hf, _, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- **The mass of the formulas a term counts is its value.** -/
theorem Term.mass_toMajs (w : List σ) : ∀ (t : Term σ) (p : Var) (ξ : Var → ℕ),
    t.pnpFree = true → 1 ≤ ξ p → ξ p ≤ w.length →
      Maj2.mass w ξ p.other (t.toMajs p) = t.val w (ξ p)
  | .countL φ, p, ξ, hf, _, h₂ => by
      rw [Term.pnpFree] at hf
      rw [Term.toMajs, Term.val, Maj2.mass, ← sum_Icc_le _ h₂]
      refine Finset.sum_congr rfl fun j hj => ?_
      rw [Finset.mem_Icc] at hj
      have e := φ.sat_toMaj w p.other (Function.update ξ p.other j) hf
        (by rw [Function.update_self]; exact hj.1) (by rw [Function.update_self]; exact hj.2)
      rw [Function.update_self] at e
      simp [Maj2.sat, e]
  | .countR φ, p, ξ, hf, h₁, h₂ => by
      rw [Term.pnpFree] at hf
      rw [Term.toMajs, Term.val, Maj2.mass, ← sum_Icc_ge _ h₁ h₂]
      refine Finset.sum_congr rfl fun j hj => ?_
      rw [Finset.mem_Icc] at hj
      have e := φ.sat_toMaj w p.other (Function.update ξ p.other j) hf
        (by rw [Function.update_self]; exact hj.1) (by rw [Function.update_self]; exact hj.2)
      rw [Function.update_self] at e
      simp [Maj2.sat, e]
  | .add t₁ t₂, p, ξ, hf, h₁, h₂ => by
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      rw [Term.toMajs, Maj2.mass_append, Term.val, t₁.mass_toMajs w p ξ hf.1 h₁ h₂,
        t₂.mass_toMajs w p ξ hf.2 h₁ h₂]
  | .one, p, ξ, _, h₁, h₂ => by
      rw [Term.toMajs, Term.val, Maj2.mass]
      refine (Finset.sum_congr rfl fun j _ => ?_).trans (sum_Icc_eq h₁ h₂)
      simp [Maj2.sat]

end

/-- The hypotheses of `Form.sat_toMaj` are satisfiable, and the translation of
`1 < ◁#[Q_b]` does hold at the second position of `bb`. -/
example : (Form.lt .one (.countL (.sym true))).pnpFree = true ∧ 1 ≤ 2 ∧ 2 ≤ [true, true].length ∧
    ((Form.lt .one (.countL (.sym true))).toMaj .x).sat [true, true] (fun _ => 2) = true := by
  decide

/-- The hypothesis of `Maj2.depth_majList_le` is satisfiable: `⊤` has depth `0`. -/
example : ∀ g ∈ [(Maj2.topv .x : Maj2 Bool)], g.depth ≤ 0 := by
  simp [Maj2.topv, Maj2.depth]

end CRASP
end Transformer
