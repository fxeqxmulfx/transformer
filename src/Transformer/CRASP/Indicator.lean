/-
# Sums and indicators in a comparison cost no depth

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A.3: the conditional term `φ ? t : t'` "does not increase
[the] expressive power or affect the depth of formulas", the elimination lemma
of Yang & Chiang.  The proof of `lem:tlclpos_reduction` in Appendix E compares
sums of counts, constants and indicators `φ ? 1 : 0`.

This is the case of that lemma the reduction needs: a comparison

    c₁ + Σ S₁ + Σ_{ψ ∈ L₁} (ψ ? 1 : 0)  <  c₂ + Σ S₂ + Σ_{ψ ∈ L₂} (ψ ? 1 : 0)

of constants, sums of terms and indicators is a plain formula.  Splitting on
each `ψ` in turn, `(ψ ∧ ⋯ + 1 < ⋯) ∨ (¬ψ ∧ ⋯ < ⋯)`, removes the indicators one
at a time, and the depth of the result is the largest depth among the terms
and the `ψ`s.  There is no constant `0`, so a sum is started from `c + 1` on
both sides.
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- `c + 1 + Σ S`, a sum of terms started from a positive constant. -/
def Term.sum (S : List (Term σ)) (c : ℕ) : Term σ := S.foldr .add (Term.ofPos c)

namespace Form

/-- `ψ ? φ₁ : φ₂` on formulas: `(ψ ∧ φ₁) ∨ (¬ψ ∧ φ₂)`. -/
def ite (ψ φ₁ φ₂ : Form σ) : Form σ := Form.or (.and ψ φ₁) (.and (.neg ψ) φ₂)

/-- `t₁ < t₂ + Σ_{ψ ∈ L₂} (ψ ? 1 : 0)`. -/
def ltIndR (t₁ : Term σ) : Term σ → List (Form σ) → Form σ
  | t₂, [] => .lt t₁ t₂
  | t₂, ψ :: L₂ => ite ψ (ltIndR t₁ (.add t₂ .one) L₂) (ltIndR t₁ t₂ L₂)

/-- `t₁ + Σ_{ψ ∈ L₁} (ψ ? 1 : 0) < t₂ + Σ_{ψ ∈ L₂} (ψ ? 1 : 0)`. -/
def ltInd : Term σ → List (Form σ) → Term σ → List (Form σ) → Form σ
  | t₁, [], t₂, L₂ => ltIndR t₁ t₂ L₂
  | t₁, ψ :: L₁, t₂, L₂ => ite ψ (ltInd (.add t₁ .one) L₁ t₂ L₂) (ltInd t₁ L₁ t₂ L₂)

/-- `c₁ + Σ S₁ + Σ_{ψ ∈ L₁} (ψ ? 1 : 0) < c₂ + Σ S₂ + Σ_{ψ ∈ L₂} (ψ ? 1 : 0)`. -/
def ltSum (c₁ : ℕ) (S₁ : List (Term σ)) (L₁ : List (Form σ)) (c₂ : ℕ) (S₂ : List (Term σ))
    (L₂ : List (Form σ)) : Form σ :=
  ltInd (Term.sum S₁ c₁) L₁ (Term.sum S₂ c₂) L₂

/-- The conditional preserves membership in `TL[◁#]_d`. -/
theorem ite_mem_TLCl {d : ℕ} {ψ φ₁ φ₂ : Form σ} (hψ : ψ ∈ TLCl σ d) (h₁ : φ₁ ∈ TLCl σ d)
    (h₂ : φ₂ ∈ TLCl σ d) : ite ψ φ₁ φ₂ ∈ TLCl σ d := by
  obtain ⟨hψp, hψn, hψd⟩ := hψ
  obtain ⟨h₁p, h₁n, h₁d⟩ := h₁
  obtain ⟨h₂p, h₂n, h₂d⟩ := h₂
  refine ⟨?_, ?_, ?_⟩
  · simp [ite, Form.or, Form.past, hψp, h₁p, h₂p]
  · simp [ite, Form.or, Form.pnpFree, hψn, h₁n, h₂n]
  · simp [ite, Form.or, Form.depth, hψd, h₁d, h₂d]

/-- Adding `1` to either side of a comparison keeps it in `TL[◁#]_d`. -/
theorem lt_add_one_mem_TLCl {d : ℕ} {t₁ t₂ : Term σ} (h : Form.lt t₁ t₂ ∈ TLCl σ d) :
    Form.lt (.add t₁ .one) t₂ ∈ TLCl σ d ∧ Form.lt t₁ (.add t₂ .one) ∈ TLCl σ d := by
  obtain ⟨hp, hn, hd⟩ := h
  simp only [TLCl, Set.mem_ofPred_eq, Form.past, Form.pnpFree, Form.depth, Term.past,
    Term.pnpFree, Term.depth, Bool.and_true, Nat.max_zero] at hp hn hd ⊢
  exact ⟨⟨hp, hn, hd⟩, ⟨hp, hn, hd⟩⟩

/-- The comparison with indicators on the right lies in `TL[◁#]_d` when the
plain comparison and every indicator formula do. -/
theorem ltIndR_mem_TLCl {d : ℕ} (t₁ : Term σ) :
    ∀ (t₂ : Term σ) (L₂ : List (Form σ)), Form.lt t₁ t₂ ∈ TLCl σ d →
      (∀ ψ ∈ L₂, ψ ∈ TLCl σ d) → ltIndR t₁ t₂ L₂ ∈ TLCl σ d
  | _, [], h, _ => h
  | t₂, ψ :: L₂, h, hL =>
      ite_mem_TLCl (hL ψ List.mem_cons_self)
        (ltIndR_mem_TLCl t₁ _ L₂ (lt_add_one_mem_TLCl h).2 fun φ hφ => hL φ (.tail ψ hφ))
        (ltIndR_mem_TLCl t₁ t₂ L₂ h fun φ hφ => hL φ (.tail ψ hφ))

/-- The comparison with indicators lies in `TL[◁#]_d` when the plain comparison
and every indicator formula do. -/
theorem ltInd_mem_TLCl {d : ℕ} :
    ∀ (t₁ : Term σ) (L₁ : List (Form σ)) (t₂ : Term σ) (L₂ : List (Form σ)),
      Form.lt t₁ t₂ ∈ TLCl σ d → (∀ ψ ∈ L₁, ψ ∈ TLCl σ d) → (∀ ψ ∈ L₂, ψ ∈ TLCl σ d) →
        ltInd t₁ L₁ t₂ L₂ ∈ TLCl σ d
  | t₁, [], t₂, L₂, h, _, h₂ => ltIndR_mem_TLCl t₁ t₂ L₂ h h₂
  | t₁, ψ :: L₁, t₂, L₂, h, h₁, h₂ =>
      ite_mem_TLCl (h₁ ψ List.mem_cons_self)
        (ltInd_mem_TLCl _ L₁ t₂ L₂ (lt_add_one_mem_TLCl h).1 (fun φ hφ => h₁ φ (.tail ψ hφ)) h₂)
        (ltInd_mem_TLCl t₁ L₁ t₂ L₂ h (fun φ hφ => h₁ φ (.tail ψ hφ)) h₂)

/-- A comparison of two sums lies in `TL[◁#]_d` when every summand does, and so
does its version with indicators. -/
theorem ltSum_mem_TLCl {d : ℕ} (c₁ : ℕ) {S₁ : List (Term σ)} {L₁ : List (Form σ)} (c₂ : ℕ)
    {S₂ : List (Term σ)} {L₂ : List (Form σ)}
    (hS₁ : ∀ s ∈ S₁, s.past = true ∧ s.pnpFree = true ∧ s.depth ≤ d)
    (hL₁ : ∀ ψ ∈ L₁, ψ ∈ TLCl σ d)
    (hS₂ : ∀ s ∈ S₂, s.past = true ∧ s.pnpFree = true ∧ s.depth ≤ d)
    (hL₂ : ∀ ψ ∈ L₂, ψ ∈ TLCl σ d) : ltSum c₁ S₁ L₁ c₂ S₂ L₂ ∈ TLCl σ d := by
  have hsum : ∀ (S : List (Term σ)) (c : ℕ),
      (∀ s ∈ S, s.past = true ∧ s.pnpFree = true ∧ s.depth ≤ d) →
        (Term.sum S c).past = true ∧ (Term.sum S c).pnpFree = true ∧ (Term.sum S c).depth ≤ d := by
    intro S c hS
    induction S with
    | nil =>
        refine ⟨Term.past_ofPos c, ?_, by simp [Term.sum]⟩
        induction c with
        | zero => rfl
        | succ c ih => simp [Term.sum, Term.ofPos, Term.pnpFree] at ih ⊢; exact ih
    | cons s S ih =>
        obtain ⟨hp, hn, hd⟩ := ih fun s' hs' => hS s' (.tail s hs')
        obtain ⟨hsp, hsn, hsd⟩ := hS s List.mem_cons_self
        simp only [Term.sum, List.foldr_cons, Term.past, Term.pnpFree, Term.depth] at hp hn hd ⊢
        exact ⟨by rw [hsp, hp]; rfl, by rw [hsn, hn]; rfl, max_le hsd hd⟩
  obtain ⟨h₁p, h₁n, h₁d⟩ := hsum S₁ c₁ hS₁
  obtain ⟨h₂p, h₂n, h₂d⟩ := hsum S₂ c₂ hS₂
  refine ltInd_mem_TLCl _ L₁ _ L₂ ⟨?_, ?_, max_le h₁d h₂d⟩ hL₁ hL₂
  · rw [Form.past, h₁p, h₂p]; rfl
  · rw [Form.pnpFree, h₁n, h₂n]; rfl

/-- The hypotheses of the membership lemmas are satisfiable: `Q_a` and `1 < 1`
are formulas of `TL[◁#]_d`, and `1` is a past-only, PNP-free term of depth `0`. -/
example (a : σ) (d : ℕ) :
    Form.sym a ∈ TLCl σ d ∧ Form.lt .one .one ∈ TLCl σ d ∧
      (Term.one : Term σ).past = true ∧ (Term.one : Term σ).pnpFree = true ∧
        (Term.one : Term σ).depth ≤ d :=
  ⟨⟨rfl, rfl, Nat.zero_le d⟩, ⟨rfl, rfl, Nat.zero_le d⟩, rfl, rfl, Nat.zero_le d⟩

variable [DecidableEq σ]

/-- The conditional means what it says. -/
@[simp] theorem sat_ite (w : List σ) (i : ℕ) (ψ φ₁ φ₂ : Form σ) :
    (ite ψ φ₁ φ₂).sat w i = if ψ.sat w i then φ₁.sat w i else φ₂.sat w i := by
  rw [ite, sat_or, Form.sat, Form.sat, Form.sat]
  cases ψ.sat w i <;> simp

/-- The value of a sum started from `c + 1`. -/
@[simp] theorem _root_.Transformer.CRASP.Term.val_sum (w : List σ) (i : ℕ) (S : List (Term σ))
    (c : ℕ) : (Term.sum S c).val w i = (S.map (·.val w i)).sum + (c + 1) := by
  induction S with
  | nil => simp [Term.sum]
  | cons s S ih =>
      simp only [Term.sum, List.foldr_cons, Term.val, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
      omega

/-- The indicators on the right of a comparison count the formulas that hold. -/
theorem sat_ltIndR (w : List σ) (i : ℕ) (t₁ : Term σ) :
    ∀ (t₂ : Term σ) (L₂ : List (Form σ)),
      (ltIndR t₁ t₂ L₂).sat w i = decide (t₁.val w i < t₂.val w i + L₂.countP (·.sat w i))
  | t₂, [] => by rw [ltIndR, Form.sat, List.countP_nil, Nat.add_zero]
  | t₂, ψ :: L₂ => by
      rw [ltIndR, sat_ite, sat_ltIndR w i t₁ _ L₂, sat_ltIndR w i t₁ t₂ L₂, List.countP_cons,
        Term.val, Term.val]
      cases ψ.sat w i <;> simp
      omega

/-- The indicators on both sides of a comparison count the formulas that hold. -/
theorem sat_ltInd (w : List σ) (i : ℕ) :
    ∀ (t₁ : Term σ) (L₁ : List (Form σ)) (t₂ : Term σ) (L₂ : List (Form σ)),
      (ltInd t₁ L₁ t₂ L₂).sat w i =
        decide (t₁.val w i + L₁.countP (·.sat w i) < t₂.val w i + L₂.countP (·.sat w i))
  | t₁, [], t₂, L₂ => by rw [ltInd, sat_ltIndR, List.countP_nil, Nat.add_zero]
  | t₁, ψ :: L₁, t₂, L₂ => by
      rw [ltInd, sat_ite, sat_ltInd w i _ L₁, sat_ltInd w i t₁ L₁, List.countP_cons, Term.val,
        Term.val]
      cases ψ.sat w i <;> simp
      omega

/-- **The comparison of sums with indicators.**  `ltSum` compares
`c₁ + Σ S₁ + Σ_{ψ ∈ L₁} (ψ ? 1 : 0)` with `c₂ + Σ S₂ + Σ_{ψ ∈ L₂} (ψ ? 1 : 0)`
(Appendix A.3, the elimination lemma for `?`, in the form the proof of
`lem:tlclpos_reduction` uses it). -/
@[simp] theorem sat_ltSum (w : List σ) (i c₁ : ℕ) (S₁ : List (Term σ)) (L₁ : List (Form σ))
    (c₂ : ℕ) (S₂ : List (Term σ)) (L₂ : List (Form σ)) :
    (ltSum c₁ S₁ L₁ c₂ S₂ L₂).sat w i =
      decide (c₁ + (S₁.map (·.val w i)).sum + L₁.countP (·.sat w i) <
        c₂ + (S₂.map (·.val w i)).sum + L₂.countP (·.sat w i)) := by
  rw [ltSum, sat_ltInd, Term.val_sum, Term.val_sum, decide_eq_decide]
  omega

/-- The indicators are not vacuous: `0 < 0 + (Q_true ? 1 : 0)` holds exactly
where the current symbol is `true`. -/
example : (ltSum 0 [] [] 0 [] [.sym true]).models [false, true] ∧
    ¬ (ltSum 0 [] [] 0 [] [.sym true]).models [true, false] := by
  constructor <;> decide

end Form

end CRASP
end Transformer
