module

public import Transformer.CRASP.GramsLogits
public import Transformer.CRASP.PreviousPredicates

/-!
# Formulas over grams read the tokens before a position

In no paper. A formula of `TL[◁#]` over the grams `grams n id w` of `Transformer.CRASP.Grams` reads
at the position `j` the tokens at `j, j - 1, …, j - n`. `symAt m a` holds where the token `m`
positions back is `a` (`sat_symAt`), as `Y^m Q_a` of `TL[◁#, Y]` (arXiv:2506.16055v3, Appendix F,
`app:tlclpos`; `Transformer.CRASP.FormP.sat_shift_sym`) does. A count over the grams equals a count
over the word when their bodies agree at every position of the word (`val_countL_grams`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {n : ℕ}

/-- The letter at the index `j` of `grams n κ w` is that of the gram of the position `j + 1` of
`⊲ · w`. -/
theorem getElem?_grams {Γ : Type v} (κ : Gram σ n → Γ) (w : List σ) {j : ℕ} (hj : j < w.length) :
    (grams n κ w)[j]? = some (κ (window w n (j + 1))) := by
  simp [grams, hj]

example := getElem?_grams (n := 1) id [true] (j := 0) (by simp)

variable [DecidableEq σ] [Fintype σ]

/-- `Y^m Q_a` over grams: the token `m` positions back is `a`. -/
noncomputable def symAt (m : Fin (n + 1)) (a : σ) : Form (Gram σ n) :=
  Form.any ((Finset.univ.filter fun c : Gram σ n => c m = some a).toList.map Form.sym)

/-- **`symAt m a` holds at the position `j` iff the token `m` positions back is `a`.** -/
theorem sat_symAt (m : Fin (n + 1)) (a : σ) {w : List σ} {j : ℕ} (hj : 0 < j)
    (hjw : j ≤ w.length) :
    (symAt m a).sat (grams n id w) j = true ↔ m.val < j ∧ w[j - m.val - 1]? = some a := by
  have hg : (grams n id w)[j - 1]? = some (window w n j) := by
    rw [getElem?_grams id w (by omega), Nat.sub_add_cancel hj]
    rfl
  have hw : (window w n j : Gram σ n) m = tok w (j - m.val) :=
    ite_eq_left (Nat.lt_succ_iff.mp m.isLt)
  have htok : tok w (j - m.val) = some a ↔ m.val < j ∧ w[j - m.val - 1]? = some a := by
    rcases Nat.lt_or_ge m.val j with h | h
    · rw [show j - m.val = j - m.val - 1 + 1 by omega, tok_succ, Nat.add_sub_cancel]
      exact ⟨fun h' => ⟨h, h'⟩, fun h' => h'.2⟩
    · rw [show j - m.val = 0 by omega, tok_zero]
      exact ⟨fun h' => absurd h' (by simp), fun h' => absurd h'.1 (by omega)⟩
  rw [← htok, ← hw, symAt, Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
    have hc' := (Finset.mem_filter.mp (Finset.mem_toList.mp hc)).2
    simp only [Form.sat, hg, decide_eq_true_eq, Option.some.injEq] at hs
    rw [hs]
    exact hc'
  · intro h
    refine ⟨Form.sym (window w n j), List.mem_map.mpr ⟨window w n j,
      Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩), rfl⟩, ?_⟩
    simp [Form.sat, hg]

/-- The hypotheses of `sat_symAt` hold at the second position of a word of two letters. -/
example : (symAt (1 : Fin 2) true).sat (grams 1 id [true, false]) 2 = true ↔
    1 < 2 ∧ [true, false][2 - 1 - 1]? = some true :=
  sat_symAt 1 true (by norm_num) (by norm_num)

/-- No gram stands in the empty word. -/
theorem sat_symAt_nil (m : Fin (n + 1)) (a : σ) (i : ℕ) :
    (symAt m a).sat ([] : List (Gram σ n)) i = false := by
  rw [← Bool.not_eq_true, symAt, Form.sat_any]
  rintro ⟨φ, hφ, h⟩
  obtain ⟨c, -, rfl⟩ := List.mem_map.mp hφ
  simp [Form.sat] at h

theorem symAt_mem (m : Fin (n + 1)) (a : σ) : symAt m a ∈ TLCl (Gram σ n) 0 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨c, -, rfl⟩ := List.mem_map.mp hφ
    exact ⟨rfl, rfl, le_rfl⟩

omit [Fintype σ] in
/-- **A count over the grams equals a count over the word** when the bodies agree at every
position of the word. -/
theorem val_countL_grams {φ : Form (Gram σ n)} {ψ : FormP σ} {w : List σ} {i : ℕ}
    (hi : i ≤ w.length) (h : ∀ j, 0 < j → j ≤ w.length → φ.sat (grams n id w) j = ψ.sat w j) :
    (Term.countL φ).val (grams n id w) i = (TermP.countL ψ).val w i := by
  simp only [Term.val, TermP.val]
  congr 1
  exact List.filter_congr fun j hj => by
    rw [List.mem_range'_1] at hj
    exact h j (by omega) (by omega)

/-- The hypotheses of `val_countL_grams` hold for the counts of `Q_true` and of its gram in a word
of two letters. -/
example : (Term.countL (symAt 0 true)).val (grams 1 id [true, false]) 2 =
    (TermP.countL (.sym true)).val [true, false] 2 :=
  val_countL_grams le_rfl fun j hj hjw => by
    apply Bool.eq_iff_iff.mpr
    rw [sat_symAt 0 true hj hjw]
    simp [FormP.sat, hj]

end Transformer.CRASP.Grams
