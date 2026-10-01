/-
# Positional formulas for finite-state attention sums

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
The auxiliary alphabet consists of finite source states. Substitution then
turns the original rounded quotient cells into positional formulas.
-/

import Transformer.CRASP.PositionalSubstitutionSemantics
import Transformer.CRASP.PositionalBoolean
import Transformer.CRASP.CountCells

namespace Transformer.CRASP.CountCells

universe u v
variable {α : Type u} {σ : Type v} [Fintype α] {p s k : ℕ}

/-- A weighted count over the auxiliary finite-state alphabet (Appendix F). -/
noncomputable def states (b : α) (f : α → Fx p s) : LinearCount α :=
  LinearCount.states Form.sym b (fun a => (f a).m)

/-- Auxiliary letter predicates have depth zero (Appendix F). -/
theorem states_good (b : α) (f : α → Fx p s) : (states b f).Good 0 :=
  LinearCount.good_states (ψ := fun a : α => Form.sym a) (k := 0)
    (fun _ => ⟨rfl, rfl, le_rfl⟩) b (fun a => (f a).m)

/-- The source-state word of a prefix, excluding BOS (Appendix F). -/
def stateWord (q : ℕ → α) (i : ℕ) : List α :=
  List.ofFn (fun j : Fin i => q (j.val + 1))

/-- Integer mantissa sums include BOS and the ordinary prefix (B.2/F). -/
def sum (b : α) (f : α → Fx p s) (q : ℕ → α) (i : ℕ) : ℤ :=
  (f b).m + ∑ j ∈ Finset.Icc 1 i, (f (q j)).m

/-- Attention obtained from finite source-state tables (B.2/F). -/
noncomputable def value (b : α) (D N V : α → Fx p s)
    (q : ℕ → α) (i : ℕ) : Fx p s :=
  if sum b D q i = 0 then
    Fx.round p s (((sum b V q i : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
  else Fx.round p s ((sum b N q i : ℝ) / (sum b D q i : ℝ))

/-- A rounded attention cell after replacing state letters (Appendix F). -/
noncomputable def stateFormula (b : α) (D N V : α → Fx p s)
    (ψ : α → FormP σ) (y : Fx p s) : FormP σ :=
  (formula (states b D) (states b N) (states b V) y).substPos ψ

/-- State substitution preserves the one-layer cost of a cell (Appendix F). -/
theorem stateFormula_mem (b : α) (D N V : α → Fx p s)
    (ψ : α → FormP σ) (hψ : ∀ a, ψ a ∈ TLClMod σ k) (y : Fx p s) :
    stateFormula b D N V ψ y ∈ TLClMod σ (k + 1) := by
  have h := formula_mem (states_good b D) (states_good b N) (states_good b V) y
  refine ⟨Form.prevFree_substPos ψ (fun a => (hψ a).1) _, ?_⟩
  have hd := Form.depth_substPos_le ψ k (fun a => (hψ a).2)
    (formula (states b D) (states b N) (states b V) y)
  change (Form.substPos ψ _).depth ≤ _
  have hb := h.2.2
  omega

variable [DecidableEq α] [DecidableEq σ]

omit [Fintype α] in
/-- Each auxiliary word position carries exactly its source state (Appendix F). -/
theorem sat_sym_stateWord (q : ℕ → α) {i j : ℕ} (hi : 0 < i)
    (hj : RTfr.Readable (stateWord q i) j) (a : α) :
    (Form.sym a).sat (stateWord q i) j = decide (q j = a) := by
  have hlen : (stateWord q i).length = i := List.length_ofFn
  have hjle : j ≤ i := by simpa only [hlen] using hj.1
  have hjpos : 0 < j := hj.2.resolve_right (by
    intro h; have := congrArg List.length h; simp [hlen] at this; omega)
  have hidx : j - 1 < i := by omega
  simp only [Form.sat, stateWord, List.getElem?_ofFn, dite_eq_left hidx,
    Option.some.injEq]
  simp only [show j - 1 + 1 = j by omega]

/-- The letter counts equal the corresponding integer table sums (Appendix F). -/
theorem val_states (b : α) (f : α → Fx p s) (q : ℕ → α) {i : ℕ} (hi : 0 < i) :
    (states b f).val (stateWord q i) i = sum b f q i := by
  apply LinearCount.val_states
  intro a j hj
  have hjb := Finset.mem_Icc.mp hj
  apply sat_sym_stateWord q hi
  exact ⟨by simpa [stateWord] using hjb.2, Or.inl (by omega)⟩

/-- A substituted cell defines the exact rounded table quotient (Appendix F). -/
theorem sat_stateFormula (b : α) (D N V : α → Fx p s)
    (hD : ∀ a, 0 ≤ (D a).m) (ψ : α → FormP σ)
    (q : ℕ → α) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat w j = decide (q j = a))
    (y : Fx p s) :
    (stateFormula b D N V ψ y).sat w i = true ↔
      (if sum b D q i = 0 then
        Fx.round p s (((sum b V q i : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
       else Fx.round p s ((sum b N q i : ℝ) / (sum b D q i : ℝ))) = y := by
  let φ := formula (states b D) (states b N) (states b V) y
  have hφ := formula_mem (states_good b D) (states_good b N) (states_good b V) y
  have hletters : ∀ j, RTfr.Readable (stateWord q i) j → ∀ a,
      (ψ a).sat w j = (Form.sym a).sat (stateWord q i) j := by
    intro j hj a
    rw [sat_sym_stateWord q hi hj a]
    have hjle : j ≤ i := by simpa [stateWord] using hj.1
    have hjpos : 0 < j := hj.2.resolve_right (by
      intro h; have := congrArg List.length h; simp [stateWord] at this; omega)
    exact hψ j (Finset.mem_Icc.mpr ⟨hjpos, hjle⟩) a
  rw [stateFormula, Form.sat_substPos ψ (stateWord q i) w hletters φ hφ.1 hφ.2.1 i
    ⟨by simp [stateWord], Or.inl hi⟩]
  change (formula (states b D) (states b N) (states b V) y).sat
    (stateWord q i) i = true ↔ _
  have hn : 0 ≤ (states b D).val (stateWord q i) i :=
    LinearCount.val_states_nonneg Form.sym b (fun a => (D a).m) hD _ _
  rw [sat_formula (states b D) (states b N) (states b V) y _ i hn]
  rw [val_states b D q hi, val_states b N q hi, val_states b V q hi]

/-- The source agreement and nonnegativity hypotheses have constant witnesses (F). -/
example : (∀ _ : Unit, 0 ≤ (0 : Fx 2 0).m) ∧ 0 < 1 ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Unit,
      (FormP.truth true : FormP Bool).sat [true] j = decide (() = a)) := by
  simp

end Transformer.CRASP.CountCells
