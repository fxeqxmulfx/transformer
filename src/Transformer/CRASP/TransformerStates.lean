/-
# Formulas for the finite activation states

arXiv:2506.16055v3, Appendix B.2, induction underlying Equation
`eq:phi_h`. State predicates include the empty-word BOS evaluation;
past counts themselves range only over ordinary token positions.
-/

import Transformer.CRASP.TransformerPrefix
import Transformer.CRASP.FixedFinite
import Transformer.CRASP.FormulaBounds
import Transformer.CRASP.Indicator

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- Valid reads are ordinary positions or the sole empty-word evaluation (B.2). -/
def Readable (w : List σ) (i : ℕ) : Prop := i ≤ w.length ∧ (0 < i ∨ w = [])

open scoped Classical in
/-- A finite family defines the activation at every valid read (Appendix B.2). -/
def StateDefinition (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    [DecidableEq σ] (ψ : (Fin d → Fx p s) → Form σ) : Prop :=
  ∀ w i, Readable w i → ∀ q, (ψ q).sat w i = decide (T.actAt w ℓ i = q)

variable [Fintype σ] [DecidableEq σ]

open scoped Classical in
/-- The base-case disjunction over the alphabet, with its BOS default (B.2). -/
noncomputable def initialFormula (T : RTfr (Option σ) p s d k)
    (q : Fin d → Fx p s) : Form σ :=
  Form.ite Form.onStr
    (Form.any ((Finset.univ.filter fun a => T.E (some a) = q).toList.map Form.sym))
    (Form.truth (decide (T.E none = q)))

omit [DecidableEq σ] in
/-- Embedding predicates require no attention/counting depth (Appendix B.2). -/
theorem initialFormula_mem (T : RTfr (Option σ) p s d k) (q : Fin d → Fx p s) :
    T.initialFormula q ∈ TLCl σ 0 := by
  classical
  apply Form.ite_mem_TLCl
  · exact ⟨Form.past_onStr, Form.pnpFree_onStr, Form.depth_onStr.le⟩
  · apply Form.any_mem
    rintro φ hφ
    obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hφ
    exact ⟨rfl, rfl, le_rfl⟩
  · exact Form.truth_mem _ _

/-- The alphabet disjunction defines exactly the embedding state (Appendix B.2). -/
theorem initialFormula_defines (T : RTfr (Option σ) p s d k) :
    T.StateDefinition 0 T.initialFormula := by
  classical
  intro w i hi q
  have hin : i < (bos w).length := by have := hi.1; rw [length_bos]; omega
  cases i with
  | zero =>
      have hw : w = [] := hi.2.resolve_left (by omega)
      subst w
      simp [initialFormula, Form.sat_ite, Form.sat_onStr, actAt, act, bos]
  | succ j =>
      have hj : j < w.length := by have := hi.1; omega
      have hon : (Form.onStr : Form σ).sat w (j + 1) = true := by
        rw [Form.sat_onStr]; simpa using hj
      rw [initialFormula, Form.sat_ite, hon]
      simp only [ite_true]
      rw [actAt, dite_eq_left hin]
      have hb : (bos w)[j + 1]'hin = some (w[j]'hj) := by simp [bos]
      change (Form.any ((Finset.univ.filter fun a => T.E (some a) = q).toList.map
        Form.sym)).sat w (j + 1) = decide (T.E ((bos w)[j + 1]'hin) = q)
      rw [hb]
      rw [Bool.eq_iff_iff, decide_eq_true_iff, Form.sat_any]
      constructor
      · rintro ⟨φ, hφ, hsat⟩
        obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hφ
        have he := (Finset.mem_filter.mp (Finset.mem_toList.mp ha)).2
        have hw : w[j]? = some a := by simpa [Form.sat] using hsat
        have hwa : w[j] = a := by
          simpa only [List.getElem?_eq_getElem hj, Option.some.injEq] using hw
        simpa only [hwa] using he
      · intro hq
        refine ⟨Form.sym w[j], List.mem_map_of_mem ?_, ?_⟩
        · exact Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq⟩)
        · simp [Form.sat]

/-- Both ordinary and empty-word reads occur in the simulation (B.2). -/
example : Readable [true] 1 ∧ Readable ([] : List Bool) 0 := by
  simp [Readable]

end Transformer.CRASP.RTfr
