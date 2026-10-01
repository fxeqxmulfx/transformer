/-
# The truth-value invariant and its base case

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
After `ℓ` layers all subformulas of counting depth at most `ℓ` are
correct. At BOS the logical word is empty, as there are no ordinary tokens.
-/

import Transformer.CRASP.TemporalProgramEvaluate
import Transformer.CRASP.Frame

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- The BOS logical input is empty; other positions use the original word (B.2). -/
def view (w : List σ) (i : ℕ) : List σ := if i = 0 then [] else w

/-- Correct Boolean memory up to the number of completed layers (Appendix B.2). -/
def Correct (φ : Form σ) (k ℓ : ℕ) : Prop :=
  ∀ (w : List σ) (i : ℕ), i ≤ w.length → ∀ ψ : Node φ, ψ.val.depth ≤ ℓ →
    read φ (state φ k w ℓ i) ψ.val = ψ.val.sat (view w i) i

/-- Initial embeddings compute all depth-zero subformulas (Appendix B.2). -/
theorem initial_correct (φ : Form σ) (k : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true) :
    Correct φ k 0 := by
  intro w i hi ψ hd
  rcases ψ with ⟨ψ, hm⟩
  have hψ := φ.subformulas_properties hp hf ψ hm
  rw [state_zero φ k w i hi]
  change read φ (embedding φ ((bos w)[i]'(by rw [length_bos]; omega))) ψ = ψ.sat (view w i) i
  rw [read, dite_eq_left hm]
  cases i with
  | zero => simp [embedding, bos, view]
  | succ j =>
      have hj : j < w.length := by omega
      have hin : j + 1 < (bos w).length := by rw [length_bos]; omega
      have hb : (bos w)[j + 1]'hin = some (w[j]'hj) := by simp [bos]
      rw [hb]
      simp only [embedding, Option.elim_some, List.length_singleton,
        view, Nat.succ_ne_zero, ite_false]
      change decide ((Fx.ofBool φ.capacity (ψ.sat [w[j]] 1)).m = 1) =
        ψ.sat w (j + 1)
      rw [Fx.read_ofBool]
      apply Form.sat_eq_of_pnpFree_depth_eq_zero (w := [w[j]]) (w' := w) (i := 1)
        (i' := j + 1) (by simp)
      · exact hψ.2.1
      · exact Nat.le_zero.mp hd

/-- Past-term values agree with the BOS logical view (Appendix B.2). -/
theorem term_val_view (t : Term σ) (hp : t.past = true) (w : List σ) (i : ℕ) :
    t.val w i = t.val (view w i) i := by
  by_cases hi : i = 0
  · subst i
    rw [t.val_countBodies hp, t.val_countBodies hp]
    simp [Term.val]
  · simp [view, hi]

/-- The root-fragment and truth-invariant hypotheses have witnesses (B.2). -/
example : (Form.lt (.countL (.sym true)) .one).past = true ∧
    (Form.lt (.countL (.sym true)) .one).pnpFree = true ∧
    Correct (Form.lt (.countL (.sym true)) .one) 1 0 :=
  ⟨rfl, rfl, initial_correct _ _ rfl rfl⟩

end Transformer.CRASP.TemporalProgram
