/-
# Recognition survives a zero-angle sinusoidal encoding

arXiv:2506.16055v3, Appendix F, the positive sinusoidal hierarchy.
An induction reads every layer from the reserved even coordinates, then
the unchanged output projection establishes BOS recognition.
-/

import Transformer.CRASP.SinusoidalLiftStep

namespace Transformer.CRASP.SinusoidalLift

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- Every ordinary activation is recovered from the even coordinates (F). -/
theorem read_act (T : RTfr (Option σ) p s d k)
    (hQ : ∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) (w : List (Option σ)) (ℓ : ℕ) :
    (fun i => read ((model T).act w ℓ i)) = T.act w ℓ := by
  induction ℓ with
  | zero => funext i; exact read_initial T w i
  | succ ℓ ih =>
      funext i
      rw [PTfr.act, read_layer T hQ, ih]
      rfl

/-- The lifted sinusoidal model recognizes the same language (Appendix F). -/
theorem model_recognizes (T : RTfr (Option σ) p s d k)
    (hQ : ∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) (L : Set (List σ)) (hT : T.Recognizes L) :
    (model T).Recognizes L := by
  intro w
  have hn : 0 < (bos w).length := by simp [length_bos]
  have ha := congrFun (read_act T hQ (bos w) k)
    (⟨(bos w).length - 1, by omega⟩ : Fin (bos w).length)
  rw [PTfr.Accepts, PTfr.out, dite_eq_left hn]
  change 0 < (T.Wout (read ((model T).act (bos w) k _))).val ↔ w ∈ L
  rw [ha]
  simpa only [RTfr.Accepts, RTfr.out, dite_eq_left hn] using hT w

/-- Uniform queries and recognition have a rejecting zero-model witness (F). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 1,
    (∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) ∧ T.Recognizes ∅ := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0 },
    fun _ _ => rfl, ?_⟩
  intro w
  simp [RTfr.Accepts, RTfr.out, length_bos, Fx.val]

end Transformer.CRASP.SinusoidalLift
