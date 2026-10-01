/-
# A common stabilization distance for finite ALiBi tables

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
There are finitely many query states, source states and coordinates in one
layer. Their individual stabilized coefficients therefore share a finite
distance bound, for either sign of the slope.
-/

import Transformer.CRASP.AlibiTails
import Transformer.CRASP.PositionalTransformers

namespace Transformer.CRASP

universe u v
variable {σ : Type u}

/-- Finitely many eventually constant coefficients share a distance threshold.
Source: arXiv:2506.16055v3, Appendix F, finite-state ALiBi construction. -/
theorem exists_uniform_constant_tail {α : Type v} [Fintype α] {β : Type u}
    (f : α → ℕ → β) (hf : ∀ a, ∃ (Δ : ℕ) (b : β), ∀ δ, Δ ≤ δ → f a δ = b) :
    ∃ (Δ : ℕ) (b : α → β), 0 < Δ ∧ ∀ a δ, Δ ≤ δ → f a δ = b a := by
  classical
  choose Δ b hb using hf
  refine ⟨Finset.univ.sup Δ + 1, b, Nat.succ_pos _, fun a δ hδ => ?_⟩
  exact hb a δ ((Finset.le_sup (Finset.mem_univ a)).trans (by omega))

namespace AlibiTables

variable {p s d k : ℕ}

/-- Finite query state, optional output coordinate, and finite source state (F). -/
abbrev Entry (p s d : ℕ) := (Fin d → Fx p s) × Option (Fin d) × (Fin d → Fx p s)

/-- A rounded denominator (`none`) or numerator (`some c`) coefficient (F/B.1). -/
noncomputable def coefficient (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (e : Entry p s d) (δ : ℕ) : Fx p s :=
  let x := ∑ c : Fin d, (T.WQ ℓ e.1 c).val * (T.WK ℓ e.2.2 c).val
  let v := e.2.1.elim 1 (fun c => (T.WV ℓ e.2.2 c).val)
  Fx.round p s (Real.exp (x - a * δ) * v)

/-- All rounded coefficients in one ALiBi layer have a common stable tail.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem exists_tail (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ) :
    ∃ (Δ : ℕ) (B : Entry p s d → Fx p s), 0 < Δ ∧
      ∀ e δ, Δ ≤ δ → coefficient T a ℓ e δ = B e := by
  apply exists_uniform_constant_tail
  intro e
  exact alibi_coefficient_stabilizes p s
    (∑ c : Fin d, (T.WQ ℓ e.1 c).val * (T.WK ℓ e.2.2 c).val)
    (e.2.1.elim 1 (fun c => (T.WV ℓ e.2.2 c).val)) a

end AlibiTables

/-- A constant coefficient satisfies the stabilization hypothesis (Appendix F). -/
example : ∀ _ : Unit, ∃ (Δ : ℕ) (b : Bool), ∀ δ, Δ ≤ δ → true = b :=
  fun _ => ⟨0, true, fun _ _ => rfl⟩

end Transformer.CRASP
