/-
# Semantics of replacing finite-state letters by positional predicates

arXiv:2506.16055v3, Appendix F, the finite-function adaptations of
`thm:rtfr_to_TLCl`. The auxiliary word contains one activation-state letter
per ordinary token; the BOS contribution remains a separate constant.
-/

import Transformer.CRASP.PositionalSubstitution
import Transformer.CRASP.TransformerStates

namespace Transformer.CRASP

universe u v
variable {α : Type u} {σ : Type v} [DecidableEq α] [DecidableEq σ]

mutual

/-- Substitution preserves satisfaction on a word of finite source states (F). -/
theorem Form.sat_substPos (ψ : α → FormP σ) (W : List α) (w : List σ)
    (hψ : ∀ j, RTfr.Readable W j → ∀ a,
      (ψ a).sat w j = (Form.sym a).sat W j) :
    ∀ (φ : Form α), φ.past = true → φ.pnpFree = true →
      ∀ i, RTfr.Readable W i → (φ.substPos ψ).sat w i = φ.sat W i
  | .sym a, _, _, i, hi => hψ i hi a
  | .lt t u, hp, hf, i, hi => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
      rw [Form.substPos, FormP.sat, Form.sat,
        t.val_substPos ψ W w hψ hp.1 hf.1 i hi,
        u.val_substPos ψ W w hψ hp.2 hf.2 i hi]
  | .neg φ, hp, hf, i, hi => by
      rw [Form.substPos, FormP.sat, Form.sat, φ.sat_substPos ψ W w hψ hp hf i hi]
  | .and φ χ, hp, hf, i, hi => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
      rw [Form.substPos, FormP.sat, Form.sat,
        φ.sat_substPos ψ W w hψ hp.1 hf.1 i hi,
        χ.sat_substPos ψ W w hψ hp.2 hf.2 i hi]
  | .pnp _, _, hf, _, _ => by simp [Form.pnpFree] at hf

/-- Substitution preserves the exact count values (Appendix F). -/
theorem Term.val_substPos (ψ : α → FormP σ) (W : List α) (w : List σ)
    (hψ : ∀ j, RTfr.Readable W j → ∀ a,
      (ψ a).sat w j = (Form.sym a).sat W j) :
    ∀ (t : Term α), t.past = true → t.pnpFree = true →
      ∀ i, RTfr.Readable W i → (t.substPos ψ).val w i = t.val W i
  | .countL φ, hp, hf, i, hi => by
      rw [Term.substPos, TermP.val, Term.val]
      apply congrArg List.length
      apply List.filter_congr
      intro j hj
      have hj' := List.mem_range'.mp hj
      have hib := hi.1
      exact φ.sat_substPos ψ W w hψ hp hf j ⟨by omega, Or.inl (by omega)⟩
  | .countR _, hp, _, _, _ => by simp [Term.past] at hp
  | .add t u, hp, hf, i, hi => by
      simp only [Term.past, Term.pnpFree, Bool.and_eq_true] at hp hf
      rw [Term.substPos, TermP.val, Term.val,
        t.val_substPos ψ W w hψ hp.1 hf.1 i hi,
        u.val_substPos ψ W w hψ hp.2 hf.2 i hi]
  | .one, _, _, _, _ => rfl

end

/-- Matching state letters and valid indices occur already on a one-token word.
Source: arXiv:2506.16055v3, Appendix F, finite-state simulation. -/
example : (∀ j, RTfr.Readable [true] j → ∀ a : Bool,
    (FormP.sym a).sat [true] j = (Form.sym a).sat [true] j) ∧
    RTfr.Readable [true] 1 := ⟨fun _ _ _ => rfl, ⟨le_rfl, Or.inl one_pos⟩⟩

end Transformer.CRASP
