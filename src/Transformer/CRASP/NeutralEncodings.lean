/-
# Plain transformers inside zero-parameter encoding families

arXiv:2506.16055v3, Appendix F, the positive halves of the three
transformer depth hierarchies. Zero-angle RoPE and zero-slope ALiBi leave
the plain embedding and logits unchanged.
-/

import Transformer.CRASP.PositionalTransformers
import Transformer.CRASP.FixedSign

namespace Transformer.CRASP

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- Equip an ordinary rounded transformer with a position encoding (F). -/
noncomputable def RTfr.withEncoding (T : RTfr σ p s d k) (pe : PosEnc) :
    PTfr σ p s d k := { T with pe := pe }

/-- Encodings with zero embeddings and unchanged logits preserve every layer.
Source: arXiv:2506.16055v3, Appendix F, Equations `eq:emb`, `eq:att_logit`. -/
theorem RTfr.withEncoding_layer (T : RTfr σ p s d k) (pe : PosEnc)
    (hlog : ∀ i j (q a : Fin d → Fx p s), pe.logit i j q a =
      ∑ c, (q c).val * (a c).val) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) :
    (T.withEncoding pe).layer ℓ h i = T.layer ℓ h i := by
  simp only [PTfr.layer, RTfr.layer, RTfr.withEncoding, hlog]

/-- Such an encoding also preserves the actual activations (Appendix F). -/
theorem RTfr.withEncoding_act (T : RTfr σ p s d k) (pe : PosEnc)
    (hemb : ∀ i, pe.emb p s d i = (fun _ => 0))
    (hlog : ∀ i j (q a : Fin d → Fx p s), pe.logit i j q a =
      ∑ c, (q c).val * (a c).val) (w : List σ) (ℓ : ℕ) :
    (T.withEncoding pe).act w ℓ = T.act w ℓ := by
  induction ℓ with
  | zero =>
      funext i c
      simp only [PTfr.act, RTfr.act, RTfr.withEncoding, hemb, Fx.add_zero]
  | succ ℓ ih =>
      funext i
      simp only [PTfr.act, RTfr.act, ih, T.withEncoding_layer pe hlog]

/-- Zero-parameter encodings preserve BOS recognition (Appendix F). -/
theorem RTfr.withEncoding_recognizes (T : RTfr (Option σ) p s d k) (pe : PosEnc)
    (hemb : ∀ i, pe.emb p s d i = (fun _ => 0))
    (hlog : ∀ i j (q a : Fin d → Fx p s), pe.logit i j q a =
      ∑ c, (q c).val * (a c).val) (L : Set (List σ)) (hT : T.Recognizes L) :
    (T.withEncoding pe).Recognizes L := by
  intro w
  have hn : 0 < (bos w).length := by simp [length_bos]
  rw [PTfr.Accepts, PTfr.out, dite_eq_left hn, T.withEncoding_act pe hemb hlog]
  simpa only [RTfr.Accepts, RTfr.out, dite_eq_left hn, RTfr.withEncoding] using hT w

/-- Zero-angle rotation is the identity (Appendix F, Equation `eq:rotation`). -/
theorem rotate_zero (i : ℕ) (v : Fin d → ℝ) : rotate d (fun _ => 0) i v = v := by
  funext c
  simp [rotate]

/-- Zero-angle RoPE has the ordinary attention logit (Appendix F). -/
theorem PosEnc.rope_zero_logit (i j : ℕ) (q a : Fin d → Fx p s) :
    (PosEnc.rope (fun _ => 0)).logit i j q a = ∑ c, (q c).val * (a c).val := by
  simp only [PosEnc.logit, rotate_zero]

/-- Zero-slope ALiBi has the ordinary attention logit (Appendix F). -/
theorem PosEnc.alibi_zero_logit (i j : ℕ) (q a : Fin d → Fx p s) :
    (PosEnc.alibi 0).logit i j q a = ∑ c, (q c).val * (a c).val := by
  simp [PosEnc.logit]

/-- The neutral-embedding and neutral-logit hypotheses have witnesses (F). -/
example : (∀ i, (PosEnc.alibi 0).emb 2 0 1 i = (fun _ => 0)) ∧
    (∀ i j (q a : Fin 1 → Fx 2 0), (PosEnc.alibi 0).logit i j q a =
      ∑ c, (q c).val * (a c).val) := ⟨fun _ => rfl, PosEnc.alibi_zero_logit⟩

end Transformer.CRASP
