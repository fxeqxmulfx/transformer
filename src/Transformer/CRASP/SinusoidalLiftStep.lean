/-
# A sinusoidal layer preserves the reserved ordinary coordinates

arXiv:2506.16055v3, Appendix F, the positive sinusoidal hierarchy, and
Appendix B.2, the uniform-attention construction. The unused coordinates
do not affect queries, values, or feed-forward evaluation.
-/

import Transformer.CRASP.SinusoidalLift

namespace Transformer.CRASP.SinusoidalLift

universe u
variable {σ : Type u} {p s d k n : ℕ}

/-- Attention on an even coordinate is the ordinary uniform attention (F). -/
theorem attention_even (T : RTfr (Option σ) p s d k)
    (hQ : ∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) (ℓ : ℕ)
    (h : Fin n → Fin (2 * d) → Fx p s) (i : Fin n) (c : Fin d) :
    PeriodicAttention.attention (model T) ℓ h i (evenIdx c) =
      T.attention ℓ (fun j => read (h j)) i c := by
  simp [PeriodicAttention.attention, RTfr.attention, model, PosEnc.logit, hQ,
    write_even, Fx.val]

/-- The residual and FFN preserve the ordinary layer on reserved features (F). -/
theorem read_layer (T : RTfr (Option σ) p s d k)
    (hQ : ∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) (ℓ : ℕ)
    (h : Fin n → Fin (2 * d) → Fx p s) (i : Fin n) :
    read ((model T).layer ℓ h i) = T.layer ℓ (fun j => read (h j)) i := by
  rw [PeriodicAttention.layer_eq_attention, RTfr.layer_eq_attention]
  change read (write (T.ff ℓ (read (fun c =>
    Fx.add (PeriodicAttention.attention (model T) ℓ h i c) (h i c))))) = _
  rw [read_write]
  congr 1
  funext c
  change Fx.add (PeriodicAttention.attention (model T) ℓ h i (evenIdx c))
    (h i (evenIdx c)) = _
  rw [attention_even T hQ]
  rfl

/-- The initial even-coordinate activation is the ordinary embedding (F). -/
theorem read_initial (T : RTfr (Option σ) p s d k) (w : List (Option σ)) (i : Fin w.length) :
    read ((model T).act w 0 i) = T.act w 0 i := by
  funext c
  change Fx.add (write (T.E w[i]) (evenIdx c))
    ((PosEnc.sinusoidal (fun _ => 0)).emb p s (2 * d) i.val (evenIdx c)) = _
  rw [write_even, emb_even, Fx.add_zero]
  rfl

/-- The uniform-query hypothesis is satisfiable (Appendix B.2/F). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 1, ∀ ℓ q, T.WQ ℓ q = (fun _ => 0) :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0 }, fun _ _ => rfl⟩

end Transformer.CRASP.SinusoidalLift
