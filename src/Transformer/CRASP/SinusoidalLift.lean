/-
# Ignoring a zero-angle sinusoidal embedding

arXiv:2506.16055v3, Appendix F, the positive sinusoidal hierarchy.
The zero-angle positional vector is zero on even coordinates. Put the
ordinary program in these coordinates and reserve the odd ones.
-/

import Transformer.CRASP.PeriodicAttention
import Transformer.CRASP.FixedSign

namespace Transformer.CRASP.SinusoidalLift

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The even coordinate reserved for an ordinary model feature (Appendix F). -/
def evenIdx (c : Fin d) : Fin (2 * d) := ⟨2 * c.val, by have := c.isLt; omega⟩

/-- Copy a vector into the even coordinates and clear the others (F). -/
def write (x : Fin d → Fx p s) : Fin (2 * d) → Fx p s := fun c =>
  if c.val % 2 = 0 then x ⟨c.val / 2, by have := c.isLt; omega⟩ else 0

/-- Read the ordinary features from the even coordinates (Appendix F). -/
def read (x : Fin (2 * d) → Fx p s) : Fin d → Fx p s := fun c => x (evenIdx c)

/-- Writing recovers the value at every reserved coordinate (Appendix F). -/
@[simp] theorem write_even (x : Fin d → Fx p s) (c : Fin d) :
    write x (evenIdx c) = x c := by
  simp [write, evenIdx]

/-- Reading a written vector recovers every ordinary feature (Appendix F). -/
@[simp] theorem read_write (x : Fin d → Fx p s) : read (write x) = x := by
  funext c
  exact write_even x c

/-- Zero-angle sinusoidal embeddings vanish on reserved coordinates (F). -/
theorem emb_even (i : ℕ) (c : Fin d) :
    (PosEnc.sinusoidal (fun _ => 0)).emb p s (2 * d) i (evenIdx c) = 0 := by
  simp [PosEnc.emb, sinusoidalVec, rotate, evenIdx]

/-- A sinusoidal model carrying an ordinary uniform-attention program (F). -/
noncomputable def model (T : RTfr (Option σ) p s d k) : PTfr (Option σ) p s (2 * d) k where
  E a := write (T.E a)
  WQ _ _ _ := 0
  WK _ _ _ := 0
  WV ℓ h := write (T.WV ℓ (read h))
  ff ℓ h := write (T.ff ℓ (read h))
  Wout h := T.Wout (read h)
  pe := .sinusoidal (fun _ => 0)

/-- The lifted model's attention logit is zero (Appendix F/B.2). -/
theorem logit_zero (T : RTfr (Option σ) p s d k) (ℓ i j : ℕ)
    (q a : Fin (2 * d) → Fx p s) :
    (model T).pe.logit i j ((model T).WQ ℓ q) ((model T).WK ℓ a) = 0 := by
  simp [model, PosEnc.logit, Fx.val]

/-- Uniform ordinary queries satisfy the logit condition (Appendix B.2). -/
theorem ordinary_logit_zero (T : RTfr (Option σ) p s d k)
    (hQ : ∀ ℓ q, T.WQ ℓ q = (fun _ => 0)) (ℓ : ℕ)
    (q a : Fin d → Fx p s) :
    (∑ c, (T.WQ ℓ q c).val * (T.WK ℓ a c).val) = 0 := by
  rw [hQ ℓ q]
  simp [Fx.val]

/-- The uniform-query hypothesis is satisfiable in a zero model (B.2/F). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 1, ∀ ℓ q, T.WQ ℓ q = (fun _ => 0) :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0 }, fun _ _ => rfl⟩

end Transformer.CRASP.SinusoidalLift
