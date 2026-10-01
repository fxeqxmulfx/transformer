/-
# ALiBi coefficient sums from a stable table and a recent profile

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
The same correction identity applies to denominator and numerator entries;
short prefixes include their known BOS coefficient explicitly.
-/

import Transformer.CRASP.AlibiAttention
import Transformer.CRASP.WindowTailSums

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k m Δ : ℕ}

/-- A finite recent-profile correction for one coefficient entry (Appendix F). -/
noncomputable def correction (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (o : Option (Fin d))
    (P : Fin m → (Fin d → Fx p s)) : ℤ :=
  WindowTailSums.correction (fun δ r => (coefficient T a ℓ (q, o, r) δ).m)
    (fun r => (B (q, o, r)).m) P

/-- A short profile's additional BOS correction at distance `m` (Appendix F). -/
noncomputable def shortCorrection (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (o : Option (Fin d))
    (P : Fin m → (Fin d → Fx p s)) : ℤ :=
  correction T a ℓ B q o P + (coefficient T a ℓ (q, o, T.actAt [] ℓ 0) m).m -
    (B (q, o, T.actAt [] ℓ 0)).m

/-- For a long prefix, the recent profile gives the exact coefficient sum (F). -/
theorem integerSum_long (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s)
    (hB : ∀ e δ, Δ ≤ δ → coefficient T a ℓ e δ = B e)
    (q : Fin d → Fx p s) (o : Option (Fin d)) (w : List σ) (i : ℕ) (hi : Δ ≤ i)
    (P : Fin Δ → (Fin d → Fx p s))
    (hP : ∀ δ, T.actAt w ℓ (i - δ.val) = P δ) :
    integerSum T a ℓ q o w i =
      CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, o, r)) (T.actAt w ℓ) i +
        correction T a ℓ B q o P := by
  exact WindowTailSums.sum_long
    (fun δ r => (coefficient T a ℓ (q, o, r) δ).m) (fun r => (B (q, o, r)).m)
    (fun δ hδ r => congrArg Fx.m (hB _ δ hδ)) (T.actAt w ℓ) (T.actAt [] ℓ 0)
    (T.actAt_bos w ℓ) i hi P hP

/-- For a short prefix, the ordinary profile and BOS give the exact sum (F). -/
theorem integerSum_short (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s)
    (hB : ∀ e δ, Δ ≤ δ → coefficient T a ℓ e δ = B e)
    (q : Fin d → Fx p s) (o : Option (Fin d)) (w : List σ) (hm : m < Δ)
    (P : Fin m → (Fin d → Fx p s))
    (hP : ∀ δ, T.actAt w ℓ (m - δ.val) = P δ) :
    integerSum T a ℓ q o w m =
      CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, o, r)) (T.actAt w ℓ) m +
        shortCorrection T a ℓ B q o P := by
  have h := WindowTailSums.sum_short
    (fun δ r => (coefficient T a ℓ (q, o, r) δ).m) (fun r => (B (q, o, r)).m)
    (fun δ hδ r => congrArg Fx.m (hB _ δ hδ)) (T.actAt w ℓ) (T.actAt [] ℓ 0)
    (T.actAt_bos w ℓ) hm P hP
  change integerSum T a ℓ q o w m =
    CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, o, r)) (T.actAt w ℓ) m +
    (correction T a ℓ B q o P + ((coefficient T a ℓ (q, o, T.actAt [] ℓ 0) m).m -
      (B (q, o, T.actAt [] ℓ 0)).m)) at h
  simpa only [shortCorrection, sub_eq_add_neg, add_assoc] using h

/-- Stable-table and recent-profile hypotheses occur in the zero model (F). -/
example : ∃ T : PTfr (Option Bool) 2 0 0 0,
    (∀ e δ, 1 ≤ δ → coefficient T 0 0 e δ = coefficient T 0 0 e 0) ∧
    (∀ δ : Fin 1, T.actAt [true] 0 (1 - δ.val) = (fun _ => 0)) := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .alibi 0 }, ?_, ?_⟩
  · intro e δ hδ
    simp [coefficient]
  · intro δ
    funext c
    exact Fin.elim0 c

end Transformer.CRASP.AlibiTables
