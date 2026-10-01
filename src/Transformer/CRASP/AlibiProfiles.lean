/-
# Enumerating ALiBi's recent windows

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
A profile of length `Δ` describes a long prefix; shorter profiles describe
their exact positive index and leave the BOS contribution explicit.
-/

import Transformer.CRASP.AlibiWindowCells
import Transformer.CRASP.WindowPredicates

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k j Δ : ℕ}

/-- The finite collection of profiles of lengths at most `Δ` (Appendix F). -/
abbrev Window (p s d Δ : ℕ) :=
  (m : Fin (Δ + 1)) × (Fin m.val → (Fin d → Fx p s))

/-- The profile's length agrees with a long or an exact short prefix (F). -/
def Fits (i : ℕ) (z : Window p s d Δ) : Prop :=
  if z.1.val = Δ then Δ ≤ i else 0 < z.1.val ∧ i = z.1.val

open scoped Classical in
/-- The previous-position formula specifying a profile and its length (F). -/
noncomputable def guard (ψ : (Fin d → Fx p s) → FormP σ)
    (z : Window p s d Δ) : FormP σ :=
  .and (if z.1.val = Δ then FormP.shift (Δ - 1) (FormP.truth true)
    else .and (FormP.truth (decide (0 < z.1.val))) (WindowPredicates.atPosition z.1.val))
    (WindowPredicates.profile ψ z.2)

/-- Reading the window adds no counting layer (Appendix F). -/
theorem guard_mem (ψ : (Fin d → Fx p s) → FormP σ)
    (hψ : ∀ r, ψ r ∈ TLClY σ j) (z : Window p s d Δ) :
    guard ψ z ∈ TLClY σ j := by
  classical
  apply FormP.and_mem_Y _ (WindowPredicates.profile_mem ψ hψ z.2)
  split
  · exact FormP.shift_mem_Y (FormP.truth_mem_Y _ _) _
  · exact FormP.and_mem_Y (FormP.truth_mem_Y _ _) (WindowPredicates.atPosition_mem _ _)

/-- The numerical correction selected by the window length (Appendix F). -/
noncomputable def windowCorrection (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (o : Option (Fin d))
    (z : Window p s d Δ) : ℤ :=
  if z.1.val = Δ then correction T a ℓ B q o z.2
  else shortCorrection T a ℓ B q o z.2

variable [DecidableEq σ]

/-- A guard reads exactly the profile at its admissible prefix length.
Source: arXiv:2506.16055v3, Appendix F, `Y` window construction. -/
theorem sat_guard (hΔ : 0 < Δ) (ψ : (Fin d → Fx p s) → FormP σ)
    (w : List σ) (i : ℕ) (hi : 0 < i) (q : ℕ → (Fin d → Fx p s))
    (hψ : ∀ t ∈ Finset.Icc 1 i, ∀ r, (ψ r).sat w t = decide (q t = r))
    (z : Window p s d Δ) :
    (guard ψ z).sat w i = true ↔ Fits i z ∧
      ∀ δ : Fin z.1.val, δ.val < i ∧ q (i - δ.val) = z.2 δ := by
  classical
  unfold guard Fits
  by_cases hz : z.1.val = Δ
  · rw [ite_eq_left hz, ite_eq_left hz, FormP.sat, Bool.and_eq_true_iff,
      FormP.sat_shift _ _ _ _ hi]
    simp only [FormP.sat_truth, Bool.and_true, decide_eq_true_eq]
    rw [WindowPredicates.sat_profile ψ q w i hi hψ z.2]
    have he : Δ - 1 < i ↔ Δ ≤ i := by omega
    rw [he]
  · rw [ite_eq_right hz, ite_eq_right hz, FormP.sat, Bool.and_eq_true_iff,
      FormP.sat, Bool.and_eq_true_iff, FormP.sat_truth]
    rw [WindowPredicates.sat_profile ψ q w i hi hψ z.2]
    by_cases hm : 0 < z.1.val
    · rw [WindowPredicates.sat_atPosition _ hm w i hi]
      simp only [decide_eq_true_eq]
    · simp [hm]

/-- Every positive position has an admissible window guard (Appendix F). -/
theorem exists_guard (hΔ : 0 < Δ) (ψ : (Fin d → Fx p s) → FormP σ)
    (w : List σ) (i : ℕ) (hi : 0 < i) (q : ℕ → (Fin d → Fx p s))
    (hψ : ∀ t ∈ Finset.Icc 1 i, ∀ r, (ψ r).sat w t = decide (q t = r)) :
    ∃ z : Window p s d Δ, (guard ψ z).sat w i = true := by
  let m : Fin (Δ + 1) := ⟨min Δ i, by omega⟩
  let z : Window p s d Δ := ⟨m, fun δ => q (i - δ.val)⟩
  refine ⟨z, (sat_guard hΔ ψ w i hi q hψ z).mpr ⟨?_, ?_⟩⟩
  · change (if min Δ i = Δ then Δ ≤ i else 0 < min Δ i ∧ i = min Δ i)
    split <;> omega
  · intro δ
    refine ⟨?_, rfl⟩
    have := δ.isLt
    change δ.val < min Δ i at this
    omega

/-- A guard supplies the exact corrections for all coefficient coordinates (F). -/
theorem integerSum_window (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s)
    (hB : ∀ e δ, Δ ≤ δ → coefficient T a ℓ e δ = B e) (hΔ : 0 < Δ)
    (ψ : (Fin d → Fx p s) → FormP σ) (w : List σ) (i : ℕ) (hi : 0 < i)
    (hψ : ∀ t ∈ Finset.Icc 1 i, ∀ r, (ψ r).sat w t = decide (T.actAt w ℓ t = r))
    (z : Window p s d Δ) (hz : (guard ψ z).sat w i = true)
    (q : Fin d → Fx p s) (o : Option (Fin d)) :
    integerSum T a ℓ q o w i =
      CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, o, r)) (T.actAt w ℓ) i +
        windowCorrection T a ℓ B q o z := by
  obtain ⟨hb, hp⟩ := (sat_guard hΔ ψ w i hi (T.actAt w ℓ) hψ z).mp hz
  obtain ⟨m, P⟩ := z
  by_cases hm : m.val = Δ
  · obtain ⟨mv, hmv⟩ := m
    change mv = Δ at hm
    subst mv
    simp only [Fits, ite_true] at hb
    simpa only [windowCorrection, ite_true] using
      integerSum_long T a ℓ B hB q o w i hb P (fun δ => (hp δ).2)
  · have hshort : m.val < Δ := by have := m.isLt; omega
    simp only [Fits, hm, ite_false] at hb
    rw [hb.2] at hp ⊢
    simpa only [windowCorrection, hm, ite_false] using
      integerSum_short T a ℓ B hB q o w hshort P (fun δ => (hp δ).2)

/-- A genuine ALiBi model satisfies all window, stabilization and source
agreement hypotheses, with a guard at a positive index (Appendix F). -/
example : ∃ (T : PTfr (Option Bool) 2 0 1 0) (Δ : ℕ)
    (B : Entry 2 0 1 → Fx 2 0) (ψ : (Fin 1 → Fx 2 0) → FormP Bool),
    0 < Δ ∧ (∀ e δ, Δ ≤ δ → coefficient T 1 0 e δ = B e) ∧
    (∀ r, ψ r ∈ TLClY Bool 0) ∧
    (∀ t ∈ Finset.Icc 1 1, ∀ r, (ψ r).sat [true] t = decide (T.actAt [true] 0 t = r)) ∧
    ∃ z : Window 2 0 1 Δ, (guard ψ z).sat [true] 1 = true := by
  classical
  let T : PTfr (Option Bool) 2 0 1 0 := {
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .alibi 1 }
  let ψ := fun r : Fin 1 → Fx 2 0 => (FormP.truth (decide ((fun _ => 0) = r)) : FormP Bool)
  obtain ⟨Δ, B, hΔ, hB⟩ := exists_tail T 1 0
  have hs : ∀ t ∈ Finset.Icc 1 1, ∀ r,
      (ψ r).sat [true] t = decide (T.actAt [true] 0 t = r) := by
    intro t ht r
    simp [ψ, T, PTfr.actAt, PTfr.act, PosEnc.emb, Fx.add_zero]
  exact ⟨T, Δ, B, ψ, hΔ, hB, fun _ => FormP.truth_mem_Y _ _, hs,
    exists_guard hΔ ψ [true] 1 one_pos (T.actAt [true] 0) hs⟩

end Transformer.CRASP.AlibiTables
