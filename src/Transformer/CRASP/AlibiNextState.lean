/-
# One ALiBi layer in previous-position counting logic

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
Enumerate the recent profile, current query, and rounded attention vector.
All distance dependence has become a constant correction to state counts.
-/

import Transformer.CRASP.AlibiProfiles

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k j Δ : ℕ}

open scoped Classical in
/-- The finite accepting-state test for an ALiBi layer (Appendix F). -/
noncomputable def nextFormula (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (ψ : (Fin d → Fx p s) → FormP σ)
    (r : Fin d → Fx p s) (Δ : ℕ) : FormP σ :=
  FormP.any (Finset.univ.toList.map fun z : Window p s d Δ =>
    .and (guard ψ z) (FormP.any (Finset.univ.toList.map
      fun v : (Fin d → Fx p s) × (Fin d → Fx p s) =>
      .and (ψ v.1) (.and (FormP.truth (decide (T.ff ℓ (fun c => Fx.add (v.1 c) (v.2 c)) = r)))
        (FormP.all (Finset.univ.toList.map fun c : Fin d =>
          cell T ℓ B v.1 c (windowCorrection T a ℓ B v.1 none z)
            (windowCorrection T a ℓ B v.1 (some c) z) ψ (v.2 c)))))))

/-- Exactly one new counting layer is used by an ALiBi update (Appendix F). -/
theorem nextFormula_mem (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (ψ : (Fin d → Fx p s) → FormP σ)
    (hψ : ∀ r, ψ r ∈ TLClY σ j) (r : Fin d → Fx p s) (Δ : ℕ) :
    nextFormula T a ℓ B ψ r Δ ∈ TLClY σ (j + 1) := by
  classical
  apply FormP.any_mem_Y
  rintro φ hφ
  obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
  apply FormP.and_mem_Y (FormP.mem_mono_Y (guard_mem ψ hψ z) (Nat.le_succ j))
  apply FormP.any_mem_Y
  rintro φ hφ
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hφ
  apply FormP.and_mem_Y (FormP.mem_mono_Y (hψ v.1) (Nat.le_succ j))
  apply FormP.and_mem_Y (FormP.truth_mem_Y _ _)
  apply FormP.all_mem_Y
  rintro φ hφ
  obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
  exact cell_mem _ _ _ _ _ _ _ _ hψ _

variable [DecidableEq σ]

/-- The update formula computes the exact attention and feed-forward output.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem sat_nextFormula (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (hΔ : 0 < Δ)
    (hB : ∀ e δ, Δ ≤ δ → coefficient T a ℓ e δ = B e)
    (ψ : (Fin d → Fx p s) → FormP σ) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ t ∈ Finset.Icc 1 i, ∀ q, (ψ q).sat w t = decide (T.actAt w ℓ t = q))
    (r : Fin d → Fx p s) :
    (nextFormula T a ℓ B ψ r Δ).sat w i =
      decide (T.ff ℓ (fun c => Fx.add (T.actAt w ℓ i c)
        (attentionValue T a ℓ w i (T.actAt w ℓ i) c)) = r) := by
  classical
  let q₀ := T.actAt w ℓ i
  let v₀ := fun c => attentionValue T a ℓ w i q₀ c
  have hcur : ∀ q, (ψ q).sat w i = decide (q₀ = q) :=
    hψ i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩)
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold nextFormula
  rw [FormP.sat_any]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
    have hh := Bool.and_eq_true_iff.mp hsat
    obtain ⟨χ, hχ, hχsat⟩ := (FormP.sat_any _ w i).mp hh.2
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hχ
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq] at hχsat
    have hq : q₀ = v.1 := of_decide_eq_true (by rw [← hcur]; exact hχsat.1)
    have ha : v₀ = v.2 := by
      funext c
      have hc := (FormP.sat_all _ w i).mp hχsat.2.2 _
        (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ c)))
      have hs := (sat_cell T a ℓ B v.1 c _ _ ψ w hi hψ
        (integerSum_window T a ℓ B hB hΔ ψ w i hi hψ z hh.1 v.1 none)
        (integerSum_window T a ℓ B hB hΔ ψ w i hi hψ z hh.1 v.1 (some c))
        (v.2 c)).mp hc
      simpa only [v₀, hq] using hs
    change T.ff ℓ (fun c => Fx.add (q₀ c) (v₀ c)) = r
    rw [ha, hq]
    exact hχsat.2.1
  · intro hr
    obtain ⟨z, hz⟩ := exists_guard hΔ ψ w i hi (T.actAt w ℓ) hψ
    refine ⟨_, List.mem_map_of_mem (a := z)
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    apply Bool.and_eq_true_iff.mpr
    refine ⟨hz, (FormP.sat_any _ w i).mpr ?_⟩
    refine ⟨_, List.mem_map_of_mem (a := (q₀, v₀))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq]
    refine ⟨by rw [hcur]; simp, hr, (FormP.sat_all _ w i).mpr ?_⟩
    rintro χ hχ
    obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hχ
    exact (sat_cell T a ℓ B q₀ c _ _ ψ w hi hψ
      (integerSum_window T a ℓ B hB hΔ ψ w i hi hψ z hz q₀ none)
      (integerSum_window T a ℓ B hB hΔ ψ w i hi hψ z hz q₀ (some c))
      (v₀ c)).mpr rfl

/-- A one-dimensional model satisfies the layer's fragment, threshold, stable
table and source-agreement hypotheses (Appendix F). -/
example : ∃ (T : PTfr (Option Bool) 2 0 1 0) (Δ : ℕ)
    (B : Entry 2 0 1 → Fx 2 0) (ψ : (Fin 1 → Fx 2 0) → FormP Bool),
    0 < Δ ∧ (∀ e δ, Δ ≤ δ → coefficient T 1 0 e δ = B e) ∧
    (∀ r, ψ r ∈ TLClY Bool 0) ∧
    (∀ t ∈ Finset.Icc 1 1, ∀ r, (ψ r).sat [true] t = decide (T.actAt [true] 0 t = r)) := by
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
  refine ⟨T, Δ, B, ψ, hΔ, hB, fun _ => FormP.truth_mem_Y _ _, ?_⟩
  intro t ht r
  simp [ψ, T, PTfr.actAt, PTfr.act, PosEnc.emb, Fx.add_zero]

end Transformer.CRASP.AlibiTables
