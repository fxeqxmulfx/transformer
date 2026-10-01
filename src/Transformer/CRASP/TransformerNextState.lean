/-
# The induction step for activation-state formulas

arXiv:2506.16055v3, Appendix B.2, `thm:rtfr_to_TLCl`.
Enumerating the current state and the rounded attention vector makes
residual addition and the feed-forward function a finite Boolean test.
-/

import Transformer.CRASP.AttentionFormulas

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} {p s d k j : ℕ}

open scoped Classical in
/-- The next-state predicate, enumerating query and attention states (B.2). -/
noncomputable def nextFormula (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (r : Fin d → Fx p s) : Form σ :=
  Form.any (Finset.univ.toList.map fun z : (Fin d → Fx p s) × (Fin d → Fx p s) =>
    .and (ψ z.1) (.and (Form.truth (decide (T.ff ℓ (fun c => Fx.add (z.2 c) (z.1 c)) = r)))
      (Form.all (Finset.univ.toList.map fun c : Fin d => T.attentionFormula ℓ ψ z.1 c (z.2 c)))))

/-- One transformer layer costs at most one level of past counting (B.2). -/
theorem nextFormula_mem (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : ∀ q, ψ q ∈ TLCl σ j)
    (r : Fin d → Fx p s) : T.nextFormula ℓ ψ r ∈ TLCl σ (j + 1) := by
  classical
  apply Form.any_mem
  rintro φ hφ
  obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
  apply Form.and_mem (Form.mem_mono (hψ z.1) (Nat.le_succ j))
  apply Form.and_mem (Form.truth_mem _ _)
  apply Form.all_mem
  rintro φ hφ
  obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
  exact T.attentionFormula_mem ℓ hψ z.1 c (z.2 c)

variable [DecidableEq σ]

/-- The next-state predicate agrees with the actual layer at every valid read.
Source: arXiv:2506.16055v3, Appendix B.2, inductive step of Equation `eq:phi_h`. -/
theorem nextFormula_defines (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : T.StateDefinition ℓ ψ) :
    T.StateDefinition (ℓ + 1) (T.nextFormula ℓ ψ) := by
  classical
  intro w i hi r
  have hin : i < (bos w).length := by have := hi.1; rw [length_bos]; omega
  let idx : Fin (bos w).length := ⟨i, hin⟩
  let q₀ := T.actAt w ℓ i
  let a₀ := fun c => T.attention ℓ (T.act (bos w) ℓ) idx c
  have hnext : T.actAt w (ℓ + 1) i = T.ff ℓ (fun c => Fx.add (a₀ c) (q₀ c)) := by
    rw [actAt, dite_eq_left hin, act, layer_eq_attention]
    simp only [q₀, a₀, idx, actAt, dite_eq_left hin]
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold nextFormula
  rw [Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
    simp only [Form.sat, Bool.and_eq_true, Form.sat_truth, decide_eq_true_eq] at hsat
    have hq : q₀ = z.1 := of_decide_eq_true (by rw [← hψ w i hi z.1]; exact hsat.1)
    have ha : a₀ = z.2 := by
      funext c
      have hc := (Form.sat_all w i _).mp hsat.2.2 _
        (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ c)))
      have hh := (T.sat_attentionFormula ℓ ψ z.1 c (z.2 c) w i).mp hc
      rw [← T.attention_eq_countAttention ℓ hψ w idx z.1 hq c] at hh
      exact hh
    rw [hnext, hq, ha]
    exact hsat.2.1
  · intro hr
    have hcoords : (Form.all (Finset.univ.toList.map fun c : Fin d =>
        T.attentionFormula ℓ ψ q₀ c (a₀ c))).sat w i = true := by
      apply (Form.sat_all w i _).mpr
      rintro φ hφ
      obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
      apply (T.sat_attentionFormula ℓ ψ q₀ c (a₀ c) w i).mpr
      exact (T.attention_eq_countAttention ℓ hψ w idx q₀ rfl c).symm
    refine ⟨_, List.mem_map_of_mem (a := (q₀, a₀))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    have hcur : (ψ q₀).sat w i = true := by rw [hψ w i hi]; simp [q₀]
    have hout : T.ff ℓ (fun c => Fx.add (a₀ c) (q₀ c)) = r := by rwa [← hnext]
    simp only [Form.sat, Bool.and_eq_true, Form.sat_truth, decide_eq_true_eq]
    exact ⟨hcur, hout, hcoords⟩

/-- The state-bound and state-definition hypotheses have witnesses (B.2). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 0,
    (∀ q, T.initialFormula q ∈ TLCl Bool 0) ∧ T.StateDefinition 0 T.initialFormula :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0 },
    initialFormula_mem _, initialFormula_defines _⟩

end Transformer.CRASP.RTfr
