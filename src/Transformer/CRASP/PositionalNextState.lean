/-
# One finite-state attention step in positional logic

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
After enumerating a query state and an attention vector, the update is a
finite Boolean test. Only the weighted prefix sums add counting depth.
-/

import Transformer.CRASP.PositionalStateCounts

namespace Transformer.CRASP.PositionalNext

universe u v
variable {α : Type u} {σ : Type v} [Fintype α] {p s d k : ℕ}

open scoped Classical in
/-- A predicate for a finite-state update from attention-table cells (F). -/
noncomputable def formula (b : α) (D : α → α → Fx p s)
    (N : α → Fin d → α → Fx p s) (V : Fin d → α → Fx p s)
    (update : α → (Fin d → Fx p s) → α) (ψ : α → FormP σ) (r : α) : FormP σ :=
  FormP.any (Finset.univ.toList.map fun z : α × (Fin d → Fx p s) =>
    .and (ψ z.1) (.and (FormP.truth (decide (update z.1 z.2 = r)))
      (FormP.all (Finset.univ.toList.map fun c : Fin d =>
        CountCells.stateFormula b (D z.1) (N z.1 c) (V c) ψ (z.2 c)))))

/-- One update adds at most one counting layer (Appendix F). -/
theorem formula_mem (b : α) (D : α → α → Fx p s)
    (N : α → Fin d → α → Fx p s) (V : Fin d → α → Fx p s)
    (update : α → (Fin d → Fx p s) → α) (ψ : α → FormP σ)
    (hψ : ∀ a, ψ a ∈ TLClMod σ k) (r : α) :
    formula b D N V update ψ r ∈ TLClMod σ (k + 1) := by
  classical
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
  apply FormP.and_mem (FormP.mem_mono (hψ z.1) (Nat.le_succ k))
  apply FormP.and_mem (FormP.truth_mem _ _)
  apply FormP.all_mem
  rintro φ hφ
  obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
  exact CountCells.stateFormula_mem b (D z.1) (N z.1 c) (V c) ψ hψ (z.2 c)

variable [DecidableEq α] [DecidableEq σ]

/-- The update formula agrees with the finite-state attention step.
Source: arXiv:2506.16055v3, Appendix F, enumeration of finite queries. -/
theorem sat_formula (b : α) (D : α → α → Fx p s)
    (N : α → Fin d → α → Fx p s) (V : Fin d → α → Fx p s)
    (update : α → (Fin d → Fx p s) → α) (ψ : α → FormP σ)
    (hD : ∀ q a, 0 ≤ (D q a).m) (q : ℕ → α) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat w j = decide (q j = a)) (r : α) :
    (formula b D N V update ψ r).sat w i =
      decide (update (q i) (fun c => CountCells.value b (D (q i)) (N (q i) c)
        (V c) q i) = r) := by
  classical
  let a₀ := fun c => CountCells.value b (D (q i)) (N (q i) c) (V c) q i
  have hcur : ∀ a, (ψ a).sat w i = decide (q i = a) :=
    hψ i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩)
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold formula
  rw [FormP.sat_any]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hφ
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq] at hsat
    have hq : q i = z.1 := of_decide_eq_true (by rw [← hcur]; exact hsat.1)
    have ha : a₀ = z.2 := by
      funext c
      have hc := (FormP.sat_all _ w i).mp hsat.2.2 _
        (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ c)))
      have hh := (CountCells.sat_stateFormula b (D z.1) (N z.1 c) (V c)
        (hD z.1) ψ q w hi hψ (z.2 c)).mp hc
      simpa only [a₀, CountCells.value, hq] using hh
    change update (q i) a₀ = r
    rw [ha, hq]
    exact hsat.2.1
  · intro hr
    have hcoords : (FormP.all (Finset.univ.toList.map fun c : Fin d =>
        CountCells.stateFormula b (D (q i)) (N (q i) c) (V c) ψ (a₀ c))).sat w i = true := by
      apply (FormP.sat_all _ w i).mpr
      rintro φ hφ
      obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hφ
      exact (CountCells.sat_stateFormula b (D (q i)) (N (q i) c) (V c)
        (hD (q i)) ψ q w hi hψ (a₀ c)).mpr rfl
    refine ⟨_, List.mem_map_of_mem (a := (q i, a₀))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq]
    exact ⟨by rw [hcur]; simp, hr, hcoords⟩

/-- Constant tables and a singleton source alphabet satisfy the hypotheses (F). -/
example : (∀ _ _ : Unit, 0 ≤ (0 : Fx 2 0).m) ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Unit,
      (FormP.truth true : FormP Bool).sat [true] j = decide (() = a)) ∧
    (∀ _ : Unit, (FormP.truth true : FormP Bool) ∈ TLClMod Bool 0) := by
  simp [FormP.truth_mem]

end Transformer.CRASP.PositionalNext
