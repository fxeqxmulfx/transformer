/-
# State predicates for every ALiBi layer

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
The initial predicates read the finite alphabet. Each later layer uses the
uniform stable-tail table and the finite recent-window enumeration.
-/

import Transformer.CRASP.AlibiNextState
import Transformer.CRASP.PositionalStateInduction
import Transformer.CRASP.FixedSign

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The actual layer recurrence expressed by the integer attention value (F). -/
theorem state_succ (T : PTfr (Option σ) p s d k) (a : ℝ) (hT : T.pe = .alibi a)
    (ℓ : ℕ) (w : List σ) (i : ℕ) (hi : i ≤ w.length) :
    T.actAt w (ℓ + 1) i = T.ff ℓ (fun c => Fx.add (T.actAt w ℓ i c)
      (attentionValue T a ℓ w i (T.actAt w ℓ i) c)) := by
  have hin : i < (bos w).length := by rw [length_bos]; omega
  let idx : Fin (bos w).length := ⟨i, hin⟩
  simp only [PTfr.actAt, dite_eq_left hin, PTfr.act, PeriodicAttention.layer_eq_attention]
  congr 1
  funext c
  have hh := attention_eq_value T a hT ℓ w idx (T.actAt w ℓ i) rfl c
  simp only [PTfr.actAt, idx, dite_eq_left hin] at hh
  rw [hh]
  simp only [Fx.add, add_comm]

variable [Fintype σ]

open scoped Classical in
/-- The finite disjunction of letters with a specified embedding (Appendix F). -/
noncomputable def initialFormula (T : PTfr (Option σ) p s d k)
    (q : Fin d → Fx p s) : FormP σ :=
  FormP.any ((Finset.univ.filter fun a => T.E (some a) = q).toList.map FormP.sym)

/-- The initial ALiBi predicates have counting depth zero (Appendix F). -/
theorem initialFormula_mem (T : PTfr (Option σ) p s d k) (q : Fin d → Fx p s) :
    initialFormula T q ∈ TLClY σ 0 := by
  classical
  apply FormP.any_mem_Y
  rintro φ hφ
  obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hφ
  exact ⟨rfl, le_rfl⟩

variable [DecidableEq σ]

/-- The embedding formula defines the actual ALiBi initial state.
Source: arXiv:2506.16055v3, Appendix F, ALiBi has no additive embedding. -/
theorem initialFormula_defines (T : PTfr (Option σ) p s d k) (a : ℝ)
    (hT : T.pe = .alibi a) :
    PositionalNext.Defines (fun ℓ w i => T.actAt w ℓ i) 0 (initialFormula T) := by
  classical
  intro w i hi hile q
  have hj : i - 1 < w.length := by omega
  have hin : i < (bos w).length := by rw [length_bos]; omega
  have hwi : (bos w)[i]'hin = some (w[i - 1]'hj) := by
    cases i with
    | zero => omega
    | succ j => simp [bos]
  have hvec : T.actAt w 0 i = T.E (some (w[i - 1])) := by
    rw [PTfr.actAt, dite_eq_left hin]
    change (fun c => Fx.add (T.E ((bos w)[i]) c) (T.pe.emb p s d i c)) = _
    rw [hwi, hT]
    simp only [PosEnc.emb, Fx.add_zero]
  change (initialFormula T q).sat w i = decide (T.actAt w 0 i = q)
  rw [Bool.eq_iff_iff, decide_eq_true_iff, initialFormula, FormP.sat_any, hvec]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨b, hb, rfl⟩ := List.mem_map.mp hφ
    have he := (Finset.mem_filter.mp (Finset.mem_toList.mp hb)).2
    have hw : w[i - 1] = b := by
      simpa only [FormP.sat, List.getElem?_eq_getElem hj, Option.some.injEq,
        decide_eq_true_eq] using hsat
    rw [hw]
    exact he
  · intro hq
    refine ⟨FormP.sym w[i - 1], List.mem_map_of_mem ?_, ?_⟩
    · exact Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq⟩)
    · simp only [FormP.sat, List.getElem?_eq_getElem hj, decide_true]

/-- ALiBi activations have same-depth predicates in the MOD-free fragment.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem exists_formulas (T : PTfr (Option σ) p s d k) (a : ℝ)
    (hT : T.pe = .alibi a) (ℓ : ℕ) :
    ∃ ψ : (Fin d → Fx p s) → FormP σ, (∀ q, ψ q ∈ TLClY σ ℓ) ∧
      PositionalNext.Defines (fun j w i => T.actAt w j i) ℓ ψ := by
  induction ℓ with
  | zero => exact ⟨initialFormula T, initialFormula_mem T, initialFormula_defines T a hT⟩
  | succ ℓ ih =>
      obtain ⟨ψ, hb, hs⟩ := ih
      obtain ⟨Δ, B, hΔ, hB⟩ := exists_tail T a ℓ
      refine ⟨fun q => nextFormula T a ℓ B ψ q Δ,
        fun q => nextFormula_mem T a ℓ B ψ hb q Δ, ?_⟩
      intro w i hi hile q
      change (nextFormula T a ℓ B ψ q Δ).sat w i = decide (T.actAt w (ℓ + 1) i = q)
      rw [sat_nextFormula T a ℓ B hΔ hB ψ w hi, state_succ T a hT ℓ w i hile]
      intro t ht q
      have htb := Finset.mem_Icc.mp ht
      exact hs w t (by omega) (by omega) q

/-- The encoding and valid-position hypotheses have a zero-model witness (F). -/
example : ∃ T : PTfr (Option Bool) 2 0 0 0,
    T.pe = .alibi 1 ∧ 0 < 1 ∧ 1 ≤ [true].length :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0,
     pe := .alibi 1 }, rfl, one_pos, le_rfl⟩

end Transformer.CRASP.AlibiTables
