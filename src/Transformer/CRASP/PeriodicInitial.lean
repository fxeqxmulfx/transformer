/-
# Embedding predicates for periodic position encodings

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
At an ordinary position the initial state depends only on its letter and
the residue of its position. Both are depth-zero positional predicates.
-/

import Transformer.CRASP.PeriodicAttentionStep

namespace Transformer.CRASP.PeriodicAttention

universe u
variable {σ : Type u} [Fintype σ] {p s d k M : ℕ}

/-- The initial vector for a fixed letter and position residue (Appendix F). -/
noncomputable def initialValue (T : PTfr (Option σ) p s d k) (r : Fin M) (a : σ) :
    Fin d → Fx p s := fun c => Fx.add (T.E (some a) c) (T.pe.emb p s d r.val c)

open scoped Classical in
/-- The letter/residue test for a possible initial decorated state (F). -/
noncomputable def initialFormula (T : PTfr (Option σ) p s d k)
    (q : State p s d M) : FormP σ :=
  .and (.mod M q.1.val) (FormP.any
    ((Finset.univ.filter fun a => initialValue T q.1 a = q.2).toList.map FormP.sym))

/-- Initial periodic predicates add no counting depth (Appendix F). -/
theorem initialFormula_mem (T : PTfr (Option σ) p s d k) (q : State p s d M) :
    initialFormula T q ∈ TLClMod σ 0 := by
  classical
  apply FormP.and_mem ⟨rfl, le_rfl⟩
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hφ
  exact ⟨rfl, le_rfl⟩

variable [DecidableEq σ]

/-- The initial-state predicate defines the actual positionally augmented
embedding. Source: arXiv:2506.16055v3, Appendix F, Equation `eq:emb`. -/
theorem initialFormula_defines (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (hemb : ∀ i, T.pe.emb p s d i = T.pe.emb p s d (i % M)) :
    PositionalNext.Defines (state T hM) 0 (initialFormula T) := by
  classical
  intro w i hi hile q
  have hj : i - 1 < w.length := by omega
  have hin : i < (bos w).length := by rw [length_bos]; omega
  have hwi : (bos w)[i]'hin = some (w[i - 1]'hj) := by
    cases i with
    | zero => omega
    | succ j => simp [bos]
  have hvec : T.actAt w 0 i = initialValue T (phase M hM i) (w[i - 1]) := by
    rw [PTfr.actAt, dite_eq_left hin]
    change (fun c => Fx.add (T.E ((bos w)[i]) c) (T.pe.emb p s d i c)) = _
    rw [hwi, hemb i]
    rfl
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  simp only [initialFormula, FormP.sat, Bool.and_eq_true, decide_eq_true_eq]
  rw [FormP.sat_any]
  have hmod : q.1.val % M = q.1.val := Nat.mod_eq_of_lt q.1.isLt
  rw [hmod]
  constructor
  · rintro ⟨hr, φ, hφ, hsat⟩
    obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hφ
    have he := (Finset.mem_filter.mp (Finset.mem_toList.mp ha)).2
    have hwa : w[i - 1] = a := by
      simpa only [FormP.sat, List.getElem?_eq_getElem hj, Option.some.injEq,
        decide_eq_true_eq] using hsat
    have hp : phase M hM i = q.1 := Fin.ext hr
    apply Prod.ext hp
    rw [state, hvec, hp, hwa]
    exact he
  · intro hq
    have hp : phase M hM i = q.1 := congrArg Prod.fst hq
    have hv : T.actAt w 0 i = q.2 := congrArg Prod.snd hq
    refine ⟨congrArg Fin.val hp, FormP.sym w[i - 1], List.mem_map_of_mem ?_, ?_⟩
    · apply Finset.mem_toList.mpr
      apply Finset.mem_filter.mpr
      refine ⟨Finset.mem_univ _, ?_⟩
      rw [← hp, ← hvec]
      exact hv
    · simp only [FormP.sat, List.getElem?_eq_getElem hj, decide_true]

/-- Zero-angle sinusoidal embeddings meet the periodicity hypothesis (F). -/
example : ∀ i : ℕ, (PosEnc.sinusoidal (fun _ => 0)).emb 2 0 1 i =
    (PosEnc.sinusoidal (fun _ => 0)).emb 2 0 1 (i % 1) := by
  intro i
  funext c
  simp [PosEnc.emb, sinusoidalVec, rotate]

end Transformer.CRASP.PeriodicAttention
