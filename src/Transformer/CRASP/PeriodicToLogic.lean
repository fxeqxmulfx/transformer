/-
# A periodic positional transformer has a same-depth counting formula

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod` and the reverse
direction of `thm:rtfr_eq_tlclmod`. The finite residue/activation induction
is followed by the accepting-state test, with a separate BOS empty input.
-/

import Transformer.CRASP.PeriodicInitial

namespace Transformer.CRASP

universe u
variable {σ : Type u} [Fintype σ] [DecidableEq σ] {p s d k M : ℕ}

/-- The ordinary-position predicate, embedded in positional syntax (F). -/
noncomputable def FormP.onStr : FormP σ :=
  (Form.onStr : Form σ).substPos FormP.sym

omit [DecidableEq σ] in
/-- The ordinary-position test is a depth-zero periodic formula (Appendix F). -/
theorem FormP.onStr_mem (k : ℕ) : (FormP.onStr : FormP σ) ∈ TLClMod σ k := by
  refine ⟨Form.prevFree_substPos FormP.sym (fun _ => rfl) _, ?_⟩
  have h := Form.depth_substPos_le (fun a : σ => FormP.sym a) 0
    (fun _ => le_rfl) (Form.onStr : Form σ)
  have h0 : (Form.substPos FormP.sym (Form.onStr : Form σ)).depth ≤ 0 := by
    simpa only [Form.depth_onStr, Nat.zero_add] using h
  exact h0.trans (Nat.zero_le k)

/-- At the final position the ordinary-position test detects nonempty words (F). -/
theorem FormP.sat_onStr_end (w : List σ) :
    (FormP.onStr : FormP σ).sat w w.length = decide (0 < w.length) := by
  have hr : RTfr.Readable w w.length := ⟨le_rfl, by
    by_cases hw : w = []
    · exact Or.inr hw
    · exact Or.inl (List.length_pos_iff.mpr hw)⟩
  rw [FormP.onStr, Form.sat_substPos FormP.sym w w (fun _ _ _ => rfl)
    Form.onStr Form.past_onStr Form.pnpFree_onStr _ hr, Form.sat_onStr]
  apply decide_eq_decide.mpr
  omega

/-- Every transformer with periodic embedding and logits has a same-depth
`MOD` formula. The hypotheses specify the actual common period.

Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod` and
`thm:rtfr_eq_tlclmod`. -/
theorem exists_mem_TLClMod_of_periodic (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (hemb : ∀ i, T.pe.emb p s d i = T.pe.emb p s d (i % M))
    (hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a) :
    ∃ φ ∈ TLClMod σ k, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  classical
  let H := PeriodicAttention.state T hM
  let b := fun ℓ => H ℓ [] 0
  obtain ⟨ψ, hb, hs⟩ := PositionalNext.exists_formulas H b
    (PeriodicAttention.weight T) (PeriodicAttention.numerator T)
    (PeriodicAttention.value T) (PeriodicAttention.update T)
    (PeriodicAttention.weight_nonneg T)
    ⟨PeriodicAttention.initialFormula T, PeriodicAttention.initialFormula_mem T,
      PeriodicAttention.initialFormula_defines T hM hemb⟩
    (fun ℓ w i _ hi => PeriodicAttention.state_succ T hM hlogit ℓ w i hi) k
  let A := (Finset.univ : Finset (PeriodicAttention.State p s d M)).filter
    (fun q => 0 < (T.Wout q.2).val)
  let φ := FormP.any (A.toList.map ψ)
  let e : FormP σ := FormP.truth (decide (T.Accepts (bos ([] : List σ))))
  refine ⟨FormP.or (.and FormP.onStr φ) (.and (.neg FormP.onStr) e),
    FormP.or_mem (FormP.and_mem (FormP.onStr_mem _) (FormP.any_mem _ ?_))
      (FormP.and_mem (FormP.neg_mem (FormP.onStr_mem _)) (FormP.truth_mem _ _)), ?_⟩
  · rintro χ hχ
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hχ
    exact hb q
  · ext w
    change (FormP.or (.and FormP.onStr φ) (.and (.neg FormP.onStr) e)).sat w w.length =
      true ↔ T.Accepts (bos w)
    rw [FormP.sat_or, FormP.sat, FormP.sat, FormP.sat, FormP.sat_onStr_end]
    by_cases hw : w = []
    · subst w
      simp [e]
    · have hn : 0 < w.length := List.length_pos_iff.mpr hw
      simp only [hn, decide_true, Bool.true_and, Bool.not_true, Bool.false_and,
        Bool.or_false]
      rw [FormP.sat_any, PTfr.Accepts, PTfr.out_bos]
      constructor
      · rintro ⟨χ, hχ, hsat⟩
        obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hχ
        have heq : H k w w.length = q := of_decide_eq_true (by
          rw [← hs w w.length hn le_rfl q]; exact hsat)
        have hv : T.actAt w k w.length = q.2 := congrArg Prod.snd heq
        rw [hv]
        exact (Finset.mem_filter.mp (Finset.mem_toList.mp hq)).2
      · intro hacc
        refine ⟨ψ (H k w w.length), List.mem_map_of_mem ?_, ?_⟩
        · exact Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hacc⟩)
        · rw [hs w w.length hn le_rfl]
          simp

/-- Periodic zero-dimensional transformers provide witnesses for both hypotheses.
Source: arXiv:2506.16055v3, Appendix F, periodic simulation. -/
example : ∃ T : PTfr (Option Bool) 2 0 0 0,
    (∀ i, T.pe.emb 2 0 0 i = T.pe.emb 2 0 0 (i % 1)) ∧
    (∀ i j (q a : Fin 0 → Fx 2 0), T.pe.logit i j q a =
      T.pe.logit (i % 1) (j % 1) q a) := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .plain }, ?_, ?_⟩
  · intro i; rfl
  · intro i j q a; rfl

end Transformer.CRASP
