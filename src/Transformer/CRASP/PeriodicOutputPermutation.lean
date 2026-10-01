/-
# A one-layer periodic transformer cannot recover source order

arXiv:2506.16055v3, Appendix F, `thm:rtfr_eq_tlclmod`.
Once the final query is fixed, the first layer reads only the multiset of
initial decorated states. This detects the extra power accidentally allowed
by `MOD` with modulus zero in the former Lean syntax.
-/

import Transformer.CRASP.PeriodicInitialWord

namespace Transformer.CRASP.PeriodicAttention

universe u
variable {σ : Type u} {p s d M : ℕ}

/-- At layer zero the last token and its index determine the final query (F). -/
theorem initial_last {k : ℕ} (T : PTfr (Option σ) p s d k) (v : List σ) (a : σ) :
    T.actAt (v ++ [a]) 0 (v.length + 1) =
      (fun c => Fx.add (T.E (some a) c) (T.pe.emb p s d (v.length + 1) c)) := by
  have hin : v.length + 1 < (bos (v ++ [a])).length := by simp
  have hb : (bos (v ++ [a]))[v.length + 1]'hin = some a := by
    apply Option.some.inj
    rw [← List.getElem?_eq_getElem hin]
    simp [bos, List.map_append]
  rw [PTfr.actAt, dite_eq_left hin]
  change (fun c => Fx.add (T.E ((bos (v ++ [a]))[v.length + 1]) c)
    (T.pe.emb p s d (v.length + 1) c)) = _
  rw [hb]

/-- Equal final queries and permuted initial source states give equal outputs.
Source: arXiv:2506.16055v3, Appendix F, finite query/source enumeration in
`thm:rtfr_eq_tlclmod`. -/
theorem out_eq_of_source_perm (T : PTfr (Option σ) p s d 1) (hM : 0 < M)
    (hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a)
    (w w' : List σ)
    (hcur : state T hM 0 w w.length = state T hM 0 w' w'.length)
    (hperm : (CountCells.stateWord (state T hM 0 w) w.length).Perm
      (CountCells.stateWord (state T hM 0 w') w'.length)) :
    T.out (bos w) = T.out (bos w') := by
  have hn : state T hM 1 w w.length = state T hM 1 w' w'.length := by
    rw [state_succ T hM hlogit 0 w w.length le_rfl,
      state_succ T hM hlogit 0 w' w'.length le_rfl, hcur]
    congr 1
    funext c
    exact CountCells.value_eq_of_perm _ _ _ _ _ _ _ _ hperm
  rw [PTfr.out_bos, PTfr.out_bos]
  exact congrArg T.Wout (congrArg Prod.snd hn)

/-- Swapping the first and one-period-later letter preserves a one-layer output.
Source: arXiv:2506.16055v3, Appendix F, periodic finite-state simulation. -/
theorem out_gap_swap (T : PTfr (Option σ) p s d 1) (hM : 0 < M)
    (hemb : ∀ i, T.pe.emb p s d i = T.pe.emb p s d (i % M))
    (hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a)
    (a f b c : σ) :
    T.out (bos (a :: (List.replicate (M - 1) f ++ [b, c]))) =
      T.out (bos (b :: (List.replicate (M - 1) f ++ [a, c]))) := by
  apply out_eq_of_source_perm T hM hlogit
  · apply Prod.ext
    · simp [state]
    · change T.actAt _ 0 _ = T.actAt _ 0 _
      have hleft : a :: (List.replicate (M - 1) f ++ [b, c]) =
          (a :: (List.replicate (M - 1) f ++ [b])) ++ [c] := by simp
      have hright : b :: (List.replicate (M - 1) f ++ [a, c]) =
          (b :: (List.replicate (M - 1) f ++ [a])) ++ [c] := by simp
      rw [hleft, hright]
      simp only [List.length_append, List.length_singleton]
      rw [initial_last, initial_last]
      simp
  · rw [stateWord_zero T hM hemb, stateWord_zero T hM hemb]
    exact initialWord_swap T hM a f b c

/-- The hypotheses permit a genuine two-source permutation (Appendix F). -/
example : ∃ T : PTfr (Option Bool) 2 0 0 1,
    (∀ i j (q a : Fin 0 → Fx 2 0), T.pe.logit i j q a =
      T.pe.logit (i % 1) (j % 1) q a) ∧
    state T (M := 1) one_pos 0 [true, false] 2 =
      state T (M := 1) one_pos 0 [false, true] 2 ∧
    (CountCells.stateWord (state T (M := 1) one_pos 0 [true, false]) 2).Perm
      (CountCells.stateWord (state T (M := 1) one_pos 0 [false, true]) 2) := by
  let T : PTfr (Option Bool) 2 0 0 1 := {
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .plain }
  have hconst : ∀ w i, state T (M := 1) one_pos 0 w i =
      (⟨0, one_pos⟩, fun _ => 0) := by
    intro w i
    apply Prod.ext
    · apply Fin.ext; simp [state, phase]
    · funext c; exact Fin.elim0 c
  refine ⟨T, fun _ _ _ _ => rfl, ?_, ?_⟩
  · rw [hconst, hconst]
  · change (List.ofFn (fun j : Fin 2 => state T one_pos 0 [true, false] (j.val + 1))).Perm
      (List.ofFn (fun j : Fin 2 => state T one_pos 0 [false, true] (j.val + 1)))
    simp only [hconst]
    exact List.Perm.refl _

end Transformer.CRASP.PeriodicAttention
