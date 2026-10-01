/-
# Source permutations at positions with the same periodic encoding

arXiv:2506.16055v3, Appendix F, periodicity assumption of
`thm:rtfr_eq_tlclmod`. At the first layer, swapping two equal-residue input
positions permutes the decorated source states.
-/

import Transformer.CRASP.PeriodicInitial
import Transformer.CRASP.StatePermutation

namespace Transformer.CRASP.PeriodicAttention

universe u
variable {σ : Type u} {p s d k M : ℕ}

/-- The list of ordinary initial decorated states (Appendix F). -/
noncomputable def initialWord (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (w : List σ) : List (State p s d M) :=
  w.mapIdx (fun j a => (phase M hM (j + 1), initialValue T (phase M hM (j + 1)) a))

/-- A periodic embedding makes the initial source word exactly this list (F). -/
theorem stateWord_zero (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (hemb : ∀ i, T.pe.emb p s d i = T.pe.emb p s d (i % M)) (w : List σ) :
    CountCells.stateWord (state T hM 0 w) w.length = initialWord T hM w := by
  apply List.ext_getElem (by simp [CountCells.stateWord, initialWord])
  intro j hj hj'
  have hjw : j < w.length := by simpa [CountCells.stateWord] using hj
  simp only [CountCells.stateWord, List.getElem_ofFn, initialWord, List.getElem_mapIdx,
    state]
  congr 1
  have hin : j + 1 < (bos w).length := by rw [length_bos]; omega
  rw [PTfr.actAt, dite_eq_left hin]
  change (fun c => Fx.add (T.E ((bos w)[j + 1]) c) (T.pe.emb p s d (j + 1) c)) = _
  have hwi : (bos w)[j + 1]'hin = some (w[j]'hjw) := by simp [bos]
  rw [hwi, hemb (j + 1)]
  rfl

/-- Two positions separated by one common period have the same residue (F). -/
theorem phase_add_period (hM : 0 < M) (i : ℕ) :
    phase M hM (M + i) = phase M hM i := by
  apply Fin.ext
  simp [phase]

/-- Expanding a word with a full-period gap exposes its two equal-residue states.
Source: arXiv:2506.16055v3, Appendix F, periodic embeddings. -/
theorem initialWord_gap (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (a f b c : σ) :
    initialWord T hM (a :: (List.replicate (M - 1) f ++ [b, c])) =
      (phase M hM 1, initialValue T (phase M hM 1) a) ::
        ((List.replicate (M - 1) f).mapIdx (fun j a =>
          (phase M hM (j + 2), initialValue T (phase M hM (j + 2)) a)) ++
        [(phase M hM 1, initialValue T (phase M hM 1) b),
         (phase M hM (M + 2), initialValue T (phase M hM (M + 2)) c)]) := by
  simp only [initialWord, List.mapIdx_cons, List.mapIdx_append, List.length_replicate,
    List.mapIdx_nil, Nat.zero_add, Nat.add_assoc]
  rw [show M - 1 + (1 + 1) = M + 1 by omega,
    show 1 + (M + 1) = M + 2 by omega, phase_add_period hM 1]

/-- Swapping letters across a full-period gap permutes the initial source states.
Source: arXiv:2506.16055v3, Appendix F, the scope of periodic encodings. -/
theorem initialWord_swap (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (a f b c : σ) :
    (initialWord T hM (a :: (List.replicate (M - 1) f ++ [b, c]))).Perm
      (initialWord T hM (b :: (List.replicate (M - 1) f ++ [a, c]))) := by
  rw [initialWord_gap, initialWord_gap]
  exact (List.perm_middle.cons _).trans
    ((List.Perm.swap _ _ _).trans (List.perm_middle.symm.cons _))

/-- A nonempty common period and embedding periodicity have a plain witness (F). -/
example : ∃ T : PTfr (Option Bool) 2 0 0 0,
    (∀ i, T.pe.emb 2 0 0 i = T.pe.emb 2 0 0 (i % 1)) ∧ 0 < 1 := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .plain }, fun _ => rfl, one_pos⟩

end Transformer.CRASP.PeriodicAttention
