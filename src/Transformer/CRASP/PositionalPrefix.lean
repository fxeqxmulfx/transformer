/-
# The BOS state and sums over a masked prefix

arXiv:2506.16055v3, Appendix F, simulation on `BOS · w`.
The initial position has the same activation on every word; later
attention sums split into this one BOS contribution and past counts.
-/

import Transformer.CRASP.PositionalTransformers

namespace Transformer.CRASP.PTfr

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The activation at a zero-based index of `BOS · w`; zero off the word.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:phi_h`. -/
noncomputable def actAt (T : PTfr (Option σ) p s d k) (w : List σ) (ℓ i : ℕ) :
    Fin d → Fx p s :=
  if h : i < (bos w).length then T.act (bos w) ℓ ⟨i, h⟩ else fun _ => 0

/-- At a valid index, the natural-indexed notation is the original activation.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:phi_h`. -/
theorem actAt_valid (T : PTfr (Option σ) p s d k) (w : List σ) (ℓ : ℕ)
    (i : Fin (bos w).length) : T.actAt w ℓ i.val = T.act (bos w) ℓ i := by
  unfold actAt
  rw [dite_eq_left i.isLt]

/-- A layer at the first position depends only on that position's state.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:att`. -/
theorem layer_first_congr (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    {n m : ℕ} (hn : 0 < n) (hm : 0 < m)
    (h : Fin n → Fin d → Fx p s) (h' : Fin m → Fin d → Fx p s)
    (heq : h ⟨0, hn⟩ = h' ⟨0, hm⟩) :
    T.layer ℓ h ⟨0, hn⟩ = T.layer ℓ h' ⟨0, hm⟩ := by
  simp only [layer, RTfr.masked_first, Finset.sum_singleton, Finset.card_singleton, heq]

/-- The BOS activation does not depend on the word (Appendix F). -/
theorem actAt_bos (T : PTfr (Option σ) p s d k) (w : List σ) (ℓ : ℕ) :
    T.actAt w ℓ 0 = T.actAt [] ℓ 0 := by
  have hw : 0 < (bos w).length := by rw [length_bos]; omega
  have he : 0 < (bos ([] : List σ)).length := by simp
  induction ℓ with
  | zero =>
      rw [actAt, actAt, dite_eq_left hw, dite_eq_left he]
      simp [act, bos]
  | succ ℓ ih =>
      simp only [actAt, dite_eq_left hw, dite_eq_left he, act]
      apply layer_first_congr
      simpa only [actAt, dite_eq_left hw, dite_eq_left he] using ih

/-- A masked sum splits into BOS and the one-based positions `1,...,i`.
Source: arXiv:2506.16055v3, Appendix F, numerator and denominator counts. -/
theorem sum_masked_actAt (T : PTfr (Option σ) p s d k) (w : List σ) (ℓ : ℕ)
    (i : Fin (bos w).length) (f : (Fin d → Fx p s) → ℝ) :
    (∑ j ∈ RTfr.masked i, f (T.act (bos w) ℓ j)) =
      f (T.actAt [] ℓ 0) + ∑ j ∈ Finset.Icc 1 i.val, f (T.actAt w ℓ j) := by
  have hsum : (∑ j ∈ RTfr.masked i, f (T.act (bos w) ℓ j)) =
      ∑ j ∈ Finset.Icc 0 i.val, f (T.actAt w ℓ j) := by
    refine Finset.sum_bij (fun j _ => j.val) ?_ ?_ ?_ ?_
    · intro j hj
      simp only [RTfr.masked, Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_def] at hj
      exact Finset.mem_Icc.mpr ⟨Nat.zero_le _, hj⟩
    · intro j hj t ht heq
      exact Fin.ext heq
    · intro j hj
      have hjle := (Finset.mem_Icc.mp hj).2
      refine ⟨⟨j, lt_of_le_of_lt hjle i.isLt⟩, ?_, rfl⟩
      simpa only [RTfr.masked, Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_def] using hjle
    · intro j hj
      rw [actAt_valid]
  have hsplit : Finset.Icc 0 i.val = insert 0 (Finset.Icc 1 i.val) := by
    ext j
    simp only [Finset.mem_Icc, Finset.mem_insert]
    omega
  rw [hsum, hsplit, Finset.sum_insert (by simp), actAt_bos]

/-- The last activation of `BOS · w` is the one used by recognition (B.2). -/
theorem out_bos (T : PTfr (Option σ) p s d k) (w : List σ) :
    T.out (bos w) = T.Wout (T.actAt w k w.length) := by
  have hw : 0 < (bos w).length := by rw [length_bos]; omega
  rw [out, dite_eq_left hw, actAt, dite_eq_left (by rw [length_bos]; omega)]
  congr 2
  ext
  simp [length_bos]

/-- The positive-length and first-state hypotheses above have witnesses. -/
example : 0 < 1 ∧
    (fun _ : Fin 1 => (fun _ : Fin 1 => (0 : Fx 2 0))) ⟨0, one_pos⟩ =
      (fun _ : Fin 1 => (fun _ : Fin 1 => (0 : Fx 2 0))) ⟨0, one_pos⟩ :=
  ⟨one_pos, rfl⟩

end Transformer.CRASP.PTfr
