/-
# Agreement of periodic attention with its finite tables

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
Reducing the two positional indices modulo a common period changes neither
the rounded sums nor the residual/feed-forward update.
-/

import Transformer.CRASP.PeriodicAttention

namespace Transformer.CRASP.PeriodicAttention

universe u
variable {σ : Type u} {p s d k M : ℕ}

/-- Periodic logits make the finite-table attention exact (Appendix F). -/
theorem attention_eq_value (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a)
    (ℓ : ℕ) (w : List σ) (i : Fin (bos w).length) (c : Fin d) :
    attention T ℓ (T.act (bos w) ℓ) i c =
      CountCells.value (state T hM ℓ [] 0) (weight T ℓ (state T hM ℓ w i.val))
        (numerator T ℓ (state T hM ℓ w i.val) c) (value T ℓ c)
        (state T hM ℓ w) i.val := by
  let q := state T hM ℓ w i.val
  let b := state T hM ℓ [] 0
  let Q := state T hM ℓ w
  let D := CountCells.sum b (weight T ℓ q) Q i.val
  let N := CountCells.sum b (numerator T ℓ q c) Q i.val
  let V := CountCells.sum b (value T ℓ c) Q i.val
  have hscore (j : Fin (bos w).length) :
      T.pe.logit i.val j.val (T.WQ ℓ (T.act (bos w) ℓ i)) (T.WK ℓ (T.act (bos w) ℓ j)) =
        T.pe.logit q.1.val (phase M hM j.val).val (T.WQ ℓ q.2)
          (T.WK ℓ (T.act (bos w) ℓ j)) := by
    simp only [q, state, phase, PTfr.actAt_valid]
    exact hlogit _ _ _ _
  have hD : (∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp
      (T.pe.logit i.val j.val (T.WQ ℓ (T.act (bos w) ℓ i))
        (T.WK ℓ (T.act (bos w) ℓ j))))).val) = (D : ℝ) / 2 ^ s := by
    have h := sum_states T hM ℓ w i (weight T ℓ q)
    rw [← h]
    apply Finset.sum_congr rfl
    intro j hj
    exact congrArg (fun x => (Fx.round p s (Real.exp x)).val) (hscore j)
  have hN : (∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp
      (T.pe.logit i.val j.val (T.WQ ℓ (T.act (bos w) ℓ i))
        (T.WK ℓ (T.act (bos w) ℓ j))) * (T.WV ℓ (T.act (bos w) ℓ j) c).val)).val) =
      (N : ℝ) / 2 ^ s := by
    have h := sum_states T hM ℓ w i (numerator T ℓ q c)
    rw [← h]
    apply Finset.sum_congr rfl
    intro j hj
    exact congrArg (fun x => (Fx.round p s
      (Real.exp x * (T.WV ℓ (T.act (bos w) ℓ j) c).val)).val) (hscore j)
  have hV : (∑ j ∈ RTfr.masked i, (T.WV ℓ (T.act (bos w) ℓ j) c).val) =
      (V : ℝ) / 2 ^ s := sum_states T hM ℓ w i (value T ℓ c)
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  simp only [attention, hD, hN, hV, RTfr.card_masked]
  change (if (D : ℝ) / 2 ^ s = 0 then
      Fx.round p s (((V : ℝ) / 2 ^ s) / (i.val + 1 : ℕ))
    else Fx.round p s (((N : ℝ) / 2 ^ s) / ((D : ℝ) / 2 ^ s))) = _
  change _ = (if D = 0 then Fx.round p s (((V : ℝ) / (i.val + 1 : ℕ)) / 2 ^ s)
    else Fx.round p s ((N : ℝ) / (D : ℝ)))
  by_cases hz : D = 0
  · simp only [hz, Int.cast_zero, zero_div, ite_true]
    congr 1
    ring
  · have hzr : (D : ℝ) ≠ 0 := by exact_mod_cast hz
    simp only [hz, div_eq_zero_iff, hzr, hs, or_self, ite_false]
    congr 1
    field_simp

/-- A periodic transformer's actual layer obeys the finite-state recurrence (F). -/
theorem state_succ (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a)
    (ℓ : ℕ) (w : List σ) (i : ℕ) (hi : i ≤ w.length) :
    state T hM (ℓ + 1) w i = update T ℓ (state T hM ℓ w i) (fun c =>
      CountCells.value (state T hM ℓ [] 0) (weight T ℓ (state T hM ℓ w i))
        (numerator T ℓ (state T hM ℓ w i) c) (value T ℓ c) (state T hM ℓ w) i) := by
  have hin : i < (bos w).length := by rw [length_bos]; omega
  let idx : Fin (bos w).length := ⟨i, hin⟩
  simp only [state, PTfr.actAt, dite_eq_left hin, PTfr.act, layer_eq_attention, update]
  congr 2
  funext c
  rw [attention_eq_value T hM hlogit ℓ w idx c]
  simp only [state, PTfr.actAt, idx, dite_eq_left hin]

/-- Zero angles give periodic logits and valid positions (Appendix F). -/
example : (∀ i j (q a : Fin 1 → Fx 2 0),
    (PosEnc.sinusoidal (fun _ => 0)).logit i j q a =
      (PosEnc.sinusoidal (fun _ => 0)).logit (i % 1) (j % 1) q a) ∧
    0 < 1 ∧ 1 ≤ [true].length := ⟨fun _ _ _ _ => rfl, one_pos, le_rfl⟩

end Transformer.CRASP.PeriodicAttention
