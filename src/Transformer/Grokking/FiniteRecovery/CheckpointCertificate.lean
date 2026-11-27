module

public import Transformer.Grokking.FiniteRecovery.StaircaseDecoder
public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Mathlib.Probability.ProductMeasure

/-! Observable certificates for finite training runs. The binary staircase
certificate applies to any real factor proposal, including a leading singular
component of the model's phase table. The independent-seed bound concerns a
fixed success event and a fixed cohort size. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

/-- With binary codes +1 and -1, a negative phase expectation predicts token 1.
Strict recovery also requires a nonzero expectation. -/
noncomputable def negativeBit (x : ℝ) : ZMod 2 := if x < 0 then 1 else 0

private theorem negativeBit_mul {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) :
    negativeBit (a * b) = negativeBit a + negativeBit b := by
  rcases lt_or_gt_of_ne ha with ha | ha <;> rcases lt_or_gt_of_ne hb with hb | hb
  · simp [negativeBit, ha, hb, (mul_pos_of_neg_of_neg ha hb).not_gt]
    decide
  · simp [negativeBit, ha, hb.not_gt, mul_neg_of_neg_of_pos ha hb]
  · simp [negativeBit, ha.not_gt, hb, mul_neg_of_pos_of_neg ha hb]
  · simp [negativeBit, ha.not_gt, hb.not_gt, (mul_pos ha hb).not_gt]

private theorem strict_sign_of_residual {x y : ℝ} (h : |x-y| < |y|) :
    y ≠ 0 ∧ x ≠ 0 ∧ negativeBit x = negativeBit y := by
  rcases lt_trichotomy y 0 with hy | rfl | hy
  · rw [abs_of_neg hy, abs_lt] at h
    have hx : x < 0 := by linarith [h.2]
    simp [negativeBit, hy, hx, hy.ne, hx.ne]
  · simp only [sub_zero, abs_zero] at h
    exact ((abs_nonneg x).not_gt h).elim
  · rw [abs_of_pos hy, abs_lt] at h
    have hx : 0 < x := by linarith [h.1]
    simp [negativeBit, hy.not_gt, hx.not_gt, hy.ne', hx.ne']

/-- For the entire additive binary family, a rank-one candidate need only
match the supplied staircase signs. If the entrywise phase residual is
smaller than the candidate magnitude everywhere, every model sign is strict
and correct. No hidden target appears in either test. -/
theorem staircase_rank_one_checkpoint {s : ℕ} (row col : Fin (s+1) → ZMod 2)
    (u v : Fin (s+1) → ℝ) (score : Fin (s+1) → Fin (s+1) → ℝ)
    (hd : ∀ i, negativeBit (u i * v i) = row i + col i)
    (he : ∀ i : Fin s, negativeBit (u i.castSucc * v i.succ) = row i.castSucc + col i.succ)
    (hr : ∀ i j, |score i j - u i * v j| < |u i * v j|) :
    ∀ i j, score i j ≠ 0 ∧ negativeBit (score i j) = row i + col j := by
  have prod (i j) : negativeBit (u i * v j) = negativeBit (u i) + negativeBit (v j) := by
    have hn := (strict_sign_of_residual (hr i j)).1
    exact negativeBit_mul (fun h => hn (by rw [h, zero_mul])) (fun h => hn (by rw [h, mul_zero]))
  have diagonal : (fun i => negativeBit (u i) + negativeBit (v i)) = fun i => row i + col i :=
    funext fun i => (prod i i).symm.trans (hd i)
  have edge : (fun i : Fin s => negativeBit (u i.castSucc) + negativeBit (v i.succ)) =
      fun i : Fin s => row i.castSucc + col i.succ :=
    funext fun i => (prod _ _).symm.trans (he i)
  intro i j
  obtain ⟨_, hx, hs⟩ := strict_sign_of_residual (hr i j)
  refine ⟨hx, hs.trans ((prod i j).trans ?_)⟩
  rw [← staircaseDecode_additive (fun t => negativeBit (u t)) (fun t => negativeBit (v t)),
    diagonal, edge, staircaseDecode_additive]

/-- A two-by-two checkerboard with a genuine hidden cell and a nonzero
residual. The sign conclusion follows from three observed labels. -/
example :
    let u := fun i : Fin 2 => if i = 0 then (1:ℝ) else -1
    let a := fun i : Fin 2 => if i = 0 then (0:ZMod 2) else 1
    ∀ i j, (3/4:ℝ)*u i*u j ≠ 0 ∧ negativeBit ((3/4:ℝ)*u i*u j) = a i+a j := by
  dsimp only
  apply staircase_rank_one_checkpoint (s := 1) _ _
    (fun i => if i = 0 then 1 else -1) (fun i => if i = 0 then 1 else -1)
  · intro i
    fin_cases i <;> norm_num [negativeBit]
    decide
  · intro i
    fin_cases i
    norm_num [negativeBit]
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num

namespace Argand

/-- Real phase expectation of the full binary softmax at the prediction position. -/
noncomputable def Model.binaryPhase {T H d L : ℕ} (m : Model (ZMod 2) H d L) (r : ℕ)
    (word : Word (ZMod 2) (T+1)) : ℝ :=
  m.probability r word (Fin.last T) 0 - m.probability r word (Fin.last T) 1

/-- The output of the certificate is a strict Argand logit inequality. Width,
layer count, pass count and the optimizer that supplied the weights are arbitrary. -/
theorem binaryPhase_realizes {T H d L : ℕ} (m : Model (ZMod 2) H d L) (r : ℕ)
    (target : Word (ZMod 2) (T+1) → ZMod 2)
    (hs : ∀ w, m.binaryPhase r w ≠ 0 ∧ negativeBit (m.binaryPhase r w) = target w) :
    m.realizes r target := by
  intro w b hb
  obtain ⟨hn, hbit⟩ := hs w
  have htotal : 0 < ∑ a, Real.exp (m.logits r w (Fin.last T) a) :=
    Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le) ⟨0, Finset.mem_univ _, Real.exp_pos _⟩
  have order (a b : ZMod 2) :
      m.probability r w (Fin.last T) a < m.probability r w (Fin.last T) b ↔
      m.logits r w (Fin.last T) a < m.logits r w (Fin.last T) b := by
    simp only [Model.probability, div_lt_div_iff_of_pos_right htotal, Real.exp_lt_exp]
  have either : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  by_cases h : m.binaryPhase r w < 0
  · have ht : target w = 1 := by simpa [negativeBit, h] using hbit.symm
    rw [ht] at hb ⊢
    rw [(either b).resolve_right hb]
    exact (order 0 1).mp (sub_neg.mp h)
  · have ht : target w = 0 := by simpa [negativeBit, h] using hbit.symm
    rw [ht] at hb ⊢
    rw [(either b).resolve_left hb]
    exact (order 1 0).mp (sub_pos.mp (lt_of_le_of_ne (le_of_not_gt h) hn.symm))

/-- A finite checkpoint on an arbitrary cut of the word domain. The only
target values used to test the proposed factors are the staircase labels. -/
theorem realizes_of_rank_one_checkpoint {s T H d L : ℕ}
    (cut : Fin (s+1) × Fin (s+1) ≃ Word (ZMod 2) (T+1))
    (m : Model (ZMod 2) H d L) (r : ℕ) (row col : Fin (s+1) → ZMod 2) (u v : Fin (s+1) → ℝ)
    (hd : ∀ i, negativeBit (u i * v i) = row i + col i)
    (he : ∀ i : Fin s, negativeBit (u i.castSucc * v i.succ) = row i.castSucc + col i.succ)
    (hr : ∀ i j, |m.binaryPhase r (cut (i,j)) - u i * v j| < |u i * v j|) :
    m.realizes r (fun w => row (cut.symm w).1 + col (cut.symm w).2) := by
  apply binaryPhase_realizes
  intro w
  obtain ⟨⟨i,j⟩, rfl⟩ := cut.surjective w
  simpa only [Equiv.symm_apply_apply] using
    staircase_rank_one_checkpoint row col u v (fun i j => m.binaryPhase r (cut (i,j))) hd he hr i j

end Argand

open MeasureTheory
open scoped ENNReal

/-- The same fixed terminal experiment is run on independent seeds. Its
success event may include all hidden predictions and all stochastic choices. -/
theorem iid_checkpoint_all {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hU : MeasurableSet U) (m : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | ∀ i ∈ Finset.range m, z i ∈ U} = (μ U)^m := by
  change Measure.infinitePi _ (Set.pi (Finset.range m : Set ℕ) (fun _ => U)) = _
  rw [Measure.infinitePi_pi _ (fun _ _ => hU)]
  simp

/-- Under a single-draw success probability at most three quarters, an
independent twelve-out-of-twelve confirmation has size below five percent. -/
theorem twelve_checkpoint_null {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hU : MeasurableSet U) (hnull : (μ U).toReal ≤ 3/4) :
    (Measure.infinitePi (fun _ : ℕ => μ) {z | ∀ i ∈ Finset.range 12, z i ∈ U}).toReal < 1/20 := by
  rw [iid_checkpoint_all μ hU, ENNReal.toReal_pow]
  exact (pow_le_pow_left₀ ENNReal.toReal_nonneg hnull 12).trans_lt (by norm_num)

example : (Measure.infinitePi (fun _ : ℕ => Measure.dirac true)
    {z | ∀ i ∈ Finset.range 12, z i ∈ ({false} : Set Bool)}).toReal < 1/20 := by
  apply twelve_checkpoint_null _ (measurableSet_singleton false)
  norm_num

end Transformer.Grokking.FiniteRecovery
