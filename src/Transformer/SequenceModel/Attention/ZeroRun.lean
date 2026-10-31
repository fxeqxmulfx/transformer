module

public import Transformer.SequenceModel.Attention.Stationary

/-!
# Width and depth do not imply learning from every initialization

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, S2.
The actual Schedule-Free AdamW recurrence stays at zero query/base buffers
when initialized there. This follows from the genuine smooth CE gradient,
not an assumed zero update oracle. Every logit remains tied forever.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini ScheduleFree

/-- S2: every actual sampled step keeps both parameter buffers zero at this
initialization. The Adam history may evolve, and decay and averaging remain. -/
theorem zero_trajectory_buffers (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ} {A : Type*} [Fintype A]
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (samples : ℕ → Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (offset : ℕ) (history : AdamW.History (Index cfg)) (n : ℕ) :
    let s := trajectory (predictor cfg eps xeps angle inputs readout) samples target p offset
      ⟨0, 0, history⟩ n
    s.point = 0 ∧ s.base = 0 := by
  induction n with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih =>
    dsimp only at ih
    simp only [trajectory]
    have hg : sampledOracle (predictor cfg eps xeps angle inputs readout) samples target
        (offset + n) 0 = 0 := by
      unfold sampledOracle
      change (gradient (loss (predictor cfg eps xeps angle inputs readout)
        (samples (offset + n)) target) 0).ofLp = 0
      rw [loss_gradient_zero cfg heps hxeps]
      rfl
    simp only [ih.1, hg, AdamW.trainingStep, AdamW.direction, ih.2, Pi.zero_apply,
      zero_div, mul_zero, add_zero, sub_zero]
    exact ⟨rfl, rfl⟩

/-- S2: all evaluated logits remain exactly tied at every finite step, for
every finite architecture size and input table, despite genuine CE training. -/
theorem zero_trajectory_score (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ} {A : Type*} [Fintype A]
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (samples : ℕ → Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (offset : ℕ) (history : AdamW.History (Index cfg))
    (n : ℕ) (a : A) (v : Fin cfg.vocab_size) :
    predictor cfg eps xeps angle inputs readout
      (Rotate90.Learning.evaluated p (trajectory (predictor cfg eps xeps angle inputs readout)
        samples target p offset ⟨0, 0, history⟩ n)) a v = 0 := by
  have hs := zero_trajectory_buffers cfg heps hxeps angle inputs readout samples target p offset history n
  have hw : Rotate90.Learning.evaluated p (trajectory
      (predictor cfg eps xeps angle inputs readout) samples target p offset ⟨0, 0, history⟩ n) = 0 := by
    ext i
    simp [Rotate90.Learning.evaluated, AdamW.evaluation, hs.1, hs.2]
  rw [hw]
  exact score_zero cfg eps xeps angle (readout a) (inputs a) v

example := zero_trajectory_buffers witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (T := 3) (fun _ => true) (fun j => j.val))
  (fun a j : Fin 3 => if a = j then 0 else 1) (fun _ => 2) (fun _ => {0, 1}) id
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 0 ⟨0, 0, 0⟩ 1000000
example := zero_trajectory_score witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (T := 3) (fun _ => true) (fun j => j.val))
  (fun a j : Fin 3 => if a = j then 0 else 1) (fun _ => 2) (fun _ => {0, 1}) id
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 0 ⟨0, 0, 0⟩ 1000000 1 2

end Transformer.SequenceModel.Attention
