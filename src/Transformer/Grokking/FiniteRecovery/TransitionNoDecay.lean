module

public import Transformer.Grokking.FiniteRecovery.TransitionTraining

/-! The same compiled Argand transitions with no decay. The actual scalar
AMSGrad run keeps its history-dependent denominator between one and two.
With a competing token the margin diverges and cross-entropy tends to zero. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.AMSGradW
open scoped Topology

noncomputable section

/-- The deterministic AMSGradW update with zero decay and the same
moment coefficients as the decayed scalar example. -/
def marginNoDecayRun (competitors : ℕ) : ℕ → TrainingState 1 :=
  trainingRun 1 1 0 (9 / 10) (98 / 100) (marginObjective competitors) 0

private theorem marginGradient_bounds (competitors : ℕ) (x : TrainingSpace 1) :
    -1 ≤ gradient (marginObjective competitors) x 0 ∧
      gradient (marginObjective competitors) x 0 ≤ 0 := by
  rw [marginObjective_gradient]
  unfold marginDerivative
  have hpos : 0 < Real.exp (x 0 + 1) + (competitors : ℝ) := by positivity
  constructor
  · apply (le_div_iff₀ hpos).2
    linarith [Real.exp_pos (x 0 + 1)]
  · exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (Nat.cast_nonneg _)) hpos.le

private theorem noDecay_position_succ (competitors step : ℕ) :
    (marginNoDecayRun competitors (step + 1)).position 0 =
      (marginNoDecayRun competitors step).position 0 -
        (marginNoDecayRun competitors (step + 1)).momentum 0 /
          Transformer.AMSGrad.trainingDenominator 1 (marginNoDecayRun competitors (step + 1)) 0 := by
  have hs : marginNoDecayRun competitors (step + 1) =
      trainingStep 1 1 0 (9 / 10) (98 / 100) (marginObjective competitors)
        (marginNoDecayRun competitors step) := rfl
  rw [hs, (trainingStep_coordinates 1 1 0 (9 / 10) (98 / 100) (marginObjective competitors)
    (marginNoDecayRun competitors step) 0).2, mul_zero, sub_zero, one_mul, one_mul]

private theorem noDecay_momentum_succ (competitors step : ℕ) :
    (marginNoDecayRun competitors (step + 1)).momentum 0 =
      (9 / 10 : ℝ) * (marginNoDecayRun competitors step).momentum 0 +
        (1 - 9 / 10) * gradient (marginObjective competitors) (marginNoDecayRun competitors step).position 0 :=
  (trainingStep_coordinates 1 1 0 (9 / 10) (98 / 100) (marginObjective competitors)
    (marginNoDecayRun competitors step) 0).1

private theorem noDecay_state_bounds (competitors step : ℕ) :
    (marginNoDecayRun competitors step).momentum 0 ≤ 0 ∧
      (marginNoDecayRun competitors step).second 0 ≤ 1 ∧
      (marginNoDecayRun competitors step).maximum 0 ≤ 1 := by
  induction step with
  | zero => simp [marginNoDecayRun, trainingRun]
  | succ step ih =>
    let s := marginNoDecayRun competitors step
    let v := marginNoDecayRun competitors (step + 1)
    let g := gradient (marginObjective competitors) s.position 0
    have hg : -1 ≤ g ∧ g ≤ 0 := marginGradient_bounds competitors s.position
    have hg2 : g ^ 2 ≤ 1 := by
      nlinarith [mul_nonneg (show 0 ≤ 1 + g by linarith [hg.1])
        (show 0 ≤ 1 - g by linarith [hg.2])]
    have hv : v.second 0 ≤ 1 := by
      change (98 / 100 : ℝ) * s.second 0 + (1 - 98 / 100) * g ^ 2 ≤ 1
      linarith [ih.2.1]
    refine ⟨?_, hv, ?_⟩
    · rw [noDecay_momentum_succ]
      linarith [ih.1, hg.2]
    · change max (s.maximum 0) (v.second 0) ≤ 1
      exact max_le ih.2.2 hv

private theorem noDecay_position_monotone (competitors : ℕ) :
    Monotone (fun step => (marginNoDecayRun competitors step).position 0) := by
  apply monotone_nat_of_le_succ
  intro step
  rw [noDecay_position_succ]
  have h := (noDecay_state_bounds competitors (step + 1)).1
  have hd := denominator_lower 1 (marginNoDecayRun competitors (step + 1)) 0
  linarith [div_nonpos_of_nonpos_of_nonneg h (zero_le_one.trans hd)]

/-- A finite limiting margin would keep every gradient uniformly negative.
The bounded AMSGrad denominator would then force a fixed positive increase
per step, a contradiction. -/
theorem marginNoDecay_tendsto_atTop {competitors : ℕ} (hC : 0 < competitors) :
    Tendsto (fun step => (marginNoDecayRun competitors step).position 0 + 1) atTop atTop := by
  let x := fun step => (marginNoDecayRun competitors step).position 0
  have hmon : Monotone x := noDecay_position_monotone competitors
  have hx : Tendsto x atTop atTop := by
    apply hmon.tendsto_atTop_atTop_iff.mpr
    intro B
    by_contra! hB
    let δ := (competitors : ℝ) / (Real.exp (B + 1) + competitors)
    have hδ : 0 < δ :=
      div_pos (Nat.cast_pos.mpr hC) (by positivity)
    have hstep (step : ℕ) : x step + δ / 20 ≤ x (step + 1) := by
      let s := marginNoDecayRun competitors step
      let v := marginNoDecayRun competitors (step + 1)
      let den := Transformer.AMSGrad.trainingDenominator 1 v 0
      have hdpos : 0 < den := one_pos.trans_le (denominator_lower 1 v 0)
      have hdle : den ≤ 2 := by
        have hs := Real.sqrt_le_sqrt (noDecay_state_bounds competitors (step + 1)).2.2
        rw [Real.sqrt_one] at hs
        change 1 + Real.sqrt (v.maximum 0) ≤ 2
        linarith
      have hg : gradient (marginObjective competitors) s.position 0 ≤ -δ := by
        rw [marginObjective_gradient]
        have hden : Real.exp (s.position 0 + 1) + (competitors : ℝ) ≤
            Real.exp (B + 1) + competitors :=
          add_le_add_left (Real.exp_le_exp.mpr (add_le_add_left (hB step).le 1)) _
        have hfrac : δ ≤ (competitors : ℝ) / (Real.exp (s.position 0 + 1) + competitors) :=
          div_le_div_of_nonneg_left (Nat.cast_nonneg _) (by positivity) hden
        rw [marginDerivative, neg_div]
        exact neg_le_neg hfrac
      have hm : v.momentum 0 ≤ -δ / 10 := by
        change (marginNoDecayRun competitors (step + 1)).momentum 0 ≤ -δ / 10
        rw [noDecay_momentum_succ]
        linarith [(noDecay_state_bounds competitors step).1]
      have hquot : δ / 20 ≤ -v.momentum 0 / den :=
        (le_div_iff₀ hdpos).2 (calc
          δ / 20 * den ≤ δ / 20 * 2 := mul_le_mul_of_nonneg_left hdle (by positivity)
          _ ≤ -v.momentum 0 := by linarith)
      change x step + δ / 20 ≤ (marginNoDecayRun competitors (step + 1)).position 0
      rw [noDecay_position_succ]
      rw [neg_div] at hquot
      change x step + δ / 20 ≤ x step - v.momentum 0 / den
      linarith
    have hlin (step : ℕ) : (step : ℝ) * (δ / 20) ≤ x step := by
      induction step with
      | zero => simp [x, marginNoDecayRun, trainingRun]
      | succ step ih =>
        rw [Nat.cast_succ, add_mul, one_mul]
        linarith [hstep step]
    obtain ⟨step, hstep⟩ := exists_nat_gt (B / (δ / 20))
    have hlarge := (div_lt_iff₀ (by positivity : 0 < δ / 20)).mp hstep
    exact (not_lt_of_ge (hlin step) (lt_trans (hB step) hlarge))
  exact tendsto_atTop_mono (fun step => by linarith : ∀ step, x step ≤ x step + 1) hx

/-- After learning the transition table from a covering mask, fitting
the margin of the ordinary Argand model gives zero limiting cross-entropy
on every word and exact greedy forecasts at every step. -/
theorem transitionNoDecay_converges {H d T : ℕ} (hc : 2 ≤ H * d)
    (next : Fin (H * d) → Fin (H * d)) (inputs : Finset (Word (Fin (H * d)) (T + 1)))
    (hcover : ∀ a, ∃ word ∈ inputs, word (Fin.last T) = a) :
    let law := transitionFromData (observe inputs (transitionTask next))
    let models := fun step => transitionModel law ((marginNoDecayRun (H * d - 1) step).position 0 + 1)
    (∀ step, (models step).realizes 1 (transitionTask (T := T) next)) ∧
      ∀ word : Word (Fin (H * d)) (T + 1),
        Tendsto (fun step => (models step).labelLoss 1 word (transitionTask next word)) atTop (𝓝 0) := by
  dsimp only
  rw [transitionFromData_eq_of_covers next inputs hcover]
  have hm (step : ℕ) : 0 < (marginNoDecayRun (H * d - 1) step).position 0 + 1 := by
    have h := noDecay_position_monotone (H * d - 1) (Nat.zero_le step)
    change (0 : ℝ) ≤ (marginNoDecayRun (H * d - 1) step).position 0 at h
    linarith
  refine ⟨fun step => transitionModel_realizes (by omega) next (hm step), ?_⟩
  intro word
  have hC : 0 < H * d - 1 := by omega
  have ht := (marginLoss_tendsto_zero (H * d - 1)).comp (marginNoDecay_tendsto_atTop hC)
  have he (step : ℕ) := transitionModel_labelLoss (show 0 < H * d by omega) next (hm step) word
  simpa only [Function.comp_def, he] using ht

example := marginNoDecay_tendsto_atTop (competitors := 255) (by decide)

example (next : Fin 256 → Fin 256) :=
  transitionNoDecay_converges (H := 4) (d := 64) (T := 31) (by decide) next (transitionInputs (Fin 256) 31)
    (fun a => ⟨fun _ => a, Finset.mem_map.mpr ⟨a, Finset.mem_univ _, rfl⟩, rfl⟩)

example (next : Fin 64 → Fin 64) :=
  transitionNoDecay_converges (H := 4) (d := 16) (T := 31) (by decide) next (transitionInputs (Fin 64) 31)
    (fun a => ⟨fun _ => a, Finset.mem_map.mpr ⟨a, Finset.mem_univ _, rfl⟩, rfl⟩)

-- These are the two constant tasks and copying, after relabeling Bool as Fin 2.
example (T : ℕ) (mode : Option Bool) :=
  transitionNoDecay_converges (H := 1) (d := 2) (T := T) (by decide)
    (fun a => finTwoEquiv.symm (mode.getD (finTwoEquiv a))) (transitionInputs (Fin 2) T)
    (fun a => ⟨fun _ => a, Finset.mem_map.mpr ⟨a, Finset.mem_univ _, rfl⟩, rfl⟩)

end
end Transformer.Grokking.FiniteRecovery.Argand
