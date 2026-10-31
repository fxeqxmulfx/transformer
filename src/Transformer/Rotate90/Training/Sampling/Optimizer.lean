module

public import Transformer.Rotate90.Training.Sampling.Gradient
public import Transformer.Rotate90.Training.Optimizer

/-!
# A source Schedule-Free AdamW update can separate equal coordinates

Source: `review/flat_spectrum.md`, *Sampled updates can break aliases*, equation
(S6), and the source recurrence cited in `ScheduleFree.AdamW.Defs`. The initial
query and base agree and their second moment is zero. A positive first rate
makes the averaging coefficient one; bias correction gives |g| + epsilon.
Opposite gradient signs therefore separate equal coordinates, independently of
weight decay, beta1, and the averaging powers. This is one actual optimizer step,
not a replacement by gradient descent or an assertion about later convergence.

All formulas concern real arithmetic and fresh optimizer history. A separate
external schedule that begins at zero rate does not satisfy the positive-rate
hypothesis at that first update; its later history is not silently reset here.
-/

@[expose] public section

namespace Transformer.Rotate90.Training.Sampling

open Landscape.Neural ScheduleFree

variable {ι : Type*}

/-- The first positive-rate weight gives averaging coefficient one, including
the upstream maximum-rate sentinel and arbitrary real averaging powers. -/
theorem initial_coefficient (p : AdamW.Parameters) (hrate : 0 < AdamW.rate p 0) :
    AdamW.coefficient p 0 (⟨fun _ : ι => 0, 0, -1⟩ : AdamW.History ι) = 1 := by
  have hm : 0 < AdamW.nextMaximum p 0 (⟨fun _ : ι => 0, 0, -1⟩ : AdamW.History ι) :=
    hrate.trans_le (le_max_left _ _)
  have hw : 0 < AdamW.weight p 0 (⟨fun _ : ι => 0, 0, -1⟩ : AdamW.History ι) := by
    exact mul_pos (Real.rpow_pos_of_pos (by norm_num) _) (Real.rpow_pos_of_pos hm _)
  simpa only [AdamW.coefficient, zero_add] using div_self (ne_of_gt hw)

/-- Zero initial variance cancels the first bias correction exactly. The
inequality on beta2 includes all source-valid values 0 <= beta2 < 1. -/
theorem initial_denominator (p : AdamW.Parameters) (hbeta : p.adam.betaTwo < 1)
    (g : ι → ℝ) (i : ι) :
    AdamW.denominator p 0 (⟨fun _ => 0, 0, -1⟩ : AdamW.History ι) g i =
      |g i| + p.adam.epsilon := by
  have hb : 1 - p.adam.betaTwo ≠ 0 := ne_of_gt (sub_pos.mpr hbeta)
  simp only [AdamW.denominator, AdamW.nextSecond, AdamW.correction,
    pow_one, mul_zero, zero_add, mul_div_cancel_left₀ _ hb,
    Real.sqrt_sq_eq_abs]

/-- At initialization, the source query and base both take the normalized
gradient-plus-decay step. No gradient is evaluated at a different buffer. -/
theorem initial_step_point (p : AdamW.Parameters) (hrate : 0 < AdamW.rate p 0)
    (hbeta : p.adam.betaTwo < 1) (E g : ι → ℝ) (i : ι) :
    (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).point i =
      E i - AdamW.rate p 0 * (g i / (|g i| + p.adam.epsilon) + p.adam.decay * E i) := by
  simp only [AdamW.trainingStep, initial_coefficient p hrate, sub_self, mul_zero,
    add_zero, zero_sub, mul_neg, mul_one, neg_mul]
  rw [AdamW.direction, initial_denominator p hbeta]
  ring

/-- The base equals the query after this first step; hence averaging cannot
restore the just-broken coordinate equality. -/
theorem initial_step_base (p : AdamW.Parameters) (hrate : 0 < AdamW.rate p 0)
    (hbeta : p.adam.betaTwo < 1) (E g : ι → ℝ) :
    (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).base =
      (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).point := by
  funext i
  rw [initial_step_point p hrate hbeta]
  change E i - AdamW.rate p 0 * AdamW.direction p 0 _ E g i = _
  rw [AdamW.direction, initial_denominator p hbeta]

/-- Evaluation equals that same first-step embedding. The algebra also holds
at beta1=0; the actual upstream eval operation uses nonzero beta1. -/
theorem initial_step_evaluation (p : AdamW.Parameters) (hrate : 0 < AdamW.rate p 0)
    (hbeta : p.adam.betaTwo < 1) (E g : ι → ℝ) :
    AdamW.evaluation p (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g) =
      (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).point := by
  funext i
  unfold AdamW.evaluation
  rw [initial_step_base p hrate hbeta]
  ring

/-- **One source update separates initially equal coordinates.** The target
coordinate's negative gradient moves it strictly above a positive-gradient
competitor. Equal weight decay cancels; positive epsilon controls both divisors. -/
theorem initial_step_separates (p : AdamW.Parameters) (hrate : 0 < AdamW.rate p 0)
    (hbeta : p.adam.betaTwo < 1) (heps : 0 < p.adam.epsilon)
    (E g : ι → ℝ) (i j : ι) (hE : E i = E j) (hi : g i < 0) (hj : 0 < g j) :
    (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).point j <
      (AdamW.trainingStep p 0 ⟨E, E, ⟨fun _ => 0, 0, -1⟩⟩ g).point i := by
  rw [initial_step_point p hrate hbeta, initial_step_point p hrate hbeta, hE]
  have hdi : g i / (|g i| + p.adam.epsilon) < 0 :=
    div_neg_of_neg_of_pos hi (add_pos_of_nonneg_of_pos (abs_nonneg _) heps)
  have hdj : 0 < g j / (|g j| + p.adam.epsilon) :=
    div_pos hj (add_pos_of_nonneg_of_pos (abs_nonneg _) heps)
  nlinarith

variable {V K H : Type*} [Fintype V] [Fintype K] [Fintype H]

/-- The embedding component of a fresh optimizer step on one supervised row.
The branch weights may be updated simultaneously: this component uses their
old values, exactly as a synchronous joint gradient step does. -/
noncomputable def rowStep (params : AdamW.Parameters) (eps : ℝ)
    (blocks : List (Power.BlockWeights K H)) (a b target : V)
    (E : Table V (Channels K)) : AdamW.TrainingState (V × Channels K) :=
  AdamW.trainingStep params 0 ⟨E.ofLp, E.ofLp, ⟨fun _ => 0, 0, -1⟩⟩
    (gradient (rowLoss eps blocks a b target) E).ofLp

/-- **The actual singleton update separates the two formerly identical rows.**
The theorem is local and does not assert a correct classifier after one step. -/
theorem rowStep_separates [Nontrivial V] (params : AdamW.Parameters)
    (hrate : 0 < AdamW.rate params 0) (hbeta : params.adam.betaTwo < 1)
    (hopt : 0 < params.adam.epsilon) {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target other : V) (hne : target ≠ other) (i : Channels K)
    (hh : 0 < hidden eps blocks v v i) :
    (rowStep params eps blocks a b target (constantTable v)).point (other, i) <
      (rowStep params eps blocks a b target (constantTable v)).point (target, i) := by
  obtain ⟨ht, ho⟩ := row_gradient_signs heps blocks v a b target other hne i hh
  exact initial_step_separates params hrate hbeta hopt _ _ (target, i) (other, i) rfl ht ho

/-- Legal parameters used by the nonvacuity witnesses: the tested beta pair,
positive decay and epsilons, and a positive first rate without warmup. -/
noncomputable def witnessParameters : AdamW.Parameters :=
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩

example := rowStep_separates (V := Fin 3) (K := Fin 8) (H := Fin 43) witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (by norm_num [witnessParameters])
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  1 1 1 2 (by decide) (0, 1) (hidden_ones_pos (by norm_num) 1 0)
example := initial_coefficient (ι := Fin 2) witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
example := initial_denominator witnessParameters (by norm_num [witnessParameters])
  (fun i : Fin 2 => if i = 0 then -2 else 1) 0
example := initial_step_point witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (fun _ : Fin 2 => 1) (fun i => if i = 0 then -2 else 1) 0
example := initial_step_base witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (fun _ : Fin 2 => 1) (fun i => if i = 0 then -2 else 1)
example := initial_step_evaluation witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (fun _ : Fin 2 => 1) (fun i => if i = 0 then -2 else 1)
example := initial_step_separates witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (by norm_num [witnessParameters])
  (fun _ : Fin 2 => 1) (fun i => if i = 0 then -2 else 1) 0 1 rfl (by norm_num) (by norm_num)

end Transformer.Rotate90.Training.Sampling
