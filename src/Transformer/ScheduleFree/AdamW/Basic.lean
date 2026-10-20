module

public import Transformer.ScheduleFree.AdamW.Defs

/-!
# Equivalence of the revised AdamW representations

Source: facebookresearch/schedule_free, commit `59ed9b662d5ec5799aded7390b2482758423653c`,
`adamw_schedulefree.py`, scalar/foreach `step` and the `train`/`eval` lerps;
`adamw_schedulefree_reference.py`, explicit `z`, `x`, `y` updates.
The identities concern exact real arithmetic. The step identity allows beta1=0;
recovering evaluation parameters from training buffers requires beta1≠0, as
does the upstream `eval` implementation. Equality of entire oracle-driven
trajectories proves that both representations query exactly the same points.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- Upstream linear warmup is in `[0,1]`, including no warmup and the first step. -/
theorem warmup_bounds (warmup t : ℕ) :
    0 ≤ warmupFactor warmup t ∧ warmupFactor warmup t ≤ 1 := by
  unfold warmupFactor
  split_ifs with hw
  · norm_num
  · have hpos : (0 : ℝ) < warmup := by exact_mod_cast Nat.pos_of_ne_zero hw
    constructor
    · exact le_min zero_le_one (div_nonneg (by positivity) hpos.le)
    · exact min_le_left _ _

/-- The manuscript's `min` convention agrees with the code's `k < warmup_steps`. -/
theorem warmup_eq_code (warmup t : ℕ) :
    warmupFactor warmup t =
      if t < warmup then ((t : ℝ) + 1) / warmup else 1 := by
  by_cases hw : warmup = 0
  · simp [warmupFactor, hw]
  · have hpos : (0 : ℝ) < warmup := by exact_mod_cast Nat.pos_of_ne_zero hw
    simp only [warmupFactor, hw, ↓reduceIte]
    split_ifs with ht
    · apply min_eq_right
      apply (div_le_one hpos).mpr
      exact_mod_cast Nat.succ_le_of_lt ht
    · apply min_eq_left
      apply (one_le_div hpos).mpr
      have hle : (warmup : ℝ) ≤ t := by exact_mod_cast Nat.le_of_not_gt ht
      linarith

/-- Version 1.3's warmup-only rate is nonnegative for a nonnegative base rate. -/
theorem rate_nonneg (p : Parameters) (t : ℕ) (h : 0 ≤ p.adam.learningRate) :
    0 ≤ rate p t := mul_nonneg h (warmup_bounds _ _).1

example (t : ℕ) : 0 ≤ rate ⟨⟨1, 0, 10, 9/10, 999/1000, 1/10⟩, 0, 2⟩ t :=
  rate_nonneg _ _ zero_le_one

/-- The `x=z` initialization queries the supplied parameter, for every beta1. -/
theorem query_initial (p : Parameters) (point : ι → ℝ) :
    query p (initial point) = point := by
  funext i
  dsimp [query, initial]
  ring

/-- `train` maps the equal-iterate initialization to the code's initial buffers. -/
theorem pack_initial (p : Parameters) (point : ι → ℝ) :
    pack p (initial point) = ⟨point, point, ⟨fun _ => 0, 0, -1⟩⟩ := by
  unfold pack
  rw [query_initial]
  rfl

/-- Upstream `eval` exactly recovers `x` from `y=(1-beta1)z+beta1*x`. -/
theorem evaluation_pack (p : Parameters) (s : State ι) (hβ : p.adam.betaOne ≠ 0) :
    evaluation p (pack p s) = s.evaluation := by
  funext i
  dsimp [evaluation, pack, query]
  field_simp
  ring

example (x z : ℝ) :
    evaluation ⟨⟨1, 0, 0, 9/10, 999/1000, 1/10⟩, 0, 2⟩
      (pack ⟨⟨1, 0, 0, 9/10, 999/1000, 1/10⟩, 0, 2⟩
        ⟨fun _ : Fin 1 => x, fun _ => z, ⟨fun _ => 0, 0, -1⟩⟩) = (fun _ => x) :=
  evaluation_pack _ _ (by norm_num)

/-- The two-buffer lerp/update equals stepping `z`, averaging `x`, and forming `y`.
The old query is used for both gradient normalization and decay in each program. -/
theorem trainingStep_pack (p : Parameters) (t : ℕ) (s : State ι) (g : ι → ℝ) :
    trainingStep p t (pack p s) g = pack p (step p t s g) := by
  simp only [trainingStep, pack, step]
  congr 1
  funext i
  dsimp only [query]
  ring

/-- Independent executions with a query-dependent oracle agree at every step. -/
theorem trainingRun_pack (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) (t : ℕ) :
    trainingRun p oracle point t = pack p (run p oracle point t) := by
  induction t with
  | zero => exact (pack_initial p point).symm
  | succ t ih =>
    change trainingStep p t (trainingRun p oracle point t)
        (oracle t (trainingRun p oracle point t).point) =
      pack p (step p t (run p oracle point t) (oracle t (query p (run p oracle point t))))
    rw [ih]
    exact trainingStep_pack p t _ _

/-- `eval` recovers the entire reference evaluation trajectory, not just one step. -/
theorem evaluation_trainingRun (p : Parameters)
    (oracle : ℕ → (ι → ℝ) → (ι → ℝ)) (point : ι → ℝ)
    (hβ : p.adam.betaOne ≠ 0) (t : ℕ) :
    evaluation p (trainingRun p oracle point t) = (run p oracle point t).evaluation := by
  rw [trainingRun_pack]
  exact evaluation_pack p _ hβ

example (t : ℕ) :
    evaluation ⟨⟨1, 0, 0, 9/10, 999/1000, 1/10⟩, 0, 2⟩
        (trainingRun ⟨⟨1, 0, 0, 9/10, 999/1000, 1/10⟩, 0, 2⟩
          (fun _ y => y) (fun _ : Fin 1 => 2) t) =
      (run ⟨⟨1, 0, 0, 9/10, 999/1000, 1/10⟩, 0, 2⟩
        (fun _ y => y) (fun _ : Fin 1 => 2) t).evaluation :=
  evaluation_trainingRun _ _ _ (by norm_num) _

/-- The group total implements the code's polynomial/rate weighting exactly. -/
theorem weighted_balance (p : Parameters) (t : ℕ) (s : State ι) (g : ι → ℝ)
    (h : s.history.total + weight p t s.history ≠ 0) (i : ι) :
    (step p t s g).history.total * (step p t s g).evaluation i =
      s.history.total * s.evaluation i + weight p t s.history * (step p t s g).base i := by
  change (s.history.total + weight p t s.history) *
      ((1 - coefficient p t s.history) * s.evaluation i +
        coefficient p t s.history * (step p t s g).base i) = _
  unfold coefficient
  field_simp
  ring

example :
    let p : Parameters := ⟨⟨1, 0, 0, 9/10, 0, 1⟩, 0, 2⟩
    let s := initial (fun _ : Fin 1 => 2)
    (step p 0 s (fun _ => 3)).history.total * (step p 0 s (fun _ => 3)).evaluation 0 =
      s.history.total * s.evaluation 0 + weight p 0 s.history * (step p 0 s (fun _ => 3)).base 0 :=
  weighted_balance _ _ _ _
    (by norm_num [initial, weight, nextMaximum, rate, warmupFactor]) _

/-- The default rate power is a square, while the code retains a running maximum. -/
theorem default_weight (p : AdamParameters) (t : ℕ) (h : History ι) :
    weight ⟨p, 0, 2⟩ t h = nextMaximum ⟨p, 0, 2⟩ t h ^ 2 := by
  simp only [weight, Real.rpow_zero, one_mul, Real.rpow_two]

end Transformer.ScheduleFree.AdamW
