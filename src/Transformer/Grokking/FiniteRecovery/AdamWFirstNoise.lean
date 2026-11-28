module

public import Transformer.Grokking.FiniteRecovery.AdamWReachability

/-! AdamW with one Gaussian perturbation of the weights after the first
ordinary update, as implemented in experiments/argand/src/engine/adamw.rs.
The first gradient and both moment buffers are those of the native start.
All later updates use the original rates, decay and bias corrections. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

open MeasureTheory ProbabilityTheory Set Transformer.AMSGradW.Kernite
open scoped NNReal

variable {I : Type*} [Fintype I]

/- The replay expression builds the first two moments from the original
gradient while keeping the already updated and perturbed weights. -/
def afterFirstRate (rate : ℕ → ℝ) (t : ℕ) := if t = 0 then 0 else rate t

def afterFirstGradient (g : ℕ → (I → ℝ) → I → ℝ) (start : I → ℝ) (t : ℕ) (w : I → ℝ) :=
  if t = 0 then g 0 start else g t w

/-- State zero is the ordinary initialization. At update one the weights
receive the noise vector; its coordinates are sampled independently in
the Gaussian specialization below. -/
noncomputable def firstNoiseRun (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (start noise : I → ℝ) : ℕ → State I
  | 0 => run rate g start 0
  | n+1 => run (afterFirstRate rate) (afterFirstGradient g start)
      ((run rate g start 1).weights + noise) (n+1)

theorem firstNoiseRun_one (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ) (start noise : I → ℝ) :
    let s := run rate g start 1
    firstNoiseRun rate g start noise 1 = ⟨s.weights + noise,s.first,s.second⟩ := by
  simp [firstNoiseRun,run,afterFirstRate,afterFirstGradient]
  rfl

/-- On each subsequent update all three buffers obey the ordinary AdamW
recurrence with the current minibatch gradient and the original time index. -/
theorem firstNoiseRun_step (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ) (start noise : I → ℝ) (n : ℕ) :
    let s := firstNoiseRun rate g start noise (n+1)
    let h := clip (g (n+1) s.weights)
    let m i := (9/10:ℝ) * s.first i + (1/10:ℝ) * h i
    let v i := (95/100:ℝ) * s.second i + (5/100:ℝ) * (h i)^2
    firstNoiseRun rate g start noise (n+2) =
      ⟨fun i => (1-rate (n+1)/10) * s.weights i - rate (n+1) * direction (n+1) (m i) (v i),m,v⟩ := rfl

theorem firstNoiseRun_zero_noise (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (start : I → ℝ) (n : ℕ) : firstNoiseRun rate g start 0 n = run rate g start n := by
  cases n with
  | zero => rfl
  | succ n =>
    induction n with
    | zero => simp [firstNoiseRun_one]
    | succ n ih => rw [firstNoiseRun_step,ih]; rfl

omit [Fintype I] in
private theorem continuous_afterFirstGradient (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (start : I → ℝ) (t : ℕ) :
    Continuous (afterFirstGradient g start t) := by
  cases t with
  | zero => exact continuous_const
  | succ t => exact hg (t+1)

theorem continuous_firstNoise_weights (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (start : I → ℝ) (n : ℕ) :
    Continuous (fun noise => (firstNoiseRun rate g start noise (n+1)).weights) :=
  (continuous_run (afterFirstRate rate) _ (continuous_afterFirstGradient g hg start) (n+1)).1.comp
    (continuous_const.add continuous_id)

/-- The unperturbed first gradient can vary with the native initialization
when both the initialization and the noise are sampled on each trial. -/
theorem continuous_firstNoise_joint (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (n : ℕ) :
    Continuous (fun p : (I → ℝ) × (I → ℝ) => (firstNoiseRun rate g p.1 p.2 (n+1)).weights) := by
  let grad (t : ℕ) (p : (I → ℝ) × (I → ℝ)) := afterFirstGradient g p.1 t
  have hc (t) : Continuous fun p : ((I → ℝ) × (I → ℝ)) × (I → ℝ) => grad t p.1 p.2 := by
    cases t with
    | zero => exact (hg 0).comp (continuous_fst.comp continuous_fst)
    | succ t => exact (hg (t+1)).comp continuous_snd
  exact (continuous_run_with_data (afterFirstRate rate) grad hc
    (fun p => (run rate g p.1 1).weights + p.2)
    (((continuous_run rate g hg 1).1.comp continuous_fst).add continuous_snd) (n+1)).1

/-- Even conditional on an arbitrary fixed starting point and a fixed
minibatch stream, varying this single perturbation reaches every terminal
weight vector. The standard WSD schedule satisfies the rate hypotheses. -/
theorem firstNoise_weights_surjective (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (start : I → ℝ) (n : ℕ)
    (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10) :
    Function.Surjective (fun noise => (firstNoiseRun rate g start noise (n+1)).weights) := by
  have hr' (t : ℕ) (ht : t < n+1) : 0 ≤ afterFirstRate rate t ∧ afterFirstRate rate t < 10 := by
    cases t with
    | zero => norm_num [afterFirstRate]
    | succ t => exact hr (t+1) ht
  intro target
  obtain ⟨w,hw⟩ := reachable_coordinates (afterFirstRate rate) (afterFirstGradient g start)
    (continuous_afterFirstGradient g hg start) id id continuous_id (fun _ _ => rfl) (n+1) hr' target
  refine ⟨w - (run rate g start 1).weights,?_⟩
  simpa only [firstNoiseRun,add_sub_cancel,id_eq] using hw

/-- Under independent centered real Gaussian coordinates of any positive
variance, every nonempty open target region has positive endpoint mass. -/
theorem firstNoise_open_probability (rate : ℕ → ℝ) (g : ℕ → (I → ℝ) → I → ℝ)
    (hg : ∀ t, Continuous (g t)) (start : I → ℝ) (n : ℕ)
    (hr : ∀ t < n+1, 0 ≤ rate t ∧ rate t < 10) (v : ℝ≥0) (hv : 0 < v)
    (U : Set (I → ℝ)) (ho : IsOpen U) (hne : U.Nonempty) :
    0 < Measure.pi (fun _ : I => gaussianReal 0 v)
      {z | (firstNoiseRun rate g start z (n+1)).weights ∈ U} := by
  let : (gaussianReal 0 v).IsOpenPosMeasure :=
    (gaussianReal_absolutelyContinuous' 0 hv.ne').isOpenPosMeasure
  exact (ho.preimage (continuous_firstNoise_weights rate g hg start n)).measure_pos _
    (hne.preimage (firstNoise_weights_surjective rate g hg start n hr))

example (w : Fin 2 → ℝ) : Function.Surjective
    (fun noise => (firstNoiseRun (fun _ => 1/1000) (fun _ x i => x i + x 0) w noise 100000).weights) :=
  firstNoise_weights_surjective _ _ (fun _ => continuous_pi fun i =>
    (continuous_apply i).add (continuous_apply 0)) w 99999 (by intro; norm_num)

end Transformer.Grokking.FiniteRecovery.AdamW
