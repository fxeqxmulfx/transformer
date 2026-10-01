/-
# The finite-step optimizer law is normalized

arXiv:2506.12543v1, Section 4.3, Theorem 1.
Smooth losses give measurable fresh-noise updates, so the pushforward
used in the weak-approximation statement is a genuine probability law.
-/

import Transformer.BatchSize.Section4_Sampling

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Joint measurability of the state and innovation in the optimizer
update, Section 4.3, equations (2)--(3). This also justifies state-dependent
Markov expectations, rather than only sampling at a fixed state. -/
theorem stochasticStep_measurable {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ) :
    Measurable (fun p : EucSpace d × (Fin d → ℝ) =>
      stochasticStep method η B f σ p.1 p.2) := by
  have hgrad : Continuous (gradient f) :=
    (InnerProductSpace.toDual ℝ (EucSpace d)).symm.continuous.comp
      (hf.continuous_fderiv (by norm_num))
  have hg (k : Fin d) : Measurable (fun p : EucSpace d × (Fin d → ℝ) =>
      sampledGradient B f σ p.1 p.2 k) :=
    (((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp hgrad).measurable.comp
      measurable_fst).add
      ((((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp hσ).measurable.comp
        measurable_fst).div_const _
        |>.mul ((measurable_pi_apply k).comp measurable_snd))
  have hm : Measurable (fun p : EucSpace d × (Fin d → ℝ) =>
      sampledGradient B f σ p.1 p.2) :=
    (WithLp.measurable_toLp 2 (Fin d → ℝ)).comp (Measurable.of_eval hg)
  cases method
  · simpa [stochasticStep, Pi.sub_def, Pi.smul_def] using
      measurable_fst.sub ((measurable_const (a := η)).smul hm)
  · simpa [stochasticStep, Function.comp_def, Pi.sub_def, Pi.smul_def] using
      measurable_fst.sub ((measurable_const (a := η)).smul
        ((WithLp.measurable_toLp 2 (Fin d → ℝ)).comp
          (Measurable.of_eval (fun k => measurable_real_sign.comp (hg k)))))

/-- Nonvacuity of joint update measurability, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

/-- The discrete optimizer is a measurable function of its innovations,
Section 4.3. Smoothness and continuous noise scales make the sampler measurable. -/
theorem discreteEndpoint_measurable {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (x₀ : EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ) (n : ℕ) :
    Measurable (discreteEndpoint method η B f σ x₀ n) := by
  change Measurable (fun z => discreteEndpoint method η B f σ x₀ n z)
  have hs := stochasticStep_measurable method η B f σ hf hσ
  induction n with
  | zero => simp [discreteEndpoint]
  | succ n ih =>
    have htail : Measurable (fun z : Fin (n + 1) → Fin d → ℝ =>
        fun i : Fin n => z i.castSucc) :=
      Measurable.of_eval (fun i => measurable_pi_apply i.castSucc)
    have h := hs.comp ((ih.comp htail).prodMk (measurable_pi_apply (Fin.last n)))
    change Measurable (fun z : Fin (n + 1) → Fin d → ℝ =>
      (List.ofFn z).foldl (stochasticStep method η B f σ) x₀)
    simp_rw [List.ofFn_succ', List.concat_eq_append, List.foldl_concat]
    exact h

/-- Nonvacuity of the sampler's regularity assumptions, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

/-- The n-step law in the main weak-approximation claim has total mass
one, Section 4.3, Theorem 1. -/
theorem discreteLaw_probability {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (x₀ : EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ) (n : ℕ) :
    IsProbabilityMeasure (discreteLaw method η B f σ x₀ n) := by
  constructor
  rw [discreteLaw, Measure.map_apply
    (discreteEndpoint_measurable method η B f σ x₀ hf hσ n) MeasurableSet.univ]
  simp

/-- Nonvacuity of normalization, Section 4.3. -/
example : ContDiff ℝ 1 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) :=
  ⟨contDiff_const, continuous_const⟩

end Transformer.BatchSize
