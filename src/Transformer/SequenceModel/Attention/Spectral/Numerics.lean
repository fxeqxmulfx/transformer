module

public import Transformer.SequenceModel.Attention.Execution.Certificate
public import Transformer.SequenceModel.Attention.Spectral.Scores

/-!
# Positive execution tolerance for a positive spectral source margin

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J6. The existing operation-error recurrence is a continuous expression that
vanishes at zero error. Every fixed finite model with a positive margin
therefore has a strictly positive local arithmetic/table error allowance.
This does not assert that a chosen hardware format meets that allowance.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Stability Execution
open scoped BigOperators

/-- The existing class-specific execution radii fit below a prescribed margin. -/
def ExecutionBelow (cfg : Config) (θ : ModelParams cfg) (eps xeps δ τ γ : ℝ) : Prop :=
  ∀ u v, logitRadius cfg eps (θ.embedding u) (modelBudget cfg θ eps xeps δ τ cfg.n_layers) δ +
    logitRadius cfg eps (θ.embedding v) (modelBudget cfg θ eps xeps δ τ cfg.n_layers) δ < γ

/-- The independent fixed-point counting-model error fits below the same margin. -/
def CountingBelow (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (s : ℕ) (γ : ℝ) : Prop :=
  ∀ u v, (rmsLip eps * errorBudget cfg θ eps xeps s cfg.n_layers) *
    (‖θ.embedding u‖ + ‖θ.embedding v‖) < γ

/-- J6: the derived stack budget is nonnegative at nonnegative local errors. -/
theorem modelBudget_nonneg (cfg : Config) (θ : ModelParams cfg) (eps xeps δ τ : ℝ)
    (hx : 0 ≤ xeps) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ) (n : ℕ) :
    0 ≤ modelBudget cfg θ eps xeps δ τ n := by
  induction n with
  | zero => dsimp [modelBudget, rmsLip]; positivity
  | succ n ih =>
    rw [modelBudget]
    split_ifs with hn
    · have hL : 0 ≤ blockLip cfg (θ.blocks ⟨n, hn⟩) eps xeps := by
        unfold blockLip ffnGain ffnLip attentionGain averageLip directionGain componentLip valueBound rmsLip
        positivity
      have hR : 0 ≤ blockRadius cfg (θ.blocks ⟨n, hn⟩) eps xeps δ τ := by
        dsimp [blockRadius, attentionBudget, ffnBudget, ffnGain, ffnLip, valueBound, rmsLip]
        positivity
      exact add_nonneg (mul_nonneg hL ih) hR
    · exact ih

/-- J6: exact local operations and exact tables incur zero derived stack error. -/
theorem modelBudget_zero (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (n : ℕ) :
    modelBudget cfg θ eps xeps 0 0 n = 0 := by
  induction n with
  | zero => simp [modelBudget]
  | succ n ih => simp [modelBudget, ih, blockRadius, attentionBudget, ffnBudget]

/-- J6: at every finite depth the execution recurrence is continuous in
a common arithmetic/table tolerance, with weights and architecture fixed. -/
theorem modelBudget_continuous (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (n : ℕ) :
    Continuous (fun η : ℝ => modelBudget cfg θ eps xeps η η n) := by
  induction n with
  | zero => simp only [modelBudget]; fun_prop
  | succ n ih =>
    by_cases hn : n < cfg.n_layers
    · simp only [modelBudget, dite_eq_left hn]
      have hr : Continuous (fun η : ℝ => blockRadius cfg (θ.blocks ⟨n, hn⟩) eps xeps η η) := by
        dsimp only [blockRadius, attentionBudget, ffnBudget]
        fun_prop
      exact (continuous_const.mul ih).add hr
    · simpa only [modelBudget, dite_eq_right hn] using ih

/-- A finite aggregate dominates every pair of nonnegative class radii. -/
def totalRadius (cfg : Config) (θ : ModelParams cfg) (eps xeps η : ℝ) : ℝ :=
  ∑ v, 2 * logitRadius cfg eps (θ.embedding v)
    (modelBudget cfg θ eps xeps η η cfg.n_layers) η

/-- J6: a positive source margin admits a genuinely positive operation
tolerance, uniformly across all class pairs and input words at this model. -/
theorem exists_execution_tolerance (cfg : Config) (θ : ModelParams cfg) (eps xeps γ : ℝ)
    (hx : 0 ≤ xeps) (hγ : 0 < γ) :
    ∃ η > 0, ExecutionBelow cfg θ eps xeps η η γ := by
  have hb := modelBudget_continuous cfg θ eps xeps cfg.n_layers
  have hc : Continuous (totalRadius cfg θ eps xeps) := by
    unfold totalRadius logitRadius
    fun_prop
  have hz : totalRadius cfg θ eps xeps 0 = 0 := by
    simp [totalRadius, modelBudget_zero, logitRadius]
  obtain ⟨r, hr, he⟩ := Metric.continuousAt_iff.mp hc.continuousAt γ hγ
  let η := r / 2
  have hη : 0 < η := by dsimp [η]; positivity
  have hdist : dist η (0 : ℝ) < r := by
    rw [Real.dist_eq, sub_zero, abs_of_pos hη]
    dsimp [η]
    linarith
  have ht : totalRadius cfg θ eps xeps η < γ := by
    have h := he hdist
    rw [Real.dist_eq, hz, sub_zero] at h
    exact (le_abs_self _).trans_lt h
  have hn (v : Fin cfg.vocab_size) : 0 ≤ logitRadius cfg eps (θ.embedding v)
      (modelBudget cfg θ eps xeps η η cfg.n_layers) η := by
    have hE := modelBudget_nonneg cfg θ eps xeps η η hx hη.le hη.le cfg.n_layers
    unfold logitRadius rmsLip
    positivity
  refine ⟨η, hη, ?_⟩
  intro u v
  have hu : 2 * logitRadius cfg eps (θ.embedding u)
      (modelBudget cfg θ eps xeps η η cfg.n_layers) η ≤ totalRadius cfg θ eps xeps η :=
    Finset.single_le_sum (fun v _ => mul_nonneg (by norm_num) (hn v)) (Finset.mem_univ u)
  have hv : 2 * logitRadius cfg eps (θ.embedding v)
      (modelBudget cfg θ eps xeps η η cfg.n_layers) η ≤ totalRadius cfg θ eps xeps η :=
    Finset.single_le_sum (fun v _ => mul_nonneg (by norm_num) (hn v)) (Finset.mem_univ v)
  linarith

example := modelBudget_nonneg witnessConfig exampleModel (3 / 4) 1 (1 / 1024) 0
  (by norm_num) (by norm_num) (by norm_num) 2
example := exists_execution_tolerance witnessConfig exampleModel (3 / 4) 1 1 (by norm_num) (by norm_num)

end Transformer.SequenceModel.Attention.Spectral
