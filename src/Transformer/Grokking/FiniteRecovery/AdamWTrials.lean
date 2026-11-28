module

public import Transformer.Grokking.FiniteRecovery.AdamWFloor
public import Transformer.Grokking.FiniteRecovery.VerifiedRestarts
public import Transformer.Grokking.FiniteRecovery.ArgandGaussianRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandBatchDifferentiability

/-! Finite ordinary AdamW trials from native Gaussian initialization. Strict
success of one exact-real trajectory has positive probability at the same
step size and horizon. Certified independent trials therefore generalize
almost surely on every identifiable task with such a witness. Inputs may
range over any finite set of valid prompts inside the token domain. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

variable {ι : Type*} [Fintype ι]

private theorem continuous_clip : Continuous (Transformer.AMSGradW.Kernite.clip (ι := ι)) := by
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : ι => ℝ)
  have hn : Continuous (fun g : ι → ℝ => ‖(WithLp.toLp 2 g : Transformer.AMSGradW.Kernite.Space ι)‖) :=
    C.symm.continuous.norm
  apply continuous_pi
  intro i
  exact (continuous_apply i).div (hn.max continuous_const)
    (fun g => ((zero_lt_one.trans_le (le_max_right _ _)) : (0:ℝ) < max _ 1).ne')

/-- Joint continuity when both the start and the gradient stream depend
continuously on auxiliary data, such as the original initialization. -/
theorem continuous_run_with_data {D : Type*} [TopologicalSpace D] (rate : ℕ → ℝ)
    (grad : ℕ → D → (ι → ℝ) → ι → ℝ) (hg : ∀ t, Continuous (fun p : D × (ι → ℝ) => grad t p.1 p.2))
    (start : D → ι → ℝ) (hs : Continuous start) (n : ℕ) :
    Continuous (fun x => (run rate (fun t => grad t x) (start x) n).weights) ∧
    Continuous (fun x => (run rate (fun t => grad t x) (start x) n).first) ∧
    Continuous (fun x => (run rate (fun t => grad t x) (start x) n).second) := by
  induction n with
  | zero => exact ⟨hs,continuous_const,continuous_const⟩
  | succ t ih =>
    let s x := run rate (fun j => grad j x) (start x) t
    let g x := Transformer.AMSGradW.Kernite.clip (grad t x (s x).weights)
    have hc : Continuous g := continuous_clip.comp ((hg t).comp (continuous_id.prodMk ih.1))
    have hm (i : ι) : Continuous fun x => (run rate (fun j => grad j x) (start x) (t+1)).first i := by
      change Continuous fun x => (9/10:ℝ)*(s x).first i + (1/10:ℝ)*g x i
      exact ((continuous_apply i |>.comp ih.2.1).const_mul _).add
        ((continuous_apply i |>.comp hc).const_mul _)
    have hv (i : ι) : Continuous fun x => (run rate (fun j => grad j x) (start x) (t+1)).second i := by
      change Continuous fun x => (95/100:ℝ)*(s x).second i + (5/100:ℝ)*(g x i)^2
      exact ((continuous_apply i |>.comp ih.2.2).const_mul _).add
        (((continuous_apply i |>.comp hc).pow 2).const_mul _)
    refine ⟨continuous_pi fun i => ?_,continuous_pi hm,continuous_pi hv⟩
    let u x := run rate (fun j => grad j x) (start x) (t+1)
    have hd : Continuous fun x => scale t ((u x).second i) :=
      (((Real.continuous_sqrt.comp (hv i)).div_const _).add_const _)
    have hdir : Continuous fun x => direction t ((u x).first i) ((u x).second i) := by
      apply (hm i).div (hd.const_mul _)
      intro x
      have hb := Transformer.AMSGradW.Kernite.firstBias_lower t
      have hz : 0 < scale t ((u x).second i) := by unfold scale; positivity
      exact (mul_pos (by linarith) hz).ne'
    exact (((continuous_apply i).comp ih.1).const_mul _).sub (hdir.const_mul _)

/-- All three buffers depend continuously on their initial weights through
the actual finite composition of clipped, bias-corrected AdamW steps. -/
theorem continuous_run (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (hg : ∀ t, Continuous (grad t)) (n : ℕ) :
    Continuous (fun x => (run rate grad x n).weights) ∧
    Continuous (fun x => (run rate grad x n).first) ∧
    Continuous (fun x => (run rate grad x n).second) :=
  continuous_run_with_data rate (fun t _ => grad t) (fun t => (hg t).comp continuous_snd) id continuous_id n

end Transformer.Grokking.FiniteRecovery.AdamW

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set Filter
open scoped ENNReal NNReal Topology

variable {A X : Type*} [Fintype A] [Finite X] {H d L T : ℕ}

/-- Strict correctness on any finite corpus is open in the full weight
space, including every trainable gate and possible competing token. -/
theorem strict_prompt_weights_open (r : ℕ)
    (prompt : X → Word A (T+1)) (f : X → A) :
    IsOpen {w : Weights A H d L | StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) f} := by
  simp only [StrictOn, ofPred_forall]
  apply isOpen_iInter_of_finite
  intro x
  apply isOpen_iInter_of_finite
  intro a
  by_cases he : a = f x
  · simp [he]
  · simp only [he, ne_eq, not_false_eq_true, iInter_true]
    exact isOpen_lt (continuous_logits_fromWeights r (prompt x) _ a)
      (continuous_logits_fromWeights r (prompt x) _ (f x))

/-- A nonempty strict basin is detected without continuity of argmax:
each of finitely many strict inequalities is open in the raw draw. -/
theorem adamW_strict_open (r K : ℕ) (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) :
    IsOpen {z : ComplexParameter A H d L → ℂ | StrictOn
      (fun x => (Model.fromWeights (adamWRun r batch rate z K).weights).logits r (prompt x) (Fin.last T)) f} :=
  (strict_prompt_weights_open r prompt f).preimage ((AdamW.continuous_run rate
    (fun t => batchGradient r (batch t)) (fun t => continuous_batchGradient r (batch t)) K).1.comp
      (continuous_sgdInitialWeights true))

/-- The learning rates and horizon stay fixed. The positive basin comes
from a successful trained trajectory in the native draw space. The verifier
receives only the observed answers and can select the first successful
attempt without scoring against held-out labels. -/
theorem adamW_gaussian_certified_restarts (r K : ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (S : Set X)
    (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (variance : ComplexParameter A H d L → ℝ≥0)
    (hv : ∀ i, 0 < variance i)
    (witness : ComplexParameter A H d L → ℂ)
    (hw : StrictOn (fun x => (Model.fromWeights (adamWRun r batch rate witness K).weights).logits r
      (prompt x) (Fin.last T)) f) :
    let scores z x := (Model.fromWeights (adamWRun r batch rate z K).weights).logits r (prompt x) (Fin.last T)
    let ok z := classCertificate S family (fun x => f x) (scores z)
    let μ := independentGaussianDraws variance
    ∃ p > (0 : ℝ≥0∞), (∀ m, Measure.infinitePi (fun _ : ℕ => μ)
      {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) f ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  let μ := independentGaussianDraws variance
  let scores (z : ComplexParameter A H d L → ℂ) x :=
    (Model.fromWeights (adamWRun r batch rate z K).weights).logits r (prompt x) (Fin.last T)
  let U := {z | StrictOn (scores z) f}
  have ho : IsOpen U := adamW_strict_open r K rate batch prompt f
  let := independentGaussianDraws_openPos variance hv
  have hp : 0 < μ U := ho.measure_pos _ ⟨witness, hw⟩
  exact ⟨μ U,hp,certified_restarts μ S family hid f hf scores ho.measurableSet hp⟩

example (g : ℕ → (Fin 3 → ℝ) → Fin 3 → ℝ) (hc : ∀ t, Continuous (g t)) :=
  AdamW.continuous_run (fun _ => 1/1000) g hc 20000

end Transformer.Grokking.FiniteRecovery.Argand
