module

public import Transformer.Grokking.FiniteRecovery.AdamWTrials

/-! One-moment softsign training. The two coefficients correspond to the blend
and the stored moving average in Chen et al., *Symbolic Discovery of Optimization
Algorithms*, Algorithm 2. The continuous direction and epsilon below are the
variant in `experiments/argand/src/engine/softsign.rs`. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.SoftsignMomentum

open Transformer.AMSGradW.Kernite

variable {ι : Type*} [Fintype ι]

/-- A continuous approximation to the sign with the native positive floor. -/
noncomputable def direction (c : ℝ) : ℝ := c / (|c| + 1/1000000)

theorem direction_bound (c : ℝ) : |direction c| ≤ 1 := by
  have hd : 0 < |c| + 1/1000000 := by positivity
  rw [direction, abs_div, abs_of_pos hd, div_le_one hd]
  linarith

theorem continuous_direction : Continuous direction :=
  continuous_id.div (continuous_abs.add_const _) (fun c => (by positivity : 0 < |c| + 1/1000000).ne')

structure State (ι : Type*) where
  weights : ι → ℝ
  moment : ι → ℝ

/-- Both blends read the previous momentum. Clipping spans all real coordinates
of the minibatch gradient; decay acts on every trainable weight. -/
noncomputable def run (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) :
    ℕ → State ι
  | 0 => ⟨initial, 0⟩
  | t+1 =>
    let s := run decay rate grad initial t
    let g := clip (grad t s.weights)
    let c i := (9/10 : ℝ)*s.moment i + (1/10 : ℝ)*g i
    ⟨fun i => (1-rate t*decay)*s.weights i-rate t*direction (c i),
     fun i => (99/100 : ℝ)*s.moment i+(1/100 : ℝ)*g i⟩

variable (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ)

/-- Positive decoupled decay bounds every native weight for any minibatch sequence
and any WSD budget, even with the positive indefinitely continued WSD floor. -/
theorem weight_bound (initial : ι → ℝ) (hdc : 0 < decay) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1) (i : ι) (n : ℕ) :
    |(run decay rate grad initial n).weights i| ≤ max ‖initial‖ decay⁻¹ := by
  let B := max ‖initial‖ decay⁻¹
  have hB : 1 ≤ decay*B := (mul_inv_cancel₀ hdc.ne').ge.trans
    (mul_le_mul_of_nonneg_left (le_max_right _ _) hdc.le)
  induction n with
  | zero => exact (norm_le_pi_norm initial i).trans (le_max_left _ _)
  | succ t ih =>
    let s := run decay rate grad initial t
    let c := (9/10 : ℝ)*s.moment i + 1/10 * clip (grad t s.weights) i
    have hd : 0 ≤ 1-rate t*decay := by linarith [(hr t).2]
    change |(1-rate t*decay)*s.weights i - rate t*direction c| ≤ B
    calc
      _ ≤ |(1-rate t*decay)*s.weights i| + |rate t*direction c| := abs_sub _ _
      _ = (1-rate t*decay)*|s.weights i| + rate t*|direction c| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hd, abs_of_nonneg (hr t).1]
      _ ≤ (1-rate t*decay)*B + rate t*(decay*B) :=
        add_le_add (mul_le_mul_of_nonneg_left ih hd) (mul_le_mul_of_nonneg_left
          ((direction_bound c).trans hB) (hr t).1)
      _ = B := by ring

/-- Finite ordinary training is continuous in its initial weights. Each of the
two buffers follows the implemented recurrence, including at zero gradient. -/
theorem continuous_run (hg : ∀ t, Continuous (grad t)) (n : ℕ) :
    Continuous (fun w => (run decay rate grad w n).weights) ∧ Continuous (fun w => (run decay rate grad w n).moment) := by
  induction n with
  | zero => exact ⟨continuous_id, continuous_const⟩
  | succ t ih =>
    let s w := run decay rate grad w t
    let g w := clip (grad t (s w).weights)
    have hc : Continuous g := AdamW.continuous_clip.comp ((hg t).comp ih.1)
    have hm (i : ι) : Continuous (fun w => (s w).moment i) := (continuous_apply i).comp ih.2
    have hg' (i : ι) : Continuous (fun w => g w i) := (continuous_apply i).comp hc
    constructor
    · apply continuous_pi
      intro i
      exact (((continuous_apply i).comp ih.1).const_mul _).sub
        ((continuous_direction.comp (((hm i).const_mul _).add ((hg' i).const_mul _))).const_mul _)
    · exact continuous_pi fun i => ((hm i).const_mul _).add ((hg' i).const_mul _)

example (i : Fin 3) (t : ℕ) :
    |(run (1/10) (fun _ => 1/1000) (fun _ w => w) (fun _ : Fin 3 => 2) t).weights i| ≤
      max ‖(fun _ : Fin 3 => (2 : ℝ))‖ (1/10 : ℝ)⁻¹ :=
  weight_bound _ _ _ _ (by norm_num) (fun _ => by norm_num) i t

example (t : ℕ) := continuous_run (ι := Fin 3) (1/10) (fun _ => 1/1000) (fun _ w => w)
  (fun _ => continuous_id) t

end Transformer.Grokking.FiniteRecovery.SoftsignMomentum
