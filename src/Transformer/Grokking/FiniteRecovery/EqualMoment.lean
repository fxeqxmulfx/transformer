module

public import Transformer.Grokking.FiniteRecovery.AdamWTrials
public import Transformer.Grokking.FiniteRecovery.DecoupledDecay

/-! Adam with a single coefficient for both moments, uncorrected directions,
and decoupled weight decay, as in the equal-moment kernel of Argand. The
scalar variance identity bounds every coordinate without a bias clock. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.EqualMoment

open Transformer.AMSGradW.Kernite (clip)

variable {ι : Type*} [Fintype ι]

def blend (β x g : ℝ) : ℝ := β*x+(1-β)*g

/-- The gap between the second moment and the squared first moment gains
a nonnegative term for every new gradient, even when gradients alternate. -/
theorem variance_step (β m v g : ℝ) :
    blend β v (g^2) - (blend β m g)^2 =
      β*(v-m^2) + β*(1-β)*(m-g)^2 := by unfold blend; ring

noncomputable def direction (ε m v : ℝ) : ℝ := m / (Real.sqrt v + ε)

/-- Ordinary uncorrected Adam with two exponential moving averages.
Both native comparisons share the first decay, rate and epsilon. -/
noncomputable def uncorrectedRun (β γ ε decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) : ℕ → AdamW.State ι
  | 0 => ⟨initial,0,0⟩
  | t+1 =>
    let s := uncorrectedRun β γ ε decay rate grad initial t
    let g := clip (grad t s.weights)
    let m i := blend β (s.first i) (g i)
    let v i := blend γ (s.second i) ((g i)^2)
    ⟨fun i => (1-rate t*decay)*s.weights i-rate t*direction ε (m i) (v i),m,v⟩

/-- Matched decay in both moment buffers. -/
noncomputable abbrev run (β : ℝ) := @uncorrectedRun ι _ β β

variable {β ε : ℝ} (decay : ℝ) (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)

theorem moment_bounds (hβ : β ∈ Set.Icc 0 1) (t : ℕ) (i : ι) :
    let s := run β ε decay rate grad initial t
    (s.first i)^2 ≤ s.second i := by
  induction t with
  | zero => norm_num [run,uncorrectedRun]
  | succ t ih =>
    let s := run β ε decay rate grad initial t
    let g := clip (grad t s.weights) i
    change (blend β (s.first i) g)^2 ≤ blend β (s.second i) (g^2)
    apply sub_nonneg.mp
    rw [variance_step]
    exact add_nonneg (mul_nonneg hβ.1 (sub_nonneg.mpr ih))
      (mul_nonneg (mul_nonneg hβ.1 (sub_nonneg.mpr hβ.2)) (sq_nonneg _))

/-- At every update the uncorrected step is bounded by one, independent of
the architecture, minibatches, dimension and positive choice of epsilon. -/
theorem direction_bound {m v : ℝ} (hε : 0 < ε) (h : m^2 ≤ v) : |direction ε m v| ≤ 1 := by
  have hm : |m| ≤ Real.sqrt v := by
    rw [← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt h
  have hd : 0 < Real.sqrt v + ε := add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε
  rw [direction,abs_div,abs_of_pos hd,div_le_one hd]
  exact hm.trans (le_add_of_nonneg_right hε.le)

theorem weight_bound (hβ : β ∈ Set.Icc 0 1) (hε : 0 < ε) {decay : ℝ} (hd : 0 < decay)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1) (i : ι) (t : ℕ) :
    |(run β ε decay rate grad initial t).weights i| ≤ max ‖initial‖ (1/decay) := by
  let s := run β ε decay rate grad initial
  exact DecoupledDecay.coordinate_bound (fun t => (s t).weights)
    (fun t j => direction ε ((s (t+1)).first j) ((s (t+1)).second j)) rate hd hr
    (fun _ _ => rfl) (fun k j => direction_bound hε (moment_bounds decay rate grad initial hβ (k+1) j)) i t

/-- Ordinary training at a fixed horizon depends continuously on the native
start when each observed minibatch gradient does. -/
theorem continuous_uncorrectedRun (β γ : ℝ) (hε : 0 < ε) (hg : ∀ t, Continuous (grad t)) (t : ℕ) :
    Continuous (fun w => (uncorrectedRun β γ ε decay rate grad w t).weights) ∧
    Continuous (fun w => (uncorrectedRun β γ ε decay rate grad w t).first) ∧
    Continuous (fun w => (uncorrectedRun β γ ε decay rate grad w t).second) := by
  induction t with
  | zero => exact ⟨continuous_id,continuous_const,continuous_const⟩
  | succ t ih =>
    let s w := uncorrectedRun β γ ε decay rate grad w t
    let g w := clip (grad t (s w).weights)
    have hg' : Continuous g := AdamW.continuous_clip.comp ((hg t).comp ih.1)
    have hm (i : ι) : Continuous (fun w => blend β ((s w).first i) (g w i)) :=
      (((continuous_apply i).comp ih.2.1).const_mul β).add (((continuous_apply i).comp hg').const_mul _)
    have hv (i : ι) : Continuous (fun w => blend γ ((s w).second i) ((g w i)^2)) :=
      (((continuous_apply i).comp ih.2.2).const_mul γ).add ((((continuous_apply i).comp hg').pow 2).const_mul _)
    refine ⟨continuous_pi fun i => ?_, continuous_pi hm, continuous_pi hv⟩
    have hd := (hm i).div ((Real.continuous_sqrt.comp (hv i)).add_const ε)
      (fun w => (add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε).ne')
    exact (((continuous_apply i).comp ih.1).const_mul _).sub (hd.const_mul _)

theorem continuous_run (β : ℝ) (hε : 0 < ε) (hg : ∀ t, Continuous (grad t)) (t : ℕ) :
    Continuous (fun w => (run β ε decay rate grad w t).weights) ∧
    Continuous (fun w => (run β ε decay rate grad w t).first) ∧
    Continuous (fun w => (run β ε decay rate grad w t).second) :=
  continuous_uncorrectedRun decay rate grad β β hε hg t

example (i : Fin 3) (t : ℕ) := weight_bound (β := 9/10) (ε := 1/100000000)
  (fun _ => 2/1000) (fun _ w => w) (fun _ => 2) (by norm_num [Set.mem_Icc]) (by norm_num)
  (decay := 1/10) (by norm_num) (fun _ => by norm_num) i t

end Transformer.Grokking.FiniteRecovery.EqualMoment
