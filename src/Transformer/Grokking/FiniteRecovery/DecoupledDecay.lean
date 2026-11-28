module

public import Transformer.AMSGradW.Kernite
public import Transformer.Grokking.AdamW.MomentLimits

/-! A uniform coordinate bound for decoupled weight decay with any bounded
optimizer direction. It is independent of clipping, data order and model. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.DecoupledDecay

open Filter
open scoped Topology

/-- A scalar exponential moving average inherits the limit of its inputs. -/
theorem moving_average_limit {q l : ℝ} {x z : ℕ → ℝ}
    (hq : 0 ≤ q) (hq' : q < 1) (hz : Tendsto z atTop (𝓝 l))
    (h : ∀ t, x (t+1) = q*x t + (1-q)*z t) : Tendsto x atTop (𝓝 l) := by
  have he (t : ℕ) : x t = Grokking.AdamW.momentAt q z (x 0) t := by
    induction t with
    | zero => rfl
    | succ t ih => rw [h, ih]; rfl
  exact (Grokking.AdamW.moment_tendsto_of_input_tendsto q (x 0) l z hq hq' hz).congr
    (fun t => (he t).symm)

/-- On a positive constant tail the weights of any decoupled update converge
to minus the limiting direction divided by decay. All earlier rates may vary. -/
theorem tendsto_of_direction_limit {I : Type*} (w d : ℕ → I → ℝ) (D : I → ℝ)
    {decay η : ℝ} (hd : 0 < decay) (hη : 0 < η) (hc : η*decay ≤ 1)
    (rate : ℕ → ℝ) (hr : ∀ᶠ t in atTop, rate t = η)
    (hs : ∀ t i, w (t+1) i = (1-rate t*decay)*w t i-rate t*d t i)
    (hD : Tendsto d atTop (𝓝 D)) : Tendsto w atTop (𝓝 fun i => -D i / decay) := by
  obtain ⟨N,hN⟩ := eventually_atTop.mp hr
  apply tendsto_pi_nhds.mpr
  intro i
  apply (tendsto_add_atTop_iff_nat N).mp
  apply moving_average_limit (q := 1-η*decay) (sub_nonneg.mpr hc) (by nlinarith [mul_pos hη hd])
    ((tendsto_add_atTop_iff_nat N).2 ((tendsto_pi_nhds.mp hD i).neg.div_const decay))
  intro t
  rw [show t+1+N = (t+N)+1 by omega,hs,hN (t+N) (Nat.le_add_left _ _)]
  field_simp [hd.ne']
  ring

example := tendsto_of_direction_limit (fun _ _ : ℕ => (0:ℝ)) (fun _ _ => 0) (fun _ => 0)
  (decay := 1) (η := 1/2) (by norm_num) (by norm_num) (by norm_num) (fun _ => 1/2)
  (Eventually.of_forall fun _ => rfl) (by intros; norm_num) tendsto_const_nhds

variable {ι : Type*} [Fintype ι]

/-- Shrinkage by 1 − ηλ keeps the box of radius max(‖w₀‖, D/λ) invariant
for arbitrary sequences of directions bounded by D and rates ηλ in [0,1]. -/
theorem coordinate_bound_of_abs_le (w d : ℕ → ι → ℝ) (rate : ℕ → ℝ) {decay D : ℝ} (hdc : 0 < decay)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (hstep : ∀ t i, |w (t+1) i| ≤ |(1-rate t*decay)*w t i-rate t*d t i|)
    (hd : ∀ t i, |d t i| ≤ D) (i : ι) (n : ℕ) :
    |w n i| ≤ max ‖w 0‖ (D/decay) := by
  let B := max ‖w 0‖ (D/decay)
  have hB : D ≤ decay*B := by
    calc
      D = decay*(D/decay) := by field_simp
      _ ≤ _ := mul_le_mul_of_nonneg_left (le_max_right _ _) hdc.le
  induction n with
  | zero => exact (norm_le_pi_norm (w 0) i).trans (le_max_left _ _)
  | succ t ih =>
    have hpos : 0 ≤ 1-rate t*decay := sub_nonneg.mpr (hr t).2
    apply (hstep t i).trans
    change |(1-rate t*decay)*w t i-rate t*d t i| ≤ B
    calc
      _ ≤ |(1-rate t*decay)*w t i| + |rate t*d t i| := abs_sub _ _
      _ = (1-rate t*decay)*|w t i| + rate t*|d t i| := by
        rw [abs_mul,abs_mul,abs_of_nonneg hpos,abs_of_nonneg (hr t).1]
      _ ≤ (1-rate t*decay)*B + rate t*(decay*B) := add_le_add
        (mul_le_mul_of_nonneg_left ih hpos) (mul_le_mul_of_nonneg_left ((hd t i).trans hB) (hr t).1)
      _ = B := by ring

/-- The ordinary decoupled update satisfies the same bound with equality
before taking coordinate magnitudes. -/
theorem coordinate_bound (w d : ℕ → ι → ℝ) (rate : ℕ → ℝ) {decay D : ℝ} (hdc : 0 < decay)
    (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (hstep : ∀ t i, w (t+1) i = (1-rate t*decay)*w t i-rate t*d t i)
    (hd : ∀ t i, |d t i| ≤ D) (i : ι) (n : ℕ) :
    |w n i| ≤ max ‖w 0‖ (D/decay) :=
  coordinate_bound_of_abs_le w d rate hdc hr (fun t i => (congrArg abs (hstep t i)).le) hd i n

example (n : ℕ) (i : Fin 2) :=
  coordinate_bound (fun _ _ => (0:ℝ)) (fun _ _ => 0) (fun _ => 1/100)
    (ι := Fin 2) (decay := 1/10) (D := 1) (by norm_num) (fun _ => by norm_num)
    (fun _ _ => by norm_num) (fun _ _ => by norm_num) i n

end Transformer.Grokking.FiniteRecovery.DecoupledDecay
