module

public import Transformer.AMSGradW.Kernite

/-! A uniform coordinate bound for decoupled weight decay with any bounded
optimizer direction. It is independent of clipping, data order and model. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.DecoupledDecay

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
