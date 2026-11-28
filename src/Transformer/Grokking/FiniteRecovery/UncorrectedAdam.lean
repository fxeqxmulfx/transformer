module

public import Transformer.Grokking.FiniteRecovery.EqualMomentLimit
public import Transformer.Grokking.FiniteRecovery.EqualMomentConfidence

/-! Uncorrected AdamW with distinct moment decays. Completing a square
gives a coordinate bound whenever the second decay exceeds the square of
the first. The native 0.9/0.95 comparison then needs no bias clock. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.EqualMoment

open Transformer.AMSGradW.Kernite (clip)

/-- Multiplier in the invariant m² ≤ K v for decays a and b. -/
noncomputable def momentRatio (a b : ℝ) : ℝ := b*(1-a)^2 / ((1-b)*(b-a^2))

/-- The next slack is a nonnegative multiple of the previous slack plus
a square. This applies to any gradient history including alternating signs. -/
theorem unequal_variance_step {a b : ℝ} (hb : b < 1) (hab : a^2 < b) (m v g : ℝ) :
    (b-a^2) * (momentRatio a b * blend b v (g^2) - (blend a m g)^2) =
    b*(b-a^2)*(momentRatio a b*v-m^2) + ((b-a^2)*m-a*(1-a)*g)^2 := by
  unfold momentRatio blend
  field_simp [(sub_pos.mpr hb).ne',(sub_pos.mpr hab).ne']
  ring

variable {ι : Type*} [Fintype ι]

theorem unequal_moment_bounds {a b : ℝ} (hb : b < 1) (hab : a^2 < b) (ε decay : ℝ)
    (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) (t : ℕ) (i : ι) :
    let s := uncorrectedRun a b ε decay rate grad initial t
    (s.first i)^2 ≤ momentRatio a b * s.second i := by
  induction t with
  | zero => simp [uncorrectedRun]
  | succ t ih =>
    let s := uncorrectedRun a b ε decay rate grad initial t
    let g := clip (grad t s.weights) i
    change (blend a (s.first i) g)^2 ≤ momentRatio a b * blend b (s.second i) (g^2)
    have hpos := sub_pos.mpr hab
    apply sub_nonneg.mp
    apply (mul_nonneg_iff_of_pos_left hpos).mp
    rw [unequal_variance_step hb hab]
    exact add_nonneg (mul_nonneg (mul_nonneg ((sq_nonneg a).trans hab.le) hpos.le) (sub_nonneg.mpr ih)) (sq_nonneg _)

/-- The square-root bound is uniform in time, dimension and minibatch order. -/
theorem unequal_direction_bound {K m v ε : ℝ} (hK : 0 < K) (hε : 0 < ε) (hm : m^2 ≤ K*v) :
    |direction ε m v| ≤ Real.sqrt K := by
  have h : |m| ≤ Real.sqrt K * Real.sqrt v := by
    rw [← Real.sqrt_mul hK.le,← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_le_sqrt hm
  have hden := add_pos_of_nonneg_of_pos (Real.sqrt_nonneg v) hε
  rw [direction,abs_div,abs_of_pos hden,div_le_iff₀ hden]
  exact h.trans (mul_le_mul_of_nonneg_left (le_add_of_nonneg_right hε.le) (Real.sqrt_nonneg _))

/-- The second memory incorporates the new squared gradient at half the
first memory's gain. At a = 0.9 this gives 19/14 rather than 1 for m²/v. -/
theorem half_gain_ratio {a : ℝ} (ha : 0 ≤ a) (ha' : a < 1) :
    momentRatio a ((1+a)/2) = 2*(1+a)/(1+2*a) := by
  have h : 0 < (1+a)/2-a^2 := by nlinarith [mul_pos (sub_pos.mpr ha') (by linarith : 0 < 1+2*a)]
  unfold momentRatio
  apply (div_eq_div_iff (mul_ne_zero (by linarith) h.ne') (by linarith)).2
  ring

theorem uncorrected_weight_bound (ε : ℝ) (hε : 0 < ε) (decay : ℝ) (hd : 0 < decay)
    (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) (t : ℕ) (i : ι) :
    |(uncorrectedRun (9/10) (19/20) ε decay rate grad initial t).weights i| ≤
      max ‖initial‖ (Real.sqrt (19/14)/decay) := by
  let s := uncorrectedRun (9/10) (19/20) ε decay rate grad initial
  apply DecoupledDecay.coordinate_bound (fun k => (s k).weights)
    (fun k j => direction ε ((s (k+1)).first j) ((s (k+1)).second j)) rate hd hr (fun _ _ => rfl) _ i t
  intro k j
  apply unequal_direction_bound (by norm_num) hε
  have hm := unequal_moment_bounds (a := 9/10) (b := 19/20) (by norm_num) (by norm_num) ε decay rate grad initial (k+1) j
  rw [show momentRatio (9/10) (19/20) = 19/14 by norm_num [momentRatio]] at hm
  exact hm

example (w : Fin 5 → ℝ) (n : ℕ) (i : Fin 5) := uncorrected_weight_bound (1/100000000) (by norm_num)
  (1/10) (by norm_num) (fun t => 3*Argand.wsdRate 20000 t)
  (fun t => by obtain ⟨hl,hu⟩ := Argand.wsdRate_bounds 20000 t; constructor <;> linarith) (fun _ v => v) w n i

end Transformer.Grokking.FiniteRecovery.EqualMoment
