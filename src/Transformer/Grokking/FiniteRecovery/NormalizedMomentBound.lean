module

public import Transformer.Grokking.FiniteRecovery.NormalizedAdam
public import Transformer.Grokking.FiniteRecovery.UncorrectedAdam

/-! A time-uniform direction bound for bias-corrected AdamW. Retaining the
finite geometric sum in the raw moments cancels the apparent amplification
from bias correction, including on the first update. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.NormalizedAdam

open EqualMoment (blend momentRatio)
open Transformer.AMSGradW.Kernite (clip)

def momentHistory (a : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t+1 => blend a (momentHistory a g t) (g t)

private theorem history_nonneg {a : ℝ} (ha : 0 ≤ a) (ha' : a ≤ 1) (g : ℕ → ℝ)
    (hg : ∀ t, 0 ≤ g t) (t : ℕ) : 0 ≤ momentHistory a g t := by
  induction t with
  | zero => exact le_rfl
  | succ t ih => exact add_nonneg (mul_nonneg ha ih) (mul_nonneg (sub_nonneg.mpr ha') (hg t))

private theorem weighted_step {a b k m v : ℝ} (hb : 0 < b) (hb' : b < 1) (hk : 0 ≤ k)
    (hv : 0 ≤ v) (h : m^2 ≤ k*v) (g : ℝ) :
    (blend a m g)^2 ≤ (a^2/b*k+(1-a)^2/(1-b))*blend b v (g^2) := by
  have h₁ := sub_pos.mpr hb'
  rcases hk.eq_or_lt with rfl | hk
  · have hm : m = 0 := by nlinarith [sq_nonneg m]
    rw [hm]
    have eq : ((1-a)^2/(1-b))*blend b v (g^2) - (blend a 0 g)^2 = ((1-a)^2/(1-b))*b*v := by
      unfold blend
      field_simp [h₁.ne']
      ring
    simp only [mul_zero,zero_add]
    exact sub_nonneg.mp (eq.symm ▸ mul_nonneg (mul_nonneg (div_nonneg (sq_nonneg _) h₁.le) hb.le) hv)
  · have eq : k*b*(1-b)*((a^2/b*k+(1-a)^2/(1-b))*blend b v (g^2)-(blend a m g)^2) =
        b*(a^2*k*(1-b)+b*(1-a)^2)*(k*v-m^2) + (a*(1-b)*k*g-b*(1-a)*m)^2 := by
      unfold blend
      field_simp [hb.ne',h₁.ne']
      ring
    apply sub_nonneg.mp
    apply (mul_nonneg_iff_of_pos_left (mul_pos (mul_pos hk hb) h₁)).mp
    rw [eq]
    exact add_nonneg (mul_nonneg (by positivity) (sub_nonneg.mpr h)) (sq_nonneg _)

/-- A bound for the two normalized scalar moments of *any* gradient
history. The learning rate, gradient scale and number of updates are free. -/
theorem corrected_history_bound {a b : ℝ} (ha : 0 ≤ a) (ha' : a < 1) (hab : a^2 < b)
    (hb' : b < 1) (g : ℕ → ℝ) (t : ℕ) :
    (momentHistory a g t / (1-a^t))^2 ≤ momentRatio a b * (momentHistory b (fun j => (g j)^2) t / (1-b^t)) := by
  have hb := (sq_nonneg a).trans_lt hab
  let q := a^2/b
  let K := momentRatio a b
  have hq : 0 ≤ q := div_nonneg (sq_nonneg a) hb.le
  have hq' : q < 1 := (div_lt_one hb).2 hab
  have hK : 0 ≤ K := div_nonneg (mul_nonneg hb.le (sq_nonneg _))
    (mul_nonneg (sub_nonneg.mpr hb'.le) (sub_nonneg.mpr hab.le))
  have fixed : K*(1-q) = (1-a)^2/(1-b) := by
    dsimp [K,q,momentRatio]
    field_simp [hb.ne',(sub_pos.mpr hb').ne',(sub_pos.mpr hab).ne']
  have raw (n : ℕ) : (momentHistory a g n)^2 ≤ K*(1-q^n)*momentHistory b (fun j => (g j)^2) n := by
    induction n with
    | zero => simp [momentHistory]
    | succ n ih =>
      have next : K*(1-q^(n+1)) = q*(K*(1-q^n))+(1-a)^2/(1-b) := by
        rw [← fixed,pow_succ]
        ring
      rw [next]
      exact weighted_step hb hb' (mul_nonneg hK (sub_nonneg.mpr (pow_le_one₀ hq hq'.le)))
        (history_nonneg hb.le hb'.le _ (fun j => sq_nonneg (g j)) n) ih (g n)
  cases t with
  | zero => simp [momentHistory]
  | succ n =>
    let t := n+1
    have A : 0 < 1-a^t := sub_pos.mpr (pow_lt_one₀ ha ha' (Nat.succ_ne_zero n))
    have B : 0 < 1-b^t := sub_pos.mpr (pow_lt_one₀ hb.le hb' (Nat.succ_ne_zero n))
    have prod : b^t*q^t = (a^t)^2 := by
      rw [← mul_pow,show b*q=a^2 from mul_div_cancel₀ _ hb.ne',← pow_mul,← pow_mul,Nat.mul_comm 2]
    have am : 0 ≤ b^t+q^t-2*a^t := by
      apply (mul_nonneg_iff_of_pos_left (pow_pos hb t)).mp
      nlinarith [sq_nonneg (b^t-a^t),prod]
    have factor : (1-q^t)*(1-b^t) ≤ (1-a^t)^2 := by nlinarith [prod]
    let v := momentHistory b (fun j => (g j)^2) t
    have hv : 0 ≤ v := history_nonneg hb.le hb'.le _ (fun j => sq_nonneg (g j)) t
    change (momentHistory a g t / (1-a^t))^2 ≤ K*(v/(1-b^t))
    rw [div_pow,mul_div_assoc',le_div_iff₀ B,div_mul_eq_mul_div,div_le_iff₀ (sq_pos_of_pos A)]
    calc
      _ ≤ (K*(1-q^t)*v)*(1-b^t) := mul_le_mul_of_nonneg_right (raw t) B.le
      _ = (K*v)*((1-q^t)*(1-b^t)) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left factor (mul_nonneg hK hv)

variable {ι : Type*} [Fintype ι]

/-- The normalized kernel satisfies the 19/14 cone along the complete
state-dependent optimizer trajectory, with both stock bias corrections. -/
theorem corrected_moment_bounds (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (w : ι → ℝ) (t : ℕ) (i : ι) :
    ((run rate grad w t).first i)^2 ≤ 19/14 * (run rate grad w t).second i := by
  let s := run rate grad w
  let g (k : ℕ) := clip (grad k (s k).weights) i
  have iso (k : ℕ) : momentHistory (9/10) g k = (1-(9/10:ℝ)^k)*(s k).first i ∧
      momentHistory (95/100) (fun j => (g j)^2) k = (1-(95/100:ℝ)^k)*(s k).second i := by
    induction k with
    | zero => simp [momentHistory]
    | succ k ih =>
      constructor
      · exact (congrArg (fun m => blend (9/10) m (g k)) ih.1).trans
          (mass_blend (by norm_num) (by norm_num) k ((s k).first i) (g k)).symm
      · exact (congrArg (fun v => blend (95/100) v ((g k)^2)) ih.2).trans
          (mass_blend (by norm_num) (by norm_num) k ((s k).second i) ((g k)^2)).symm
  cases t with
  | zero => norm_num [run]
  | succ k =>
    have h := corrected_history_bound (a := 9/10) (b := 95/100) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) g (k+1)
    rw [(iso _).1,(iso _).2,mul_div_cancel_left₀ _ (bias_pos (by norm_num) (by norm_num) k).ne',
      mul_div_cancel_left₀ _ (bias_pos (by norm_num) (by norm_num) k).ne'] at h
    norm_num [momentRatio] at h
    exact h

theorem corrected_direction_bound (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (w : ι → ℝ) (t : ℕ) (i : ι) :
    let s := run rate grad w (t+1)
    |EqualMoment.direction (1/100000000) (s.first i) (s.second i)| ≤ Real.sqrt (19/14) :=
  EqualMoment.unequal_direction_bound (by norm_num) (by norm_num) (corrected_moment_bounds rate grad w _ i)

/-- Decoupled decay 0.1 bounds every trainable coordinate, with no
assumption on the minibatch stream or convergence of its gradients. -/
theorem corrected_weight_bound (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10)
    (grad : ℕ → (ι → ℝ) → ι → ℝ) (w : ι → ℝ) (t : ℕ) (i : ι) :
    |(run rate grad w t).weights i| ≤ max ‖w‖ (10*Real.sqrt (19/14)) := by
  let s := run rate grad w
  let dir k j := EqualMoment.direction (1/100000000) ((s (k+1)).first j) ((s (k+1)).second j)
  have update (k : ℕ) (j : ι) : (s (k+1)).weights j = (1-rate k*(1/10))*(s k).weights j-rate k*dir k j := by
    change (1-rate k/10)*(s k).weights j-rate k*dir k j = _
    ring
  calc
    _ ≤ max ‖w‖ (Real.sqrt (19/14)/(1/10)) :=
      DecoupledDecay.coordinate_bound (fun k => (s k).weights) dir rate (by norm_num)
        (fun k => ⟨(hr k).1,by linarith [(hr k).2]⟩) update (corrected_direction_bound rate grad w) i t
    _ = _ := by congr 1; ring

theorem stock_weight_bound (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10)
    (grad : ℕ → (ι → ℝ) → ι → ℝ) (w : ι → ℝ) (t : ℕ) (i : ι) :
    |(AdamW.run rate grad w t).weights i| ≤ max ‖w‖ (10*Real.sqrt (19/14)) := by
  rw [← (agrees_with_stock rate grad w t).1]
  exact corrected_weight_bound rate hr grad w t i

example (n : ℕ) (i : Fin 5) (w : Fin 5 → ℝ) :=
  stock_weight_bound (fun k => 3*Argand.wsdRate 20000 k)
    (fun k => by obtain ⟨hl,hu⟩ := Argand.wsdRate_bounds 20000 k; constructor <;> linarith)
    (fun _ v => v) w n i

end Transformer.Grokking.FiniteRecovery.NormalizedAdam
