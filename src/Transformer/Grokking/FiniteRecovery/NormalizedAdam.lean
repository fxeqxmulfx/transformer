module

public import Transformer.Grokking.FiniteRecovery.EqualMoment
public import Transformer.Grokking.FiniteRecovery.NativeFirstPerfect
public import Transformer.Grokking.FiniteRecovery.AdamWFloor

/-! AdamW written as two normalized moving averages. Two scalar gains start at
one and depend only on their previous values. For the stock Argand coefficients
every weight on the real trajectory agrees with the ordinary AdamW kernel. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.NormalizedAdam

open Transformer.AMSGradW.Kernite (clip firstBias)
open EqualMoment (blend)

/-- Fraction of the new observation in the normalized exponential average. -/
noncomputable def gain (β : ℝ) : ℕ → ℝ
  | 0 => 1
  | t+1 => gain β t / (β+gain β t)

theorem bias_pos {β : ℝ} (h0 : 0 ≤ β) (h1 : β < 1) (t : ℕ) : 0 < 1-β^(t+1) :=
  sub_pos.mpr (pow_lt_one₀ h0 h1 (Nat.succ_ne_zero t))

theorem gain_eq {β : ℝ} (h0 : 0 ≤ β) (h1 : β < 1) (t : ℕ) :
    gain β t = (1-β)/(1-β^(t+1)) := by
  induction t with
  | zero => simp [gain,(sub_pos.mpr h1).ne']
  | succ t ih =>
    rw [gain,ih]
    have den := bias_pos h0 h1 t
    have sum := add_pos_of_nonneg_of_pos h0 (div_pos (sub_pos.mpr h1) den)
    apply (div_eq_div_iff sum.ne' (bias_pos h0 h1 (t+1)).ne').2
    rw [pow_succ β (t+1)]
    field_simp [den.ne']
    ring

/-- Multiplying the normalized moment by its accumulated mass recovers
the ordinary zero-initialized moment recurrence at every update. -/
theorem mass_blend {β : ℝ} (h0 : 0 ≤ β) (h1 : β < 1) (t : ℕ) (m g : ℝ) :
    (1-β^(t+1))*blend (1-gain β t) m g = blend β ((1-β^t)*m) g := by
  rw [gain_eq h0 h1]
  unfold blend
  field_simp [(bias_pos h0 h1 t).ne']
  rw [pow_succ]
  ring

variable {ι : Type*} [Fintype ι]

/-- The same accumulator update is used for each normalized moment. All
coordinates share the two scalar gains. The decay uses the original rate. -/
noncomputable def run (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) :
    ℕ → AdamW.State ι
  | 0 => ⟨initial,0,0⟩
  | t+1 =>
    let s := run rate grad initial t
    let g := clip (grad t s.weights)
    let m i := blend (1-gain (9/10) t) (s.first i) (g i)
    let v i := blend (1-gain (95/100) t) (s.second i) ((g i)^2)
    ⟨fun i => (1-rate t/10)*s.weights i-rate t*EqualMoment.direction (1/100000000) (m i) (v i),m,v⟩

theorem scaled_direction (t : ℕ) (m v : ℝ) :
    AdamW.direction t ((1-(9/10:ℝ)^(t+1))*m) ((1-(95/100:ℝ)^(t+1))*v) =
      EqualMoment.direction (1/100000000) m v := by
  have hb := bias_pos (β := 95/100) (by norm_num) (by norm_num) t
  have ha := bias_pos (β := 9/10) (by norm_num) (by norm_num) t
  simp only [AdamW.direction,AdamW.scale,firstBias,Real.sqrt_mul hb.le,mul_div_cancel_left₀ _ (Real.sqrt_pos.mpr hb).ne',
    EqualMoment.direction]
  rw [mul_div_mul_left _ _ ha.ne']

/-- All rates, model starts and state-dependent gradient callbacks are
allowed. The weights agree at each update; the raw moment differences
are precisely the accumulated masses. -/
theorem agrees_with_stock (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) (t : ℕ) :
    let s := run rate grad initial t
    let stock := AdamW.run rate grad initial t
    s.weights = stock.weights ∧ (∀ i, stock.first i = (1-(9/10:ℝ)^t)*s.first i) ∧
      (∀ i, stock.second i = (1-(95/100:ℝ)^t)*s.second i) := by
  induction t with
  | zero => simp [run,AdamW.run]
  | succ t ih =>
    obtain ⟨hw,hm,hv⟩ := ih
    let s := run rate grad initial t
    let g := clip (grad t s.weights)
    have m (i : ι) : (AdamW.run rate grad initial (t+1)).first i =
        (1-(9/10:ℝ)^(t+1))*(run rate grad initial (t+1)).first i := by
      change (9/10:ℝ)*_ + 1/10*_ = _*blend _ _ _
      rw [← hw,hm]
      convert (mass_blend (β := 9/10) (by norm_num) (by norm_num) t (s.first i) (g i)).symm using 1
      unfold blend
      ring
    have v (i : ι) : (AdamW.run rate grad initial (t+1)).second i =
        (1-(95/100:ℝ)^(t+1))*(run rate grad initial (t+1)).second i := by
      change (95/100:ℝ)*_ + 5/100*_ = _*blend _ _ _
      rw [← hw,hv]
      convert (mass_blend (β := 95/100) (by norm_num) (by norm_num) t (s.second i) ((g i)^2)).symm using 1
      unfold blend
      ring
    refine ⟨funext fun i => ?_,m,v⟩
    change (1-rate t/10)*s.weights i-rate t*EqualMoment.direction _ _ _ =
      (1-rate t/10)*(AdamW.run rate grad initial t).weights i-rate t*AdamW.direction t
        ((AdamW.run rate grad initial (t+1)).first i) ((AdamW.run rate grad initial (t+1)).second i)
    rw [m,v,scaled_direction,← hw]
    rfl

/-- Every finite-grid first-hit time, including failure throughout the
grid, agrees for any evaluation rule on the current weights. -/
theorem first_hit_eq_stock (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)
    (grid : Finset ℕ) (accept : ℕ → (ι → ℝ) → Prop) :
    firstPerfect grid (fun t => accept t (run rate grad initial t).weights) =
      firstPerfect grid (fun t => accept t (AdamW.run rate grad initial t).weights) := by
  simp_rw [(agrees_with_stock rate grad initial _).1]

open Filter
open scoped Topology

/-- On the positive WSD floor a common clipped gradient limit determines
the full weight limit of the recursively normalized training run. -/
theorem common_gradient_limit (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial G : ι → ℝ) {η : ℝ} (hη : 0 < η) (hc : η ≤ 10) (hr : ∀ᶠ t in atTop, rate t = η)
    (hg : Tendsto (fun t => clip (grad t (run rate grad initial t).weights)) atTop (𝓝 G)) :
    Tendsto (fun t => (run rate grad initial t).weights) atTop
      (𝓝 (fun i => -10*(G i/(|G i|+1/100000000)))) := by
  simp_rw [(agrees_with_stock rate grad initial _).1] at hg ⊢
  exact AdamW.limit_of_common_clipped_gradient rate grad initial G hη hc hr hg

example (w c : Fin 3 → ℝ) := common_gradient_limit (fun _ => 3/100000000) (fun _ _ => c)
  w (clip c) (by norm_num) (by norm_num) (Eventually.of_forall fun _ => rfl) tendsto_const_nhds

example (batch : ℕ → List (Argand.LabeledPosition (Fin 239)))
    (draw : Argand.ComplexParameter (Fin 239) 4 16 6 → ℂ) (S : Set Argand.DivisionInput) :=
  first_hit_eq_stock (fun t => 3*Argand.wsdRate 20000 t) (fun t => Argand.batchGradient 1 (batch t))
    (Argand.sgdInitialWeights true draw) (evaluationGrid 256 20000) (fun _ w => StrictOn
      (fun x : (Sᶜ : Set Argand.DivisionInput) => (Argand.Model.fromWeights w).logits 1 (Argand.divisionPrompt x) (Fin.last 4))
      (fun x => Argand.divisionAnswer x))

end Transformer.Grokking.FiniteRecovery.NormalizedAdam
