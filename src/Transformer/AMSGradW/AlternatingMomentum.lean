module

public import Transformer.AMSGradW.KerniteAlternation

/-! Momentum SGD on the same two interpolating quadratic minibatches.
An explicit stable momentum step converges without decay. At a positive
limiting rate, either form of positive decay rules out a common weight limit. -/

@[expose] public section

namespace Transformer.AMSGradW.Kernite

open Filter
open scoped Topology

/-- Scalar Polyak momentum with zero initial velocity. The L2 coefficient
adds to the batch gradient. Decoupled decay acts directly on the weight. -/
noncomputable def alternatingMomentumRun (rate : ℕ → ℝ) (β l2 decay initial : ℝ) : ℕ → ℝ × ℝ
  | 0 => (initial, 0)
  | t + 1 =>
    let s := alternatingMomentumRun rate β l2 decay initial t
    let v := β * s.2 + alternatingWeight t * (s.1 - 1) + l2 * s.1
    ((1 - rate t * decay) * s.1 - rate t * v, v)

private theorem alternatingWeight_even (n : ℕ) : alternatingWeight (2 * n) = 1 := by
  simp [alternatingWeight]

private theorem alternatingWeight_odd (n : ℕ) : alternatingWeight (2 * n + 1) = 2 := by
  simp [alternatingWeight]

/-- With step (1+β)/2, every other position error satisfies a geometric
recurrence. The other parity has the same contraction with vanishing input. -/
theorem alternatingMomentum_tendsto (β initial : ℝ) (hβ : 0 ≤ β) (hβ' : β < 1) :
    Tendsto (fun t => (alternatingMomentumRun (fun _ => (1 + β) / 2) β 0 0 initial t).1) atTop (𝓝 1) := by
  let η := (1 + β) / 2
  let s := alternatingMomentumRun (fun _ => η) β 0 0 initial
  let e := fun t => (s t).1 - 1
  have hp (t : ℕ) : e (t + 1) = e t - η * (s (t + 1)).2 := by
    change (1 - η * 0) * (s t).1 - η * (s (t + 1)).2 - 1 = (s t).1 - 1 - η * (s (t + 1)).2
    ring
  have hrec (t : ℕ) : e (t + 2) =
      (1 + β - η * alternatingWeight (t + 1)) * e (t + 1) - β * e t := by
    have hv : (s (t + 2)).2 = β * (s (t + 1)).2 + alternatingWeight (t + 1) * e (t + 1) := by
      change β * (s (t + 1)).2 + alternatingWeight (t + 1) * e (t + 1) + 0 * (s (t + 1)).1 = _
      rw [zero_mul, add_zero]
    have he : η * (s (t + 1)).2 = e t - e (t + 1) := by linarith [hp t]
    calc
      e (t + 2) = e (t + 1) - η * (s (t + 2)).2 := hp (t + 1)
      _ = e (t + 1) - β * (η * (s (t + 1)).2) - η * alternatingWeight (t + 1) * e (t + 1) := by
        rw [hv]
        ring
      _ = _ := by rw [he]; ring
  have heven (n : ℕ) : e (2 * (n + 1)) = -β * e (2 * n) := by
    rw [show 2 * (n + 1) = 2 * n + 2 by omega, hrec, alternatingWeight_odd]
    dsimp only [η]
    ring
  have hodd (n : ℕ) : e (2 * (n + 1) + 1) = -β * e (2 * n + 1) + (1 + β - η) * e (2 * (n + 1)) := by
    have h := hrec (2 * n + 1)
    rw [show 2 * n + 1 + 2 = 2 * (n + 1) + 1 by omega,
      show 2 * n + 1 + 1 = 2 * (n + 1) by omega, alternatingWeight_even, mul_one] at h
    rw [h]
    ring
  have htEven : Tendsto (fun n => |e (2 * n)|) atTop (𝓝 0) := by
    apply contraction_forcing_tendsto_zero β _ (fun _ => 0) hβ hβ' (fun _ => abs_nonneg _) tendsto_const_nhds
    intro n
    rw [heven, abs_mul, abs_neg, abs_of_nonneg hβ, add_zero]
  have hf : Tendsto (fun n => |1 + β - η| * |e (2 * (n + 1))|) atTop (𝓝 0) := by
    simpa only [mul_zero] using ((Filter.tendsto_add_atTop_iff_nat 1).2 htEven).const_mul |1 + β - η|
  have htOdd : Tendsto (fun n => |e (2 * n + 1)|) atTop (𝓝 0) := by
    apply contraction_forcing_tendsto_zero β _ _ hβ hβ' (fun _ => abs_nonneg _) hf
    intro n
    rw [hodd]
    refine (abs_add_le _ _).trans_eq ?_
    rw [abs_mul, abs_mul, abs_neg, abs_of_nonneg hβ]
  have hhalf : Tendsto (fun n : ℕ => n / 2) atTop atTop := by
    apply tendsto_atTop.mpr
    intro n
    exact (eventually_ge_atTop (2 * n)).mono fun k hk => by change n ≤ k / 2; omega
  have hub : Tendsto (fun t => |e (2 * (t / 2))| + |e (2 * (t / 2) + 1)|) atTop (𝓝 0) := by
    simpa only [Function.comp_def, add_zero] using (htEven.comp hhalf).add (htOdd.comp hhalf)
  have hlim : Tendsto (fun t => |e t|) atTop (𝓝 0) := by
    apply squeeze_zero (fun _ => abs_nonneg _) (fun t => ?_) hub
    rcases Nat.mod_two_eq_zero_or_one t with h | h
    · rw [show 2 * (t / 2) = t by omega]
      exact le_add_of_nonneg_right (abs_nonneg _)
    · rw [show 2 * (t / 2) + 1 = t by omega]
      exact le_add_of_nonneg_left (abs_nonneg _)
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  exact hlim.congr fun _ => (Real.norm_eq_abs _).symm

/-- At a positive limiting rate, both batches must obey the same momentum
balance. Their only common gradient is zero at one. Any positive effective
decay, coupled or decoupled, conflicts with that balance. -/
theorem alternatingMomentum_not_tendsto (rate : ℕ → ℝ) (β l2 decay initial : ℝ) {η : ℝ}
    (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η)) (hdecay : 0 < (1 - β) * decay + l2) :
    ¬ ∃ w, Tendsto (fun t => (alternatingMomentumRun rate β l2 decay initial t).1) atTop (𝓝 w) := by
  rintro ⟨w, hw⟩
  let s := alternatingMomentumRun rate β l2 decay initial
  let m := -decay * w
  have hv (t : ℕ) (hr : rate t ≠ 0) :
      (s (t + 1)).2 = ((1 - rate t * decay) * (s t).1 - (s (t + 1)).1) / rate t := by
    change (s (t + 1)).2 = ((1 - rate t * decay) * (s t).1 -
      ((1 - rate t * decay) * (s t).1 - rate t * (s (t + 1)).2)) / rate t
    field_simp
    ring
  have hm' : Tendsto (fun t => (s (t + 1)).2) atTop (𝓝 m) := by
    have hcoef := (hηlim.mul_const decay).const_sub 1
    have h := ((hcoef.mul hw).sub ((Filter.tendsto_add_atTop_iff_nat 1).2 hw)).div hηlim hη.ne'
    rw [show ((1 - η * decay) * w - w) / η = m by dsimp only [m]; field_simp; ring] at h
    apply h.congr'
    filter_upwards [hηlim.eventually (lt_mem_nhds hη)] with t ht
    exact (hv t (ne_of_gt ht)).symm
  have hm : Tendsto (fun t => (s t).2) atTop (𝓝 m) := (Filter.tendsto_add_atTop_iff_nat 1).1 hm'
  have hg (t : ℕ) : (s (t + 1)).2 - β * (s t).2 - l2 * (s t).1 =
      alternatingWeight t * ((s t).1 - 1) := by
    change β * (s t).2 + alternatingWeight t * ((s t).1 - 1) + l2 * (s t).1 - β * (s t).2 -
      l2 * (s t).1 = _
    ring
  have ht : Tendsto (fun t => alternatingWeight t * ((s t).1 - 1)) atTop (𝓝 (m - β * m - l2 * w)) :=
    ((hm'.sub (hm.const_mul β)).sub (hw.const_mul l2)).congr hg
  have he : StrictMono (fun t : ℕ => 2 * t) := fun a b h => by change 2 * a < 2 * b; omega
  have ho : StrictMono (fun t : ℕ => 2 * t + 1) := fun a b h => by change 2 * a + 1 < 2 * b + 1; omega
  have heq := tendsto_nhds_unique (ht.comp he.tendsto_atTop) (show
    Tendsto (fun t => alternatingWeight (2 * t) * ((s (2 * t)).1 - 1)) atTop (𝓝 (w - 1)) by
      simpa only [Function.comp_def, alternatingWeight_even, one_mul, s] using
        (hw.comp he.tendsto_atTop).sub_const 1)
  have hoq := tendsto_nhds_unique (ht.comp ho.tendsto_atTop) (show
    Tendsto (fun t => alternatingWeight (2 * t + 1) * ((s (2 * t + 1)).1 - 1)) atTop (𝓝 (2 * (w - 1))) by
      simpa only [Function.comp_def, alternatingWeight_odd, s] using
        ((hw.comp ho.tendsto_atTop).sub_const 1).const_mul 2)
  have hw₁ : w = 1 := by linarith
  have hm₁ : m = -decay := by simp [m, hw₁]
  rw [hm₁, hw₁, sub_self] at heq
  have h := hdecay.ne'
  apply h
  linarith

example (initial : ℝ) :=
  alternatingMomentum_tendsto (9 / 10) initial (by norm_num) (by norm_num)

example (initial : ℝ) :=
  alternatingMomentum_not_tendsto (fun _ => 1 / 1000) (9 / 10) 0 (1 / 10) initial
    (by norm_num) tendsto_const_nhds (by norm_num)

example (initial : ℝ) :=
  alternatingMomentum_not_tendsto (fun _ => 1 / 1000) (9 / 10) (1 / 10) 0 initial
    (by norm_num) tendsto_const_nhds (by norm_num)

end Transformer.AMSGradW.Kernite
