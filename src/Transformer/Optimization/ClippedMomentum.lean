module

public import Transformer.AMSGradW.Kernite
public import Transformer.AMSGradW.Perturbation
public import Mathlib.Topology.Algebra.InfiniteSum.Real

/-! The exact-real version of the default Argand update in
experiments/phasor_value_transport/src/argand/kernels.rs. Global clipping
has threshold one, Polyak momentum is 0.9, and weight decay is zero. -/

@[expose] public section

namespace Transformer.Optimization.ClippedMomentum

open Transformer.AMSGradW Filter
open scoped Topology

variable {ι : Type*} [Fintype ι]

structure State (ι : Type*) where
  weights : ι → ℝ
  velocity : ι → ℝ

noncomputable def run (rate : ℕ → ℝ) (gradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) : ℕ → State ι
  | 0 => ⟨initial, 0⟩
  | t + 1 =>
    let s := run rate gradient initial t
    let v := (9 / 10 : ℝ) • s.velocity + Kernite.clip (gradient t s.weights)
    ⟨s.weights - rate t • v, v⟩

noncomputable def clippedGradient (rate : ℕ → ℝ) (gradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (t : ℕ) : ι → ℝ :=
  Kernite.clip (gradient t (run rate gradient initial t).weights)

theorem run_velocity_abs_le (rate : ℕ → ℝ) (gradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (i : ι) (t : ℕ) : |(run rate gradient initial t).velocity i| ≤ 10 := by
  induction t with
  | zero => simp [run]
  | succ t ih =>
    let s := run rate gradient initial t
    let g := Kernite.clip (gradient t s.weights) i
    change |(9 / 10 : ℝ) * s.velocity i + g| ≤ 10
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10)]
    linarith [Kernite.abs_clip_le (gradient t s.weights) i, ih]

theorem velocity_tendsto_zero_of_clippedGradient (rate : ℕ → ℝ)
    (gradient : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)
    (hg : Tendsto (clippedGradient rate gradient initial) atTop (𝓝 0)) :
    Tendsto (fun t => (run rate gradient initial t).velocity) atTop (𝓝 0) := by
  apply tendsto_pi_nhds.mpr
  intro i
  let v := fun t => (run rate gradient initial t).velocity i
  let g := fun t => clippedGradient rate gradient initial t i
  have hgi : Tendsto g atTop (𝓝 0) := tendsto_pi_nhds.mp hg i
  have hga : Tendsto (fun t => |g t|) atTop (𝓝 0) := by simpa only [abs_zero] using hgi.abs
  have hv : Tendsto (fun t => |v t|) atTop (𝓝 0) := by
    apply contraction_forcing_tendsto_zero (9 / 10) _ _ (by norm_num) (by norm_num)
      (fun _ => abs_nonneg _) hga
    intro t
    change |(9 / 10 : ℝ) * v t + g t| ≤ _
    simpa only [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10)] using
      abs_add_le ((9 / 10 : ℝ) * v t) (g t)
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  simpa only [Pi.zero_apply, sub_zero, Real.norm_eq_abs] using hv

/-- When the learning rate tends to a positive value, a finite weight limit
forces both the velocity and the actual clipped gradient to vanish. -/
theorem zero_limits_of_weights_tendsto (rate : ℕ → ℝ) (gradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial w : ι → ℝ) {η : ℝ} (hη : 0 < η) (hr : Tendsto rate atTop (𝓝 η))
    (hw : Tendsto (fun t => (run rate gradient initial t).weights) atTop (𝓝 w)) :
    Tendsto (fun t => (run rate gradient initial t).velocity) atTop (𝓝 0) ∧
      Tendsto (clippedGradient rate gradient initial) atTop (𝓝 0) := by
  let s := run rate gradient initial
  have hv : Tendsto (fun t => (s (t + 1)).velocity) atTop (𝓝 0) := by
    apply tendsto_pi_nhds.mpr
    intro i
    have hx := tendsto_pi_nhds.mp hw i
    have hx' := (Filter.tendsto_add_atTop_iff_nat 1).2 hx
    have hq : Tendsto (fun t => ((s t).weights i - (s (t + 1)).weights i) / rate t) atTop (𝓝 0) := by
      have h := (hx.sub hx').div hr hη.ne'
      change Tendsto (fun t => ((s t).weights i - (s (t + 1)).weights i) / rate t)
        atTop (𝓝 ((w i - w i) / η)) at h
      simpa only [sub_self, zero_div] using h
    apply hq.congr'
    filter_upwards [hr.eventually (lt_mem_nhds hη)] with t ht
    change ((s t).weights i - ((s t).weights i - rate t * (s (t + 1)).velocity i)) / rate t = _
    rw [sub_sub_cancel, mul_div_cancel_left₀ _ (ne_of_gt ht)]
  have hv₀ : Tendsto (fun t => (s t).velocity) atTop (𝓝 0) :=
    (Filter.tendsto_add_atTop_iff_nat 1).1 hv
  refine ⟨hv₀, ?_⟩
  apply tendsto_pi_nhds.mpr
  intro i
  have hg (t : ℕ) : clippedGradient rate gradient initial t i =
      (s (t + 1)).velocity i - (9 / 10 : ℝ) * (s t).velocity i := by
    change _ = ((9 / 10 : ℝ) * (s t).velocity i + clippedGradient rate gradient initial t i) -
      (9 / 10 : ℝ) * (s t).velocity i
    ring
  simp only [hg]
  simpa only [Pi.zero_apply, mul_zero, sub_zero] using
    (tendsto_pi_nhds.mp hv i).sub ((tendsto_pi_nhds.mp hv₀ i).const_mul (9 / 10))

private theorem norm_clip (g : ι → ℝ) :
    ‖(WithLp.toLp 2 (Kernite.clip g) : Kernite.Space ι)‖ =
      ‖(WithLp.toLp 2 g : Kernite.Space ι)‖ / max ‖(WithLp.toLp 2 g : Kernite.Space ι)‖ 1 := by
  let x : Kernite.Space ι := WithLp.toLp 2 g
  let c := max ‖x‖ 1
  have h : (WithLp.toLp 2 (Kernite.clip g) : Kernite.Space ι) = c⁻¹ • x := by
    ext i
    change g i / c = c⁻¹ * g i
    rw [div_eq_mul_inv, mul_comm]
  rw [h, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr (zero_le_one.trans (le_max_right _ _)))]
  exact (div_eq_inv_mul _ _).symm

/-- When clipped gradients tend to zero their norms eventually fall below
the clipping threshold. From that point the raw and clipped gradients agree. -/
theorem tendsto_zero_of_clip {g : ℕ → ι → ℝ}
    (h : Tendsto (fun t => Kernite.clip (g t)) atTop (𝓝 0)) : Tendsto g atTop (𝓝 0) := by
  have hn : Tendsto (fun t => ‖(WithLp.toLp 2 (Kernite.clip (g t)) : Kernite.Space ι)‖) atTop (𝓝 0) := by
    have ht := ((PiLp.continuous_toLp 2 _).tendsto (0 : ι → ℝ)).comp h
    simpa only [Function.comp_def, WithLp.toLp_zero, norm_zero] using ht.norm
  apply h.congr'
  filter_upwards [hn.eventually (gt_mem_nhds zero_lt_one)] with t ht
  have hf : ‖(WithLp.toLp 2 (g t) : Kernite.Space ι)‖ < 1 := by
    by_contra! ha
    rw [norm_clip, max_eq_left ha, div_self (ne_of_gt (zero_lt_one.trans_le ha))] at ht
    exact lt_irrefl _ ht
  funext i
  simp only [Kernite.clip, max_eq_right hf.le, div_one]

/-- A summable rate bounds total movement, even for a changing sequence
of gradients. Applied to WSD this gives a guarantee for a summable tail. -/
theorem weights_tendsto_of_summable (rate : ℕ → ℝ)
    (gradient : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)
    (hr : ∀ t, 0 ≤ rate t) (hs : Summable rate) :
    ∃ w : ι → ℝ, Tendsto (fun t => (run rate gradient initial t).weights) atTop (𝓝 w) := by
  let s := run rate gradient initial
  have hdist (i : ι) (n : ℕ) : dist ((s n).weights i) ((s (n + 1)).weights i) ≤ 10 * rate n := by
    rw [Real.dist_eq]
    change |(s n).weights i - ((s n).weights i - rate n * (s (n + 1)).velocity i)| ≤ _
    rw [sub_sub_cancel, abs_mul, abs_of_nonneg (hr n), mul_comm (10 : ℝ)]
    exact mul_le_mul_of_nonneg_left (run_velocity_abs_le rate gradient initial i (n + 1)) (hr n)
  have hlim (i : ι) : ∃ w : ℝ, Tendsto (fun t => (s t).weights i) atTop (𝓝 w) :=
    cauchySeq_tendsto_of_complete (cauchySeq_of_dist_le_of_summable _ (hdist i) (hs.mul_left 10))
  choose w hw using hlim
  exact ⟨w, tendsto_pi_nhds.mpr hw⟩

example (gradient : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  weights_tendsto_of_summable (fun t => (1 / 2 : ℝ) ^ t) gradient initial
    (fun _ => by positivity) (summable_geometric_of_lt_one (by norm_num) (by norm_num))

example := tendsto_zero_of_clip (ι := Fin 2) (g := fun _ => 0) (by
  have h : Kernite.clip (0 : Fin 2 → ℝ) = 0 := by
    funext i
    simp [Kernite.clip]
  simpa only [h] using (tendsto_const_nhds (x := (0 : Fin 2 → ℝ))))

end Transformer.Optimization.ClippedMomentum
