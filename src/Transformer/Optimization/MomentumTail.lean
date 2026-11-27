module

public import Transformer.Optimization.ClippedMomentum

/-! With a positive constant tail rate, full-sequence weight convergence
of clipped momentum SGD is equivalent to convergence of accumulated
clipped gradients on that tail. -/

@[expose] public section

namespace Transformer.Optimization.ClippedMomentum

open Transformer.AMSGradW Filter
open scoped Topology

variable {ι : Type*} [Fintype ι]

theorem tail_gradient_sum_eq (rate : ℕ → ℝ) (gradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (N : ℕ) {η : ℝ} (hr : ∀ t ≥ N, rate t = η) (i : ι) (n : ℕ) :
    let s := run rate gradient initial
    (s (N + n)).weights i - 9 * η * (s (N + n)).velocity i =
      ((s N).weights i - 9 * η * (s N).velocity i) -
        10 * η * ∑ j ∈ Finset.range n, clippedGradient rate gradient initial (N + j) i := by
  let s := run rate gradient initial
  dsimp only
  induction n with
  | zero => simp
  | succ n ih =>
    let t := N + n
    have hp : (s (t + 1)).weights i = (s t).weights i - rate t * (s (t + 1)).velocity i := rfl
    have hv : (s (t + 1)).velocity i =
        (9 / 10 : ℝ) * (s t).velocity i + clippedGradient rate gradient initial t i := rfl
    rw [Nat.add_succ, Finset.sum_range_succ]
    change (s (t + 1)).weights i - 9 * η * (s (t + 1)).velocity i = _
    rw [hp, hr t (Nat.le_add_right _ _), hv]
    change (s t).weights i - 9 * η * (s t).velocity i = _ at ih
    linear_combination ih

/-- A zero gradient limit alone does not suffice. The summed gradients
must have finite limits in every real coordinate. -/
theorem weights_tendsto_iff_tail_gradient_sum (rate : ℕ → ℝ)
    (gradient : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) (N : ℕ)
    {η : ℝ} (hη : 0 < η) (hr : ∀ t ≥ N, rate t = η) :
    (∃ w, Tendsto (fun t => (run rate gradient initial t).weights) atTop (𝓝 w)) ↔
      ∀ i, ∃ G : ℝ,
        Tendsto (fun n => ∑ j ∈ Finset.range n, clippedGradient rate gradient initial (N + j) i)
          atTop (𝓝 G) := by
  let s := run rate gradient initial
  let g := clippedGradient rate gradient initial
  let G (i : ι) (n : ℕ) := ∑ j ∈ Finset.range n, g (N + j) i
  let C (i : ι) := (s N).weights i - 9 * η * (s N).velocity i
  have htel (i : ι) (n : ℕ) : (s (N + n)).weights i - 9 * η * (s (N + n)).velocity i =
      C i - 10 * η * G i n := tail_gradient_sum_eq rate gradient initial N hr i n
  have hrate : Tendsto rate atTop (𝓝 η) :=
    tendsto_const_nhds.congr' ((eventually_ge_atTop N).mono fun t ht => (hr t ht).symm)
  have hshift : Tendsto (fun n => N + n) atTop atTop :=
    tendsto_atTop.mpr fun b => (eventually_ge_atTop b).mono fun n hn => hn.trans (Nat.le_add_left _ _)
  constructor
  · rintro ⟨w, hw⟩ i
    have hv := (zero_limits_of_weights_tendsto rate gradient initial w hη hrate hw).1
    have h := (tendsto_pi_nhds.mp hw i).comp hshift
    have hz : Tendsto (fun n => 9 * η * (s (N + n)).velocity i) atTop (𝓝 0) := by
      simpa only [Function.comp_def, Pi.zero_apply, mul_zero, s] using
        ((tendsto_pi_nhds.mp hv i).comp hshift).const_mul (9 * η)
    have hu := ((h.sub hz).const_sub (C i)).div_const (10 * η)
    rw [sub_zero] at hu
    refine ⟨_, hu.congr fun n => ?_⟩
    change ((C i - ((s (N + n)).weights i - 9 * η * (s (N + n)).velocity i)) / (10 * η)) = G i n
    apply (div_eq_iff (by positivity : (10 : ℝ) * η ≠ 0)).2
    linarith [htel i n]
  · intro hg
    choose limit hlimit using hg
    have hzero : Tendsto g atTop (𝓝 0) := tendsto_pi_nhds.mpr fun i => by
      apply (Filter.tendsto_add_atTop_iff_nat N).1
      have hs := (((Filter.tendsto_add_atTop_iff_nat 1).2 (hlimit i)).sub (hlimit i))
      change Tendsto (fun n => G i (n + 1) - G i n) atTop (𝓝 (limit i - limit i)) at hs
      have he (n : ℕ) : G i (n + 1) - G i n = g (n + N) i := by
        simp only [G, Finset.sum_range_succ, add_sub_cancel_left, Nat.add_comm N n]
      simpa only [sub_self, Pi.zero_apply, he] using hs
    have hv := velocity_tendsto_zero_of_clippedGradient rate gradient initial hzero
    refine ⟨fun i => C i - 10 * η * limit i, tendsto_pi_nhds.mpr fun i => ?_⟩
    apply (Filter.tendsto_add_atTop_iff_nat N).1
    have hz : Tendsto (fun n => 9 * η * (s (N + n)).velocity i) atTop (𝓝 0) := by
      simpa only [Function.comp_def, Pi.zero_apply, mul_zero, s] using
        ((tendsto_pi_nhds.mp hv i).comp hshift).const_mul (9 * η)
    have hp := (((hlimit i).const_mul (10 * η)).const_sub (C i)).add hz
    rw [add_zero] at hp
    apply hp.congr
    intro n
    change (C i - 10 * η * G i n) + 9 * η * (s (N + n)).velocity i = (s (n + N)).weights i
    rw [Nat.add_comm n N]
    linarith [htel i n]

example (gradient : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  weights_tendsto_iff_tail_gradient_sum (fun _ => 1 / 1000) gradient initial 10
    (by norm_num) (fun _ _ => rfl)

end Transformer.Optimization.ClippedMomentum
