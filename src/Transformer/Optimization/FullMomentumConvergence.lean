module

public import Transformer.Optimization.FullMomentumLimits
public import Transformer.Landscape.SaddleAvoidance.Sequences
public import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

/-! Finite length for clipped momentum once the full gradient is small.
The Łojasiewicz estimate is required only near a stationary cluster point
of the unclipped energy. This is the sufficient-decrease argument of
Attouch, Bolte and Svaiter (2013), Theorem 2.9, using the trapping and
power-drop estimates in Landscape.SaddleAvoidance.Sequences. -/

@[expose] public section

open Filter Metric
open scoped Topology

namespace Transformer.Optimization

open Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E]

private theorem tendsto_of_local_length_bound {x : ℕ → E} {star : E}
    {u : ℕ → ℕ} (hu : StrictMono u) (hsub : Tendsto (x ∘ u) atTop (𝓝 star))
    {φ : ℕ → ℝ} (hφ : ∀ t, 0 ≤ φ t) (hlim : Tendsto φ atTop (𝓝 0))
    {r c : ℝ} (hr : 0 < r) (hc : 0 < c)
    (hstep : ∀ᶠ t in atTop, x t ∈ ball star r →
      c * ‖x (t + 1) - x t‖ ≤ φ t - φ (t + 1)) :
    Tendsto x atTop (𝓝 star) := by
  apply Metric.tendsto_atTop.mpr
  intro ε hε
  let s := min r ε
  have hs : 0 < s := lt_min hr hε
  obtain ⟨N, hN⟩ := eventually_atTop.mp hstep
  have hdist : Tendsto (fun t => ‖x (u t) - star‖) atTop (𝓝 0) := by
    simpa only [Function.comp_def, sub_self, norm_zero] using (hsub.sub_const star).norm
  have hsmall₁ := hdist.eventually (gt_mem_nhds (half_pos hs))
  have hsmall₂ := (hlim.comp hu.tendsto_atTop).eventually
    (gt_mem_nhds (mul_pos hc (half_pos hs)))
  obtain ⟨j, hj₁, hj₂, hjN⟩ := (hsmall₁.and (hsmall₂.and
    (hu.tendsto_atTop.eventually (eventually_ge_atTop N)))).exists
  simp only [Function.comp_apply] at hj₂
  let k := u j
  let y (n : ℕ) := x (k + n)
  let ψ (n : ℕ) := φ (k + n)
  have hin : c * ‖y 0 - star‖ + ψ 0 < c * s := by
    have h := mul_lt_mul_of_pos_left hj₁ hc
    simpa only [y, ψ, Nat.add_zero] using (show c * ‖x (u j) - star‖ + φ (u j) < c * s by
      linarith only [h, hj₂])
  have hloc (n : ℕ) (hn : y n ∈ ball star s) :
      c * ‖y (n + 1) - y n‖ ≤ ψ n - ψ (n + 1) := by
    have hn' : x (k + n) ∈ ball star r := ball_subset_ball (min_le_left _ _) hn
    simpa only [y, ψ, Nat.add_assoc] using hN (k + n) (by omega : N ≤ k + n) hn'
  refine ⟨k, fun n hn => ?_⟩
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hn
  exact (mem_ball.mp (mem_ball_of_step hc (fun _ => hφ _) hloc hin j)).trans_le (min_le_right _ _)

private theorem mul_step_le_energy_power_drop {s t d g m a c : ℝ}
    (hs : 0 < s) (ht : 0 ≤ t) (hts : t ≤ s)
    (hm : 0 < m) (ha0 : 0 ≤ a) (ha1 : a < 1)
    (hdec : c * d * g ≤ s - t) (hKL : m * s ^ a ≤ g) :
    m * (1 - a) * c * d ≤ s ^ (1 - a) - t ^ (1 - a) := by
  have hg : 0 < g := (mul_pos hm (Real.rpow_pos_of_pos hs a)).trans_le hKL
  have hconc := mul_sub_le_rpow_mul_sub ha0 ha1.le hs.le ht
  have hdiff : 0 ≤ s ^ (1 - a) - t ^ (1 - a) :=
    sub_nonneg.mpr (Real.rpow_le_rpow ht hts (by linarith))
  apply le_of_mul_le_mul_left _ hg
  calc
    g * (m * (1 - a) * c * d) = m * ((1 - a) * (c * d * g)) := by ring
    _ ≤ m * ((1 - a) * (s - t)) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hdec (by linarith)) hm.le
    _ ≤ m * (s ^ a * (s ^ (1 - a) - t ^ (1 - a))) := mul_le_mul_of_nonneg_left hconc hm.le
    _ = m * s ^ a * (s ^ (1 - a) - t ^ (1 - a)) := by ring
    _ ≤ g * (s ^ (1 - a) - t ^ (1 - a)) := mul_le_mul_of_nonneg_right hKL hdiff

variable [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The unclipped objective and kinetic energy near a stationary cluster
point. This is a local gradient inequality for the energy on the product
of the weight and velocity spaces, in terms of the two gradient norms. -/
def MomentumEnergyInequality (f : E → ℝ) (η : ℝ) (star : E) : Prop :=
  ∃ r > 0, ∃ m > 0, ∃ a : ℝ, 0 ≤ a ∧ a < 1 ∧
    ∀ w ∈ ball star r, ∀ v : E, ‖v‖ < r →
      0 < f w + η / 2 * ‖v‖ ^ 2 - f star →
      m * (f w + η / 2 * ‖v‖ ^ 2 - f star) ^ a ≤ ‖gradient f w‖ + ‖v‖

/-- A local gradient inequality for the energy upgrades a stationary
subsequence to convergence of the entire weight sequence. -/
theorem fullMomentum_tendsto_of_energy_inequality {f : E → ℝ} (hf : Continuous f)
    (hnonneg : ∀ x, 0 ≤ f x) {η : ℝ} (hη : 0 < η) (x v : ℕ → E)
    (hvel : ∀ t, v (t + 1) = (9 / 10 : ℝ) • v t +
      (clipScale (gradient f (x t)))⁻¹ • gradient f (x t))
    (hpos : ∀ t, x (t + 1) = x t - η • v (t + 1))
    (hdec : ∀ t, fullMomentumEnergy f η (x (t + 1)) (v (t + 1)) +
      η / 40 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) ≤ fullMomentumEnergy f η (x t) (v t))
    {star : E} {u : ℕ → ℕ} (hu : StrictMono u) (hsub : Tendsto (x ∘ u) atTop (𝓝 star))
    (hKL : MomentumEnergyInequality f η star) :
    Tendsto x atTop (𝓝 star) := by
  obtain ⟨value, hvalue, hv, _, hclip⟩ := fullMomentum_limits hnonneg hη x v hvel hdec
  have hvalue' := tendsto_nhds_unique (hvalue.comp hu.tendsto_atTop) ((hf.tendsto star).comp hsub)
  subst value
  let e (t : ℕ) := fullMomentumEnergy f η (x t) (v t)
  have hvn : Tendsto (fun t => ‖v t‖) atTop (𝓝 0) := by simpa only [norm_zero] using hv.norm
  have hk : Tendsto (fun t => η / 2 * ‖v t‖ ^ 2) atTop (𝓝 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), mul_zero] using (hvn.pow 2).const_mul (η / 2)
  have he : Tendsto e atTop (𝓝 (f star)) := by
    have hh : Tendsto (fun t => f (x t) + η / 2 * ‖v t‖ ^ 2) atTop (𝓝 (f star)) := by
      simpa only [add_zero] using hvalue.add hk
    exact hh.congr' (hclip.mono fun t ht => by simp [e, fullMomentumEnergy, ht])
  have ha : Antitone e := antitone_nat_of_succ_le fun t =>
    (le_add_of_nonneg_right (by positivity)).trans (hdec t)
  have he₀ (t : ℕ) : f star ≤ e t :=
    le_of_tendsto he ((eventually_ge_atTop t).mono fun _ hn => ha hn)
  obtain ⟨r, hr, m, hm, a, ha₀, ha₁, hloc⟩ := hKL
  let φ (t : ℕ) := (e t - f star) ^ (1 - a)
  have hφ₀ (t : ℕ) : 0 ≤ φ t := Real.rpow_nonneg (sub_nonneg.mpr (he₀ t)) (1 - a)
  have hφ : Tendsto φ atTop (𝓝 0) := by
    have hp : 0 < 1 - a := sub_pos.mpr ha₁
    have h : Tendsto (fun t => e t - f star) atTop (𝓝 0) := by
      simpa only [sub_self] using he.sub_const (f star)
    exact h.rpow_const_nhds_zero hp
  apply tendsto_of_local_length_bound hu hsub hφ₀ hφ hr
    (show 0 < m * (1 - a) * (1 / 80) by positivity)
  filter_upwards [hclip, hvn.eventually (gt_mem_nhds hr)] with t hcl hsmall hball
  have hx : ‖x (t + 1) - x t‖ = η * ‖v (t + 1)‖ := by
    rw [hpos t, sub_sub_cancel_left, norm_neg, norm_smul, Real.norm_of_nonneg hη.le]
  have hΔ : η / 40 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) ≤ e t - e (t + 1) := by
    dsimp only [e]
    linarith only [hdec t]
  rcases (he₀ t).eq_or_lt with hz | hposgap
  · have hsum : ‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2 ≤ 0 :=
      nonpos_of_mul_nonpos_right (hΔ.trans (sub_nonpos.mpr ((le_of_eq hz.symm).trans (he₀ (t + 1)))))
        (by positivity)
    have hvzero : ‖v (t + 1)‖ = 0 := by nlinarith [sq_nonneg ‖v t‖, norm_nonneg (v (t + 1))]
    have hnext : e (t + 1) = f star := le_antisymm ((ha (Nat.le_succ t)).trans_eq hz.symm) (he₀ _)
    simp [φ, ← hz, hnext, hx, hvzero]
  have hg_eq : gradient f (x t) = v (t + 1) - (9 / 10 : ℝ) • v t := by
    simpa only [hcl, inv_one, one_smul, add_sub_cancel_left] using congrArg
      (fun z => z - (9 / 10 : ℝ) • v t) (hvel t).symm
  have hg : ‖gradient f (x t)‖ + ‖v t‖ ≤ ‖v (t + 1)‖ + (19 / 10 : ℝ) * ‖v t‖ := by
    rw [hg_eq]
    have h := norm_sub_le (v (t + 1)) ((9 / 10 : ℝ) • v t)
    rw [norm_smul, Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10)] at h
    linarith
  have hprod : (1 / 80 : ℝ) * ‖x (t + 1) - x t‖ *
      (‖gradient f (x t)‖ + ‖v t‖) ≤ e t - e (t + 1) := by
    rw [hx]
    have h₁ := mul_le_mul_of_nonneg_left hg (show 0 ≤ (1 / 80 : ℝ) * (η * ‖v (t + 1)‖) by positivity)
    have h₂ : ‖v (t + 1)‖ * (‖v (t + 1)‖ + (19 / 10 : ℝ) * ‖v t‖) ≤
        2 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) := by
      nlinarith [sq_nonneg (‖v t‖ - ‖v (t + 1)‖), sq_nonneg ‖v t‖, sq_nonneg ‖v (t + 1)‖]
    have h₃ := mul_le_mul_of_nonneg_left h₂ (show 0 ≤ η / 80 by positivity)
    nlinarith only [h₁, h₃, hΔ]
  have hkl := hloc (x t) hball (v t) hsmall
  have heclip : e t = f (x t) + η / 2 * ‖v t‖ ^ 2 := by simp [e, fullMomentumEnergy, hcl]
  rw [← heclip] at hkl
  apply mul_step_le_energy_power_drop (g := ‖gradient f (x t)‖ + ‖v t‖) (sub_pos.mpr hposgap) (sub_nonneg.mpr (he₀ _))
    (sub_le_sub_right (ha (Nat.le_succ t)) _) hm ha₀ ha₁
  · linarith only [hprod]
  · exact hkl (sub_pos.mpr hposgap)

end Transformer.Optimization
