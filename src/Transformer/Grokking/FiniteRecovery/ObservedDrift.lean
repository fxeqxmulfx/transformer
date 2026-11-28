module

public import Transformer.Grokking.FiniteRecovery.ObservedConstraints
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Analysis.SpecificLimits.Basic
public import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-! One uninterrupted trajectory. The slope is evaluated on the actual
weight update, using a score risk defined by the observations and the
structural rules. Markov's inequality is over initializations or complete
training executions. Optimizer steps need not be independent. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.ObservedConstraints

open MeasureTheory Set Filter
open scoped Topology

variable {D W : Type*} [MeasurableSpace D] [AddCommGroup W] [Module ℝ W]

/-- Restrict the observable risk to the segment traversed by one update. -/
def segmentRisk (V : W → ℝ) (w Δ : W) (u : ℝ) := V (w + u • Δ)

/-- The mean value theorem converts a signed derivative bound along the
actual displacement into an upper bound on the change of risk. -/
theorem segment_step (V : W → ℝ) (w Δ : W) (b : ℝ)
    (hc : ContinuousOn (segmentRisk V w Δ) (Icc 0 1))
    (hd : DifferentiableOn ℝ (segmentRisk V w Δ) (Ioo 0 1))
    (slope : ∀ u ∈ Ioo 0 1, deriv (segmentRisk V w Δ) u ≤ b) :
    V (w+Δ) - V w ≤ b := by
  obtain ⟨u,hu,he⟩ := exists_deriv_eq_slope _ (by norm_num : (0:ℝ)<1) hc hd
  have h := slope u hu
  rw [he] at h
  simpa [segmentRisk] using h

/-- Finite-time estimate with an additive error allowed at every step. -/
theorem drift_envelope (v : ℕ → ℝ) (B : ℕ) {ρ r e : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1)
    (steps : ∀ t ≥ B, v (t+1) ≤ ρ*v t+(1-ρ)*r+e) (k : ℕ) :
    let c := r + e/(1-ρ)
    v (B+k) ≤ c + ρ^k*(v B-c) := by
  intro c
  have equilibrium : ρ*c+(1-ρ)*r+e = c := by
    dsimp [c]
    field_simp [sub_ne_zero.mpr hlt.ne']; ring
  induction k with
  | zero => simp
  | succ k ih =>
    calc
      v (B+k+1) ≤ ρ*v (B+k)+(1-ρ)*r+e := steps _ (Nat.le_add_right _ _)
      _ ≤ ρ*(c+ρ^k*(v B-c))+(1-ρ)*r+e := by gcongr
      _ = c+ρ^(k+1)*(v B-c) := by rw [pow_succ]; linarith only [equilibrium]

/-- A small mean bound on the excess slope gives a high-probability set
of starts with eventually permanent, simultaneous strict recovery.
Both r and the excess concern observable risk. The target enters only
in the identification proof for the risk threshold. -/
theorem permanent_of_observed_drift {X A J : Type*} [Fintype X] [Fintype A] [Nonempty A]
    [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    (μ : Measure D) [IsProbabilityMeasure μ] (score : W → X → A → ℝ)
    (S : Finset X) (rule : J → Set (X → A)) (f : X → A) (hf : ∀ j, f ∈ rule j)
    (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (V : W → ℝ) (above : ∀ v, risk (tableLaw (score v)) S (fun x => f x) rule ≤ V v)
    (w Δ : D → ℕ → W) (update : ∀ z t, w z (t+1) = w z t + Δ z t)
    (B : ℕ) {ρ r q : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (excess : D → ℝ) (hmeas : Measurable excess) (hint : Integrable excess μ)
    (hpos : ∀ᵐ z ∂μ, 0 ≤ excess z)
    (small : ∫ z, excess z ∂μ < (1-q)*((1-ρ)*(1/2-r))) :
    (∀ z t, B ≤ t →
      let line := segmentRisk V (w z t) (Δ z t)
      ContinuousOn line (Icc 0 1) ∧ DifferentiableOn ℝ line (Ioo 0 1) ∧
      ∀ u ∈ Ioo 0 1, deriv line u ≤ -(1-ρ)*(V (w z t)-r)+excess z) →
    ∃ G : Set D, MeasurableSet G ∧ q < μ.real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (w z t)) f := by
  intro localSlope
  let a := (1-ρ)*(1/2-r)
  have ha : 0 < a := mul_pos (sub_pos.mpr hlt) (sub_pos.mpr hr)
  let bad := {z | a ≤ excess z}
  have measurableBad : MeasurableSet bad := measurableSet_le measurable_const hmeas
  have mark : a * μ.real bad < (1-q)*a :=
    (mul_meas_ge_le_integral_of_nonneg hpos hint a).trans_lt small
  have badProb : μ.real bad < 1-q := by
    rw [mul_comm] at mark
    exact (mul_lt_mul_iff_left₀ ha).mp mark
  refine ⟨badᶜ,measurableBad.compl,?_,?_⟩
  · rw [measureReal_compl measurableBad,probReal_univ]
    linarith
  intro z hz
  have ez : excess z < a := lt_of_not_ge hz
  let c := r+excess z/(1-ρ)
  have low : c < 1/2 := by
    have he : excess z/(1-ρ) < 1/2-r := (div_lt_iff₀ (sub_pos.mpr hlt)).mpr (by simpa [a,mul_comm] using ez)
    dsimp [c]; linarith
  have steps t (ht : B ≤ t) : V (w z (t+1)) ≤ ρ*V (w z t)+(1-ρ)*r+excess z := by
    obtain ⟨hc,hd,hs⟩ := localSlope z t ht
    have he := segment_step V (w z t) (Δ z t) _ hc hd hs
    rw [update]
    linarith only [he]
  have limit : Tendsto (fun k : ℕ => c+ρ^k*(V (w z B)-c)) atTop (𝓝 c) := by
    simpa using tendsto_const_nhds (x := c) |>.add
      ((tendsto_pow_atTop_nhds_zero_of_lt_one hρ hlt).mul_const (V (w z B)-c))
  obtain ⟨n,hn⟩ := eventually_atTop.mp ((tendsto_order.mp limit).2 _ low)
  refine eventually_atTop.mpr ⟨B+n,fun t ht => ?_⟩
  have hk : n ≤ t-B := by omega
  have later := (drift_envelope (fun k => V (w z k)) B hρ hlt steps (t-B)).trans_lt (hn _ hk)
  rw [show B+(t-B)=t by omega] at later
  exact strict_of_risk _ S rule id f hf ((above _).trans_lt later)

end Transformer.Grokking.FiniteRecovery.ObservedConstraints
