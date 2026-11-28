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

/-- The recovery theorem with the drift bound required only for starts in a
measurable set E. The recovered set lies inside E, and the starts outside E
are lost from the probability. -/
theorem permanent_of_observed_drift_on {X A J : Type*} [Fintype X] [Fintype A] [Nonempty A]
    [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    (μ : Measure D) [IsProbabilityMeasure μ] (score : W → X → A → ℝ)
    (S : Finset X) (rule : J → Set (X → A)) (f : X → A) (hf : ∀ j, f ∈ rule j)
    (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (V : W → ℝ) (above : ∀ v, risk (tableLaw (score v)) S (fun x => f x) rule ≤ V v)
    (w Δ : D → ℕ → W) (update : ∀ z t, w z (t+1) = w z t + Δ z t)
    (B : ℕ) {ρ r q : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (excess : D → ℝ) (hmeas : Measurable excess) (hint : Integrable excess μ)
    (hpos : ∀ᵐ z ∂μ, 0 ≤ excess z)
    (small : ∫ z, excess z ∂μ < (1-q)*((1-ρ)*(1/2-r))) {E : Set D} (hE : MeasurableSet E) :
    (∀ z ∈ E, ∀ t, B ≤ t →
      let line := segmentRisk V (w z t) (Δ z t)
      ContinuousOn line (Icc 0 1) ∧ DifferentiableOn ℝ line (Ioo 0 1) ∧
      ∀ u ∈ Ioo 0 1, deriv line u ≤ -(1-ρ)*(V (w z t)-r)+excess z) →
    ∃ G ⊆ E, MeasurableSet G ∧ q - μ.real Eᶜ < μ.real G ∧
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
  refine ⟨E ∩ badᶜ,inter_subset_left,hE.inter measurableBad.compl,?_,?_⟩
  · have hdiff : μ.real E - μ.real bad ≤ μ.real (E ∩ badᶜ) := le_measureReal_sdiff
    rw [measureReal_compl hE,probReal_univ] at *
    linarith
  intro z ⟨hzE,hz⟩
  have ez : excess z < a := lt_of_not_ge hz
  let c := r+excess z/(1-ρ)
  have low : c < 1/2 := by
    have he : excess z/(1-ρ) < 1/2-r := (div_lt_iff₀ (sub_pos.mpr hlt)).mpr (by simpa [a,mul_comm] using ez)
    dsimp [c]; linarith
  have steps t (ht : B ≤ t) : V (w z (t+1)) ≤ ρ*V (w z t)+(1-ρ)*r+excess z := by
    obtain ⟨hc,hd,hs⟩ := localSlope z hzE t ht
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
  obtain ⟨G,-,hG,hq,hs⟩ := permanent_of_observed_drift_on μ score S rule f hf id V above w Δ update
    B hρ hlt hr excess hmeas hint hpos small MeasurableSet.univ fun z _ => localSlope z
  exact ⟨G,hG,by simpa using hq,hs⟩

section Examples

example := permanent_of_observed_drift_on (D := ℝ) (W := ℝ) (Measure.dirac 0)
  (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅ (fun _ : Fin 1 => Set.univ) (fun _ => 0)
  (fun _ => Set.mem_univ _) (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => 0)
  (fun _ => by simp [risk]) (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => by simp) 0 (ρ := 0) (r := 0)
  (q := 0) le_rfl one_pos (by norm_num) (fun _ => 0) measurable_const (integrable_const _)
  (ae_of_all _ fun _ => le_rfl) (by norm_num) (E := Set.Iic 0) measurableSet_Iic
  (fun _ _ _ _ => ⟨continuousOn_const, differentiableOn_const _, fun _ _ => by
    unfold segmentRisk
    simp⟩)

example := permanent_of_observed_drift (D := ℝ) (W := ℝ) (Measure.dirac 0)
  (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅ (fun _ : Fin 1 => Set.univ) (fun _ => 0)
  (fun _ => Set.mem_univ _) (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => 0)
  (fun _ => by simp [risk]) (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => by simp) 0 (ρ := 0) (r := 0)
  (q := 0) le_rfl one_pos (by norm_num) (fun _ => 0) measurable_const (integrable_const _)
  (ae_of_all _ fun _ => le_rfl) (by norm_num)
  (fun _ _ _ => ⟨continuousOn_const, differentiableOn_const _, fun _ _ => by
    unfold segmentRisk
    simp⟩)

end Examples

end Transformer.Grokking.FiniteRecovery.ObservedConstraints
