module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Handoff
public import Transformer.Grokking.FiniteRecovery.ObservedDrift

/-! The width transfer inside the recovery theorem. The finite slope along
each actual update is bounded by the transferred residual; a bound r̄ on
the limiting residual over B ≤ t ≤ H and an allowance for the tail after
H then give permanent recovery with probability above q as soon as
max(r̄, 0) + widthError + E[tail] < (1 - q)(1 - ρ)(1/2 - r). The error term
`widthError` is what the width limit adds to a bound on the limiting drift. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset Filter Transformer.Grokking.FiniteRecovery.ObservedConstraints

variable {P K J : Type*}

/-- The weights of the N-particle system after t steps, as one point. -/
noncomputable def weightsAt (A : Training P K) {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) :
    Fin N → P → ℝ :=
  fun i => (system A init t i).weights

/-- Permanent recovery from a bound on the limiting drift plus the width error. -/
theorem permanent_of_width_transfer {X Act R : Type*} [Fintype X] [Fintype Act] [Nonempty Act]
    [MeasurableSpace Act] [MeasurableSingletonClass Act] [Fintype R] [Fintype P] [Fintype K]
    [Fintype J] (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) {B H n : ℕ} (hn : 0 < n)
    (score : (Fin N → P → ℝ) → X → Act → ℝ) (S : Finset X) (rule : R → Set (X → Act))
    (f : X → Act) (hf : ∀ j, f ∈ rule j)
    (hid : Set.InjOn (fun g : X → Act => fun x : S => g x) (⋂ j, rule j))
    (V : (Fin N → P → ℝ) → ℝ) (above : ∀ v, risk (tableLaw (score v)) S (fun x => f x) rule ≤ V v)
    {ρ r q rbar : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ Γ χ t u ≤ rbar)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A Γ χ z t u ≤ tail z)
    (small : max rbar 0 +
        widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n +
        ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) < (1-q)*((1-ρ)*(1/2-r)))
    (slope : ∀ z t, B ≤ t →
      let line := segmentRisk V (weightsAt A z t) (weightsAt A z (t+1) - weightsAt A z t)
      ContinuousOn line (Set.Icc 0 1) ∧ DifferentiableOn ℝ line (Set.Ioo 0 1) ∧
      ∀ u ∈ Set.Ioo (0:ℝ) 1, deriv line u + (1-ρ)*(V (weightsAt A z t) - r) ≤ residual A Γ χ z t u) :
    ∃ G : Set (Fin N → P → ℝ), MeasurableSet G ∧ q < (Measure.pi fun _ : Fin N => μ).real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (weightsAt A z t)) f := by
  obtain ⟨hm, hi, hpos, hint, hle⟩ := drift_excess A μ h hN hn hlim tail htm hti htn htail
  refine permanent_of_observed_drift (Measure.pi fun _ : Fin N => μ) score S rule f hf hid V above
    (weightsAt A) (fun z t => weightsAt A z (t+1) - weightsAt A z t) (fun z t => by abel) B hρ hlt hr
    _ hm hi (ae_of_all _ hpos) (hint.trans_lt small) ?_
  intro z t ht
  obtain ⟨hc, hd, hs⟩ := slope z t ht
  exact ⟨hc, hd, fun u hu => by linarith [hs u hu, hle z t ht u hu]⟩

section Examples

example := permanent_of_width_transfer identityTraining (Measure.dirac 0) identityRegular
  (N := 2) two_pos (B := 1) (H := 3) (n := 4) (by norm_num)
  (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅ (fun _ : Fin 1 => Set.univ) (fun _ => 0)
  (fun _ => Set.mem_univ _) (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => 0)
  (fun _ => by simp [risk]) (ρ := 0) (r := 0) (q := 0) (rbar := 0) le_rfl one_pos (by norm_num)
  (fun _ _ _ _ => by simp [limitResidual]) (fun _ => 0) measurable_const (integrable_const _)
  (fun _ => le_rfl) (fun _ _ _ _ _ => by simp [residual]) (by norm_num [widthError])
  (fun _ _ _ => ⟨continuousOn_const, differentiableOn_const _, fun _ _ => by
    unfold segmentRisk
    simp [residual]⟩)

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
