module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer
public import Transformer.Grokking.FiniteRecovery.ObservedDrift

/-! The training tail and the width transfer inside the recovery theorem.
The Gronwall factor of `coupling_le` grows without bound in the horizon,
so it controls no infinite tail. The estimate that passes beyond a finite
horizon H is a contraction of the coupled distance,
coupling(t+1) ≤ a · coupling(t) + b · fluctuation(t) with a < 1, together
with run-wise bounds Z on the fluctuations and S on the segment errors
over all later updates (a law of large numbers uniform in time). Under
them the finite residual after H stays within an explicit allowance of
the limiting bound. With such an allowance, a bound r̄ on the limiting
residual over B ≤ t ≤ H gives permanent recovery with probability above q
as soon as max(r̄, 0) + widthError + E[tail] < (1 - q)(1 - ρ)(1/2 - r). The
error term `widthError` is what the width limit adds to a bound on the
limiting drift. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset Filter Transformer.Grokking.FiniteRecovery.ObservedConstraints

variable {P K J : Type*}


/-- A contracting recursion stays within D(H) + b Z/(1 - a) after H. -/
theorem le_of_contraction {D ζ : ℕ → ℝ} {a b Z : ℝ} {H : ℕ} (ha0 : 0 ≤ a) (ha1 : a < 1)
    (hb : 0 ≤ b) (hZ : 0 ≤ Z) (hD : 0 ≤ D H) (hrec : ∀ t, H ≤ t → D (t+1) ≤ a * D t + b * ζ t)
    (hζ : ∀ t, H ≤ t → ζ t ≤ Z) : ∀ t, H ≤ t → D t ≤ D H + b * Z / (1 - a) := by
  have h1 : 0 < 1 - a := by linarith
  have hq : 0 ≤ b * Z / (1 - a) := div_nonneg (mul_nonneg hb hZ) h1.le
  intro t ht
  induction t, ht using Nat.le_induction with
  | base => linarith
  | succ t ht ih =>
    have e : a * (b * Z / (1 - a)) + b * Z = b * Z / (1 - a) := by field_simp; ring
    have h2 : a * D t ≤ a * (D H + b * Z / (1 - a)) := mul_le_mul_of_nonneg_left ih ha0
    have h3 : b * ζ t ≤ b * Z := mul_le_mul_of_nonneg_left (hζ t ht) hb
    have h4 : a * D H ≤ D H := mul_le_of_le_one_left hD ha1.le
    calc D (t+1) ≤ a * D t + b * ζ t := hrec t ht
      _ ≤ a * (D H + b * Z / (1 - a)) + b * Z := add_le_add h2 h3
      _ ≤ _ := by nlinarith

theorem coupling_nonneg [Fintype P] (A : Training P K) (μ : Measure (P → ℝ)) {N : ℕ}
    (init : Fin N → P → ℝ) (t : ℕ) : 0 ≤ coupling A μ init t :=
  div_nonneg (sum_nonneg fun _ _ => gap_nonneg _ _) (Nat.cast_nonneg _)

/-- The tail estimate: after H the finite residual is at most the limiting
bound plus LΓ (2 Lχ (coupling(H) + b Z/(1 - a)) + S). -/
theorem residual_le_tail [Fintype P] [Fintype K] [Fintype J] (A : Training P K)
    (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ}
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) (z : Fin N → P → ℝ)
    {H : ℕ} {a b r Z S : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (hb : 0 ≤ b)
    (hcontr : ∀ t, H ≤ t → coupling A μ z (t+1) ≤ a * coupling A μ z t + b * fluctuation A μ z t)
    (hZ : ∀ t, H ≤ t → fluctuation A μ z t ≤ Z)
    (hS : ∀ t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, segmentError A μ χ z t u ≤ S)
    (hlim : ∀ t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ Γ χ t u ≤ r) :
    ∀ t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1,
      residual A Γ χ z t u ≤ r + LΓ * (2 * Lχ * (coupling A μ z H + b * Z / (1 - a)) + S) := by
  intro t ht u hu
  have hZ0 : 0 ≤ Z := (l1_nonneg _).trans (hZ H le_rfl)
  have hc := le_of_contraction ha0 ha1 hb hZ0 (coupling_nonneg A μ z H) hcontr hZ
  have h1 := hc t ht.le
  have h2 := hc (t+1) (by omega)
  have h3 : Lχ * (coupling A μ z t + coupling A μ z (t+1)) ≤
      2 * Lχ * (coupling A μ z H + b * Z / (1 - a)) := by nlinarith [h.Lχ_nonneg]
  have h4 := hS t ht u hu
  have h5 : LΓ * (Lχ * (coupling A μ z t + coupling A μ z (t+1)) + segmentError A μ χ z t u) ≤
      LΓ * (2 * Lχ * (coupling A μ z H + b * Z / (1 - a)) + S) :=
    mul_le_mul_of_nonneg_left (by linarith) h.LΓ_nonneg
  linarith [residual_sub_le A μ h hN z t u, hlim t ht u hu]

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

example := le_of_contraction (D := fun _ => 0) (ζ := fun _ => 0) (a := 1/2) (b := 1) (Z := 0)
  (H := 0) (by norm_num) (by norm_num) zero_le_one le_rfl le_rfl (fun _ _ => by norm_num)
  (fun _ _ => le_rfl)
example := coupling_nonneg identityTraining (MeasureTheory.Measure.dirac 0) (N := 2) (fun _ => 0) 3

theorem identity_system_zero {N : ℕ} (t : ℕ) (i : Fin N) :
    system identityTraining (fun _ => 0) t i = ⟨0, 0, 0⟩ := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [system_succ, ih]
    simp [step, direction, identityTraining]
    rfl

theorem identity_flow_zero (μ : Measure (Fin 1 → ℝ)) (t : ℕ) :
    flow identityTraining μ t 0 = ⟨0, 0, 0⟩ := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [flow_succ, ih]
    simp [step, direction, identityTraining]
    rfl

theorem identity_coupling_zero {N : ℕ} (μ : Measure (Fin 1 → ℝ)) (t : ℕ) :
    coupling identityTraining μ (N := N) (fun _ => 0) t = 0 := by
  simp [coupling, identity_system_zero, identity_flow_zero, gap]

example := residual_le_tail identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (fun _ => 0) (H := 3) (a := 1/2) (b := 1) (r := 0) (Z := 0) (S := 0) (by norm_num) (by norm_num)
  zero_le_one (fun t _ => by
    rw [identity_coupling_zero, identity_coupling_zero]
    simp [fluctuation, meanField, limitField, identityTraining, l1])
  (fun _ _ => by simp [fluctuation, meanField, limitField, identityTraining, l1])
  (fun _ _ _ _ => by simp [segmentError, meanField, limitFeatures, l1])
  (fun _ _ _ _ => by simp [limitResidual])


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
