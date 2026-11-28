module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Handoff

/-! The training tail. The Gronwall factor of `coupling_le` grows without
bound in the horizon, so it controls no infinite tail. The estimate that
passes beyond a finite horizon H is a contraction of the coupled
distance, coupling(t+1) ≤ a · coupling(t) + b · fluctuation(t) with
a < 1, together with run-wise bounds Z on the fluctuations and S on the
segment errors over all later updates (a law of large numbers uniform in
time). Under them the finite residual after H stays within an explicit
allowance of the limiting bound. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

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

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
