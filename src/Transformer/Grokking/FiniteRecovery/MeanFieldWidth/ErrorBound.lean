module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Grid

/-! The explicit finite-width error. One measurable majorant bounds the
residual gap at every update B ≤ t ≤ H and every fraction of the segment.
Its expectation under N independent initial draws is `widthError`: width
enters through 1/√N, the horizon through the Gronwall factor growth^(H+1)
and the number of updates, the grid of fractions through n + 1 and 1/n. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

variable {P K J : Type*}

/-- The law of large numbers for finitely many bounded measurable features, in ℓ¹. -/
theorem integral_l1_meanField_sub_le [Fintype J] {α : Type*} [MeasurableSpace α] (μ : Measure α)
    [IsProbabilityMeasure μ] {N : ℕ} (hN : 0 < N) {Ξ : α → J → ℝ}
    (hΞ : ∀ j, Measurable fun x => Ξ x j) {C : ℝ} (hC : ∀ x j, |Ξ x j| ≤ C) :
    Integrable (fun z => l1 (meanField Ξ z - fun j => ∫ x, Ξ x j ∂μ)) (Measure.pi fun _ : Fin N => μ) ∧
      ∫ z, l1 (meanField Ξ z - fun j => ∫ x, Ξ x j ∂μ) ∂(Measure.pi fun _ : Fin N => μ) ≤
        Fintype.card J * C / Real.sqrt N := by
  have hi j := integrable_abs_empirical_sub μ N (hΞ j) fun x => hC x j
  refine ⟨integrable_finsetSum _ fun j _ => hi j, ?_⟩
  change ∫ z, ∑ j, |(∑ i, Ξ (z i) j) / N - ∫ x, Ξ x j ∂μ| ∂(Measure.pi fun _ : Fin N => μ) ≤ _
  rw [integral_finsetSum _ fun j _ => hi j]
  calc _ ≤ ∑ _j : J, C / Real.sqrt N :=
        sum_le_sum fun j _ => integral_abs_empirical_sub_le μ hN (hΞ j) fun x => hC x j
    _ = _ := by simp [mul_div_assoc]

/-- The explicit error bound for k field coordinates, j segment features,
width N, updates B ≤ t ≤ H and n + 1 grid fractions. -/
noncomputable def widthError (eps η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ) (k j N B H n : ℕ) : ℝ :=
  LΓ * (2 * Lχ * (growth eps η LG Lφ ^ (H+1) * forcing eps η LG * ((H+1) * (k * Cφ / Real.sqrt N))) +
    (H + 1 - B : ℕ) * ((n+1) * (j * Cχ / Real.sqrt N)) + 2 * Mχ / n)

/-- A pathwise majorant of the residual gap over B ≤ t ≤ H and all fractions. -/
noncomputable def majorant [Fintype K] [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
    (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ) (η LG Lφ LΓ Lχ Mχ : ℝ) {N : ℕ} (init : Fin N → P → ℝ)
    (B H n : ℕ) : ℝ :=
  LΓ * (2 * Lχ * (growth A.eps η LG Lφ ^ (H+1) * forcing A.eps η LG *
      ∑ s ∈ range (H+1), fluctuation A μ init s) +
    ∑ t ∈ Icc B H, ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n) + 2 * Mχ / n)

section Bound

variable [Fintype P] [Fintype K] [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
  [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ}
  {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}

theorem Regular.gronwall_nonneg (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (H : ℕ) :
    0 ≤ growth A.eps η LG Lφ ^ H * forcing A.eps η LG := by
  have hη : 0 ≤ η := (h.rate 0).1.trans (h.rate 0).2
  have hq : 0 ≤ 12 * η / A.eps := div_nonneg (by linarith) h.eps_pos.le
  have hl : 0 ≤ LG * (1 + Lφ) := mul_nonneg h.LG_nonneg (by linarith [h.Lφ_nonneg])
  exact mul_nonneg (pow_nonneg (mul_nonneg (by linarith) (by linarith)) _)
    (mul_nonneg (by linarith) h.LG_nonneg)

/-- The residual gap at every update B ≤ t ≤ H and every fraction 0 < u < 1
is below one majorant of the run. -/
theorem residual_sub_le_majorant (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N)
    (init : Fin N → P → ℝ) {B H n : ℕ} (hn : 0 < n) {t : ℕ} (ht : t ∈ Icc B H) {u : ℝ}
    (hu : u ∈ Set.Ioo (0:ℝ) 1) :
    residual A Γ χ init t u - limitResidual A μ Γ χ t u ≤ majorant A μ χ η LG Lφ LΓ Lχ Mχ init B H n := by
  have htH : t ≤ H := (mem_Icc.1 ht).2
  have hc (s : ℕ) (hs : s ≤ H + 1) := coupling_le A μ hN init h.eps_pos h.rate h.rate_le h.LG_nonneg
    h.Lφ_nonneg h.grad h.field (H := H+1) hs
  set X := growth A.eps η LG Lφ ^ (H+1) * forcing A.eps η LG * ∑ s ∈ range (H+1), fluctuation A μ init s
  have h1 : Lχ * (coupling A μ init t + coupling A μ init (t+1)) ≤ 2 * Lχ * X := by
    calc _ ≤ Lχ * (X + X) := mul_le_mul_of_nonneg_left
          (add_le_add (hc t (by omega)) (hc (t+1) (by omega))) h.Lχ_nonneg
      _ = 2 * Lχ * X := by ring
  have hs := segmentError_le_grid A μ h hN init t hn hu.1.le hu.2.le
  have hsum : ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n) ≤
      ∑ t ∈ Icc B H, ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n) :=
    single_le_sum (f := fun t => ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n))
      (fun _ _ => sum_nonneg fun _ _ => l1_nonneg _) ht
  exact (residual_sub_le A μ h hN init t u).trans
    (mul_le_mul_of_nonneg_left (by linarith) h.LΓ_nonneg)

omit [IsProbabilityMeasure μ] in
theorem majorant_nonneg (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (init : Fin N → P → ℝ)
    (B H n : ℕ) : 0 ≤ majorant A μ χ η LG Lφ LΓ Lχ Mχ init B H n :=
  mul_nonneg h.LΓ_nonneg (add_nonneg (add_nonneg (mul_nonneg (by linarith [h.Lχ_nonneg])
    (mul_nonneg (h.gronwall_nonneg A (H+1)) (sum_nonneg fun _ _ => l1_nonneg _)))
    (sum_nonneg fun _ _ => sum_nonneg fun _ _ => l1_nonneg _))
    (div_nonneg (by linarith [h.Mχ_nonneg]) (Nat.cast_nonneg _)))

/-- The majorant is integrable and its expectation is the explicit error. -/
theorem integral_majorant_le (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N)
    (B H n : ℕ) :
    Integrable (fun z => majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n) (Measure.pi fun _ : Fin N => μ) ∧
      ∫ z, majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n ∂(Measure.pi fun _ : Fin N => μ) ≤
        widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n := by
  set π := Measure.pi fun _ : Fin N => μ
  have hfl (s : ℕ) := integral_fluctuation_le A μ hN h.eps_pos h.continuous_grad
    (continuous_of_l1 h.Lφ_nonneg h.field) h.field_bound s
  have hseg (t : ℕ) (u : ℝ) := integral_l1_meanField_sub_le μ hN
    (Ξ := fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights)
    (fun j => (integrable_features A μ h t u j).1.measurable) fun _ j => h.feature_bound u _ _ j
  have hs1 (t : ℕ) (u : ℝ) : Integrable (fun z => segmentError A μ χ z t u) π := (hseg t u).1
  have hs2 (t : ℕ) (u : ℝ) : ∫ z, segmentError A μ χ z t u ∂π ≤ Fintype.card J * Cχ / Real.sqrt N :=
    (hseg t u).2
  have i1 : Integrable (fun z => ∑ s ∈ range (H+1), fluctuation A μ z s) π :=
    integrable_finsetSum _ fun s _ => (hfl s).1
  have i2t (t : ℕ) : Integrable (fun z => ∑ k ∈ range (n+1), segmentError A μ χ z t ((k:ℝ)/n)) π :=
    integrable_finsetSum _ fun k _ => hs1 t _
  have i2 : Integrable (fun z => ∑ t ∈ Icc B H, ∑ k ∈ range (n+1),
      segmentError A μ χ z t ((k:ℝ)/n)) π := integrable_finsetSum _ fun t _ => i2t t
  set c := growth A.eps η LG Lφ ^ (H+1) * forcing A.eps η LG
  have i3 : Integrable (fun z => 2 * Lχ * (c * ∑ s ∈ range (H+1), fluctuation A μ z s)) π :=
    (i1.const_mul _).const_mul _
  have i4 : Integrable (fun z => 2 * Lχ * (c * ∑ s ∈ range (H+1), fluctuation A μ z s) +
      ∑ t ∈ Icc B H, ∑ k ∈ range (n+1), segmentError A μ χ z t ((k:ℝ)/n)) π := i3.add i2
  refine ⟨(i4.add (integrable_const _)).const_mul _, ?_⟩
  have b1 : ∫ z, ∑ s ∈ range (H+1), fluctuation A μ z s ∂π ≤
      (H+1) * (Fintype.card K * Cφ / Real.sqrt N) := by
    rw [integral_finsetSum _ fun s _ => (hfl s).1]
    exact (sum_le_sum fun s _ => (hfl s).2).trans_eq (by simp)
  have b2 : ∫ z, ∑ t ∈ Icc B H, ∑ k ∈ range (n+1), segmentError A μ χ z t ((k:ℝ)/n) ∂π ≤
      (H + 1 - B : ℕ) * ((n+1) * (Fintype.card J * Cχ / Real.sqrt N)) := by
    rw [integral_finsetSum _ fun t _ => i2t t]
    refine (sum_le_sum fun t _ => (?_ : _ ≤ ((n:ℝ)+1) * (Fintype.card J * Cχ / Real.sqrt N))).trans_eq
      (by simp)
    rw [integral_finsetSum _ fun k _ => hs1 t _]
    exact (sum_le_sum fun k _ => hs2 t _).trans_eq (by simp)
  calc ∫ z, majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n ∂π
      = LΓ * (2 * Lχ * (c * ∫ z, ∑ s ∈ range (H+1), fluctuation A μ z s ∂π) +
          ∫ z, ∑ t ∈ Icc B H, ∑ k ∈ range (n+1), segmentError A μ χ z t ((k:ℝ)/n) ∂π + 2 * Mχ / n) := by
        change ∫ z, LΓ * (2 * Lχ * (c * ∑ s ∈ range (H+1), fluctuation A μ z s) + _ + _) ∂π = _
        rw [integral_const_mul, integral_add i4 (integrable_const _), integral_add i3 i2,
          integral_const_mul, integral_const_mul, integral_const]
        simp
    _ ≤ _ := by
        unfold widthError
        refine mul_le_mul_of_nonneg_left (add_le_add_left (add_le_add ?_ b2) _) h.LΓ_nonneg
        exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left b1 (h.gronwall_nonneg A _))
          (by linarith [h.Lχ_nonneg])

end Bound

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
