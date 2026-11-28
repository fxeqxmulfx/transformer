module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Sampling

/-! The residual of a mean-field certificate along one update. A
certificate that reads the weights through finitely many width averages
has a residual Γ of the empirical mean of segment features χ_u, which read
the weights of one particle before and after the update at the fraction u
of the segment. Its mean-field counterpart is Γ of their expectation along
one limiting particle. Their gap is bounded pathwise by the coupling
distance, by the sampling error on a grid of fractions, and by the grid
spacing. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

variable {P K J : Type*}

/-- Regularity of the training map, of the residual map Γ and of the
segment features χ, with explicit constants. Γ need only be Lipschitz on
the box of feature values. -/
structure Regular [Fintype P] [Fintype K] [Fintype J] (A : Training P K) (Γ : (J → ℝ) → ℝ)
    (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ) (η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ) : Prop where
  eps_pos : 0 < A.eps
  rate : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ η
  rate_le : η ≤ 10
  LG_nonneg : 0 ≤ LG
  Lφ_nonneg : 0 ≤ Lφ
  LΓ_nonneg : 0 ≤ LΓ
  Lχ_nonneg : 0 ≤ Lχ
  grad : ∀ t θ θ' Φ Φ', l1 (A.grad t θ Φ - A.grad t θ' Φ') ≤ LG * (l1 (θ - θ') + l1 (Φ - Φ'))
  field : ∀ θ θ', l1 (A.field θ - A.field θ') ≤ Lφ * l1 (θ - θ')
  field_bound : ∀ θ k, |A.field θ k| ≤ Cφ
  map : ∀ a a' : J → ℝ, (∀ j, |a j| ≤ Cχ) → (∀ j, |a' j| ≤ Cχ) → Γ a - Γ a' ≤ LΓ * l1 (a - a')
  features : ∀ u θ θ' ϑ ϑ', l1 (χ u θ θ' - χ u ϑ ϑ') ≤ Lχ * (l1 (θ - ϑ) + l1 (θ' - ϑ'))
  segment : ∀ u u' θ θ', l1 (χ u θ θ' - χ u' θ θ') ≤ Mχ * |u - u'|
  feature_bound : ∀ u θ θ' j, |χ u θ θ' j| ≤ Cχ

/-- The finite residual: Γ of the empirical mean of the segment features
of the N particles over update t. -/
noncomputable def residual (A : Training P K) (Γ : (J → ℝ) → ℝ) (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ)
    {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (u : ℝ) : ℝ :=
  Γ (meanField (fun q : (P → ℝ) × (P → ℝ) => χ u q.1 q.2)
    fun i => ((system A init t i).weights, (system A init (t+1) i).weights))

/-- Expected segment features of one limiting particle over update t. -/
noncomputable def limitFeatures (A : Training P K) (μ : Measure (P → ℝ))
    (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ) (t : ℕ) (u : ℝ) : J → ℝ :=
  fun j => ∫ x, χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j ∂μ

/-- The mean-field residual. -/
noncomputable def limitResidual (A : Training P K) (μ : Measure (P → ℝ)) (Γ : (J → ℝ) → ℝ)
    (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ) (t : ℕ) (u : ℝ) : ℝ :=
  Γ (limitFeatures A μ χ t u)

/-- Sampling error of the segment features over the N limiting particles
started at the same initial weights as the finite system. -/
noncomputable def segmentError [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
    (χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ) {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (u : ℝ) : ℝ :=
  l1 (meanField (fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights) init -
    limitFeatures A μ χ t u)

theorem l1_meanField_sub_le [Fintype J] {Θ Θ' : Type*} {N : ℕ} (φ : Θ → J → ℝ) (φ' : Θ' → J → ℝ)
    (θ : Fin N → Θ) (θ' : Fin N → Θ') :
    l1 (meanField φ θ - meanField φ' θ') ≤ (∑ i, l1 (φ (θ i) - φ' (θ' i))) / N := by
  unfold l1 meanField
  simp only [Pi.sub_apply]
  calc ∑ j, |(∑ i, φ (θ i) j) / N - (∑ i, φ' (θ' i) j) / N|
      = ∑ j, |∑ i, (φ (θ i) j - φ' (θ' i) j)| / N := by
        refine sum_congr rfl fun j _ => ?_
        rw [← sub_div, ← sum_sub_distrib, abs_div, Nat.abs_cast]
    _ ≤ ∑ j, (∑ i, |φ (θ i) j - φ' (θ' i) j|) / N :=
        sum_le_sum fun j _ => div_le_div_of_nonneg_right (abs_sum_le_sum_abs _ _) (Nat.cast_nonneg _)
    _ = _ := by rw [← sum_div, sum_comm]

theorem abs_meanField_le {Θ : Type*} {N : ℕ} (hN : 0 < N) (φ : Θ → J → ℝ) {C : ℝ}
    (hC : ∀ q j, |φ q j| ≤ C) (θ : Fin N → Θ) (j : J) : |meanField φ θ j| ≤ C := by
  have hNr : (0:ℝ) < N := Nat.cast_pos.2 hN
  rw [meanField, abs_div, abs_of_pos hNr, div_le_iff₀ hNr]
  refine (abs_sum_le_sum_abs _ _).trans ?_
  simpa [mul_comm] using sum_le_sum fun i (_ : i ∈ univ) => hC (θ i) j

theorem abs_limitFeatures_le (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {C : ℝ} (hC : ∀ u θ θ' j, |χ u θ θ' j| ≤ C) (t : ℕ) (u : ℝ)
    (j : J) : |limitFeatures A μ χ t u j| ≤ C := by
  have h := norm_integral_le_of_norm_le_const (μ := μ)
    (f := fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j)
    (ae_of_all _ fun x => by simpa [Real.norm_eq_abs] using hC u _ _ j)
  simpa [Real.norm_eq_abs, limitFeatures] using h

/-- The finite residual exceeds the mean-field residual by at most the
coupling distance at both ends of the update plus the sampling error. -/
theorem residual_sub_le [Fintype P] [Fintype K] [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
    [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) (init : Fin N → P → ℝ)
    (t : ℕ) (u : ℝ) :
    residual A Γ χ init t u - limitResidual A μ Γ χ t u ≤
      LΓ * (Lχ * (coupling A μ init t + coupling A μ init (t+1)) + segmentError A μ χ init t u) := by
  let φ := fun q : (P → ℝ) × (P → ℝ) => χ u q.1 q.2
  let a := meanField φ fun i => ((system A init t i).weights, (system A init (t+1) i).weights)
  let b := meanField φ fun i => ((flow A μ t (init i)).weights, (flow A μ (t+1) (init i)).weights)
  let c := limitFeatures A μ χ t u
  have hφ : ∀ q j, |φ q j| ≤ Cχ := fun q j => h.feature_bound u q.1 q.2 j
  have hΓ : Γ a - Γ c ≤ LΓ * l1 (a - c) :=
    h.map a c (abs_meanField_le hN φ hφ _) (abs_limitFeatures_le A μ h.feature_bound t u)
  have hab : l1 (a - b) ≤ Lχ * (coupling A μ init t + coupling A μ init (t+1)) := by
    refine (l1_meanField_sub_le φ φ _ _).trans ?_
    rw [coupling, coupling, ← add_div, mul_div_assoc', ← sum_add_distrib, mul_sum]
    refine div_le_div_of_nonneg_right (sum_le_sum fun i _ => ?_) (Nat.cast_nonneg _)
    refine (h.features u _ _ _ _).trans (mul_le_mul_of_nonneg_left (add_le_add ?_ ?_) h.Lχ_nonneg)
    · exact l1_weights_le_gap _ _
    · exact l1_weights_le_gap _ _
  have hbc : l1 (b - c) = segmentError A μ χ init t u := rfl
  calc residual A Γ χ init t u - limitResidual A μ Γ χ t u = Γ a - Γ c := rfl
    _ ≤ LΓ * l1 (a - c) := hΓ
    _ ≤ LΓ * (l1 (a - b) + l1 (b - c)) := mul_le_mul_of_nonneg_left (l1_sub_le a b c) h.LΓ_nonneg
    _ ≤ _ := by
        rw [hbc]
        exact mul_le_mul_of_nonneg_left (by linarith) h.LΓ_nonneg

section Examples

/-- The identity training with a zero residual map and zero features is regular. -/
theorem identityRegular :
    Regular identityTraining (fun _ : (Fin 1 → ℝ) => (0:ℝ)) (fun _ _ _ _ => 0) 1 1 0 0 0 0 0 0 where
  eps_pos := one_pos
  rate _ := ⟨zero_le_one, le_rfl⟩
  rate_le := by norm_num
  LG_nonneg := zero_le_one
  Lφ_nonneg := le_rfl
  LΓ_nonneg := le_rfl
  Lχ_nonneg := le_rfl
  grad := identityTraining_grad
  field := identityTraining_field
  field_bound _ _ := by simp [identityTraining]
  map _ _ _ _ := by simp
  features _ _ _ _ _ := by simp [l1]
  segment _ _ _ _ := by simp [l1]
  feature_bound _ _ _ _ := by simp

example := abs_meanField_le (N := 2) two_pos (fun (_ : Unit) (_ : Fin 1) => (0:ℝ)) (C := 0)
  (fun _ _ => by simp) (fun _ => ()) 0
example := abs_limitFeatures_le identityTraining (Measure.dirac 0)
  (χ := fun _ _ _ (_ : Fin 1) => (0:ℝ)) (C := 0) (fun _ _ _ _ => by simp) 0 0 0
example := residual_sub_le identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (fun _ => 0) 3 (1/2)

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
