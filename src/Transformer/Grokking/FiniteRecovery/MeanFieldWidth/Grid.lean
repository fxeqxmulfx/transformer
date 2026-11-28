module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Residual

/-! Uniformity in the segment fraction. The features of a limiting particle
are continuous in its initial weights, hence integrable, and a grid of
fractions k/n controls the sampling error at every fraction of the segment
up to twice the feature modulus times the spacing. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

variable {P K J : Type*}

theorem continuous_of_l1₂ {Q R : Type*} [Fintype Q] [Fintype R] {f : (Q → ℝ) → (Q → ℝ) → R → ℝ}
    {L : ℝ} (hL : 0 ≤ L) (h : ∀ x x' y y', l1 (f x x' - f y y') ≤ L * (l1 (x - y) + l1 (x' - y'))) :
    Continuous fun q : (Q → ℝ) × (Q → ℝ) => f q.1 q.2 := by
  refine (LipschitzWith.of_dist_le_mul (K := ⟨2 * L * Fintype.card Q, by positivity⟩)
    fun q q' => ?_).continuous
  have hd : 0 ≤ 2 * L * Fintype.card Q * dist q q' := by positivity
  refine (dist_pi_le_iff hd).2 fun r => ?_
  have hl (x y : Q → ℝ) (hxy : dist x y ≤ dist q q') : l1 (x - y) ≤ Fintype.card Q * dist q q' := by
    calc l1 (x - y) ≤ ∑ _p : Q, dist q q' := sum_le_sum fun p _ => by
          simpa [Real.dist_eq] using (dist_le_pi_dist x y p).trans hxy
      _ = _ := by simp
  have h1 := hl q.1 q'.1 ((le_max_left _ _).trans (Prod.dist_eq ..).ge)
  have h2 := hl q.2 q'.2 ((le_max_right _ _).trans (Prod.dist_eq ..).ge)
  calc dist (f q.1 q.2 r) (f q'.1 q'.2 r) = |(f q.1 q.2 - f q'.1 q'.2) r| := Real.dist_eq _ _
    _ ≤ l1 (f q.1 q.2 - f q'.1 q'.2) :=
        single_le_sum (f := fun r => |(f q.1 q.2 - f q'.1 q'.2) r|) (fun _ _ => abs_nonneg _) (mem_univ r)
    _ ≤ L * (Fintype.card Q * dist q q' + Fintype.card Q * dist q q') :=
        (h _ _ _ _).trans (mul_le_mul_of_nonneg_left (add_le_add h1 h2) hL)
    _ = _ := by ring

theorem Regular.continuous_grad [Fintype P] [Fintype K] [Fintype J] {A : Training P K}
    {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (t : ℕ) (Φ : K → ℝ) :
    Continuous fun θ => A.grad t θ Φ := by
  have h0 : l1 (0 : K → ℝ) = 0 := by simp [l1]
  exact continuous_of_l1 h.LG_nonneg fun x y => by simpa [h0] using h.grad t x y Φ Φ

theorem Regular.Mχ_nonneg [Fintype P] [Fintype K] [Fintype J] {A : Training P K}
    {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) : 0 ≤ Mχ := by
  simpa using (l1_nonneg _).trans (h.segment 0 1 0 0)

/-- Each segment feature of a limiting particle is integrable in its start. -/
theorem integrable_features [Fintype P] [Fintype K] [Fintype J] (A : Training P K)
    (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ}
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (t : ℕ) (u : ℝ) (j : J) :
    Continuous (fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j) ∧
      Integrable (fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j) μ := by
  have hc : Continuous fun x => χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j :=
    (continuous_apply j).comp ((continuous_of_l1₂ h.Lχ_nonneg (h.features u)).comp
      ((continuous_flow A μ h.eps_pos h.continuous_grad t).1.prodMk
        (continuous_flow A μ h.eps_pos h.continuous_grad (t+1)).1))
  exact ⟨hc, Integrable.of_bound hc.aestronglyMeasurable Cχ
    (ae_of_all _ fun x => by simpa [Real.norm_eq_abs] using h.feature_bound u _ _ j)⟩

/-- The sampling error at any fraction is at most its sum over the grid
k/n plus twice the feature modulus times the spacing. -/
theorem segmentError_le_grid [Fintype P] [Fintype K] [Fintype J] (A : Training P K)
    (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ}
    {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ} {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}
    (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) (init : Fin N → P → ℝ)
    (t : ℕ) {n : ℕ} (hn : 0 < n) {u : ℝ} (hu0 : 0 ≤ u) (hu1 : u ≤ 1) :
    segmentError A μ χ init t u ≤
      ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n) + 2 * Mχ / n := by
  have hnr : (0:ℝ) < n := Nat.cast_pos.2 hn
  have hk : ⌊u * n⌋₊ ∈ range (n+1) :=
    mem_range.2 (Nat.lt_succ_of_le (Nat.floor_le_of_le (by nlinarith)))
  have h1 : ((⌊u * n⌋₊ : ℕ) : ℝ) ≤ u * n := Nat.floor_le (by positivity)
  have h2 : u * n < ⌊u * n⌋₊ + 1 := Nat.lt_floor_add_one _
  generalize ⌊u * n⌋₊ = k at hk h1 h2
  have hd : |u - k/n| ≤ 1/n := by
    rw [show u - k/n = (u*n - k)/n by field_simp, abs_of_nonneg (div_nonneg (by linarith) hnr.le)]
    exact div_le_div_of_nonneg_right (by linarith) hnr.le
  let E (v : ℝ) := meanField (fun x => χ v (flow A μ t x).weights (flow A μ (t+1) x).weights) init
  let c (v : ℝ) := limitFeatures A μ χ t v
  have hE : l1 (E u - E (k/n)) ≤ Mχ / n := by
    refine (l1_meanField_sub_le _ _ init init).trans ?_
    have hs := sum_le_sum fun i (_ : i ∈ univ) => h.segment u (k/n) (flow A μ t (init i)).weights
      (flow A μ (t+1) (init i)).weights
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at hs
    have hM := h.Mχ_nonneg
    rw [div_le_div_iff₀ (Nat.cast_pos.2 hN) hnr]
    calc _ ≤ (N * (Mχ * |u - k/n|)) * n := mul_le_mul_of_nonneg_right hs hnr.le
      _ ≤ (N * (Mχ * (1/n))) * n := by gcongr
      _ = Mχ * N := by field_simp
  have hc : l1 (c (k/n) - c u) ≤ Mχ / n := by
    have hi := integrable_features A μ h t
    calc l1 (c (k/n) - c u)
        = ∑ j, |∫ x, (χ (k/n) (flow A μ t x).weights (flow A μ (t+1) x).weights j -
            χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j) ∂μ| := by
          refine sum_congr rfl fun j _ => ?_
          rw [integral_sub (hi _ j).2 (hi _ j).2]
          rfl
      _ ≤ ∑ j, ∫ x, |χ (k/n) (flow A μ t x).weights (flow A μ (t+1) x).weights j -
            χ u (flow A μ t x).weights (flow A μ (t+1) x).weights j| ∂μ :=
          sum_le_sum fun j _ => abs_integral_le_integral_abs
      _ = ∫ x, l1 (χ (k/n) (flow A μ t x).weights (flow A μ (t+1) x).weights -
            χ u (flow A μ t x).weights (flow A μ (t+1) x).weights) ∂μ :=
          (integral_finsetSum _ fun j _ => ((hi _ j).2.sub (hi _ j).2).abs).symm
      _ ≤ ∫ _x, Mχ / n ∂μ := integral_mono
          (integrable_finsetSum _ fun j _ => ((hi _ j).2.sub (hi _ j).2).abs) (integrable_const _)
          fun x => (h.segment _ _ _ _).trans (by rw [abs_sub_comm]; exact
            (mul_le_mul_of_nonneg_left hd h.Mχ_nonneg).trans_eq (mul_one_div _ _))
      _ = Mχ / n := by simp
  have hk' : segmentError A μ χ init t (k/n) ≤ ∑ k ∈ range (n+1), segmentError A μ χ init t ((k:ℝ)/n) :=
    single_le_sum (f := fun k : ℕ => segmentError A μ χ init t ((k:ℝ)/n)) (fun _ _ => l1_nonneg _) hk
  calc segmentError A μ χ init t u = l1 (E u - c u) := rfl
    _ ≤ l1 (E u - E (k/n)) + (l1 (E (k/n) - c (k/n)) + l1 (c (k/n) - c u)) :=
        (l1_sub_le _ (E (k/n)) _).trans (add_le_add_right (l1_sub_le _ _ _) _)
    _ ≤ Mχ / n + (segmentError A μ χ init t (k/n) + Mχ / n) := add_le_add hE (add_le_add le_rfl hc)
    _ ≤ _ := by linarith [show 2 * Mχ / n = Mχ / n + Mχ / n by ring]

section Examples

example := continuous_of_l1₂ (f := fun (x _ : Fin 1 → ℝ) => x) zero_le_one fun x x' y y' => by
  show l1 (x - y) ≤ 1 * (l1 (x - y) + l1 (x' - y'))
  linarith [l1_nonneg (x' - y')]
example := identityRegular.continuous_grad 0 0
example := identityRegular.Mχ_nonneg
example := integrable_features identityTraining (Measure.dirac 0) identityRegular 2 (1/2) 0
example := segmentError_le_grid identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (fun _ => 0) 3 (n := 4) (by norm_num) (u := 1/2) (by norm_num) (by norm_num)

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
