module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Initialization
public import Mathlib.Probability.Distributions.Gaussian.Real

/-! Gaussian initialization. Argand draws every real and imaginary part of a
weight independently from N(0, σ²) with σ = 0.02. By the Chernoff bound one
coordinate leaves [-C₀, C₀] with probability at most 2 · exp(-C₀²/(2σ²)), so
a start of N units leaves the box of `Initialization` with probability at
most 2 · N · |P| · exp(-C₀²/(2σ²)), and recovery from the certificate
F ∘ mean ψ loses no more than that. At σ = 0.02 and C₀ = 120, where the
clamp of `certificate_residual_of_init` is the one of every smaller C₀, the
exponent is -1.8 · 10⁷. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset Filter ProbabilityTheory
  Transformer.Grokking.FiniteRecovery.ObservedConstraints
open scoped NNReal

variable {P K J : Type*}

/-- **The Gaussian tail**: a centred Gaussian of variance v leaves [-C, C] with
probability at most 2 · exp(-C²/(2v)), by the Chernoff bound at t = C/v on
either side. For v = 0 the right side is 2. -/
theorem gaussianReal_abs_gt_le (v : ℝ≥0) {C : ℝ} (hC : 0 ≤ C) :
    (gaussianReal 0 v).real {x | C < |x|} ≤ 2 * Real.exp (-C^2 / (2 * v)) := by
  rcases eq_or_ne v 0 with hv | hv
  · subst hv
    simp only [NNReal.coe_zero, mul_zero, div_zero, Real.exp_zero, mul_one]
    linarith [measureReal_le_one (μ := gaussianReal 0 0) (s := {x : ℝ | C < |x|})]
  have hvpos : (0:ℝ) < v := lt_of_le_of_ne v.2 (by exact_mod_cast hv.symm)
  have hv' : (v:ℝ) ≠ 0 := hvpos.ne'
  have ht : 0 ≤ C / v := div_nonneg hC hvpos.le
  have expo (s : ℝ) (hs : s ^ 2 = (C / v) ^ 2) :
      -(C / v) * C + (0 * s + v * s ^ 2 / 2) = -C^2 / (2 * v) := by
    rw [hs]
    field_simp
    ring
  have upper : (gaussianReal 0 v).real {x | C ≤ id x} ≤ Real.exp (-C^2 / (2 * v)) := by
    have h := measure_ge_le_exp_mul_mgf (μ := gaussianReal 0 v) (X := id) C ht
      (integrable_exp_mul_gaussianReal (C / v))
    rwa [mgf_id_gaussianReal, ← Real.exp_add, expo _ rfl] at h
  have lower : (gaussianReal 0 v).real {x | C ≤ (-id) x} ≤ Real.exp (-C^2 / (2 * v)) := by
    have hint : Integrable (fun x => Real.exp (C / v * (-id) x)) (gaussianReal 0 v) := by
      simpa [mul_neg, ← neg_mul] using integrable_exp_mul_gaussianReal (μ := 0) (v := v) (-(C / v))
    have h := measure_ge_le_exp_mul_mgf (μ := gaussianReal 0 v) (X := -id) C ht hint
    rwa [mgf_neg, mgf_id_gaussianReal, ← Real.exp_add, expo _ (neg_sq _)] at h
  have cover : {x : ℝ | C < |x|} ⊆ {x | C ≤ id x} ∪ {x | C ≤ (-id) x} := by
    intro x hx
    rcases lt_abs.mp (show C < |x| from hx) with h | h
    · exact Or.inl (by simpa using h.le)
    · exact Or.inr (by simpa using h.le)
  calc _ ≤ (gaussianReal 0 v).real ({x | C ≤ id x} ∪ {x | C ≤ (-id) x}) := measureReal_mono cover
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ _ := by linarith

/-- Under independent N(0, v) coordinates, each coordinate leaves [-C, C] with
probability at most 2 · exp(-C²/(2v)). -/
theorem gaussian_coordinate_tail [Fintype P] (v : ℝ≥0) {C : ℝ} (hC : 0 ≤ C) (p : P) :
    (Measure.pi fun _ : P => gaussianReal 0 v).real {θ | C < |θ p|} ≤
      2 * Real.exp (-C^2 / (2 * v)) := by
  have hs : MeasurableSet {x : ℝ | C < |x|} :=
    measurableSet_lt measurable_const continuous_abs.measurable
  have e := (measurePreserving_eval (fun _ : P => gaussianReal 0 v) p).measure_preimage
    hs.nullMeasurableSet
  exact (congrArg ENNReal.toReal e).trans_le (gaussianReal_abs_gt_le v hC)

/-- **Permanent recovery from Gaussian initialization.** When every coordinate
of every unit is drawn independently from N(0, v), the certificate F ∘ mean ψ
recovers with probability above q - 2 · N · |P| · exp(-C₀²/(2v)), on starts
within C₀. -/
theorem permanent_of_gaussian_init {X Act R : Type*} [Fintype X] [Fintype Act] [Nonempty Act]
    [MeasurableSpace Act] [MeasurableSingletonClass Act] [Fintype R] [Fintype P] [Fintype K]
    [Fintype J] (A : Training P K) (v : ℝ≥0) (F : (J → ℝ) → ℝ) (ψ : (P → ℝ) → J → ℝ)
    (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ)
    {η LG Lφ Cφ LΓ Lχ Mχ Cχ C₀ ρ r q rbar : ℝ} (hC : 0 ≤ C₀)
    (h : Regular A (certificateMap F ρ r) (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12)))
      η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N) {B H n : ℕ} (hn : 0 < n)
    (score : (Fin N → P → ℝ) → X → Act → ℝ) (S : Finset X) (rule : R → Set (X → Act))
    (f : X → Act) (hf : ∀ j, f ∈ rule j)
    (hid : Set.InjOn (fun g : X → Act => fun x : S => g x) (⋂ j, rule j))
    (above : ∀ w, risk (tableLaw (score w)) S (fun x => f x) rule ≤ F (meanField ψ w))
    (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1,
      limitResidual A (Measure.pi fun _ : P => gaussianReal 0 v) (certificateMap F ρ r)
        (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) t u ≤ rbar)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => Measure.pi fun _ : P => gaussianReal 0 v))
    (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A (certificateMap F ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) z t u ≤ tail z)
    (small : max rbar 0 + widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K)
        (Fintype.card (Fin 3 × J)) N B H n +
        ∫ z, tail z ∂(Measure.pi fun _ : Fin N => Measure.pi fun _ : P => gaussianReal 0 v) <
      (1-q)*((1-ρ)*(1/2-r))) :
    ∃ G ⊆ box N C₀, MeasurableSet G ∧
      q - 2 * N * Fintype.card P * Real.exp (-C₀^2 / (2 * v)) <
        (Measure.pi fun _ : Fin N => Measure.pi fun _ : P => gaussianReal 0 v).real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (score (weightsAt A z t)) f := by
  obtain ⟨G, hGE, hG, hq, hs⟩ := permanent_of_certificate A _ F ψ hF hψ
    (gaussian_coordinate_tail v hC) h hN hn score S rule f hf hid above hρ hlt hr hlim tail htm
    hti htn htail small
  exact ⟨G, hGE, hG, by linarith, hs⟩

/-- At Argand's initialization scale σ = 0.02 a coordinate leaves [-120, 120]
with probability at most 2 · exp(-1.8 · 10⁷). -/
theorem argand_gaussian_tail :
    (gaussianReal 0 (1/2500)).real {x | 120 < |x|} ≤ 2 * Real.exp (-18000000) := by
  have h := gaussianReal_abs_gt_le (1/2500) (C := 120) (by norm_num)
  have e : -(120:ℝ)^2 / (2 * ((1/2500 : ℝ≥0) : ℝ)) = -18000000 := by
    push_cast
    norm_num
  rwa [e] at h

section Examples

example := gaussianReal_abs_gt_le 1 (C := 1) zero_le_one
example := gaussian_coordinate_tail (P := Fin 1) 1 (C := 1) zero_le_one 0
example := permanent_of_gaussian_init identityTraining 1 (fun _ : Fin 1 → ℝ => (0:ℝ))
  (fun _ (_ : Fin 1) => (0:ℝ)) (differentiable_const _) (differentiable_const _) (C₀ := 1)
  zero_le_one (identityCertificateRegular 0 0 _) (N := 2) two_pos (B := 1) (H := 3) (n := 4)
  (by norm_num) (fun _ (_ : Fin 1) (_ : Fin 1) => (0:ℝ)) ∅ (fun _ : Fin 1 => Set.univ) (fun _ => 0)
  (fun _ => Set.mem_univ _) (fun _ _ _ _ _ => Subsingleton.elim _ _) (fun _ => by simp [risk])
  (q := 0) (rbar := 0) le_rfl one_pos (by norm_num)
  (fun _ _ _ _ => by simp [limitResidual, certificateMap]) (fun _ => 0) measurable_const
  (integrable_const _) (fun _ => le_rfl) (fun _ _ _ _ _ => by simp [residual, certificateMap])
  (by norm_num [widthError])

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
