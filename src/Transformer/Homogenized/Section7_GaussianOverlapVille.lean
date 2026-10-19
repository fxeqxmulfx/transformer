module

public import Transformer.Homogenized.Section7_GramFieldContinuity
public import Transformer.Homogenized.Section7_IntegratingFactorVille

/-!
# Maximal control of the original Gaussian exponential observable

ArXiv:2604.01978v1, §7, the maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
Apply the proved integrating-factor maximal inequality to the
actual observable exp(cR), its original Gaussian generator,
and coefficient cD+c²(3/d)/2. The original Gram estimates give
the required continuity and measurable composition. The unit
configuration bounds give every domination constant.

The stochastic premise is exactly the compensated martingale
identity for the selected smooth exponential observable. The
weighted martingale property and probability bound are proved,
with neither supplied as an input. Both signs of c are allowed.
The actual time integral of the original overlap drift remains
in the event. The noise coefficient is the proved upper bound
3/d at the source's standard value variance σV²=1/d.
The accompanying identities prepare the original scalar-noise
threshold and cancel its deterministic initial exponential.
-/

@[expose] public section

open Set MeasureTheory
open scoped NNReal
namespace Transformer.Homogenized

/-- The original integrating factor gives the original compensated overlap exponent.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem exp_overlap_compensation_eq (c q r t : ℝ) (D : ℝ → ℝ)
    (hD : IntervalIntegrable D volume 0 t) :
    Real.exp (-(∫ s in (0 : ℝ)..t, c * D s + c ^ 2 / 2 * q)) * Real.exp (c * r) =
      Real.exp (c * (r - ∫ s in (0 : ℝ)..t, D s) - c ^ 2 / 2 * q * t) := by
  have hconst : IntervalIntegrable (fun _ : ℝ => c ^ 2 / 2 * q) volume 0 t := intervalIntegrable_const
  rw [intervalIntegral.integral_add (hD.const_mul c) hconst, intervalIntegral.integral_const_mul,
    intervalIntegral.integral_const, smul_eq_mul, ← Real.exp_add]
  congr 1
  ring

/-- The initial exponential cancels in the actual maximal-bound quotient.
Source: arXiv:2604.01978v1, §7, the exponential martingale tail bound. -/
theorem ofReal_exp_initial_div_exp_compensation (c γ r q T : ℝ) :
    ENNReal.ofReal (Real.exp (c * γ)) / ENNReal.ofReal (Real.exp (c * γ + c * r - c ^ 2 / 2 * q * T)) =
      ENNReal.ofReal (Real.exp (-c * r + c ^ 2 / 2 * q * T)) := by
  rw [← ENNReal.ofReal_div_of_pos (Real.exp_pos _), ← Real.exp_sub]
  congr 2
  ring

/-- The maximal inequality for the original Gaussian overlap exponential,
from the actual compensated exponential test martingale at σV²=1/d.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem gaussian_overlap_exponential_ville {d n : ℕ} (hd : 2 ≤ d) (c β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d)) (i j : Idx n)
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T))
    (hM : IsMartingaleOn T ℱ P (fun t ω => exponentialOverlap c i j (X t ω) -
      exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (X s ω)))
    (r : ℝ) (hr : 0 < r) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T,
      r < Real.exp (-(∫ s in (0 : ℝ)..t, c * overlapDrift β ρ (X s ω) i j + c ^ 2 / 2 * (3 / (d : ℝ)))) *
        exponentialOverlap c i j (X t ω)} ≤
      ENNReal.ofReal (∫ ω, exponentialOverlap c i j (X 0 ω) ∂P) / ENNReal.ofReal r := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  let a (t : ℝ) (ω : Ω) := c * overlapDrift β ρ (X t ω) i j + c ^ 2 / 2 * (3 / (d : ℝ))
  let Y (t : ℝ) (ω : Ω) := exponentialOverlap c i j (X t ω)
  let H (t : ℝ) (ω : Ω) := sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
    (exponentialOverlap c i j) (X t ω)
  let A := 2 * |c| + c ^ 2 / 2 * (3 / (d : ℝ))
  have hD := continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j
  have hGen := continuousOn_sphGenerator_exponentialOverlap_gaussian hd c β σV σA ρ hρ hσ i j
  have hφ : Continuous (exponentialOverlap (d := d) c i j) := (contDiff_exponentialOverlap c i j).continuous
  have ha (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (a t) := by
    have hDm : StronglyMeasurable[ℱ t] (fun ω => overlapDrift β ρ (X t ω) i j) :=
      stronglyMeasurable_comp_of_continuousOn_range _ _ hD (X t) (hadapt t ht).measurable (hunit t ht)
    exact (hDm.const_mul c).add_const _
  have hY (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (Y t) :=
    hφ.comp_stronglyMeasurable (hadapt t ht)
  have hc (ω : Ω) : ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => Y t ω) (Icc (0 : ℝ) T) ∧
      ContinuousOn (fun t => H t ω) (Icc (0 : ℝ) T) := by
    have hDt : ContinuousOn (fun t => overlapDrift β ρ (X t ω) i j) (Icc (0 : ℝ) T) :=
      hD.comp' (hcont ω) (fun t ht => hunit t ht ω)
    exact ⟨(continuousOn_const.mul hDt).add_const _, hφ.comp_continuousOn (hcont ω),
      hGen.comp' (hcont ω) (fun t ht => hunit t ht ω)⟩
  apply exp_neg_timeIntegral_mul_ville P ℱ T hT a Y H ha hY hc hM A (Real.exp |c|)
    (Real.exp |c| * A) (exponential_overlap_bound_nonneg d c)
    (mul_nonneg (Real.exp_nonneg _) (exponential_overlap_bound_nonneg d c))
    (fun t ht ω => abs_overlap_compensation_coefficient_le hd c β ρ (X t ω) (hunit t ht ω) i j)
    (fun t ht ω => by
      change |exponentialOverlap c i j (X t ω)| ≤ Real.exp |c|
      rw [abs_of_pos (exponentialOverlap_pos c i j (X t ω))]
      exact exponentialOverlap_le_exp_abs c i j (X t ω) (hunit t ht ω))
    (fun t ht ω => abs_sphGenerator_exponentialOverlap_le hd c β hρ hσ (X t ω) (hunit t ht ω) i j)
    (fun t ht ω => (exponentialOverlap_pos c i j (X t ω)).le) ?_ r hr
  intro t ht ω
  exact (sphGenerator_exponentialOverlap_le c β hρ hσ (X t ω) (hunit t ht ω) i j).trans_eq (mul_comm _ _)

/-- A genuine positive-value-variance Gaussian head and a frozen
consensus configuration satisfy the selected exponential observable's
compensated martingale identity. The scalar drift zero is also
integrable for the compensation identity. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)),
    (2 : ℕ) ≤ 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 0 (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ (0 : ℝ) < 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω k, ‖X t ω k‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (X t)) ∧
    (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) 1)) ∧
    IsMartingaleOn 1 ℱ (Measure.dirac ()) (fun t ω => exponentialOverlap 1 0 1 (X t ω) -
      exponentialOverlap 1 0 1 (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 0) (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 0))
          (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0)) (exponentialOverlap 1 0 1) (X s ω)) ∧
    (0 : ℝ) < 1 / 2 ∧ IntervalIntegrable (fun _ : ℝ => (0 : ℝ)) volume 0 1 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let K : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ => K
  let ρ := gaussHeadLaw 2 (stdSigmaV 2) 0
  have hρ : IsGaussianHeadLaw 2 (stdSigmaV 2) 0 ρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 0
  have hK : ∀ k, ‖K k‖ = 1 := fun k => by simp [K]
  have hR : inner (𝕜 := ℝ) (K 0) (K 1) = 1 := by
    change inner (𝕜 := ℝ) (K 0) (K 0) = 1
    rw [real_inner_self_eq_norm_sq, hK 0, one_pow]
  have hD : overlapDrift 1 ρ K 0 1 = 0 := by
    change overlapDrift 1 ρ K 0 0 = 0
    exact overlapDrift_self 1 ρ K 0 (hK 0)
  have hq : (∫ θ, overlapNoiseIntegrand 1 ρ K 0 1 θ ^ 2 ∂ρ) = 0 := by
    rw [integral_overlapNoiseIntegrand_sq_factor 1 hρ K hK, hR]
    norm_num
  have hGen : sphGenerator ρ (bField 1 ρ) (noiseField 1 1 1 ρ) (exponentialOverlap 1 0 1) K = 0 := by
    rw [sphGenerator_exponentialOverlap_gaussian 1 1 hρ (stdSigmaV_sq 2) K hK, hD, hq]
    ring
  refine ⟨ℱ, X, le_rfl, hρ, stdSigmaV_sq 2, one_pos, fun t ht ω => hK,
    fun t ht => stronglyMeasurable_const, fun ω => continuousOn_const, ?_, by norm_num, intervalIntegrable_const⟩
  have hz : (fun t ω => exponentialOverlap 1 0 1 (X t ω) - exponentialOverlap 1 0 1 (X 0 ω) -
      ∫ s in (0 : ℝ)..t, sphGenerator ρ (bField 1 ρ) (noiseField 1 1 1 ρ) (exponentialOverlap 1 0 1) (X s ω)) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
    funext t ω
    simp only [X, hGen, sub_self, intervalIntegral.integral_zero]
  rw [hz]
  exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0

end Transformer.Homogenized
