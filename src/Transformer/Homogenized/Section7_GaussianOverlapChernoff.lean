import Transformer.Homogenized.Section7_GaussianOverlapVille

/-!
# Chernoff control of the actual compensated Gaussian overlap

ArXiv:2604.01978v1, §7, the martingale maximal inequality after
`eq:Gronwall_E_selfcontained` and `eq:martingale_sup_bound_clean`.
The original Gaussian exponential maximal bound now controls the
actual residual R(t)-R(0)-∫₀ᵗD, with deterministic initial overlap.
The sample law is explicitly a probability measure. Its initial
exponential expectation cancels in the maximal-bound quotient.

For every real c and r, the event c r < c(R(t)-γ-∫D) has probability
at most exp(-cr+c²(3/d)T/2). This single statement retains either
sign of c for the two sides of the final noise bound. The drift
integral is the original one and its time integrability is derived
from the actual Gaussian field continuity and unit paths.
The only stochastic input is the original compensated exponential
test martingale, rather than a tail or weighted-process premise.
-/

open Set MeasureTheory
open scoped NNReal
namespace Transformer.Homogenized

/-- The actual optimizing coefficient gives the improved exponent -dr²/(6T).
Source: arXiv:2604.01978v1, §7, optimizing the exponential maximal bound
using the proved noise coefficient 3/d. -/
theorem overlap_chernoff_optimal_exponent (d T r : ℝ) (hd : d ≠ 0) (hT : T ≠ 0) :
    -(d * r / (3 * T)) * r + (d * r / (3 * T)) ^ 2 / 2 * (3 / d) * T = -(d * r ^ 2 / (6 * T)) := by
  field_simp
  ring

/-- The original compensated overlap obeys the signed Chernoff maximal bound.
Source: arXiv:2604.01978v1, §7, the noise estimate following
`eq:Gronwall_E_selfcontained`, with the proved variance coefficient 3/d. -/
theorem gaussian_overlap_signed_chernoff {d n : ℕ} (hd : 2 ≤ d) (c β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d)) (i j : Idx n)
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ : ℝ)
    (hinit : ∀ ω, inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = γ)
    (hM : IsMartingaleOn T ℱ P (fun t ω => exponentialOverlap c i j (X t ω) -
      exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (X s ω)))
    (r : ℝ) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, c * r < c * (inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ -
      ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s ω) i j)} ≤
        ENNReal.ofReal (Real.exp (-c * r + c ^ 2 / 2 * (3 / (d : ℝ)) * T)) := by
  let a := Real.exp (c * γ + c * r - c ^ 2 / 2 * (3 / (d : ℝ)) * T)
  have hmax := gaussian_overlap_exponential_ville hd c β σV σA ρ hρ hσ P ℱ T hT X i j
    hunit hadapt hcont hM a (Real.exp_pos _)
  have hD (ω : Ω) : ContinuousOn (fun s => overlapDrift β ρ (X s ω) i j) (Icc (0 : ℝ) T) :=
    (continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j).comp' (hcont ω) (fun s hs => hunit s hs ω)
  have hsub : {ω | ∃ t ∈ Icc (0 : ℝ) T, c * r < c * (inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ -
      ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s ω) i j)} ⊆
      {ω | ∃ t ∈ Icc (0 : ℝ) T,
        a < Real.exp (-(∫ s in (0 : ℝ)..t, c * overlapDrift β ρ (X s ω) i j + c ^ 2 / 2 * (3 / (d : ℝ)))) *
          exponentialOverlap c i j (X t ω)} := by
    rintro ω ⟨t, ht, hc⟩
    refine ⟨t, ht, ?_⟩
    have hint : IntervalIntegrable (fun s => overlapDrift β ρ (X s ω) i j) volume 0 t := by
      apply ContinuousOn.intervalIntegrable
      rw [uIcc_of_le ht.1]
      exact (hD ω).mono (fun s hs => ⟨hs.1, hs.2.trans ht.2⟩)
    change Real.exp (c * γ + c * r - c ^ 2 / 2 * (3 / (d : ℝ)) * T) <
      Real.exp (-(∫ s in (0 : ℝ)..t, c * overlapDrift β ρ (X s ω) i j + c ^ 2 / 2 * (3 / (d : ℝ)))) *
        Real.exp (c * inner (𝕜 := ℝ) (X t ω i) (X t ω j))
    rw [exp_overlap_compensation_eq c (3 / (d : ℝ)) _ t _ hint]
    apply Real.exp_lt_exp.mpr
    have hq : 0 ≤ c ^ 2 / 2 * (3 / (d : ℝ)) := by positivity
    have htime := mul_le_mul_of_nonneg_left ht.2 hq
    nlinarith
  have h := (measure_mono hsub).trans hmax
  have hi : (∫ ω, exponentialOverlap c i j (X 0 ω) ∂P) = Real.exp (c * γ) := by
    have heq : (fun ω => exponentialOverlap c i j (X 0 ω)) = (fun _ : Ω => Real.exp (c * γ)) := by
      funext ω
      simp only [exponentialOverlap, hinit ω]
    rw [heq, integral_const, probReal_univ, one_smul]
  rw [hi] at h
  exact h.trans_eq (ofReal_exp_initial_div_exp_compensation c γ r (3 / (d : ℝ)) T)

/-- The optimized signed noise bound has exponent -dr²/(6T).
Source: arXiv:2604.01978v1, §7, the two-sided maximal inequality,
with the proved improved noise coefficient 3/d. -/
theorem gaussian_overlap_signed_maximal {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d)) (i j : Idx n)
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ : ℝ)
    (hinit : ∀ ω, inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = γ) (r : ℝ) (hr : 0 < r)
    (ε : ℝ) (hε : ε ^ 2 = 1)
    (hM : IsMartingaleOn T ℱ P (fun t ω => exponentialOverlap (ε * ((d : ℝ) * r / (3 * T))) i j (X t ω) -
      exponentialOverlap (ε * ((d : ℝ) * r / (3 * T))) i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
          (exponentialOverlap (ε * ((d : ℝ) * r / (3 * T))) i j) (X s ω))) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, r < ε * (inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ -
      ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s ω) i j)} ≤
        ENNReal.ofReal (Real.exp (-((d : ℝ) * r ^ 2 / (6 * T)))) := by
  let b := (d : ℝ) * r / (3 * T)
  have hdN : 0 < d := by omega
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hdN
  have hb : 0 < b := by dsimp [b]; positivity
  have h := gaussian_overlap_signed_chernoff hd (ε * b) β σV σA ρ hρ hσ P ℱ T hT X i j
    hunit hadapt hcont γ hinit hM (ε * r)
  have hcomp (z : ℝ) : (ε * b) * (ε * r) < (ε * b) * z ↔ r < ε * z := by
    have hlin : (ε * b) * (ε * r) = b * r := by
      calc
        _ = ε ^ 2 * (b * r) := by ring
        _ = _ := by rw [hε, one_mul]
    rw [hlin, show (ε * b) * z = b * (ε * z) by ring, mul_lt_mul_iff_right₀ hb]
  simp_rw [hcomp] at h
  have hlin : -(ε * b) * (ε * r) = -b * r := by
    calc
      _ = -(ε ^ 2) * (b * r) := by ring
      _ = _ := by rw [hε]; ring
  rw [hlin, mul_pow, hε, one_mul] at h
  exact h.trans_eq (congrArg (fun z => ENNReal.ofReal (Real.exp z))
    (overlap_chernoff_optimal_exponent (d : ℝ) T r hd0.ne' hT.ne'))

/-- A genuine Gaussian law, a frozen consensus configuration, and its
actual initial overlap 1 satisfy the selected exponential test identity.
The positive dimension and horizon also realize the optimization hypotheses. -/
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
    (∀ ω, inner (𝕜 := ℝ) (X 0 ω 0) (X 0 ω 1) = 1) ∧
    (2 : ℝ) ≠ 0 ∧ (1 : ℝ) ≠ 0 ∧ (1 : ℝ) ^ 2 = 1 ∧ (0 : ℝ) < 3 / 2 := by
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
    fun t ht => stronglyMeasurable_const, fun ω => continuousOn_const, ?_, fun ω => hR,
    by norm_num, by norm_num, by norm_num, by norm_num⟩
  have hz : (fun t ω => exponentialOverlap 1 0 1 (X t ω) - exponentialOverlap 1 0 1 (X 0 ω) -
      ∫ s in (0 : ℝ)..t, sphGenerator ρ (bField 1 ρ) (noiseField 1 1 1 ρ) (exponentialOverlap 1 0 1) (X s ω)) =
        (fun _ : ℝ => fun _ : Unit => (0 : ℝ)) := by
    funext t ω
    simp only [X, hGen, sub_self, intervalIntegral.integral_zero]
  rw [hz]
  exact isMartingaleOn_const 1 ℱ (Measure.dirac ()) 0

end Transformer.Homogenized
