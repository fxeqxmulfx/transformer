import Transformer.Homogenized.Section7_GaussianOverlapChernoff

/-!
# The actual Gaussian overlap noise: two sides and all token pairs

ArXiv:2604.01978v1, §7, the maximal inequality and union bound
leading to `eq:martingale_sup_bound_clean`.
The noise residual is exactly R(t)-γ-∫₀ᵗD with the original
overlap drift. The two signed exponential test martingales give
a continuous-time two-sided bound with the proved exponent
d r²/(6T). A finite union over actual distinct token pairs then
gives the factor 2n². No noise estimate is assumed.

The selected exponential-test martingale hypotheses are explicit.
A frozen consensus configuration is proved to satisfy every such
test identity under a genuine positive-value-variance Gaussian
head law. This realizes the stochastic hypotheses used here.
The witnesses and bound concern the selected overlap observables.
-/

open Set MeasureTheory
open scoped NNReal ENNReal BigOperators
namespace Transformer.Homogenized

/-- The original overlap minus its original deterministic initial overlap and drift integral.
Source: arXiv:2604.01978v1, §7, the decomposition before `eq:Gronwall_E_selfcontained`. -/
noncomputable def overlapNoiseResidual {d n : ℕ} {Ω : Type*} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : ℝ → Ω → (Idx n → EucSpace d)) (γ : ℝ) (i j : Idx n) (t : ℝ) (ω : Ω) : ℝ :=
  inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ - ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s ω) i j

/-- Frozen unit consensus gives a genuine martingale for every selected
exponential overlap test. Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem isMartingaleOn_frozen_consensus_exponentialOverlap {d n : ℕ} (c β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (v : EucSpace d) (hv : ‖v‖ = 1) (i j : Idx n) :
    IsMartingaleOn T ℱ P (fun t _ => exponentialOverlap c i j (fun _ => v) -
      exponentialOverlap c i j (fun _ => v) - ∫ _ in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (fun _ => v)) := by
  let K : Idx n → EucSpace d := fun _ => v
  have hK : ∀ k, ‖K k‖ = 1 := fun k => show ‖K k‖ = 1 from hv
  have hR : inner (𝕜 := ℝ) (K i) (K j) = 1 := by
    change inner (𝕜 := ℝ) v v = 1
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  have hD : overlapDrift β ρ K i j = 0 := by
    change overlapDrift β ρ K i i = 0
    exact overlapDrift_self β ρ K i (hK i)
  have hq : (∫ θ, overlapNoiseIntegrand β ρ K i j θ ^ 2 ∂ρ) = 0 := by
    rw [integral_overlapNoiseIntegrand_sq_factor β hρ K hK, hR]
    norm_num
  have hGen : sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) K = 0 := by
    rw [sphGenerator_exponentialOverlap_gaussian c β hρ hσ K hK, hD, hq]
    ring
  have heq : (fun (t : ℝ) (_ : Ω) => exponentialOverlap c i j (fun _ => v) - exponentialOverlap c i j (fun _ => v) -
      ∫ _ in (0 : ℝ)..t, sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (fun _ => v)) =
        (fun _ : ℝ => fun _ : Ω => (0 : ℝ)) := by
    funext t ω
    change exponentialOverlap c i j K - exponentialOverlap c i j K -
      (∫ _ in (0 : ℝ)..t, sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) K) = 0
    simp only [hGen, sub_self, intervalIntegral.integral_zero]
  rw [heq]
  exact isMartingaleOn_const T ℱ P 0

/-- Both signs give the actual continuous-time two-sided overlap-noise bound.
Source: arXiv:2604.01978v1, §7, the maximal inequality after the bracket calculation. -/
theorem gaussian_overlap_noise_maximal {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d)) (i j : Idx n)
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ : ℝ)
    (hinit : ∀ ω, inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = γ) (r : ℝ) (hr : 0 < r)
    (hM : ∀ c : ℝ, IsMartingaleOn T ℱ P (fun t ω => exponentialOverlap c i j (X t ω) -
      exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (X s ω))) :
    P {ω | ∃ t ∈ Icc (0 : ℝ) T, r < |overlapNoiseResidual β ρ X γ i j t ω|} ≤
      2 * ENNReal.ofReal (Real.exp (-((d : ℝ) * r ^ 2 / (6 * T)))) := by
  have hp := gaussian_overlap_signed_maximal hd β σV σA ρ hρ hσ P ℱ T hT X i j
    hunit hadapt hcont γ hinit r hr 1 (by norm_num) (hM _)
  have hm := gaussian_overlap_signed_maximal hd β σV σA ρ hρ hσ P ℱ T hT X i j
    hunit hadapt hcont γ hinit r hr (-1) (by norm_num) (hM _)
  change P {ω | ∃ t ∈ Icc (0 : ℝ) T, r < 1 * overlapNoiseResidual β ρ X γ i j t ω} ≤ _ at hp
  change P {ω | ∃ t ∈ Icc (0 : ℝ) T, r < (-1) * overlapNoiseResidual β ρ X γ i j t ω} ≤ _ at hm
  have hsub : {ω | ∃ t ∈ Icc (0 : ℝ) T, r < |overlapNoiseResidual β ρ X γ i j t ω|} ⊆
      {ω | ∃ t ∈ Icc (0 : ℝ) T, r < 1 * overlapNoiseResidual β ρ X γ i j t ω} ∪
      {ω | ∃ t ∈ Icc (0 : ℝ) T, r < (-1) * overlapNoiseResidual β ρ X γ i j t ω} := by
    rintro ω ⟨t, ht, hc⟩
    rcases lt_abs.mp hc with hpos | hneg
    · exact Or.inl ⟨t, ht, by simpa only [one_mul] using hpos⟩
    · exact Or.inr ⟨t, ht, by simpa only [neg_one_mul] using hneg⟩
  exact (measure_mono hsub).trans ((measure_union_le _ _).trans
    ((add_le_add hp hm).trans_eq (by rw [two_mul])))
/-- The actual distinct-pair union has probability at most 2n²exp(-dr²/(6T)).
Source: arXiv:2604.01978v1, §7, the union bound before `eq:martingale_sup_bound_clean`. -/
theorem gaussian_overlap_noise_pairs {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (X : ℝ → Ω → (Idx n → EucSpace d))
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ : ℝ)
    (hinit : ∀ ω i j, i ≠ j → inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = γ) (r : ℝ) (hr : 0 < r)
    (hM : ∀ (c : ℝ) (i j : Idx n), i ≠ j → IsMartingaleOn T ℱ P
      (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (X s ω))) :
    P {ω | ∃ i j : Idx n, i ≠ j ∧ ∃ t ∈ Icc (0 : ℝ) T, r < |overlapNoiseResidual β ρ X γ i j t ω|} ≤
      2 * (n : ℝ≥0∞) ^ 2 * ENNReal.ofReal (Real.exp (-((d : ℝ) * r ^ 2 / (6 * T)))) := by
  classical
  let e := ENNReal.ofReal (Real.exp (-((d : ℝ) * r ^ 2 / (6 * T))))
  let Bad (p : Idx n × Idx n) : Set Ω := if p.1 = p.2 then ∅ else
    {ω | ∃ t ∈ Icc (0 : ℝ) T, r < |overlapNoiseResidual β ρ X γ p.1 p.2 t ω|}
  have hEach (p : Idx n × Idx n) : P (Bad p) ≤ 2 * e := by
    by_cases hp : p.1 = p.2
    · simp only [Bad, ite_eq_left hp, measure_empty]
      exact zero_le
    · simpa only [Bad, ite_eq_right hp] using gaussian_overlap_noise_maximal hd β σV σA ρ hρ hσ
        P ℱ T hT X p.1 p.2 hunit hadapt hcont γ (fun ω => hinit ω p.1 p.2 hp) r hr
        (fun c => hM c p.1 p.2 hp)
  have hsub : {ω | ∃ i j : Idx n, i ≠ j ∧ ∃ t ∈ Icc (0 : ℝ) T, r < |overlapNoiseResidual β ρ X γ i j t ω|} ⊆
      ⋃ p, Bad p := by
    rintro ω ⟨i, j, hij, t, ht, hrt⟩
    apply mem_iUnion.mpr
    refine ⟨(i, j), ?_⟩
    simpa only [Bad, ite_eq_right hij, mem_ofPred_eq] using (show ∃ t ∈ Icc (0 : ℝ) T,
      r < |overlapNoiseResidual β ρ X γ i j t ω| from ⟨t, ht, hrt⟩)
  calc
    _ ≤ P (⋃ p, Bad p) := measure_mono hsub
    _ ≤ ∑ p, P (Bad p) := measure_iUnion_fintype_le P Bad
    _ ≤ ∑ p : Idx n × Idx n, 2 * e := Finset.sum_le_sum (fun p hp => hEach p)
    _ = (n : ℝ≥0∞) ^ 2 * (2 * e) := by simp [Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul, pow_two]
    _ = _ := by ring


/-- Positive value variance, unit consensus paths, their actual initial
overlaps, and every exponential test identity realize all hypotheses. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (v : EucSpace 2),
    (2 : ℕ) ≤ 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 0 (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ (0 : ℝ) < 1 ∧ ‖v‖ = 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω k, ‖X t ω k‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (X t)) ∧
    (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) 1)) ∧
    (∀ ω i j, inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = 1) ∧ (0 : ℝ) < 1 / 2 ∧
    ∀ (c : ℝ) (i j : Idx 2), IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 0) (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 0))
          (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0)) (exponentialOverlap c i j) (X s ω)) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ _ => v
  have hv : ‖v‖ = 1 := by simp [v]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 0
  refine ⟨ℱ, X, v, le_rfl, hρ, stdSigmaV_sq 2, one_pos, hv,
    fun t ht ω k => hv, fun t ht => stronglyMeasurable_const, fun ω => continuousOn_const,
    ?_, by norm_num, ?_⟩
  · intro ω i j
    change inner (𝕜 := ℝ) v v = 1
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  · intro c i j
    exact isMartingaleOn_frozen_consensus_exponentialOverlap c 1 (stdSigmaV 2) 0 _ hρ (stdSigmaV_sq 2)
      (Measure.dirac ()) ℱ 1 v hv i j

end Transformer.Homogenized
