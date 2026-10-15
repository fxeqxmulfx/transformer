import Transformer.Homogenized.Section7_OverlapNoiseMaximal
import Transformer.Homogenized.Section7_SimplexNoiseReserve

/-!
# Simultaneous original overlap noise within the source's radius

ArXiv:2604.01978v1, §7, `eq:martingale_sup_bound_clean` and
`thm:clustering_random_init`.
The actual two-sided distinct-pair estimate, with noise coefficient
3/d, is evaluated at seven eighths of the original displayed radius.
The previously proved numerical reserve pays for the entire 2n²
union factor and leaves failure probability at most δ.

The original residual is adapted and continuous: its drift
integral is the original one, with regularity derived from the
Gaussian Gram estimates. Thus the actual uniform-in-time event
is measurable, and the complement identity gives probability
at least 1-δ. No tail bound or uniform noise event is assumed.
The unused eighth of the radius remains available to absorb
the original finite-dimension drift bias in the deterministic
simplex comparison already proved.
-/

open Set MeasureTheory
open scoped NNReal ENNReal
namespace Transformer.Homogenized

/-- The actual uniform residual event is measurable, using the original drift integral.
Source: arXiv:2604.01978v1, §7, `eq:martingale_sup_bound_clean`. -/
theorem measurableSet_overlapNoiseResidual_le {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m) (T : ℝ) (hT : 0 < T)
    (X : ℝ → Ω → (Idx n → EucSpace d)) (i j : Idx n)
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ r : ℝ) :
    MeasurableSet {ω | ∀ t ∈ Icc (0 : ℝ) T, |overlapNoiseResidual β ρ X γ i j t ω| ≤ r} := by
  let D (t : ℝ) (ω : Ω) := overlapDrift β ρ (X t ω) i j
  have hfield := continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j
  have hDc (ω : Ω) : ContinuousOn (fun t => D t ω) (Icc (0 : ℝ) T) :=
    hfield.comp' (hcont ω) (fun t ht => hunit t ht ω)
  have hDm (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : StronglyMeasurable[ℱ t] (D t) :=
    stronglyMeasurable_comp_of_continuousOn_range _ _ hfield (X t) (hadapt t ht).measurable (hunit t ht)
  have hInner : Continuous (fun Z : Idx n → EucSpace d => inner (𝕜 := ℝ) (Z i) (Z j)) :=
    (continuous_apply i).inner (continuous_apply j)
  apply measurableSet_continuous_time_le (fun t ω => |overlapNoiseResidual β ρ X γ i j t ω|) T hT
  · intro t ht
    have hprim := stronglyMeasurable_adapted_timeIntegral ℱ T D hDm hDc t ht
    have hres : StronglyMeasurable[ℱ t] (fun ω => overlapNoiseResidual β ρ X γ i j t ω) :=
      ((hInner.comp_stronglyMeasurable (hadapt t ht)).sub stronglyMeasurable_const).sub hprim
    have habs : Continuous (fun z : ℝ => |z|) := continuous_id.abs
    exact ((habs.comp_stronglyMeasurable hres).mono (ℱ.le t)).measurable
  · intro ω
    have hi : IntegrableOn (fun t => D t ω) (uIcc (0 : ℝ) T) volume := by
      rw [uIcc_of_le hT.le]
      exact (hDc ω).integrableOn_Icc
    have hprim : ContinuousOn (fun t => ∫ s in (0 : ℝ)..t, D s ω) (Icc (0 : ℝ) T) := by
      simpa only [uIcc_of_le hT.le] using intervalIntegral.continuousOn_primitive_interval hi
    exact (((hInner.comp_continuousOn (hcont ω)).sub continuousOn_const).sub hprim).abs

/-- The original source radius is strictly positive throughout its parameter range.
Source: arXiv:2604.01978v1, §7, `thm:clustering_random_init`. -/
theorem simplexNoiseRadius_pos (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    0 < simplexNoiseRadius d n T δ := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hl := one_le_simplex_noise_log n hn δ hδ
  apply Real.sqrt_pos.mpr
  exact mul_pos (div_pos (by positivity) hd0) (by linarith)

/-- The proved scalar reserve bounds the actual pair-union failure expression.
Source: arXiv:2604.01978v1, §7, `eq:martingale_sup_bound_clean`. -/
theorem simplex_noise_exponential_reserve_ennreal (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    2 * (n : ℝ≥0∞) ^ 2 * ENNReal.ofReal (Real.exp (-((d : ℝ) *
      ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ) ^ 2 / (6 * T)))) ≤ ENNReal.ofReal δ := by
  have h := ENNReal.ofReal_le_ofReal (simplex_noise_exponential_reserve d n hd hn T hT δ hδ)
  have he : -(d : ℝ) * ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ) ^ 2 / (6 * T) =
      -((d : ℝ) * ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ) ^ 2 / (6 * T)) := by ring
  rw [he, ENNReal.ofReal_mul (by positivity : (0 : ℝ) ≤ 2 * (n : ℝ) ^ 2),
    ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 2), ENNReal.ofReal_pow (Nat.cast_nonneg n),
    ENNReal.ofReal_natCast] at h
  norm_num at h ⊢
  exact h

/-- All actual distinct-pair noise residuals stay within seven eighths
of the original source radius with probability at least 1-δ.
Source: arXiv:2604.01978v1, §7, `eq:martingale_sup_bound_clean`, with
coefficient 3/d derived from the actual Gaussian noise law. -/
theorem gaussian_overlap_noise_probability {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ)) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    (X : ℝ → Ω → (Idx n → EucSpace d))
    (hunit : ∀ t ∈ Icc (0 : ℝ) T, ∀ ω k, ‖X t ω k‖ = 1)
    (hadapt : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t))
    (hcont : ∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (γ : ℝ)
    (hinit : ∀ ω i j, i ≠ j → inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = γ)
    (hM : ∀ (c : ℝ) (i j : Idx n), i ≠ j → IsMartingaleOn T ℱ P
      (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) (X s ω))) :
    ENNReal.ofReal (1 - δ) ≤ P {ω | ∀ i j : Idx n, i ≠ j → ∀ t ∈ Icc (0 : ℝ) T,
      |overlapNoiseResidual β ρ X γ i j t ω| ≤ (7 / 8 : ℝ) * simplexNoiseRadius d n T δ} := by
  let r := (7 / 8 : ℝ) * simplexNoiseRadius d n T δ
  have hr : 0 < r := mul_pos (by norm_num) (simplexNoiseRadius_pos d n hd hn T hT δ hδ)
  let Bad : Set Ω := {ω | ∃ i j : Idx n, i ≠ j ∧ ∃ t ∈ Icc (0 : ℝ) T,
    r < |overlapNoiseResidual β ρ X γ i j t ω|}
  have hBad : P Bad ≤ ENNReal.ofReal δ :=
    (gaussian_overlap_noise_pairs hd β σV σA ρ hρ hσ P ℱ T hT X hunit hadapt hcont γ hinit r hr hM).trans
      (simplex_noise_exponential_reserve_ennreal d n hd hn T hT δ hδ)
  have hGood : {ω | ∀ i j : Idx n, i ≠ j → ∀ t ∈ Icc (0 : ℝ) T,
      |overlapNoiseResidual β ρ X γ i j t ω| ≤ r} = Badᶜ := by
    ext ω
    simp only [Bad, mem_ofPred_eq, mem_compl_iff, not_exists, not_and, not_lt]
  have hGoodMeas : MeasurableSet {ω | ∀ i j : Idx n, i ≠ j → ∀ t ∈ Icc (0 : ℝ) T,
      |overlapNoiseResidual β ρ X γ i j t ω| ≤ r} := by
    have heq : {ω | ∀ i j : Idx n, i ≠ j → ∀ t ∈ Icc (0 : ℝ) T,
        |overlapNoiseResidual β ρ X γ i j t ω| ≤ r} =
        ⋂ i : Idx n, ⋂ j : Idx n, ⋂ (_ : i ≠ j),
          {ω | ∀ t ∈ Icc (0 : ℝ) T, |overlapNoiseResidual β ρ X γ i j t ω| ≤ r} := by
      ext ω
      simp only [mem_ofPred_eq, mem_iInter]
    rw [heq]
    exact MeasurableSet.iInter fun i => MeasurableSet.iInter fun j => MeasurableSet.iInter fun _ =>
      measurableSet_overlapNoiseResidual_le hd β σV σA ρ hρ ℱ T hT X i j hunit hadapt hcont γ r
  rw [hGood] at hGoodMeas
  have hBadMeas : MeasurableSet Bad := by simpa only [compl_compl] using hGoodMeas.compl
  rw [hGood, measure_compl hBadMeas (measure_ne_top P Bad), measure_univ, ENNReal.ofReal_sub 1 hδ.1.le]
  norm_num only [ENNReal.ofReal_one]
  exact tsub_le_tsub_left hBad 1

/-- Positive Gaussian value and key variances, unit paths, actual initial
overlaps, every selected exponential test identity, T=1 and δ=1/2
satisfy every hypothesis. The source radius is therefore nonzero. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)),
    (2 : ℕ) ≤ 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω k, ‖X t ω k‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (X t)) ∧
    (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) 1)) ∧
    (∀ ω i j, inner (𝕜 := ℝ) (X 0 ω i) (X 0 ω j) = 1) ∧
    ∀ (c : ℝ) (i j : Idx 2), IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 1) (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
          (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (exponentialOverlap c i j) (X s ω)) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ _ => v
  have hv : ‖v‖ = 1 := by simp [v]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 1
  refine ⟨ℱ, X, le_rfl, hρ, stdSigmaV_sq 2, one_pos, by norm_num,
    fun t ht ω k => hv, fun t ht => stronglyMeasurable_const, fun ω => continuousOn_const, ?_, ?_⟩
  · intro ω i j
    change inner (𝕜 := ℝ) v v = 1
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  · intro c i j
    exact isMartingaleOn_frozen_consensus_exponentialOverlap c 1 (stdSigmaV 2) 1 _ hρ (stdSigmaV_sq 2)
      (Measure.dirac ()) ℱ 1 v hv i j

end Transformer.Homogenized
