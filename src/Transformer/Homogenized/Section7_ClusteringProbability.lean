import Transformer.Homogenized.Section7_SimplexConsensus
import Transformer.Homogenized.Section7_SphereMartingale
import Transformer.Homogenized.Section7_OverlapNoiseProbability

/-!
# The source simplex clustering probability estimate

ArXiv:2604.01978v1, §2.3.4, `thm:clustering_random_init`,
and its proof in §7. The original overlap error stays within
exp(82(1+dσA⁴β²n)T) sqrt(8T/d log(2n²/δ)) with probability
at least 1-δ. Both the actual noise probability and the actual
pathwise drift comparison have been proved from the original fields.

The adopted value scaling σV²=1/d is explicit, as required by
the source after `cor:weak_error_centered`. The process law is
a probability measure. The adapted continuous unit paths carry
the conditional martingale identities supplied by `def: ito.formula`.
Only its actual exponential overlap tests are needed here;
`IsSphereMartingaleSolution.exponentialOverlap` supplies them
from the full smooth-test martingale formulation. The former
marginal expectation derivative alone did not imply these identities.

The source's bounded moments are supplied in the form proved by
`simplex_overlap_wellDefined`, including their [0,1] extensions.
The strict-interior proof uses only their actual simplex identities.
The bounds additionally permit the consensus initial endpoint,
which this theorem includes. The temperature may be any real value.
The finite-dimension drift bias omitted in the source's exact
simplex substitution is retained and absorbed by the proved reserve.
The coefficient 82 is universal over all the displayed model data.

The concrete example below checks every premise simultaneously
at positive source-normalized Gaussian value variance, positive
horizon, genuine attention moments and consensus initial data.
It verifies precisely the selected observable martingale conditions.
-/

open MeasureTheory Set
open scoped BigOperators NNReal ENNReal
namespace Transformer.Homogenized

/-- The source's uniform simplex overlap estimate, with universal K=82.
Source: arXiv:2604.01978v1, §2.3.4, `thm:clustering_random_init`, proof §7.
The source assumes γ₀<1 and β>0; both restrictions are relaxed here.
Its adopted σV²=1/d and probability law are explicit. The original
bounded attention moments, ODE and conditional exponential-test martingales
are inputs; no tail estimate, supermartingale or probability bound is assumed.
The source's full smooth-test martingale solution supplies every stochastic
input through `IsSphereMartingaleSolution` and its proved test selector. -/
theorem clustering_random_init :
    ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ), 2 ≤ d → 2 ≤ n → ∀ (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)),
      IsGaussianHeadLaw d σV σA ρ → (σV : ℝ) ^ 2 = 1 / (d : ℝ) →
      ∀ f g : ℝ → ℝ, (∀ c, f c ∈ Icc (0 : ℝ) 1) → (∀ c, g c ∈ Icc (0 : ℝ) 1) →
        (∀ (c : ℝ) (Y : Idx n → EucSpace d), IsSimplexConfig c Y → ∀ i,
          (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c) →
        (∀ (c : ℝ) (Y : Idx n → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
          (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c) →
      ∀ γ₀ ∈ Ioc (-(1 / ((n : ℝ) - 1))) 1,
      ∀ γ : ℝ → ℝ, γ 0 = γ₀ →
        (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) →
      ∀ T : ℝ, 0 < T → ∀ δ ∈ Ioo (0 : ℝ) 1,
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
        (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Ω))
        (X : ℝ → Ω → (Idx n → EucSpace d)),
        (∀ t ∈ Icc (0 : ℝ) T, ∀ ω i, ‖X t ω i‖ = 1) →
        (∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t)) →
        (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) →
        (∀ ω, IsSimplexConfig γ₀ (X 0 ω)) →
        (∀ (c : ℝ) (i j : Idx n), i ≠ j → IsMartingaleOn T ℱ P
          (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) -
            ∫ s in (0 : ℝ)..t, sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ)
              (exponentialOverlap c i j) (X s ω))) →
        ENNReal.ofReal (1 - δ) ≤ P {ω | ∀ t ∈ Icc (0 : ℝ) T, ∀ i j : Idx n, i ≠ j →
          |inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ t| ≤
            Real.exp (K * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * T) *
              Real.sqrt (8 * T / (d : ℝ) * Real.log (2 * (n : ℝ) ^ 2 / δ))} := by
  refine ⟨82, by norm_num, ?_⟩
  intro d N hd hn
  cases N with
  | zero => omega
  | succ n =>
    intro β σV σA ρ hρ hσ f g hfBound hgBound hf hg γ₀ hγ₀ γ hinit hdγ T hT δ hδ Ω m P hP ℱ X
      hunit hadapt hcont hX₀ hM
    have hp := gaussian_overlap_noise_probability hd hn β σV σA ρ hρ hσ P ℱ T hT δ hδ X
      hunit hadapt hcont γ₀ (fun ω i j hij => (hX₀ ω).2 i j hij) hM
    apply hp.trans
    apply measure_mono
    intro ω hNoise t ht i j hij
    have hE : simplexOverlapError (X t ω) (γ t) ≤
        Real.exp (82 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) * T) *
          simplexNoiseRadius d (n + 1) T δ := by
      by_cases hlt : γ₀ < 1
      · exact simplexOverlapError_le_of_noiseResidual hd hn β σV σA ρ hρ f g hf hg γ₀
          ⟨hγ₀.1, hlt⟩ γ hinit hdγ T hT.le δ hδ X ω (hX₀ ω)
          (fun s hs k => hunit s hs ω k) (hcont ω)
          (fun s hs p => hNoise p.1.1 p.1.2 p.2 s hs) t ht
      · have heq : γ₀ = 1 := le_antisymm hγ₀.2 (le_of_not_gt hlt)
        rw [heq] at hinit hX₀ hNoise
        rw [simplex_ode_eq_one f g hfBound hgBound γ hinit hdγ t ht.1]
        exact simplexOverlapError_le_of_noiseResidual_consensus hd hn β σV σA ρ hρ f g hf hg T hT.le δ hδ
          X ω (hX₀ ω) (fun s hs k => hunit s hs ω k) (hcont ω)
          (fun s hs p => hNoise p.1.1 p.1.2 p.2 s hs) t ht
    exact (abs_overlap_sub_simplex_le_error (X t ω) (γ t) i j hij).trans hE

/-- Every hypothesis is realized with d=n=2, positive Gaussian variances,
β=T=1, δ=1/2, the original moments, and actual consensus overlap tests. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (f g : ℝ → ℝ) (X : ℝ → Unit → (Idx 2 → EucSpace 2)),
    (2 : ℕ) ≤ 2 ∧ IsProbabilityMeasure (Measure.dirac ()) ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧
    (∀ c, f c ∈ Icc (0 : ℝ) 1) ∧ (∀ c, g c ∈ Icc (0 : ℝ) 1) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k ^ 2 ∂(gaussHeadLaw 2 (stdSigmaV 2) 1)) = f c) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k * attnProb 1 θ.2 Y j k ∂(gaussHeadLaw 2 (stdSigmaV 2) 1)) = g c) ∧
    (1 : ℝ) ∈ Ioc (-(1 / ((2 : ℝ) - 1))) 1 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt (fun _ : ℝ => (1 : ℝ)) (simplexDrift f g 1) (Ici 0) t) ∧
    (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω i, ‖X t ω i‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (X t)) ∧
    (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) 1)) ∧
    (∀ ω, IsSimplexConfig 1 (X 0 ω)) ∧
    ∀ (c : ℝ) (i j : Idx 2), IsMartingaleOn 1 ℱ (Measure.dirac ())
      (fun t ω => exponentialOverlap c i j (X t ω) - exponentialOverlap c i j (X 0 ω) -
        ∫ s in (0 : ℝ)..t, sphGenerator (gaussHeadLaw 2 (stdSigmaV 2) 1)
          (bField 1 (gaussHeadLaw 2 (stdSigmaV 2) 1)) (noiseField 1 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 1))
          (exponentialOverlap c i j) (X s ω)) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ _ => v
  have hv : ‖v‖ = 1 := by simp [v]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 1
  obtain ⟨f, g, hfb, hgb, hf, hg⟩ := simplex_overlap_wellDefined (d := 2) (n := 2) 1 _ _ _ hρ
  refine ⟨ℱ, f, g, X, le_rfl, inferInstance, hρ, stdSigmaV_sq 2, hfb, hgb, hf, hg,
    by norm_num, ?_, one_pos, by norm_num, fun t ht ω i => hv,
    fun t ht => stronglyMeasurable_const, fun ω => continuousOn_const, ?_, ?_⟩
  · intro t ht
    simpa [simplexDrift] using (hasDerivWithinAt_const t (Ici (0 : ℝ)) (1 : ℝ))
  · intro ω
    refine ⟨fun i => hv, ?_⟩
    intro i j hij
    change inner (𝕜 := ℝ) v v = 1
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  · intro c i j
    exact isMartingaleOn_frozen_consensus_exponentialOverlap c 1 (stdSigmaV 2) 1 _ hρ
      (stdSigmaV_sq 2) (Measure.dirac ()) ℱ 1 v hv i j

end Transformer.Homogenized
