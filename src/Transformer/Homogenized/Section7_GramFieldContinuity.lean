module

public import Transformer.Homogenized.Section7_ExponentialDriftBounds
public import Mathlib.Topology.MetricSpace.Lipschitz

/-!
# Continuity of the original Gaussian overlap fields

ArXiv:2604.01978v1, §7, `lem:Gram_stability`,
`eq:Dij_explicit_clean`, and the original overlap-noise calculation.
The norm distance between two unit configurations controls each
actual Gram entry by twice that distance. The proved Gaussian
Gram estimates therefore give continuity of the original
barycenter correlations and overlap drift on all unit configurations.

The original noise second moment is a polynomial in those same
correlations and overlaps, so it is continuous as well. The actual
exponential generator identity transfers this to the original
generator, without a continuity assumption on the time drift.
The proofs include singular Gram matrices and arbitrary temperature.
The final composition lemma transfers this regularity through
an adapted process whose actual range lies in the unit configurations.
-/

@[expose] public section

open Set MeasureTheory
open scoped NNReal
namespace Transformer.Homogenized

/-- Actual unit configurations have Gram distance at most twice their norm distance.
Source: arXiv:2604.01978v1, §7, the configurations in `lem:Gram_stability`. -/
theorem abs_unit_gram_sub_le_norm {d n : ℕ} (X Y : Idx n → EucSpace d)
    (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1) (i j : Idx n) :
    |inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (Y i) (Y j)| ≤ 2 * ‖X - Y‖ := by
  have heq : inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (Y i) (Y j) =
      inner (𝕜 := ℝ) (X i - Y i) (X j) + inner (𝕜 := ℝ) (Y i) (X j - Y j) := by
    rw [inner_sub_left, inner_sub_right]
    ring
  rw [heq]
  calc
    _ ≤ |inner (𝕜 := ℝ) (X i - Y i) (X j)| + |inner (𝕜 := ℝ) (Y i) (X j - Y j)| := abs_add_le _ _
    _ ≤ ‖X i - Y i‖ * ‖X j‖ + ‖Y i‖ * ‖X j - Y j‖ :=
      add_le_add (abs_real_inner_le_norm _ _) (abs_real_inner_le_norm _ _)
    _ = ‖X i - Y i‖ + ‖X j - Y j‖ := by rw [hX j, hY i, mul_one, one_mul]
    _ ≤ ‖X - Y‖ + ‖X - Y‖ := add_le_add (norm_le_pi_norm (X - Y) i) (norm_le_pi_norm (X - Y) j)
    _ = _ := by ring

/-- The original Gaussian barycenter correlation is continuous even at singular Grams.
Source: arXiv:2604.01978v1, §7, the expectation estimate in `lem:Gram_stability`. -/
theorem continuousOn_baryCorr_empMeasure_gaussian {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (i j : Idx n) :
    ContinuousOn (fun X : Idx n → EucSpace d => baryCorr β ρ (empMeasure X) (X i) (X j))
      {X | ∀ a, ‖X a‖ = 1} := by
  let K : ℝ≥0 := ⟨2 * (1 + 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)), by positivity⟩
  apply LipschitzOnWith.continuousOn (K := K)
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro X hX Y hY
  rw [Real.dist_eq, dist_eq_norm]
  have h := abs_baryCorr_sub_le_gram i.pos β σV σA ρ hρ X Y hX hY i j (2 * ‖X - Y‖)
    (abs_unit_gram_sub_le_norm X Y hX hY)
  change |baryCorr β ρ (empMeasure X) (X i) (X j) - baryCorr β ρ (empMeasure Y) (Y i) (Y j)| ≤
    2 * (1 + 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * ‖X - Y‖
  nlinarith

/-- The original Gaussian overlap drift is continuous on unit configurations.
Source: arXiv:2604.01978v1, §7, `lem:Gram_stability` and `eq:Dij_explicit_clean`. -/
theorem continuousOn_overlapDrift_gaussian {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (i j : Idx n) :
    ContinuousOn (fun X : Idx n → EucSpace d => overlapDrift β ρ X i j) {X | ∀ a, ‖X a‖ = 1} := by
  let K : ℝ≥0 := ⟨160 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)), by positivity⟩
  apply LipschitzOnWith.continuousOn (K := K)
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro X hX Y hY
  rw [Real.dist_eq, dist_eq_norm]
  have h := abs_overlapDrift_sub_le_gram hd β σV σA ρ hρ X Y hX hY i j (2 * ‖X - Y‖)
    (abs_unit_gram_sub_le_norm X Y hX hY)
  change |overlapDrift β ρ X i j - overlapDrift β ρ Y i j| ≤
    160 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * ‖X - Y‖
  nlinarith
/-- The actual overlap-noise second moment is continuous on unit configurations.
Source: arXiv:2604.01978v1, §7, the original Gaussian bracket calculation. -/
theorem continuousOn_overlapNoise_sq_integral_gaussian {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (i j : Idx n) :
    ContinuousOn (fun X : Idx n → EucSpace d => ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ)
      {X | ∀ a, ‖X a‖ = 1} := by
  have hR : ContinuousOn (fun X : Idx n → EucSpace d => inner (𝕜 := ℝ) (X i) (X j))
      {X | ∀ a, ‖X a‖ = 1} := ((continuous_apply i).inner (continuous_apply j)).continuousOn
  have hP := continuousOn_baryCorr_empMeasure_gaussian β σV σA ρ hρ i i
  have hQ := continuousOn_baryCorr_empMeasure_gaussian β σV σA ρ hρ j j
  have hS := continuousOn_baryCorr_empMeasure_gaussian β σV σA ρ hρ i j
  have hp : ContinuousOn (fun X : Idx n → EucSpace d =>
      (σV : ℝ) ^ 2 * (1 - inner (𝕜 := ℝ) (X i) (X j) ^ 2) *
        (baryCorr β ρ (empMeasure X) (X i) (X i) + baryCorr β ρ (empMeasure X) (X j) (X j) -
          2 * inner (𝕜 := ℝ) (X i) (X j) * baryCorr β ρ (empMeasure X) (X i) (X j)))
      {X | ∀ a, ‖X a‖ = 1} :=
    (continuousOn_const.mul (continuousOn_const.sub (hR.pow 2))).mul
      ((hP.add hQ).sub ((continuousOn_const.mul hR).mul hS))
  apply hp.congr
  intro X hX
  exact integral_overlapNoiseIntegrand_sq_factor β hρ X hX i j

/-- The original exponential generator is continuous on the actual sphere product.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`, at σV²=1/d. -/
theorem continuousOn_sphGenerator_exponentialOverlap_gaussian {d n : ℕ} (hd : 2 ≤ d) (c β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (i j : Idx n) :
    ContinuousOn (fun X : Idx n → EucSpace d =>
      sphGenerator ρ (bField β ρ) (noiseField β 1 1 ρ) (exponentialOverlap c i j) X)
      {X | ∀ a, ‖X a‖ = 1} := by
  have he : ContinuousOn (exponentialOverlap (d := d) c i j) {X | ∀ a, ‖X a‖ = 1} :=
    (contDiff_exponentialOverlap c i j).continuous.continuousOn
  have hD := continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j
  have hq := continuousOn_overlapNoise_sq_integral_gaussian β σV σA ρ hρ i j
  have hp : ContinuousOn (fun X : Idx n → EucSpace d => exponentialOverlap c i j X *
      (c * overlapDrift β ρ X i j + c ^ 2 / 2 * ∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ))
      {X | ∀ a, ‖X a‖ = 1} :=
    he.mul ((continuousOn_const.mul hD).add (continuousOn_const.mul hq))
  apply hp.congr
  intro X hX
  exact sphGenerator_exponentialOverlap_gaussian c β hρ hσ X hX i j

/-- A field continuous on the actual range of a measurable process composes measurably.
Source: arXiv:2604.01978v1, §7, adaptedness of the overlap integrating factor. -/
theorem stronglyMeasurable_comp_of_continuousOn_range {Ω E : Type*} {m : MeasurableSpace Ω}
    [TopologicalSpace E] [MeasurableSpace E] [BorelSpace E] (f : E → ℝ) (S : Set E)
    (hf : ContinuousOn f S) (X : Ω → E) (hX : Measurable X) (hXS : ∀ ω, X ω ∈ S) :
    StronglyMeasurable (fun ω => f (X ω)) := by
  have hm := hf.domRestrict.measurable.comp (hX.subtype_mk (h := hXS))
  exact hm.stronglyMeasurable

/-- Genuine positive Gaussian variances, an orthonormal configuration,
and a coincident-token configuration realize every field hypothesis.
The actual drift also composes strongly measurably through a unit-valued map. -/
example : ∃ X Y : Idx 2 → EucSpace 2,
    (2 : ℕ) ≤ 2 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧
    (∀ k, ‖X k‖ = 1) ∧ (∀ k, ‖Y k‖ = 1) ∧
    Measurable (fun _ : Unit => X) ∧
    (∀ ω : Unit, (fun _ : Unit => X) ω ∈ {Z | ∀ k, ‖Z k‖ = 1}) ∧
    StronglyMeasurable (fun _ : Unit => overlapDrift 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) X 0 1) := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  let Y : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hY : ∀ k, ‖Y k‖ = 1 := fun k => by simp [Y]
  have hρ := isGaussianHeadLaw_gaussHeadLaw 2 (stdSigmaV 2) 1
  refine ⟨X, Y, le_rfl, hρ, stdSigmaV_sq 2, hX, hY, measurable_const, fun ω => hX, ?_⟩
  exact stronglyMeasurable_comp_of_continuousOn_range (fun Z => overlapDrift 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) Z 0 1)
    {Z | ∀ k, ‖Z k‖ = 1} (continuousOn_overlapDrift_gaussian le_rfl 1 (stdSigmaV 2) 1 _ hρ 0 1)
    (fun _ : Unit => X) measurable_const (fun ω => hX)

end Transformer.Homogenized
