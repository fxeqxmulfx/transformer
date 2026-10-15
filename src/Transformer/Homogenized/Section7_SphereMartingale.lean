import Transformer.Homogenized.Section7_ExponentialObservable
import Transformer.Homogenized.MvGenerator
import Transformer.Homogenized.GaussianEnsemble

/-!
# The finite-horizon sphere martingale problem

ArXiv:2604.01978v1, Appendix, `def: ito.formula`, and §7,
`thm:clustering_random_init`. The source's adapted continuous
sphere-valued process has a martingale compensator for every
smooth test. This finite-horizon predicate records that condition
using the original Riemannian generator and ambient smooth extensions.

It retains conditional martingale identities, rather than only
their marginal expectation derivatives. Restriction to a shorter
horizon and specialization to the actual exponential overlap tests
are proved directly. Neither a noise tail nor an exponential
supermartingale is part of the solution predicate.

Concrete full-test solutions are constructed when the actual
fields vanish. The examples include orthonormal tokens under the
zero head law and antipodal tokens under positive Gaussian value
variance at uniform attention. All smooth tests are checked.
-/

open MeasureTheory Set
open scoped BigOperators NNReal
namespace Transformer.Homogenized

/-- The source's smooth-test martingale problem on [0,T], with the original
sphere generator. Source: arXiv:2604.01978v1, Appendix, `def: ito.formula`.
This is a predicate of the fields, process, filtration and measures. -/
def IsSphereMartingaleSolution {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ : Filtration ℝ m)
    (T : ℝ) (X : ℝ → Ω → (Idx n → EucSpace d)) : Prop :=
  (∀ t ∈ Icc (0 : ℝ) T, ∀ ω i, ‖X t ω i‖ = 1) ∧
    (∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t)) ∧
    (∀ ω, ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) ∧
    ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
      IsMartingaleOn T ℱ P (fun t ω => φ (X t ω) - φ (X 0 ω) -
        ∫ s in (0 : ℝ)..t, sphGenerator ρ B G φ (X s ω))

/-- A source solution remains a solution on every shorter horizon.
Source: arXiv:2604.01978v1, Appendix, `def: ito.formula`. -/
theorem IsSphereMartingaleSolution.restrict {d n : ℕ} {ρ : Measure (HeadParam d)}
    {B : (Idx n → EucSpace d) → Idx n → EucSpace d}
    {G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d}
    {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ m}
    {T U : ℝ} {X : ℝ → Ω → (Idx n → EucSpace d)}
    (h : IsSphereMartingaleSolution ρ B G P ℱ U X) (hTU : T ≤ U) :
    IsSphereMartingaleSolution ρ B G P ℱ T X := by
  have hs : Icc (0 : ℝ) T ⊆ Icc (0 : ℝ) U := Icc_subset_Icc_right hTU
  refine ⟨fun t ht => h.1 t (hs ht), fun t ht => h.2.1 t (hs ht),
    fun ω => (h.2.2.1 ω).mono hs, ?_⟩
  intro φ hφ
  obtain ⟨ha, hi, hm⟩ := h.2.2.2 φ hφ
  exact ⟨fun t ht => ha t (hs ht), fun t ht => hi t (hs ht),
    fun s hs' t ht hst => hm s (hs hs') t (hs ht) hst⟩

/-- Vanishing actual fields give a full smooth-test stationary solution.
Source: arXiv:2604.01978v1, Appendix, `def: ito.formula`. -/
theorem isSphereMartingaleSolution_constant_of_null_fields {d n : ℕ}
    (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T : ℝ) (K : Idx n → EucSpace d)
    (hK : ∀ i, ‖K i‖ = 1) (hB : B K = 0) (hG : ∀ᵐ θ ∂ρ, G θ K = 0) :
    IsSphereMartingaleSolution ρ B G P ℱ T (fun _ _ => K) := by
  refine ⟨fun t ht ω => hK, fun t ht => stronglyMeasurable_const,
    fun ω => continuousOn_const, ?_⟩
  intro φ hφ
  have hgen : sphGenerator ρ B G φ K = 0 := by
    have he : (∫ θ, sphHess φ K (G θ K) ∂ρ) = 0 := by
      calc
        _ = ∫ _ : HeadParam d, (0 : ℝ) ∂ρ := integral_congr_ae (by
          filter_upwards [hG] with θ hθ
          rw [hθ, sphHess_zero])
        _ = 0 := integral_zero _ _
    simp only [sphGenerator, hB, map_zero, he, mul_zero, add_zero]
  have heq : (fun (t : ℝ) (_ : Ω) => φ K - φ K -
      ∫ _ in (0 : ℝ)..t, sphGenerator ρ B G φ K) = fun _ _ => (0 : ℝ) := by
    funext t ω
    simp only [hgen, intervalIntegral.integral_zero, sub_self]
  rw [heq]
  exact isMartingaleOn_const T ℱ P 0

/-- A full source solution supplies every actual exponential overlap test.
Source: arXiv:2604.01978v1, §7, the martingale maximal-inequality step. -/
theorem IsSphereMartingaleSolution.exponentialOverlap {d n : ℕ} {ρ : Measure (HeadParam d)}
    {B : (Idx n → EucSpace d) → Idx n → EucSpace d}
    {G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d}
    {Ω : Type*} {m : MeasurableSpace Ω} {P : Measure Ω} {ℱ : Filtration ℝ m}
    {T : ℝ} {X : ℝ → Ω → (Idx n → EucSpace d)}
    (h : IsSphereMartingaleSolution ρ B G P ℱ T X) (c : ℝ) (i j : Idx n) :
    IsMartingaleOn T ℱ P (fun t ω => exponentialOverlap c i j (X t ω) -
      exponentialOverlap c i j (X 0 ω) - ∫ s in (0 : ℝ)..t,
        sphGenerator ρ B G (exponentialOverlap c i j) (X s ω)) :=
  h.2.2.2 _ (contDiff_exponentialOverlap c i j)

/-- At the zero head law every constant unit configuration solves all tests.
Source: arXiv:2604.01978v1, Appendix, `def: ito.formula`, at zero fields. -/
theorem isSphereMartingaleSolution_dirac_zero {d n : ℕ}
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) [IsFiniteMeasure P]
    (ℱ : Filtration ℝ m) (T β α s : ℝ) (K : Idx n → EucSpace d) (hK : ∀ i, ‖K i‖ = 1) :
    IsSphereMartingaleSolution (Measure.dirac (0 : HeadParam d))
      (bField β (Measure.dirac 0)) (noiseField β α s (Measure.dirac 0))
      P ℱ T (fun _ _ => K) := by
  apply isSphereMartingaleSolution_constant_of_null_fields _ _ _ P ℱ T K hK
  · funext i
    exact bField_dirac_zero β K i
  · rw [ae_dirac_eq]
    change noiseField β α s (Measure.dirac 0) (0 : HeadParam d) K = 0
    funext i
    simp [noiseField]

/-- Antipodal unit tokens solve the full equation at uniform attention,
for every head law, including positive value variance.
Source: arXiv:2604.01978v1, `eq:Diffusive_gaussian_case`, at β=0. -/
theorem isSphereMartingaleSolution_antipodal_zero_beta {d : ℕ}
    (ρ : Measure (HeadParam d)) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsFiniteMeasure P] (ℱ : Filtration ℝ m) (T : ℝ)
    (v : EucSpace d) (hv : ‖v‖ = 1) :
    IsSphereMartingaleSolution ρ (bField 0 ρ) (noiseField 0 1 1 ρ)
      P ℱ T (fun _ _ => ![v, -v]) := by
  let K : Idx 2 → EucSpace d := ![v, -v]
  have hfield (θ : HeadParam d) (z : EucSpace d) : attnField 0 θ K z = 0 := by
    rw [attnField_zero_beta, Fin.sum_univ_two]
    simp only [K, Matrix.cons_val_zero, Matrix.cons_val_one, valueMap, map_neg,
      add_neg_cancel, smul_zero]
  have hm (z : EucSpace d) : meanField 0 ρ K z = 0 := by
    simp only [meanField, hfield, integral_zero]
  apply isSphereMartingaleSolution_constant_of_null_fields _ _ _ P ℱ T K
  · intro i
    fin_cases i <;> simpa [K] using hv
  · funext i
    simp only [bField, hm, proj, inner_zero_right, zero_smul, sub_self, Pi.zero_apply]
  · apply ae_of_all
    intro θ
    funext i
    simp only [noiseField_one_one, Gfield, fluct, hfield, hm, sub_self,
      proj, inner_zero_right, zero_smul, Pi.zero_apply]

/-- Nonzero orthonormal tokens realize the null-field and zero-head premises. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (K : Idx 2 → EucSpace 2),
    (∀ i, ‖K i‖ = 1) ∧
    (0 : (Idx 2 → EucSpace 2) → Idx 2 → EucSpace 2) K = 0 ∧
    (∀ᵐ θ ∂(Measure.dirac (0 : HeadParam 2)),
      (0 : HeadParam 2 → (Idx 2 → EucSpace 2) → Idx 2 → EucSpace 2) θ K = 0) ∧
    IsSphereMartingaleSolution (Measure.dirac (0 : HeadParam 2))
      (bField 1 (Measure.dirac 0)) (noiseField 1 1 1 (Measure.dirac 0))
      (Measure.dirac ()) ℱ 2 (fun _ _ => K) := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let K : Idx 2 → EucSpace 2 := fun i => EuclideanSpace.single i 1
  have hK : ∀ i, ‖K i‖ = 1 := fun i => by simp [K]
  exact ⟨ℱ, K, hK, rfl, ae_of_all _ (fun θ => rfl),
    isSphereMartingaleSolution_dirac_zero (Measure.dirac ()) ℱ 2 1 1 1 K hK⟩

/-- Positive source-normalized Gaussian value variance is compatible with
the full smooth-test predicate. This solution can also be restricted and
tested against each exponential overlap observable. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace Unit))
    (X : ℝ → Unit → (Idx 2 → EucSpace 2)),
    IsGaussianHeadLaw 2 (stdSigmaV 2) 0 (gaussHeadLaw 2 (stdSigmaV 2) 0) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧
    IsSphereMartingaleSolution (gaussHeadLaw 2 (stdSigmaV 2) 0)
      (bField 0 (gaussHeadLaw 2 (stdSigmaV 2) 0))
      (noiseField 0 1 1 (gaussHeadLaw 2 (stdSigmaV 2) 0))
      (Measure.dirac ()) ℱ 2 X := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace Unit)) ⊥ bot_le
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  refine ⟨ℱ, fun _ _ => ![v, -v], isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, ?_⟩
  exact isSphereMartingaleSolution_antipodal_zero_beta _ (Measure.dirac ()) ℱ 2 v (by simp [v])

end Transformer.Homogenized
