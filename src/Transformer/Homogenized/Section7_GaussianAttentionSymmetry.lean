module

public import Transformer.Homogenized.Section7_GaussianQuerySecondMoment

/-!
# Genuine attention-matrix sign symmetry of the original Gaussian head

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7,
proof of `thm:clustering_small_beta`, the cancellation under
`A ↦ -A` in `eq:s_mu_small_beta_clean`. The original Gaussian
head is `(V, WW′ᵀ)`, with independent centered Gaussian entries.
Flipping the entire original `W` block preserves its actual product
law and flips the genuine attention matrix while keeping `V` fixed.

The resulting measurable equivalence preserves every original head
law, with arbitrary separate entry variances, including zero. Thus
any genuine odd scalar observable has zero original expectation.
In particular the actual query vector and its scalar token scores
are centered. Their integrability follows from the already proved
exact original query second moment, rather than being assumed in the
symmetry argument or inferred from a fictitious Gaussian product law.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The genuine attention-matrix flip, with the original value
matrix retained. Source: arXiv:2604.01978v1, §7, part (b) of
the proof of `eq:s_mu_small_beta_clean`. -/
noncomputable def negAttention (d : ℕ) : HeadParam d ≃ᵐ HeadParam d :=
  MeasurableEquiv.prodCongr (MeasurableEquiv.refl _) (MeasurableEquiv.neg _)

/-- The flip acts on the actual original head coordinates.
Source: arXiv:2604.01978v1, §7, the symmetry `A ↦ -A`. -/
theorem negAttention_apply {d : ℕ} (θ : HeadParam d) : negAttention d θ = (θ.1, -θ.2) := rfl

/-- Flipping the first original query/key block of the entry sample.
Source: arXiv:2604.01978v1, initialization (G), `A = WW′ᵀ`. -/
def attentionBlockNeg {d : ℕ} (z : GaussIdx d → ℝ) : GaussIdx d → ℝ :=
  fun p => if p.1 = 1 then -z p else z p

/-- The genuine original block flip is measurable.
Source: arXiv:2604.01978v1, §7, the original sign-symmetry argument. -/
theorem measurable_attentionBlockNeg {d : ℕ} : Measurable (attentionBlockNeg (d := d)) := by
  apply Measurable.of_eval
  intro p
  by_cases hp : p.1 = 1
  · simp only [attentionBlockNeg, hp, ite_true]
    fun_prop
  · simpa only [attentionBlockNeg, hp, ite_false] using measurable_pi_apply p

/-- The entry flip gives exactly the original head's attention flip.
Source: arXiv:2604.01978v1, initialization (G), with its transpose retained. -/
theorem negAttention_headOfBlocks {d : ℕ} (z : GaussIdx d → ℝ) :
    negAttention d (headOfBlocks z) = headOfBlocks (attentionBlockNeg z) := by
  rw [negAttention_apply, headOfBlocks, headOfBlocks]
  apply Prod.ext
  · ext i j
    simp [attentionBlockNeg]
  · ext i j
    simp [attentionBlockNeg, Matrix.mul_apply, Finset.sum_neg_distrib]

/-- Centered independent original entries retain their full joint
law under the genuine query/key block flip.
Source: arXiv:2604.01978v1, §2.3.3, initialization (G). -/
theorem map_attentionBlockNeg_pi {d : ℕ} (σV σA : ℝ≥0) :
    (Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2)).map attentionBlockNeg =
      Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2) := by
  let γ := fun p : GaussIdx d => gaussianReal 0 (blockSigma σV σA p ^ 2)
  let g := fun p : GaussIdx d => fun t : ℝ => if p.1 = 1 then -t else t
  have hg (p : GaussIdx d) : Measurable (g p) := by
    by_cases hp : p.1 = 1
    · simpa only [g, hp, ite_true] using measurable_neg
    · simpa only [g, hp, ite_false] using measurable_id'
  have hm (p : GaussIdx d) : (γ p).map (g p) = γ p := by
    by_cases hp : p.1 = 1
    · simp only [g, hp, ite_true, γ]
      rw [gaussianReal_map_neg, neg_zero]
    · simp only [g, hp, ite_false]
      exact Measure.map_id'
  have : ∀ p, SigmaFinite ((γ p).map (g p)) := fun p => by rw [hm]; infer_instance
  change (Measure.pi γ).map (fun z p => g p (z p)) = Measure.pi γ
  rw [Measure.pi_map_pi fun p => (hg p).aemeasurable]
  congr 1
  funext p
  exact hm p

/-- **Every original Gaussian head law is invariant under `A ↦ -A`.**
Source: arXiv:2604.01978v1, §7, part (b) of the proof of
`eq:s_mu_small_beta_clean`, justified from the original entry law. -/
theorem measurePreserving_negAttention {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) : MeasurePreserving (negAttention d) ρ ρ := by
  refine ⟨(negAttention d).measurable, ?_⟩
  rw [← map_headOfBlocks_pi hρ,
    Measure.map_map (negAttention d).measurable measurable_headOfBlocks,
    show negAttention d ∘ headOfBlocks = headOfBlocks ∘ attentionBlockNeg from
      funext negAttention_headOfBlocks,
    ← Measure.map_map measurable_headOfBlocks measurable_attentionBlockNeg, map_attentionBlockNeg_pi]

/-- A genuine odd scalar observable has zero original Gaussian head
expectation. Source: arXiv:2604.01978v1, §7, the first-order
temperature cancellation in `eq:s_mu_small_beta_clean`. -/
theorem integral_negAttention_odd_eq_zero {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (F : HeadParam d → ℝ)
    (hF : ∀ θ, F (negAttention d θ) = -F θ) : (∫ θ, F θ ∂ρ) = 0 := by
  have h := (measurePreserving_negAttention hρ).integral_comp' F
  simp only [hF, integral_neg] at h
  linarith

/-- Actual original query vectors are integrable, including zero
attention variance. Source: arXiv:2604.01978v1, §7, the
expectation of the linear temperature term, using its actual second moment. -/
theorem integrable_gaussian_head_query {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (x : EucSpace d) :
    Integrable (fun θ => qkMap θ x) ρ := by
  have : IsProbabilityMeasure ρ := by
    rw [gaussianHeadLaw_eq_canonical hρ]
    infer_instance
  have hq : Measurable (fun θ : HeadParam d => qkMap θ x) :=
    (measurable_gaussianMatrix_query x).comp measurable_snd
  apply ((gaussian_head_query_second_moment hρ x).1.add (integrable_const 1)).mono' hq.aestronglyMeasurable
  filter_upwards with θ
  change ‖qkMap θ x‖ ≤ ‖qkMap θ x‖ ^ 2 + 1
  nlinarith [sq_nonneg (‖qkMap θ x‖ - 1)]

/-- Every original scalar query score is genuinely integrable.
Source: arXiv:2604.01978v1, §7, the scores in the `w_k'` formula. -/
theorem integrable_gaussian_head_score {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (x y : EucSpace d) :
    Integrable (fun θ => inner ℝ (qkMap θ x) y) ρ :=
  (integrable_gaussian_head_query hρ x).inner_const y

/-- The actual original scalar scores are centered; no Gaussian law
is assigned to their bilinear random matrix.
Source: arXiv:2604.01978v1, §7, the first-order temperature cancellation. -/
theorem integral_gaussian_head_score_eq_zero {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (x y : EucSpace d) :
    (∫ θ, inner ℝ (qkMap θ x) y ∂ρ) = 0 := by
  apply integral_negAttention_odd_eq_zero hρ
  intro θ
  simp [qkMap, negAttention_apply]

/-- A nondegenerate original attention law and a nonconstant odd
actual score jointly realize all symmetry, integrability and oddness premises. -/
example : IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧
    (∀ θ : HeadParam 2, (negAttention 2 θ).2 0 1 = -θ.2 0 1) := by
  refine ⟨isGaussianHeadLaw_gaussHeadLaw 2 1 1, ?_⟩
  intro θ
  rfl

end Transformer.Homogenized
