module

public import Transformer.Modes.Section2_GaussianTransversality
public import Mathlib.MeasureTheory.Function.Jacobian

/-!
# Gaussian-null critical images for the original KDE parameter space

ArXiv:2412.09080v3, §2.1, the mode/upcrossing identity
following `eq:Fn`. The original sample space carries a finite product
of nondegenerate real Gaussian laws. We prove its absolute continuity
with respect to the original coordinate Lebesgue volume, including
the empty-coordinate base case, by splitting off one coordinate.

Mathlib's equal-dimension Sard lemma says that the image of a set
on which the derivative determinant vanishes has Lebesgue measure
zero. Absolute continuity transfers those critical images to the
actual Gaussian sample law; countably many such images remain null.
These are the measure estimates needed for local charts of the
regular zero set of the original field and projection to sample space.
The chart cover and the vanishing of its projection determinants
remain explicit inputs here; no generic nondegeneracy is defined
into a predicate or asserted without its geometric proof.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Modes

/-- Finite coordinate products preserve absolute continuity of sigma-finite real measures.
Source: arXiv:2412.09080v3, §2.1, the product sample law of `eq:Fn`;
proved by the original coordinate split and product absolute continuity. -/
theorem absolutelyContinuous_pi_real (n : ℕ) :
    ∀ (μ ν : Idx n → Measure ℝ) [∀ i, SigmaFinite (μ i)] [∀ i, SigmaFinite (ν i)],
      (∀ i, μ i ≪ ν i) → Measure.pi μ ≪ Measure.pi ν := by
  induction n with
  | zero =>
    intro μ ν instμ instν h
    have he : μ = ν := funext fun i => Fin.elim0 i
    rw [he]
  | succ n ih =>
    intro μ ν instμ instν h
    let i : Fin (n + 1) := 0
    let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) i
    have htail : (Measure.pi fun j => μ (i.succAbove j)) ≪
        (Measure.pi fun j => ν (i.succAbove j)) :=
      ih _ _ (fun j => h (i.succAbove j))
    have hp := (h i).prod htail
    have hm := hp.map e.symm.measurable
    have hμ := MeasurePreserving.symm e (measurePreserving_piFinSuccAbove μ i)
    have hν := MeasurePreserving.symm e (measurePreserving_piFinSuccAbove ν i)
    rw [hμ.map_eq, hν.map_eq] at hm
    exact hm

/-- The actual independent standard Gaussian sample is absolutely continuous with respect to coordinate volume.
Source: arXiv:2412.09080v3, §2.1, the original Gaussian sample in `eq:Fn`. -/
theorem gaussianSample_absolutelyContinuous_volume (n : ℕ) :
    gaussianSample n ≪ (volume : Measure (Idx n → ℝ)) := by
  rw [gaussianSample, volume_pi]
  exact absolutelyContinuous_pi_real n _ _ (fun _ => gaussianReal_absolutelyContinuous 0 (by norm_num))

/-- Every Lebesgue-null subset of the original data space is Gaussian-null.
Source: arXiv:2412.09080v3, §2.1, the original product Gaussian law in `eq:Fn`. -/
theorem gaussianSample_null_of_volume_null (n : ℕ) (S : Set (Idx n → ℝ))
    (hS : (volume : Measure (Idx n → ℝ)) S = 0) : gaussianSample n S = 0 :=
  gaussianSample_absolutelyContinuous_volume n hS

/-- A singular-derivative image in the original data dimension has zero actual Gaussian sample probability.
Source: arXiv:2412.09080v3, §2.1, the critical-value step for `eq:Fn`;
uses Mathlib's equal-dimension Sard lemma, `addHaar_image_eq_zero_of_det_fderivWithin_eq_zero`. -/
theorem gaussianSample_critical_image_null (n : ℕ) (f : (Idx n → ℝ) → Idx n → ℝ)
    (f' : (Idx n → ℝ) → (Idx n → ℝ) →L[ℝ] (Idx n → ℝ)) (S : Set (Idx n → ℝ))
    (hf : ∀ x ∈ S, HasFDerivWithinAt f (f' x) S x)
    (hdet : ∀ x ∈ S, (f' x).toLinearMap.det = 0) :
    gaussianSample n (f '' S) = 0 := by
  apply gaussianSample_null_of_volume_null
  exact addHaar_image_eq_zero_of_det_fderivWithin_eq_zero volume hf hdet

/-- Countably many critical chart images still have zero probability under the actual source sample law.
Source: arXiv:2412.09080v3, §2.1, the critical-value projection argument for the original `eq:Fn`. -/
theorem gaussianSample_countable_critical_images_null {ι : Type*} [Countable ι] (n : ℕ)
    (f : ι → (Idx n → ℝ) → Idx n → ℝ)
    (f' : ι → (Idx n → ℝ) → (Idx n → ℝ) →L[ℝ] (Idx n → ℝ)) (S : ι → Set (Idx n → ℝ))
    (hf : ∀ a x, x ∈ S a → HasFDerivWithinAt (f a) (f' a x) (S a) x)
    (hdet : ∀ a x, x ∈ S a → (f' a x).toLinearMap.det = 0) :
    gaussianSample n (⋃ a, f a '' S a) = 0 := by
  apply measure_iUnion_null_iff.mpr
  exact fun a => gaussianSample_critical_image_null n (f a) (f' a) (S a) (hf a) (hdet a)

/-- Any source-data subset covered by countably many such critical images is null for the actual Gaussian sample.
Source: arXiv:2412.09080v3, §2.1, the final null-cover step for the regular zero-set projection. -/
theorem gaussianSample_null_of_critical_cover {ι : Type*} [Countable ι] (n : ℕ)
    (A : Set (Idx n → ℝ)) (f : ι → (Idx n → ℝ) → Idx n → ℝ)
    (f' : ι → (Idx n → ℝ) → (Idx n → ℝ) →L[ℝ] (Idx n → ℝ)) (S : ι → Set (Idx n → ℝ))
    (hf : ∀ a x, x ∈ S a → HasFDerivWithinAt (f a) (f' a x) (S a) x)
    (hdet : ∀ a x, x ∈ S a → (f' a x).toLinearMap.det = 0)
    (hcover : A ⊆ ⋃ a, f a '' S a) : gaussianSample n A = 0 := by
  exact measure_mono_null hcover (gaussianSample_countable_critical_images_null n f f' S hf hdet)

/-- The original singleton field has slope one at its unique zero, for every real bandwidth.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, at one original Gaussian sample. -/
theorem fieldF_singleton_deriv_at_zero (β : ℝ) (X : Idx 1 → ℝ) (t : ℝ)
    (ht : fieldF β X t = 0) : deriv (fieldF β X) t = 1 := by
  have hf : (t - X 0) * Real.exp (-(β / 2) * (t - X 0) ^ 2) = 0 := by
    simpa only [fieldF, Fin.sum_univ_one, Nat.cast_one, Real.sqrt_one, div_one, one_mul] using ht
  have he : t - X 0 = 0 := (mul_eq_zero.mp hf).resolve_right (Real.exp_ne_zero _)
  rw [(hasDerivAt_fieldF_query β X t).deriv]
  simp [he]

/-- A proved critical chart cover gives almost-sure query nondegeneracy of the original field.
Source: arXiv:2412.09080v3, §2.1, the critical-value step for `eq:Fn`.
The geometric cover is an explicit hypothesis; it is not part of the field or sample definitions. -/
theorem ae_fieldF_query_nondegenerate_of_critical_cover {ι : Type*} [Countable ι] (n : ℕ)
    (β : ℝ) (hβ : 0 < β) (f : ι → (Idx n → ℝ) → Idx n → ℝ)
    (f' : ι → (Idx n → ℝ) → (Idx n → ℝ) →L[ℝ] (Idx n → ℝ)) (S : ι → Set (Idx n → ℝ))
    (hf : ∀ a x, x ∈ S a → HasFDerivWithinAt (f a) (f' a x) (S a) x)
    (hdet : ∀ a x, x ∈ S a → (f' a x).toLinearMap.det = 0)
    (hcover : {X | (∀ i j, i ≠ j → X i ≠ X j ∧ β * (X i - X j) ^ 2 ≠ 4) ∧
      ∃ t : ℝ, fieldF β X t = 0 ∧ deriv (fieldF β X) t = 0} ⊆ ⋃ a, f a '' S a) :
    ∀ᵐ X ∂gaussianSample n, ∀ t : ℝ, fieldF β X t = 0 → deriv (fieldF β X) t ≠ 0 := by
  have hnull := gaussianSample_countable_critical_images_null n f f' S hf hdet
  have hae : ∀ᵐ X ∂gaussianSample n, X ∉ ⋃ a, f a '' S a := by
    apply ae_iff.mpr
    simpa only [not_not, Set.ofPred_mem_eq] using hnull
  filter_upwards [ae_gaussianSample_pair_gaps n β hβ, hae] with X hX hnot
  intro t ht hd
  exact hnot (hcover ⟨hX, t, ht, hd⟩)

/-- Actual Gaussian factors jointly realize sigma-finiteness and coordinate absolute continuity.
Source: arXiv:2412.09080v3, §2.1, the independent original Gaussian sample. -/
example : ∃ μ ν : Idx 2 → Measure ℝ,
    (∀ i, SigmaFinite (μ i)) ∧ (∀ i, SigmaFinite (ν i)) ∧ (∀ i, μ i ≪ ν i) := by
  exact ⟨fun _ => gaussianReal 0 1, fun _ => volume, fun _ => inferInstance,
    fun _ => inferInstance, fun _ => gaussianReal_absolutelyContinuous 0 (by norm_num)⟩

/-- Nonempty chart domains and a countable family of nonempty images realize every critical-image premise.
The maps are actual constant charts in two-dimensional sample space, with genuine zero derivative determinant.
Source: arXiv:2412.09080v3, §2.1, the null critical-image argument for `eq:Fn`. -/
example : ∃ (f : ℕ → (Idx 2 → ℝ) → Idx 2 → ℝ)
    (f' : ℕ → (Idx 2 → ℝ) → (Idx 2 → ℝ) →L[ℝ] (Idx 2 → ℝ)) (S : ℕ → Set (Idx 2 → ℝ)),
    (∀ a, (S a).Nonempty) ∧
    (∀ a x, x ∈ S a → HasFDerivWithinAt (f a) (f' a x) (S a) x) ∧
    (∀ a x, x ∈ S a → (f' a x).toLinearMap.det = 0) ∧
    ({0} : Set (Idx 2 → ℝ)) ⊆ ⋃ a, f a '' S a := by
  let f : ℕ → (Idx 2 → ℝ) → Idx 2 → ℝ := fun a _ => ![(a : ℝ), 0]
  refine ⟨f, fun _ _ => 0, fun _ => Set.univ, fun _ => ⟨0, Set.mem_univ _⟩,
    ?_, ?_, ?_⟩
  · intro a x hx
    exact (hasFDerivAt_const (𝕜 := ℝ) (![a, 0] : Idx 2 → ℝ) x).hasFDerivWithinAt
  · intro a x hx
    simp
  · intro x hx
    have hx0 : x = 0 := Set.mem_singleton_iff.mp hx
    rw [hx0]
    exact Set.mem_iUnion.mpr ⟨0, 0, Set.mem_univ _, by ext j; fin_cases j <;> norm_num [f]⟩

/-- The conditional critical-cover premise is realized by the actual singleton field,
whose unique zero has query slope one; the chart domain is nonempty.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, for one original sample. -/
example : ∃ (f : Unit → (Idx 1 → ℝ) → Idx 1 → ℝ)
    (f' : Unit → (Idx 1 → ℝ) → (Idx 1 → ℝ) →L[ℝ] (Idx 1 → ℝ)),
    (0 : ℝ) < 1 ∧ (∀ a x, HasFDerivWithinAt (f a) (f' a x) Set.univ x) ∧
    (∀ a x, (f' a x).toLinearMap.det = 0) ∧
    ({X | (∀ i j, i ≠ j → X i ≠ X j ∧ (1 : ℝ) * (X i - X j) ^ 2 ≠ 4) ∧
      ∃ t : ℝ, fieldF 1 X t = 0 ∧ deriv (fieldF 1 X) t = 0} ⊆ ⋃ a, f a '' Set.univ) ∧
    fieldF 1 (fun _ : Idx 1 => 0) 0 = 0 := by
  refine ⟨fun _ _ => 0, fun _ _ => 0, one_pos, ?_, ?_, ?_, by simp [fieldF]⟩
  · intro a x
    exact (hasFDerivAt_const (𝕜 := ℝ) (0 : Idx 1 → ℝ) x).hasFDerivWithinAt
  · intro a x
    simp
  · rintro X ⟨_, t, ht, hd⟩
    rw [fieldF_singleton_deriv_at_zero 1 X t ht] at hd
    norm_num at hd

end Transformer.Modes
