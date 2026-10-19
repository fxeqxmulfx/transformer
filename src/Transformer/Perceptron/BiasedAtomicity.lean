module

public import Transformer.Perceptron.Section5_BiasedAtomicity

/-!
# Perceptrons and attention's mean-field landscape — atomicity with biases

`rem: ext` (i) of arXiv:2601.21366v2 in its two carried forms: `thm: circle`
and `thm: any.d` (i) for the biased perceptron of `Bias.lean`, under the
source's proviso that at least one of the hyperplanes `{x : a_j·x + b_j = 0}`
meets `𝕊^{d-1}` in more than one point.

Which of "the results above" are carried, and why the strict-SOPD ones are
not, is recorded in the docstring of `Bias.lean`.

Source: arXiv:2601.21366v2, `rem: ext` (i).
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- **The proviso of `rem: ext` (i)**: at least one of the perceptron's
hyperplanes meets the sphere in more than one point. The source calls the
section a hyperplane; the nonzero normal is made explicit here. Otherwise
`a_j = b_j = 0` describes the whole sphere and violates the source's stated
equivalence with `|b_j| < ‖a_j‖`. In dimension at least two the corrected
predicate has exactly that equivalence, proved below.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
def HasTransverseHyperplane (a : Idx d → EucSpace d) (b : Idx d → ℝ) : Prop :=
  ∃ j : Idx d, a j ≠ 0 ∧ ¬ (sphereHyperplane (a j) (b j)).Subsingleton

/-- The source's transverse-hyperplane proviso is exactly its printed
norm inequality in dimension at least two, with an actual nonzero normal.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), the parenthesis.
The explicit normal condition repairs the zero equation as documented above. -/
theorem hasTransverseHyperplane_iff_exists_abs_bias_lt_norm (hd : 2 ≤ d)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) :
    HasTransverseHyperplane a b ↔ ∃ j, |b j| < ‖a j‖ := by
  constructor
  · rintro ⟨j, ha, hj⟩
    exact ⟨j, (not_subsingleton_sphereHyperplane_iff hd ha (b j)).mp hj⟩
  · rintro ⟨j, hj⟩
    have ha : a j ≠ 0 := norm_pos_iff.mp ((abs_nonneg (b j)).trans_lt hj)
    exact ⟨j, ha, (not_subsingleton_sphereHyperplane_iff hd ha (b j)).mpr hj⟩

/-- The original transverse section supplies an actual sphere point on
an actual nonzero neuron hyperplane. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (i), and §5.3, the constant in `eq:global-Id`.
This point fixes the actual first-variation constant in both source proofs. -/
theorem exists_spherePoint_of_hasTransverseHyperplane (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (hb : HasTransverseHyperplane a b) :
    ∃ (o : SSphere d) (j : Idx d), a j ≠ 0 ∧ o ∈ sphereHyperplane (a j) (b j) := by
  obtain ⟨j, ha, hj⟩ := hb
  obtain ⟨o, ho⟩ := (Set.not_subsingleton_iff.mp hj).nonempty
  exact ⟨o, j, ha, ho⟩

/-- A perceptron with a nonzero weight and no bias has a transverse
hyperplane: `|0| < ‖a_j‖`. -/
theorem hasTransverseHyperplane_zero (hd : 2 ≤ d) {a : Idx d → EucSpace d} {j : Idx d}
    (ha : a j ≠ 0) : HasTransverseHyperplane a 0 :=
  ⟨j, ha, by
    rw [show (0 : Idx d → ℝ) j = 0 from rfl,
      not_subsingleton_sphereHyperplane_iff hd ha 0, abs_zero]
    exact norm_pos_iff.mpr ha⟩

/-- The base point of `𝕊¹ ⊂ ℝ²` is nonzero: its norm is `1`. -/
theorem basePoint_ne_zero : ((basePoint 1 : SSphere 2) : EucSpace 2) ≠ 0 :=
  norm_ne_zero_iff.mp (by rw [norm_basePoint_one]; exact (one_ne_zero : (1 : ℝ) ≠ 0))

/-- The one-neuron weight `a_0 = -basePoint 1` of the witnesses below is
nonzero. -/
theorem single_neg_basePoint_ne_zero :
    (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)) : Idx 2 → EucSpace 2)
      (0 : Idx 2) ≠ 0 := by
  rw [Pi.single_eq_same]
  exact neg_ne_zero.mpr basePoint_ne_zero

/-- The hypotheses of `hasTransverseHyperplane_zero` are satisfiable: `d = 2`
and the one-neuron weight `a_0 = -basePoint 1`, which is nonzero. -/
example : (2 : ℕ) ≤ 2 ∧
    (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)) : Idx 2 → EucSpace 2)
      (0 : Idx 2) ≠ 0 :=
  ⟨le_rfl, single_neg_basePoint_ne_zero⟩

/-- **Remark (rem: ext) (i), for `thm: circle`.**  With biases inside the
perceptron, and provided one of the hyperplanes `{x : a_j·x + b_j = 0}` meets
`𝕊¹` in more than one point, a stationary measure for the ReLU perceptron with
a non-analytic potential is still purely atomic with finite support.

Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), `thm: circle`,
with the proof in §5.2. The transverse section supplies the actual point
fixing the first-variation constant. All original biases are retained. -/
theorem biased_circle_isFinitelyAtomic (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (b : Idx 2 → ℝ) (hb : HasTransverseHyperplane a b)
    (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b) (μ : Perspective.ProbSphere 2)
    (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ) :
    IsFinitelyAtomic μ := by
  obtain ⟨o, _⟩ := exists_spherePoint_of_hasTransverseHyperplane a b hb
  exact isFinitelyAtomic_of_biased_stationary_relu_circle β hβ φ hφ ω a b hana μ hμ o

/-- The hypotheses of `biased_circle_isFinitelyAtomic` are satisfiable at zero
bias, where the biased perceptron is the unbiased one and the witness of
`thm: circle` serves: the hyperplane of the neuron `a_0 = -basePoint 1`
meets the circle in the two opposite second-axis points. This also supplies
the geometric hypotheses of the two transverse-section lemmas above. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      HasTransverseHyperplane
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) (0 : Idx 2 → ℝ) ∧
      ¬ IsBiasedAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) 0 ∧
      IsBiasedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) 0
        (Perspective.diracProb 2 (basePoint 1)) := by
  refine ⟨one_pos, hasDerivAt_reluSq, hasTransverseHyperplane_zero le_rfl single_neg_basePoint_ne_zero, ?_, ?_⟩
  · rw [isBiasedAnalyticOnSphere_zero_iff]
    exact not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis
  · rw [isBiasedStationary_zero_iff]
    exact isStationary_relu_pin 1 0 (basePoint 1)

/-- **Remark (rem: ext) (i), for `thm: any.d` (i).**  In `d ≥ 2`, with biases
and the same proviso, a stationary measure for the ReLU perceptron with a
non-analytic potential has `σ_d`-null support. The uniform law is quantified
over inside the conclusion, as in `thm: any.d`.

Source: arXiv:2601.21366v2, §3.1, `rem: ext` (i), `thm: any.d` (i),
with §5.3 after `eq:global-Id`. The transverse section fixes the actual
first-variation constant for the original affine-quadratic inverse. -/
theorem biased_any_d_measure_support_eq_zero (d : ℕ) (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (φ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * max s 0) s) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (hb : HasTransverseHyperplane a b)
    (hana : ¬ IsBiasedAnalyticOnSphere φ ω a b) (μ : Perspective.ProbSphere d)
    (hμ : IsBiasedStationary β (fun s => max s 0) ω a b μ) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      ν (μ : Measure (SSphere d)).support = 0 := by
  obtain ⟨o, _⟩ := exists_spherePoint_of_hasTransverseHyperplane a b hb
  exact biased_relu_measure_support_eq_zero hd β hβ φ hφ ω a b hana μ hμ o

/-- The hypotheses of `biased_any_d_measure_support_eq_zero` are satisfiable at
`d = 2` and zero bias, by the same witness. -/
example :
    (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
      (∀ s : ℝ, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
      HasTransverseHyperplane
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) (0 : Idx 2 → ℝ) ∧
      ¬ IsBiasedAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 (1 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) 0 ∧
      IsBiasedStationary 1 (fun s => max s 0) (Pi.single 0 (1 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) 0
        (Perspective.diracProb 2 (basePoint 1)) := by
  refine ⟨le_rfl, one_pos, hasDerivAt_reluSq, hasTransverseHyperplane_zero le_rfl single_neg_basePoint_ne_zero, ?_, ?_⟩
  · rw [isBiasedAnalyticOnSphere_zero_iff]
    exact not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis
      inner_basePoint_secondAxis
  · rw [isBiasedStationary_zero_iff]
    exact isStationary_relu_pin 1 0 (basePoint 1)

end Perceptron
end Transformer
