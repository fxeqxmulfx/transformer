import Transformer.Memorization.Section2_LiteralCounterexample
import Transformer.Memorization.Section3_UniformData

/-!
# The missing assumptions in the Kolmogorov/Shannon comparison

arXiv:2505.24832v3, Section 2.2, Proposition 4; Appendix A.7. The paper
allows an arbitrary computational model in Definition 2. That generality
does not imply a uniform additive comparison with Shannon information.
This counterexample uses one uniformly random `n`-bit sample, identity
training, and a fixed ground-truth model. Both Shannon entropies are `n`.
The permitted reversible literal decoder has zero algorithmic memorization.

This does not refute a suitably qualified theorem for a fixed universal
prefix-free machine and computable distributions. Such a theorem must
include coding constants and distribution-description complexity. The
source's proof sets ε=2K(f), but the learner's joint density generally
depends on dataset size; that does not justify an n-independent ε.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory

/-- Section 2.2, Proposition 4, counterexample: actual expected sample-level
algorithmic memorization under the synthetic uniform-bit distribution.
Every integrand is `some 0` by `reversibleLiteral_unintended`, so the
option default is never selected. -/
noncomputable def literalExpectedUnintended (n : ℕ) : ℝ :=
  ∫ x : Fin n → Bool,
    (((kolmogorovUnintended reversibleLiteralDecoder
      (List.ofFn x) [] (List.ofFn x)).getD 0 : ℤ) : ℝ)
    ∂uniformOn Set.univ

/-- Section 2.2, Proposition 4: the algorithmic expectation in the
counterexample vanishes under an actual probability distribution. -/
theorem literal_expected_unintended_zero (n : ℕ) :
    literalExpectedUnintended n = 0 := by
  simp [literalExpectedUnintended, reversibleLiteral_unintended]

/-- Section 2.2, Proposition 4: identity training retains exactly `n`
bits of conditional Shannon information about one uniform n-bit sample. -/
theorem uniform_copy_unintended (n : ℕ) :
    shannonUnintended (uniformOn (Set.univ : Set (Fin n → Bool)))
      id id (fun _ => ()) = (n : ℝ) := by
  rw [← uniform_bool_vector_entropy n]
  unfold shannonUnintended
  congr 1
  let μ : Measure (Fin n → Bool) := uniformOn Set.univ
  change condMutualInfo id id (fun _ => ()) μ = entropy id μ
  have hdiag : entropy (prod (id : (Fin n → Bool) → (Fin n → Bool)) id) μ =
      entropy id μ := by
    simpa only [Function.comp_id] using entropy_prod_comp (X := id) measurable_id μ id
  unfold condMutualInfo
  simp [cond_univ]
  rw [hdiag]
  ring

/-- Section 2.2, Proposition 4, refuted under Definition 2's stated
arbitrary-decoder generality: no additive constant independent of sample
and model bit lengths controls the comparison. Here there is one sample,
and both ℓ and ℓ' equal its growing bit length `n`. -/
theorem proposition4_arbitrary_decoder_counterexample (ε : ℝ) :
    ∃ n : ℕ, ε < |literalExpectedUnintended n -
      shannonUnintended (uniformOn (Set.univ : Set (Fin n → Bool)))
        id id (fun _ => ())| := by
  obtain ⟨n, hn⟩ := exists_nat_gt ε
  refine ⟨n, ?_⟩
  rw [literal_expected_unintended_zero, uniform_copy_unintended]
  rw [zero_sub, abs_neg, abs_of_nonneg (show (0 : ℝ) ≤ n from Nat.cast_nonneg n)]
  exact hn

end Transformer.Memorization
