import Transformer.GPTMini.Sparsemax.IncidentMemoryFeasibility

/-!
# Unique constrained parameter selection and quantitative error

Derived extra criterion for arXiv:1602.02068v2, Eq. (1). The separate-budget
domain has a proved unique complete-coordinate squared-distance minimum.
Its selected point supplies genuine embeddings, invertible actual attention
and at most three routes. A feasible reference is recovered exactly.

For any convex feasible set, comparison with the midpoint bounds squared
parameter error by twice objective suboptimality. This quantitative result
requires an attained minimum and explicit feasible endpoints. It concerns
the additional geometry criterion, not unseen target errors or existence
of a complete joint model minimum. Selection is noncomputable, not a solver.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Choose the proved unique constrained quadratic minimum in the relaxed domain.
Source: additional geometry selection for arXiv:1602.02068v2, Eq. (1). -/
def incidentMemorySelectedParameters {N : ℕ} (cap floor : ℝ) (reference : LocalMemoryParameters N)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) : LocalMemoryParameters N :=
  Classical.choose (incidentMemoryQuadratic_existsUnique cap floor reference hc hf).exists

/-- The selected parameters satisfy all separate incident and independent norm bounds.
Source: the proved constrained quadratic minimum before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemorySelectedParameters_mem {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    incidentMemorySelectedParameters cap floor reference hc hf ∈
      incidentMemoryParameterDomain N cap floor :=
  (Classical.choose_spec (incidentMemoryQuadratic_existsUnique cap floor reference hc hf).exists).1

/-- An infeasible reference gives a genuine feasible selected point. -/
example : incidentMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference
    (by norm_num) (by norm_num) ∈ incidentMemoryParameterDomain 1 4 (3 / 4) :=
  incidentMemorySelectedParameters_mem _ _ _ (by norm_num) (by norm_num)

/-- The selected point minimizes the actual additional criterion over the full relaxed domain.
Source: the constrained extra-criterion result for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemorySelectedParameters_min {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor)
      (incidentMemorySelectedParameters cap floor reference hc hf) :=
  (Classical.choose_spec (incidentMemoryQuadratic_existsUnique cap floor reference hc hf).exists).2

/-- Both actual existence-bound premises hold for an infeasible reference. -/
example : IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
    (incidentMemoryParameterDomain 1 4 (3 / 4))
    (incidentMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference (by norm_num) (by norm_num)) :=
  incidentMemorySelectedParameters_min _ _ _ (by norm_num) (by norm_num)

/-- A data reference already satisfying the structural bounds is recovered exactly.
Source: zero complete-coordinate distance and unique constrained selection for
arXiv:1602.02068v2, Eq. (1), rather than a fixed-attention feasibility constraint. -/
theorem incidentMemorySelectedParameters_eq_reference {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1)
    (hr : reference ∈ incidentMemoryParameterDomain N cap floor) :
    incidentMemorySelectedParameters cap floor reference hc hf = reference := by
  exact localMemoryQuadratic_unique_min reference _ (incidentMemoryParameterDomain_convex N cap floor)
    _ reference (incidentMemorySelectedParameters_mem cap floor reference hc hf) hr
    (incidentMemorySelectedParameters_min cap floor reference hc hf)
    (fun p hp => localMemoryQuadratic_reference_min reference (Set.mem_univ p))

/-- A nonidentity reference excluded by the former global budget is exactly recovered. -/
example : incidentMemorySelectedParameters 4 (3 / 4) incidentMemoryExampleParameters
    (by norm_num) (by norm_num) = incidentMemoryExampleParameters :=
  incidentMemorySelectedParameters_eq_reference _ _ _ (by norm_num) (by norm_num)
    incidentMemoryExampleParameters_mem

/-- Unique selected geometry gives actual bounded-width embeddings, an inverse and sparse routes.
Source: the proved local-budget Gram and variational arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemorySelected_embeddings_inverse_support {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : 1 / 2 < floor) (hf1 : floor ≤ 1) :
    ∃ p ∈ incidentMemoryParameterDomain N cap floor,
      IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) p ∧
      (∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
        localMemoryGram p = featureGram features) ∧
      IsUnit (memoryGramAttention (localMemoryGram p)).det ∧
      ∀ i, (Finset.univ.filter (fun j => memoryGramAttention (localMemoryGram p) i j ≠ 0)).card ≤ 3 := by
  let p := incidentMemorySelectedParameters cap floor reference hc hf1
  have hp := incidentMemorySelectedParameters_mem cap floor reference hc hf1
  exact ⟨p, hp, incidentMemorySelectedParameters_min cap floor reference hc hf1,
    incidentMemoryGram_fixedWidth cap floor p (by linarith) hp,
    incidentMemoryAttention_det_unit cap floor p hf hp,
    incidentMemoryAttention_support_card cap floor p (by linarith) hp⟩

/-- Infeasible preferred coordinates still yield embeddings, an actual inverse and sparse attention. -/
example : ∃ p ∈ incidentMemoryParameterDomain 1 4 (3 / 4),
    IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
      (incidentMemoryParameterDomain 1 4 (3 / 4)) p ∧
    (∃ features : Fin 4 → Sum (Fin 2) (Fin 2) → ℝ, localMemoryGram p = featureGram features) ∧
    IsUnit (memoryGramAttention (localMemoryGram p)).det ∧
    ∀ i, (Finset.univ.filter (fun j => memoryGramAttention (localMemoryGram p) i j ≠ 0)).card ≤ 3 :=
  incidentMemorySelected_embeddings_inverse_support _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- The selected geometry retains exact decoding by one global common value table.
Source: the actual attention inverse after arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemorySelected_values_exact {N D : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hc : 1 ≤ cap) (hf : 1 / 2 < floor) (hf1 : floor ≤ 1) :
    let p := incidentMemorySelectedParameters cap floor reference hc hf1
    memoryValueOutput (localMemoryGram p) (recoverMemoryValues (localMemoryGram p) Z) = Z :=
  incidentMemoryValues_exact cap floor _ Z hf (incidentMemorySelectedParameters_mem cap floor reference hc hf1)

/-- Infeasible preferences and nonconstant outputs inhabit the selected common-value decoder premises. -/
example : memoryValueOutput (localMemoryGram
    (incidentMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference (by norm_num) (by norm_num)))
    (recoverMemoryValues (localMemoryGram
      (incidentMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference (by norm_num) (by norm_num)))
      (Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)))) =
    Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)) :=
  incidentMemorySelected_values_exact 4 (3 / 4) localMemoryOutsideReference _
    (by norm_num) (by norm_num) (by norm_num)

/-- Squared coordinate error is bounded by twice the additional criterion's suboptimality.
Source: midpoint comparison and the exact quadratic gap for the derived
arXiv:1602.02068v2, Eq. (1) geometry criterion; the factor two is explicit. -/
theorem localMemoryQuadratic_min_growth {N : ℕ} (reference p q : LocalMemoryParameters N)
    (s : Set (LocalMemoryParameters N)) (hs : Convex ℝ s) (hp : p ∈ s) (hq : q ∈ s)
    (hm : IsMinOn (localMemoryQuadratic reference) s q) :
    localMemoryQuadratic q p ≤ 2 * (localMemoryQuadratic reference p - localMemoryQuadratic reference q) := by
  have hmid := hs hq hp (by norm_num : 0 ≤ (1 / 2 : ℝ))
    (by norm_num : 0 ≤ (1 / 2 : ℝ)) (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  have hmin := hm hmid
  have hg := localMemoryQuadratic_affine_gap reference p q (1 / 2) (1 / 2) (by norm_num)
  have he : (1 / 2 : ℝ) • q + (1 / 2 : ℝ) • p = (1 / 2 : ℝ) • p + (1 / 2 : ℝ) • q := add_comm _ _
  rw [he] at hmin
  change localMemoryQuadratic reference q ≤
    localMemoryQuadratic reference ((1 / 2 : ℝ) • p + (1 / 2 : ℝ) • q) at hmin
  linarith

/-- A nonidentity selected geometry and identity comparison point inhabit all growth hypotheses. -/
example : localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) ≤
    2 * (localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) -
      localMemoryQuadratic incidentMemoryExampleParameters incidentMemoryExampleParameters) :=
  localMemoryQuadratic_min_growth _ _ _ _ (incidentMemoryParameterDomain_convex 3 4 (3 / 4))
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem
    (fun p hp => localMemoryQuadratic_reference_min _ (Set.mem_univ p))

/-- An approximate quadratic minimizer has certified squared parameter error.
Source: the explicit extra-criterion growth estimate for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_suboptimal_error {N : ℕ} (reference p q : LocalMemoryParameters N)
    (s : Set (LocalMemoryParameters N)) (epsilon : ℝ) (hs : Convex ℝ s)
    (hp : p ∈ s) (hq : q ∈ s) (hm : IsMinOn (localMemoryQuadratic reference) s q)
    (he : localMemoryQuadratic reference p - localMemoryQuadratic reference q ≤ epsilon) :
    localMemoryQuadratic q p ≤ 2 * epsilon := by
  have hg := localMemoryQuadratic_min_growth reference p q s hs hp hq hm
  linarith

/-- A positive-error comparison point satisfies the approximate-minimizer premises. -/
example : localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) ≤
    2 * localMemoryQuadratic incidentMemoryExampleParameters (0 : LocalMemoryParameters 3) := by
  apply localMemoryQuadratic_suboptimal_error _ _ _ _ _
    (incidentMemoryParameterDomain_convex 3 4 (3 / 4))
    (zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num))
    incidentMemoryExampleParameters_mem
    (fun p hp => localMemoryQuadratic_reference_min _ (Set.mem_univ p))
  rw [(localMemoryQuadratic_eq_zero _ _).2 rfl, sub_zero]

end Transformer.GPTMini.Sparsemax
