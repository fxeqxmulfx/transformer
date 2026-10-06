import Transformer.GPTMini.Sparsemax.AtomicMatchingHead
import Mathlib.LinearAlgebra.Finsupp.LinearCombination
import Mathlib.Analysis.Convex.Combination

/-!
# A convex learned mixture over the complete family of matching heads

New model after arXiv:2211.11052v1, §3.1 and Appendix A.4. A state
assigns nonnegative unit-total mass to finitely many physical Q/K/value
heads. Its possible indices are all heads in a parameter box, not an
enumerated feature dictionary. Selecting a previously unused index
therefore learns a new query-key interaction with its original values.

The mixture forward evaluates actual causal sparsemax heads. Convexity
is in this distributional state, not the raw matrices of a fixed list
of heads. Head count may change. The numerical box and unit mass are
explicit architectural bounds; the paper's original four-matrix weight
decay and a practical globally optimal head search are not preserved.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Finite active support inside the full, continuously parameterized head family.
Source: the genuine matching variant of Appendix A.4's finite atomic measure. -/
abbrev MatchingMixture (V H D : ℕ) := MatchingHead V H D →₀ ℝ

/-- Total coefficient mass, not a physical matrix norm or prescribed route loss.
Source: the unit-budget atomic model following Appendix A.4. -/
def matchingMixtureMass (V H D : ℕ) : MatchingMixture V H D →ₗ[ℝ] ℝ :=
  Finsupp.linearCombination ℝ (fun _ => (1 : ℝ))

/-- Exact observed predictions from every active genuine learned matching head.
Source: sparsemax Eq. (1) inside the new Appendix A.4 atomic mixture. -/
def matchingMixtureSample {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) : MatchingMixture V H D →ₗ[ℝ] (Fin R → ℝ) :=
  Finsupp.linearCombination ℝ (matchingHeadSample tokens rows channels)

/-- All matching geometries in the box are eligible; only active ones must be stored.
Source: the new convex atomic domain after Appendix A.4, with free Q/K/value atoms. -/
def matchingMixtureDomain (V H D : ℕ) (cap : ℝ) : Set (MatchingMixture V H D) :=
  {μ | (∀ h, 0 ≤ μ h) ∧ (∀ h, μ h ≠ 0 → h ∈ matchingHeadBox V H D cap) ∧
    matchingMixtureMass V H D μ = 1}

/-- Distributional convex combinations may introduce newly learned Q/K/value heads.
Source: Appendix A.4's atomic convexification, retaining genuine sparsemax heads. -/
theorem matchingMixtureDomain_convex (V H D : ℕ) (cap : ℝ) :
    Convex ℝ (matchingMixtureDomain V H D cap) := by
  intro μ hμ ν hν a b ha hb hab
  refine ⟨?_, ?_, ?_⟩
  · intro h
    change 0 ≤ a * μ h + b * ν h
    exact add_nonneg (mul_nonneg ha (hμ.1 h)) (mul_nonneg hb (hν.1 h))
  · intro h hh
    by_cases hatom : h ∈ matchingHeadBox V H D cap
    · exact hatom
    · have hm : μ h = 0 := by
        by_contra hn
        exact hatom (hμ.2.1 h hn)
      have hn : ν h = 0 := by
        by_contra hz
        exact hatom (hν.2.1 h hz)
      change a * μ h + b * ν h ≠ 0 at hh
      simp only [hm, hn, mul_zero, add_zero, ne_eq, not_true_eq_false] at hh
  · rw [map_add, map_smul, map_smul, hμ.2.2, hν.2.2]
    simpa only [smul_eq_mul, mul_one] using hab

/-- Every single bounded learned head is a feasible atomic state without fixed routes.
Source: exact inclusion of physical heads in the new Appendix A.4 model. -/
theorem matchingMixture_single_mem {V H D : ℕ} (cap : ℝ) (h : MatchingHead V H D)
    (hh : h ∈ matchingHeadBox V H D cap) :
    Finsupp.single h (1 : ℝ) ∈ matchingMixtureDomain V H D cap := by
  classical
  refine ⟨?_, ?_, ?_⟩
  · intro g
    simp only [Finsupp.single_apply]
    split_ifs <;> norm_num
  · intro g hg
    by_cases he : g = h
    · subst g
      exact hh
    · simp only [Finsupp.single_eq_of_ne he, ne_eq, not_true_eq_false] at hg
  · norm_num [matchingMixtureMass]

/-- Distinct matching heads inhabit the same atomic feasible set. -/
example : Finsupp.single (matchingScalarHead 0) (1 : ℝ) ∈ matchingMixtureDomain 2 1 1 1 ∧
    Finsupp.single (matchingScalarHead 1) (1 : ℝ) ∈ matchingMixtureDomain 2 1 1 1 :=
  ⟨matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)),
    matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))⟩

/-- A one-atom distribution evaluates the actual physical attention head on every sample.
Source: exact forward inclusion, not an inverse or a supplied target response, after Appendix A.4. -/
theorem matchingMixture_single_output {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (h : MatchingHead V H D) :
    matchingMixtureSample tokens rows channels (Finsupp.single h 1) = matchingHeadSample tokens rows channels h := by
  rw [matchingMixtureSample, Finsupp.linearCombination_single, one_smul]

/-- The stored state computes a sum of genuine occurrence-attention/value products.
Source: sparsemax Eq. (1) and the finite matching mixture following Appendix A.4. -/
theorem matchingMixtureSample_apply {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (μ : MatchingMixture V H D) (r : Fin R) :
    matchingMixtureSample tokens rows channels μ r =
      ∑ h ∈ μ.support, μ h * matchingHeadOutput h (tokens r) (rows r) (channels r) := by
  simp only [matchingMixtureSample, Finsupp.linearCombination_apply, Finsupp.sum,
    Finset.sum_apply, Pi.smul_apply, smul_eq_mul, matchingHeadSample]

/-- Ordinary multihead evaluation absorbs each mixture coefficient into independent original values.
Source: attention-only head summation in §3.1, with no compensating attention inverse. -/
theorem matchingMixtureSample_physical {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (μ : MatchingMixture V H D) (r : Fin R) :
    matchingMixtureSample tokens rows channels μ r =
      ∑ h ∈ μ.support, matchingHeadOutput (matchingHeadScaleValues (μ h) h) (tokens r) (rows r) (channels r) := by
  rw [matchingMixtureSample_apply]
  apply Finset.sum_congr rfl
  intro h hh
  rw [matchingHeadOutput_scaleValues]

/-- Changed hidden continuations cannot affect any active or newly selected matching head.
Source: actual-head causality before the Appendix A.4 mixture sum. -/
theorem matchingMixtureSample_causal {V H D R T : ℕ} (tokens other : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (μ : MatchingMixture V H D)
    (ht : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    matchingMixtureSample tokens rows channels μ = matchingMixtureSample other rows channels μ := by
  ext r
  rw [matchingMixtureSample_apply, matchingMixtureSample_apply]
  apply Finset.sum_congr rfl
  intro h hh
  rw [matchingHeadOutput_causal h (tokens r) (other r) (rows r) (channels r) (ht r)]

/-- Two hidden continuations satisfy all distributional causality premises, for any learned mixture. -/
example (μ : MatchingMixture 2 1 1) :
    matchingMixtureSample (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0) (fun _ => 0) μ =
    matchingMixtureSample (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2))
      (fun _ => 0) (fun _ => 0) μ := by
  apply matchingMixtureSample_causal
  intro r j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- Every normalized matching mixture has a literal finite coefficient-mass representation.
Source: the stored finite-support state following Appendix A.4. -/
theorem matchingMixtureMass_eq_sum {V H D : ℕ} (μ : MatchingMixture V H D) :
    matchingMixtureMass V H D μ = ∑ h ∈ μ.support, μ h := by
  simp only [matchingMixtureMass, Finsupp.linearCombination_apply, Finsupp.sum, smul_eq_mul, mul_one]

end Transformer.GPTMini.Sparsemax
