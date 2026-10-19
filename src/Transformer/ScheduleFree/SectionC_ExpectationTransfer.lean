module

public import Transformer.ScheduleFree.Section2_SGD

/-!
# Expectations and random weights

arXiv:2405.15682v4, Appendix C, Theorem 5. The integration lemmas
retain Bregman penalties while replacing population pairings by unbiased
oracle pairings. The full stochastic Bregman identity is proved for
fixed weights, including the `α=0` endpoint.

The final section refutes unrestricted random normalization: for two
independent uniform signs, zero population loss, `z₁=1`, `z₂=0`,
`w₁=1`, and `w₂=1` only after a positive first sample, normalized
regret has expectation `-1/4`. These predictable weights belong to
`[0,1]`, all prefixes are positive, and population Bregman terms vanish.
The aggregator records the source corrections and exact scope.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

section

namespace Transformer.ScheduleFree

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [IsProbabilityMeasure μ] in
/-- Linearity and the oracle moment identities preserve any integrable penalty. -/
theorem mean_penalty_eq (w : ℕ → ℝ) (T : ℕ) (a b p : ℕ → Ω → ℝ)
    (hmean : ∀ t < T, (∫ ω, a t ω ∂μ) = ∫ ω, b t ω ∂μ)
    (ha : ∀ t < T, Integrable (a t) μ) (hb : ∀ t < T, Integrable (b t) μ)
    (hp : ∀ t < T, Integrable (p t) μ) :
    (∫ ω, ∑ t ∈ Finset.range T, w t * (a t ω - p t ω) ∂μ) =
      ∫ ω, ∑ t ∈ Finset.range T, w t * (b t ω - p t ω) ∂μ := by
  have hsa (t : ℕ) (ht : t < T) : Integrable (fun ω => w t * (a t ω - p t ω)) μ := by
    simpa only [Pi.sub_apply] using ((ha t ht).sub (hp t ht)).const_mul (w t)
  have hsb (t : ℕ) (ht : t < T) : Integrable (fun ω => w t * (b t ω - p t ω)) μ := by
    simpa only [Pi.sub_apply] using ((hb t ht).sub (hp t ht)).const_mul (w t)
  rw [integral_finsetSum (Finset.range T) (fun t ht => hsa t (Finset.mem_range.mp ht)),
    integral_finsetSum (Finset.range T) (fun t ht => hsb t (Finset.mem_range.mp ht))]
  apply Finset.sum_congr rfl
  intro t ht
  have h1 := integral_sub (ha t (Finset.mem_range.mp ht)) (hp t (Finset.mem_range.mp ht))
  have h2 := integral_sub (hb t (Finset.mem_range.mp ht)) (hp t (Finset.mem_range.mp ht))
  rw [integral_const_mul, integral_const_mul, h1, h2, hmean t (Finset.mem_range.mp ht)]

omit [IsProbabilityMeasure μ] in
/-- Integrate a conversion bound while retaining its penalty. -/
theorem expectation_penalty (w : ℕ → ℝ) (W : ℝ) (T : ℕ)
    (f : Ω → ℝ) (a b p : ℕ → Ω → ℝ)
    (hpath : ∀ ω, W * f ω ≤ ∑ t ∈ Finset.range T, w t * (a t ω - p t ω))
    (hmean : ∀ t < T, (∫ ω, a t ω ∂μ) = ∫ ω, b t ω ∂μ)
    (hf : Integrable f μ) (ha : ∀ t < T, Integrable (a t) μ)
    (hb : ∀ t < T, Integrable (b t) μ) (hp : ∀ t < T, Integrable (p t) μ) :
    W * (∫ ω, f ω ∂μ) ≤ ∫ ω, ∑ t ∈ Finset.range T, w t * (b t ω - p t ω) ∂μ := by
  have hia := integrable_finsetSum (Finset.range T) (fun t ht => by
    simpa only [Pi.sub_apply] using ((ha t (Finset.mem_range.mp ht)).sub
      (hp t (Finset.mem_range.mp ht))).const_mul (w t))
  have h := integral_mono_ae (hf.const_mul W) hia (Filter.Eventually.of_forall hpath)
  rw [integral_const_mul] at h
  have h' : W * (∫ ω, f ω ∂μ) ≤ ∫ ω, ∑ t ∈ Finset.range T, w t * (a t ω - p t ω) ∂μ := by
    simpa only [Pi.sub_apply] using h
  exact h'.trans_eq (mean_penalty_eq w T a b p hmean ha hb hp)

/-- Transfer a pathwise population bound to an observed-oracle upper bound.
The two inequalities are inputs from the conversion and regret proofs,
respectively; equality of scalar moments is the oracle interface. -/
theorem expectation_transfer (w : ℕ → ℝ) (W B : ℝ) (T : ℕ)
    (f : Ω → ℝ) (a b p : ℕ → Ω → ℝ)
    (hpath : ∀ ω, W * f ω ≤ ∑ t ∈ Finset.range T, w t * (a t ω - p t ω))
    (hupper : ∀ ω, (∑ t ∈ Finset.range T, w t * (b t ω - p t ω)) ≤ B)
    (hmean : ∀ t < T, (∫ ω, a t ω ∂μ) = ∫ ω, b t ω ∂μ)
    (hf : Integrable f μ) (ha : ∀ t < T, Integrable (a t) μ)
    (hb : ∀ t < T, Integrable (b t) μ) (hp : ∀ t < T, Integrable (p t) μ) :
    W * (∫ ω, f ω ∂μ) ≤ B := by
  have h1 := expectation_penalty w W T f a b p hpath hmean hf ha hb hp
  have hib := integrable_finsetSum (Finset.range T) (fun t ht => by
    simpa only [Pi.sub_apply] using ((hb t (Finset.mem_range.mp ht)).sub
      (hp t (Finset.mem_range.mp ht))).const_mul (w t))
  have h2 := integral_mono_ae hib (integrable_const B) (Filter.Eventually.of_forall hupper)
  apply h1.trans
  simpa only [integral_const, probReal_univ, one_smul, Pi.sub_apply] using h2

example (T : ℕ) : (1 : ℝ) * (∫ _ω : Ω, (0 : ℝ) ∂μ) ≤ 0 :=
  expectation_transfer (fun _ => 1) 1 0 T (fun _ => 0) (fun _ _ => 0) (fun _ _ => 0)
    (fun _ _ => 0) (by intro ω; simp) (by intro ω; simp) (by intro t ht; rfl)
    (by simp) (by intro t ht; simp) (by intro t ht; simp) (by intro t ht; simp)

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {P : Measure Ω}

/-- Appendix C, Theorem 5: the full equality before dividing by the fixed
total weight. No convexity is needed until the signs of the defects are used. -/
theorem expected_bregman_identity (F : E → ℝ) (grad : E → E) (w : ℕ → ℝ)
    (α : Ω → ℕ → ℝ) (z g : ℕ → Ω → E) (u : E) (T : ℕ)
    (hW : ∀ t < T, weightSum w (t + 1) ≠ 0)
    (hunbiased : UnbiasedPairings P grad
      (fun t ω => query w (α ω) (fun k => z k ω) t) z g)
    (htrue : ∀ t < T, Integrable (fun ω =>
      inner ℝ (grad (query w (α ω) (fun k => z k ω) t)) (z t ω - u)) P)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) P)
    (hdef : ∀ t < T, Integrable (fun ω => defect F grad w (α ω) (fun k => z k ω) u t) P) :
    weightSum w T * (∫ ω, F (average w (fun k => z k ω) T) - F u ∂P) =
      (∫ ω, ∑ t ∈ Finset.range T, w t * inner ℝ (g t ω) (z t ω - u) ∂P) -
        ∫ ω, ∑ t ∈ Finset.range T, defect F grad w (α ω) (fun k => z k ω) u t ∂P := by
  have hpoint (ω : Ω) := conversion_identity F grad w (α ω) (fun k => z k ω) u T hW
  simp only [Finset.sum_sub_distrib] at hpoint
  have he := integral_congr_ae (μ := P) (Filter.Eventually.of_forall hpoint)
  have hts := integrable_finsetSum (Finset.range T)
    (fun t ht => (htrue t (Finset.mem_range.mp ht)).const_mul (w t))
  have hds := integrable_finsetSum (Finset.range T)
    (fun t ht => hdef t (Finset.mem_range.mp ht))
  rw [integral_const_mul, integral_sub hts hds] at he
  have hm := mean_penalty_eq w T
    (fun t ω => inner ℝ (grad (query w (α ω) (fun k => z k ω) t)) (z t ω - u))
    (fun t ω => inner ℝ (g t ω) (z t ω - u)) (fun _ _ => 0)
    (fun t _ => hunbiased t u) htrue hobs (by intro t ht; simp)
  simp only [sub_zero] at hm
  rw [hm] at he
  exact he

example (T : ℕ) :
    weightSum (fun _ => 1) T * (∫ _ω : Ω, (0 : ℝ) - 0 ∂P) =
      (∫ _ω : Ω, ∑ _t ∈ Finset.range T, (1 : ℝ) * inner ℝ (0 : ℝ) (0 - 0) ∂P) -
        ∫ _ω : Ω, ∑ t ∈ Finset.range T,
          defect (fun _ => 0) (fun _ => 0) (fun _ => 1) (fun _ => 0) (fun _ => (0 : ℝ)) 0 t ∂P :=
  expected_bregman_identity (fun _ : ℝ => 0) (fun _ => 0) (fun _ => 1) (fun _ _ => 0)
    (fun _ _ => 0) (fun _ _ => 0) 0 T
    (fun t _ => (weightSum_pos_succ (fun _ => zero_le_one) zero_lt_one t).ne')
    (by intro t u; rfl) (by intro t ht; simp) (by intro t ht; simp)
    (by intro t ht; simp [defect, mixGap, bregman, mix])

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

/-- The two equally likely outputs of each fresh unbiased oracle call. -/
def oracleSign (s : Bool) : ℝ := if s then 1 else -1

/-- The second weight is predictable, depending only on the first sample. -/
def predictableWeights (s : Bool) (t : ℕ) : ℝ :=
  if t = 0 then 1 else if t = 1 then (if s then 1 else 0) else 0

/-- Arbitrary base vectors, as allowed in Theorems 2 and 5. -/
def counterBase (t : ℕ) : ℝ := if t = 0 then 1 else 0

/-- The two samples are the independent coordinates of `Bool × Bool`. -/
def counterOracle (s r : Bool) (t : ℕ) : ℝ :=
  if t = 0 then oracleSign s else oracleSign r

theorem predictableWeights_admissible (s : Bool) (t : ℕ) :
    0 ≤ predictableWeights s t ∧ predictableWeights s t ≤ 1 := by
  unfold predictableWeights
  split_ifs <;> norm_num

theorem predictableWeights_total (s : Bool) : 0 < weightSum (predictableWeights s) 2 := by
  cases s <;> norm_num [weightSum, predictableWeights, Finset.sum_range_succ]

theorem oracleSign_unbiased : (∑ s : Bool, oracleSign s) / 2 = 0 := by
  norm_num [Fintype.sum_bool, oracleSign]

/-- Counterexample to Appendix C, Theorem 5's random-weight equality, and to
the corresponding normalized inequality: `0 ≤ -1/4` would be required. -/
theorem random_weights_counterexample :
    (∑ s : Bool, ∑ r : Bool,
      (∑ t ∈ Finset.range 2,
        predictableWeights s t * inner ℝ (counterOracle s r t) (counterBase t - 0)) /
          weightSum (predictableWeights s) 2) / 4 = -(1 / 4 : ℝ) ∧
      ¬(0 : ℝ) ≤ -(1 / 4 : ℝ) := by
  norm_num [Fintype.sum_bool, Finset.sum_range_succ, predictableWeights,
    counterOracle, counterBase, oracleSign, weightSum, RCLike.inner_apply]

end Transformer.ScheduleFree

end
