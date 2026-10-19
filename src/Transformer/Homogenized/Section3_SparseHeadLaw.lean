module

public import Transformer.Homogenized.RademacherLaw

/-!
# A sparse subGaussian law for the kernel regularity claim

The hypothesis `ass:high_order_short` in arXiv:2604.01978v1, §2.2.3,
controls variance proxies, without a lower bound on the actual variances.
Here `V₀₁` is `+1` or `-1`, each with probability `p/2`, and is otherwise
zero. All other entries and the query/key matrices vanish. The same constants
`C = σ_V = 1`, `σ_A = 0` work for every `0 ≤ p ≤ 1`.

This law tests `lem:Kernel_regularity`, §3.1. Its proof in §6.3 uses an
isotropy identity that is absent from the stated assumption.

The source is the three-point space `Fin 3`. The head law is its actual
pushforward; the zero atom is retained in the probability and integral proofs.
-/

@[expose] public section

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- A symmetric rare pair and a remaining atom. Source: arXiv:2604.01978v1,
§3.1, counterexample to `lem:Kernel_regularity`. -/
noncomputable def sparseMixture {Ω : Type*} [MeasurableSpace Ω]
    (p : ℝ) (a b z : Ω) : Measure Ω :=
  ENNReal.ofReal (p / 2) • (Measure.dirac a + Measure.dirac b) +
    ENNReal.ofReal (1 - p) • Measure.dirac z

/-- All integrands at the three atoms are genuinely integrable. Source:
arXiv:2604.01978v1, §3.1, auxiliary to the kernel counterexample. -/
theorem integrable_sparseMixture {Ω E : Type*} [MeasurableSpace Ω]
    [MeasurableSingletonClass Ω] [NormedAddCommGroup E] [NormedSpace ℝ E]
    (p : ℝ) (a b z : Ω) (f : Ω → E) : Integrable f (sparseMixture p a b z) := by
  exact ((integrable_dirac enorm_lt_top).add_measure
    (integrable_dirac enorm_lt_top)).smul_measure ENNReal.ofReal_ne_top |>.add_measure
      ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top)

/-- The exact three-atom expectation, including the mass at zero. Source:
arXiv:2604.01978v1, §3.1, auxiliary to `lem:Kernel_regularity`. -/
theorem integral_sparseMixture {Ω E : Type*} [MeasurableSpace Ω]
    [MeasurableSingletonClass Ω] [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] {p : ℝ} (hp : p ∈ Set.Icc 0 1) (a b z : Ω) (f : Ω → E) :
    ∫ ω, f ω ∂sparseMixture p a b z =
      (p / 2) • (f a + f b) + (1 - p) • f z := by
  rw [sparseMixture, integral_add_measure
    (((integrable_dirac enorm_lt_top).add_measure (integrable_dirac enorm_lt_top)).smul_measure
      ENNReal.ofReal_ne_top)
    ((integrable_dirac enorm_lt_top).smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure,
    integral_add_measure (integrable_dirac enorm_lt_top) (integrable_dirac enorm_lt_top),
    integral_dirac, integral_dirac, integral_dirac,
    ENNReal.toReal_ofReal (by linarith [hp.1]),
    ENNReal.toReal_ofReal (sub_nonneg.mpr hp.2)]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The three masses sum to one. Source: arXiv:2604.01978v1, §3.1,
auxiliary to the kernel counterexample. -/
theorem isProbabilityMeasure_sparseMixture {Ω : Type*} [MeasurableSpace Ω]
    {p : ℝ} (hp : p ∈ Set.Icc 0 1) (a b z : Ω) :
    IsProbabilityMeasure (sparseMixture p a b z) := by
  constructor
  have hhalf : 0 ≤ p / 2 := by linarith [hp.1]
  simp only [sparseMixture, Measure.add_apply, Measure.smul_apply, measure_univ,
    smul_eq_mul, mul_add, mul_one]
  rw [← ENNReal.ofReal_add hhalf hhalf,
    ← ENNReal.ofReal_add (add_nonneg hhalf hhalf) (sub_nonneg.mpr hp.2)]
  norm_num [show p / 2 + p / 2 + (1 - p) = 1 by ring]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The three values of the only random entry. Source: arXiv:2604.01978v1,
§3.1, counterexample to `lem:Kernel_regularity`. -/
def sparseSign (b : Fin 3) : ℝ := if b = 0 then 1 else if b = 1 then -1 else 0

/-- A single off-diagonal value entry; every remaining entry is constant.
Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
def sparseMatrix (b : Fin 3) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, sparseSign b; 0, 0]

/-- The genuine source probability space. -/
noncomputable def sparseSource (p : ℝ) : Measure (Fin 3) := sparseMixture p 0 1 2

/-- The corresponding law on actual attention heads, with `A = 0`. -/
noncomputable def sparseHeadLaw (p : ℝ) : Measure (HeadParam 2) :=
  sparseMixture p (sparseMatrix 0, 0) (sparseMatrix 1, 0) (sparseMatrix 2, 0)

/-- The actual second moment of the random entry is `p`, despite the fixed
subGaussian proxy `1` used below. The isotropy identity in the source proof
cannot identify those quantities. Source: arXiv:2604.01978v1, §6.3,
proof of `lem:Kernel_regularity`(2). -/
theorem integral_sparse_entry_sq {p : ℝ} (hp : p ∈ Set.Icc 0 1) :
    ∫ b, (sparseMatrix b 0 1) ^ 2 ∂sparseSource p = p := by
  rw [sparseSource, integral_sparseMixture hp]
  norm_num [sparseMatrix, sparseSign]

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

/-- The sparse law satisfies every independence and subGaussian hypothesis
with fixed proxies, even as the actual variance tends to zero. No isotropy
or variance lower bound is added. Source: arXiv:2604.01978v1,
§2.2.3, `ass:high_order_short`. -/
theorem isHighOrderLaw_sparse {p : ℝ} (hp : p ∈ Set.Icc 0 1) :
    IsHighOrderLaw 2 1 1 0 (sparseHeadLaw p) (sparseSource p) sparseMatrix 0 0 0 0 := by
  have : IsProbabilityMeasure (sparseSource p) :=
    isProbabilityMeasure_sparseMixture hp _ _ _
  have hc : ∀ i j, ∫ b, sparseMatrix b i j ∂sparseSource p = 0 := by
    intro i j
    rw [sparseSource, integral_sparseMixture hp]
    fin_cases i <;> fin_cases j <;> simp [sparseMatrix, sparseSign]
  refine ⟨inferInstance, measurable_of_countable _, measurable_const, measurable_const,
    ?_, hc, indepFun_const_right sparseMatrix
      ((0 : Matrix (Fin 2) (Fin 2) ℝ), (0 : Matrix (Fin 2) (Fin 2) ℝ)),
    ?_, ?_, ?_, ?_, ?_⟩
  · have hm : Measurable (fun b : Fin 3 => ((sparseMatrix b, 0) : HeadParam 2)) :=
      measurable_of_countable _
    have hf : (fun b : Fin 3 =>
        (((0 : Matrix (Fin 2) (Fin 2) ℝ) + sparseMatrix b,
          (0 : Matrix (Fin 2) (Fin 2) ℝ) +
            (0 : Fin 3 → Matrix (Fin 2) (Fin 2) ℝ) b *
              ((0 : Fin 3 → Matrix (Fin 2) (Fin 2) ℝ) b).transpose) : HeadParam 2)) =
        fun b : Fin 3 => ((sparseMatrix b, 0) : HeadParam 2) := by funext b; simp
    rw [hf]
    simp [sparseSource, sparseHeadLaw, sparseMixture,
      Measure.map_add _ _ hm, Measure.map_smul _ hm.aemeasurable]
  · refine iIndepFun_of_const_of_ne _ (0, 1) ?_
    intro ij hij
    refine ⟨0, funext fun b => ?_⟩
    rcases ij with ⟨i, j⟩
    fin_cases i <;> fin_cases j <;> simp_all [sparseMatrix]
  · exact iIndepFun_of_const_of_ne _ (true, 0, 0)
      fun _ _ => ⟨0, funext fun _ => by simp⟩
  · intro i j
    have hb : ∀ᵐ b ∂sparseSource p, sparseMatrix b i j ∈ Set.Icc (-1 : ℝ) 1 := by
      apply Filter.Eventually.of_forall
      intro b
      fin_cases i <;> fin_cases j <;> fin_cases b <;>
        norm_num [sparseMatrix, sparseSign]
    have hsg := hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      (measurable_of_countable _).aemeasurable hb (hc i j)
    have hproxy : ((‖(1 : ℝ) - (-1)‖₊ / 2) ^ 2 : ℝ≥0) = 1 * 1 ^ 2 := by norm_num
    rwa [hproxy] at hsg
  · intro i j; simp
  · intro i j; simp

example : (1 / 2 : ℝ) ∈ Set.Icc 0 1 := by norm_num

end Transformer.Homogenized
