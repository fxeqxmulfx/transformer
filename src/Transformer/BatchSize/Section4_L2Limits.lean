/-
# Actual L2 limits from geometric mean-square refinement estimates

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
These completeness lemmas turn proved Euler comparison estimates into
square-integrable random limits, rather than assuming a diffusion law.
-/

import Transformer.BatchSize.Section4_DyadicEuler
import Mathlib.MeasureTheory.Function.LpSpace.Complete
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Function.ConditionalExpectation.AEMeasurable
import Mathlib.Topology.Algebra.InfiniteSum.Real

open MeasureTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The L2 norm of an actual random variable is its mean-square norm,
Section 4.3, used to pass Euler refinements to a limit. -/
theorem toLp_meanSquare {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (P : Measure Ω)
    (X : Ω → E) (hX : MemLp X 2 P) :
    ‖hX.toLp X‖ ^ 2 = ∫ ω, ‖X ω‖ ^ 2 ∂P := by
  rw [← real_inner_self_eq_norm_sq, L2.inner_def]
  apply integral_congr_ae
  filter_upwards [hX.coeFn_toLp] with ω hω
  rw [hω, real_inner_self_eq_norm_sq]

/-- L2 distance between two random variables equals their actual
mean-square difference, Section 4.3, Theorem 1. -/
theorem toLp_dist_meanSquare {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (P : Measure Ω)
    (X Y : Ω → E) (hX : MemLp X 2 P) (hY : MemLp Y 2 P) :
    dist (hX.toLp X) (hY.toLp Y) ^ 2 = ∫ ω, ‖X ω - Y ω‖ ^ 2 ∂P := by
  rw [dist_eq_norm, ← hX.toLp_sub hY, toLp_meanSquare]
  rfl

/-- A geometric bound on successive mean-square differences produces
an actual square-integrable limit, Section 4.3, Theorem 1.
Completeness is applied to L2 equivalence classes; the conclusion then
compares the original random variables to a representative of the limit. -/
theorem meanSquare_geometric_limit {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (P : Measure Ω) (X : ℕ → Ω → E) (hX : ∀ m, MemLp (X m) 2 P)
    (K : ℝ) (hK : 0 ≤ K)
    (hbound : ∀ m, (∫ ω, ‖X m ω - X (m + 1) ω‖ ^ 2 ∂P) ≤ K * (1 / 2 : ℝ) ^ m) :
    ∃ Y : Ω → E, MemLp Y 2 P ∧
      Tendsto (fun m => ∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0) := by
  let Z (m : ℕ) : Lp E 2 P := (hX m).toLp (X m)
  let r : ℝ := Real.sqrt (1 / 2)
  have hr : 0 ≤ r := Real.sqrt_nonneg _
  have hr1 : r < 1 := by
    simpa only [Real.sqrt_one] using Real.sqrt_lt_sqrt (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (by norm_num : (1 / 2 : ℝ) < 1)
  have hrsq : r ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have hdist (m : ℕ) : dist (Z m) (Z (m + 1)) ≤ Real.sqrt K * r ^ m := by
    have h := hbound m
    rw [← toLp_dist_meanSquare P (X m) (X (m + 1)) (hX m) (hX (m + 1))] at h
    have heq : (Real.sqrt K * r ^ m) ^ 2 = K * (1 / 2 : ℝ) ^ m := by
      rw [mul_pow, Real.sq_sqrt hK, ← pow_mul, Nat.mul_comm m 2, pow_mul, hrsq]
    exact (sq_le_sq₀ dist_nonneg (by positivity)).mp (h.trans_eq heq.symm)
  have hc : CauchySeq Z := cauchySeq_of_dist_le_of_summable
    (fun m => Real.sqrt K * r ^ m) hdist ((summable_geometric_of_lt_one hr hr1).mul_left _)
  obtain ⟨Y, hY⟩ := cauchySeq_tendsto_of_complete hc
  refine ⟨Y, Lp.memLp Y, ?_⟩
  have hn := ((tendsto_iff_norm_sub_tendsto_zero).mp hY).pow 2
  have heq (m : ℕ) : ‖Z m - Y‖ ^ 2 = ∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂P := by
    have h := toLp_dist_meanSquare P (X m) Y (hX m) (Lp.memLp Y)
    simpa only [Lp.toLp_coeFn, dist_eq_norm] using h
  simpa only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), heq] using hn

/-- Mean-square convergence gives convergence in the actual
L2 space, Section 4.3, Theorem 1. -/
theorem meanSquare_tendsto_toLp {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (P : Measure Ω) (X : ℕ → Ω → E) (hX : ∀ n, MemLp (X n) 2 P)
    (Y : Ω → E) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun n => ∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0)) :
    Tendsto (fun n => (hX n).toLp (X n)) atTop (𝓝 (hY.toLp Y)) := by
  have heq (n : ℕ) : (∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) =
      ‖(hX n).toLp (X n) - hY.toLp Y‖ ^ 2 := by
    simpa only [dist_eq_norm] using (toLp_dist_meanSquare P (X n) Y (hX n) hY).symm
  have hn := (Real.continuous_sqrt.tendsto (0 : ℝ)).comp hlim
  simp only [Function.comp_def, heq, Real.sqrt_sq_eq_abs, abs_norm, Real.sqrt_zero] at hn
  exact tendsto_iff_norm_sub_tendsto_zero.mpr hn

/-- An L2 limit of variables measurable in a fixed Brownian past is
still measurable in that past up to null sets, Section 4.3, Theorem 1.
The closed subspace is defined by actual almost-everywhere equality to
strongly measurable functions in the smaller measurable space. -/
theorem meanSquare_limit_adapted {Ω E : Type*} [m₀ : MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (P : Measure Ω) (m : MeasurableSpace Ω) (hm : m ≤ m₀)
    (X : ℕ → Ω → E) (hX : ∀ n, MemLp (X n) 2 P)
    (hXm : ∀ n, StronglyMeasurable[m] (X n)) (Y : Ω → E) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun n => ∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0)) :
    AEStronglyMeasurable[m] Y P := by
  let : MeasurableSpace Ω := m₀
  let Z (n : ℕ) : Lp E 2 P := (hX n).toLp (X n)
  let W : Lp E 2 P := hY.toLp Y
  have hZW : Tendsto Z atTop (𝓝 W) := meanSquare_tendsto_toLp P X hX Y hY hlim
  have hclosed := isClosed_aestronglyMeasurable (F := E) (p := 2) (μ := P) hm
  have hW : AEStronglyMeasurable[m] W P := hclosed.mem_of_tendsto hZW (Eventually.of_forall
    fun n => ((hXm n).aestronglyMeasurable).congr (hX n).coeFn_toLp.symm)
  exact hW.congr hY.coeFn_toLp

/-- Joint nonvacuity of L2 limit hypotheses, Section 4.3:
the same nonconstant Brownian random variable at each refinement level. -/
example : (∀ m : ℕ, MemLp ((fun _ : ℕ => coordinateBrownian (0 : Fin 1) 1) m)
    2 (brownianNoiseLaw 1)) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ m : ℕ, (∫ ω, ‖coordinateBrownian (0 : Fin 1) 1 ω -
      coordinateBrownian (0 : Fin 1) 1 ω‖ ^ 2 ∂brownianNoiseLaw 1) ≤ (1 : ℝ) * (1 / 2) ^ m) := by
  refine ⟨fun _ => ((coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two,
    by norm_num, ?_⟩
  intro m
  simp only [sub_self, norm_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0), integral_zero, one_mul]
  positivity

/-- Joint nonvacuity of adapted L2 limit hypotheses, Section 4.3:
the entire sequence and its limit equal a genuine past Brownian value. -/
example : brownianFiltration 1 1 ≤ (inferInstance : MeasurableSpace (BrownianSample 1)) ∧
    (∀ n : ℕ, MemLp ((fun _ : ℕ => coordinateBrownian (0 : Fin 1) 1) n)
    2 (brownianNoiseLaw 1)) ∧
    (∀ n : ℕ, StronglyMeasurable[brownianFiltration 1 1]
      ((fun _ : ℕ => coordinateBrownian (0 : Fin 1) 1) n)) ∧
    MemLp (coordinateBrownian (0 : Fin 1) 1) 2 (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω, ‖coordinateBrownian (0 : Fin 1) 1 ω -
      coordinateBrownian (0 : Fin 1) 1 ω‖ ^ 2 ∂brownianNoiseLaw 1) atTop (𝓝 0) := by
  have h := ((coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two
  refine ⟨(brownianFiltration 1).le 1, fun _ => h,
    fun _ => (coordinateBrownian_filtered (0 : Fin 1)).stronglyAdapted 1, h, ?_⟩
  simp

end Transformer.BatchSize
