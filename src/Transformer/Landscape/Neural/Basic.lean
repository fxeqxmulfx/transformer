/-
# Analytic activations and positive-epsilon RMS normalization

Sources: `landscape_grokking_plan.md`, Activation comparisons and Model and
protocol; `grokking_abmod/web/own-channel/README.md`, Architecture and
initialization, revision f49969393a7482490d45483d8a4fd3046b1adede.

SiLU is the scalar replacement for ReLU. SwiGLU multiplies independent gate
and value coordinates; its projections belong to the network, not to this
scalar activation. The browser's RMSNorm retains its positive epsilon.
These results concern real arithmetic, rather than FP32 evaluation.
-/

import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open scoped BigOperators
open Filter

namespace Transformer.Landscape.Neural

variable {P W : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P] [Fintype W]

/-- SiLU, with unit sigmoid slope: the scalar formula of the plan's
Activation comparisons. It has no additional trainable parameter. -/
noncomputable def silu (t : ℝ) : ℝ := t / (1 + Real.exp (-t))

/-- One SwiGLU coordinate, with independent gate and value inputs; see
Activation comparisons in `landscape_grokking_plan.md`. -/
noncomputable def swiglu (gate value : ℝ) : ℝ := silu gate * value

/-- Mean square over the channel index, as in the lab's RMSNorm formula.
An empty index uses the real division convention; the lab has positive width. -/
noncomputable def meanSquare (v : W → ℝ) : ℝ := (∑ i, v i ^ 2) / Fintype.card W

/-- RMSNorm with the lab's additive epsilon and no learned gain. -/
noncomputable def rmsNorm (eps : ℝ) (v : W → ℝ) (i : W) : ℝ :=
  v i / Real.sqrt (eps + meanSquare v)

/-- The denominator of SiLU is strictly positive for every real input
(Activation comparisons, scalar formula). -/
theorem silu_denominator_pos (t : ℝ) : 0 < 1 + Real.exp (-t) := by
  positivity

/-- SiLU has only one zero, unlike ReLU's negative half-line
(Activation comparisons, scalar formula). -/
theorem silu_eq_zero_iff (t : ℝ) : silu t = 0 ↔ t = 0 := by
  rw [silu, div_eq_zero_iff]
  exact or_iff_left (silu_denominator_pos t).ne'

/-- A zero SwiGLU coordinate comes from a zero gate or zero value
(Activation comparisons, gated formula). -/
theorem swiglu_eq_zero_iff (g v : ℝ) : swiglu g v = 0 ↔ g = 0 ∨ v = 0 := by
  rw [swiglu, mul_eq_zero, silu_eq_zero_iff]

/-- The scalar SiLU replacement is analytic in a varying parameter vector;
the denominator cannot vanish (Activation comparisons). -/
theorem analyticAt_silu_comp {f : P → ℝ} {x : P} (hf : AnalyticAt ℝ f x) :
    AnalyticAt ℝ (fun y => silu (f y)) x := by
  exact hf.fun_div (analyticAt_const.fun_add hf.fun_neg.rexp')
    (silu_denominator_pos (f x)).ne'

/-- SiLU is analytic on all real inputs (Activation comparisons). -/
theorem analyticAt_silu (t : ℝ) : AnalyticAt ℝ silu t := by
  exact analyticAt_silu_comp analyticAt_id

/-- Both inputs of a SwiGLU coordinate may depend on the same parameters;
no independence of their derivatives is assumed (Activation comparisons). -/
theorem analyticAt_swiglu_comp {g v : P → ℝ} {x : P}
    (hg : AnalyticAt ℝ g x) (hv : AnalyticAt ℝ v x) :
    AnalyticAt ℝ (fun y => swiglu (g y) (v y)) x := by
  exact (analyticAt_silu_comp hg).fun_mul hv

/-- Square root preserves analyticity where its argument is positive.
This is the positive-denominator step of the plan's RMSNorm formula. -/
theorem analyticAt_sqrt_pos {f : P → ℝ} {x : P} (hf : AnalyticAt ℝ f x)
    (hpos : 0 < f x) : AnalyticAt ℝ (fun y => Real.sqrt (f y)) x := by
  have ha : AnalyticAt ℝ (fun y => Real.exp (Real.log (f y) * (1 / 2 : ℝ))) x :=
    (((analyticAt_log hpos).comp hf).fun_mul analyticAt_const).rexp'
  apply ha.congr
  filter_upwards [continuousAt_const.eventually_lt hf.continuousAt hpos] with y hy
  exact ((Real.sqrt_eq_rpow (f y)).trans (Real.rpow_def_of_pos hy (1 / 2))).symm

/-- A channel mean square is nonnegative (Model and protocol, RMSNorm). -/
theorem meanSquare_nonneg (v : W → ℝ) : 0 ≤ meanSquare v := by
  exact div_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg (v i)) (Nat.cast_nonneg _)

/-- Positive epsilon keeps the radicand positive at every vector, including
the zero vector (Model and protocol, RMSNorm). -/
theorem rmsNorm_radicand_pos {eps : ℝ} (heps : 0 < eps) (v : W → ℝ) :
    0 < eps + meanSquare v := by
  exact add_pos_of_pos_of_nonneg heps (meanSquare_nonneg v)

/-- Finite channel means of squared analytic coordinates are analytic
(Model and protocol, RMSNorm). -/
theorem analyticAt_meanSquare {v : P → W → ℝ} {x : P}
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) :
    AnalyticAt ℝ (fun y => meanSquare (v y)) x := by
  exact (Finset.analyticAt_fun_sum _ fun i _ => (hv i).pow 2).div_const

/-- RMSNorm is analytic in all varying coordinates when epsilon is positive.
The proof includes zero activations; no nonzero-vector hypothesis is needed
(Model and protocol, epsilon 1e-5). -/
theorem analyticAt_rmsNorm {eps : ℝ} (heps : 0 < eps)
    {v : P → W → ℝ} {x : P} (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (i : W) :
    AnalyticAt ℝ (fun y => rmsNorm eps (v y) i) x := by
  have hr : AnalyticAt ℝ (fun y => eps + meanSquare (v y)) x :=
    analyticAt_const.fun_add (analyticAt_meanSquare hv)
  exact (hv i).fun_div (analyticAt_sqrt_pos hr (rmsNorm_radicand_pos heps (v x)))
    (Real.sqrt_pos.mpr (rmsNorm_radicand_pos heps (v x))).ne'

/- Concrete witnesses for every theorem with hypotheses. They allow changing
parameters and pass through the zero input of the browser's normalization. -/

example : AnalyticAt ℝ (fun t : ℝ => t ^ 2 - 1) 0 ∧
    AnalyticAt ℝ (fun t : ℝ => silu (t ^ 2 - 1)) 0 := by
  have h : AnalyticAt ℝ (fun t : ℝ => t ^ 2 - 1) 0 := by fun_prop
  exact ⟨h, analyticAt_silu_comp h⟩

example : AnalyticAt ℝ (fun t : ℝ => t) 0 ∧
    AnalyticAt ℝ (fun t : ℝ => t ^ 2 + 1) 0 ∧
    AnalyticAt ℝ (fun t : ℝ => swiglu t (t ^ 2 + 1)) 0 := by
  have hg : AnalyticAt ℝ (fun t : ℝ => t) 0 := by fun_prop
  have hv : AnalyticAt ℝ (fun t : ℝ => t ^ 2 + 1) 0 := by fun_prop
  exact ⟨hg, hv, analyticAt_swiglu_comp hg hv⟩

example : AnalyticAt ℝ (fun t : ℝ => t ^ 2 + 1) 0 ∧
    0 < (0 : ℝ) ^ 2 + 1 ∧
    AnalyticAt ℝ (fun t : ℝ => Real.sqrt (t ^ 2 + 1)) 0 := by
  have h : AnalyticAt ℝ (fun t : ℝ => t ^ 2 + 1) 0 := by fun_prop
  exact ⟨h, by norm_num, analyticAt_sqrt_pos h (by norm_num)⟩

example : (0 : ℝ) < 1 / 100000 ∧
    0 < 1 / 100000 + meanSquare (fun _ : Fin 2 => (0 : ℝ)) := by
  exact ⟨by norm_num, rmsNorm_radicand_pos (by norm_num) _⟩

example : (∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t + (i.val : ℝ)) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => meanSquare (fun i : Fin 2 => t + (i.val : ℝ))) 0 := by
  have h : ∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t + (i.val : ℝ)) 0 := by
    intro i
    fun_prop
  exact ⟨h, analyticAt_meanSquare h⟩

example : (0 : ℝ) < 1 / 100000 ∧
    (∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => (fun _ : Fin 2 => t) i) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => rmsNorm (1 / 100000) (fun _ : Fin 2 => t) 0) 0 := by
  have h : ∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => (fun _ : Fin 2 => t) i) 0 := by
    intro i
    fun_prop
  exact ⟨by norm_num, h, analyticAt_rmsNorm (by norm_num) h 0⟩

end Transformer.Landscape.Neural
