import Transformer.GPTMini.QKNorm
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add

/-!
# Unit keys realizing prescribed QKNorm scores

Derived upstream construction for arXiv:1602.02068v2, §2.5, with the
actual normalization and score in `QKNormScores.forward` at `73f8a0b`,
with rotary positional maps absent.
For orthogonal unit vectors `query` and `transverse`, a score `s` strictly
between `-gain` and `gain` is realized by the unit key
`(s / gain) • query + sqrt (1 - (s / gain)^2) • transverse`.

The ordinary query/key score is evaluated and proved equal to `s`;
the score is not redefined to return its input. The key chart is
differentiable throughout its open interval. This is a restriction on
key vectors, additional to QKNorm: unconstrained key learning need not
stay in this chart. A fixed orthogonal frame requires head dimension
at least two. Normalization epsilon may be any value at most one.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped Topology

/-- A positive transverse unit-key chart for the actual QKNorm operator.
Derived score realization for §2.5 of arXiv:1602.02068v2 and the
normalized dot product at `73f8a0b`; no attention weights are changed. -/
def qkKeyChart {h : ℕ} (query transverse : EucSpace h) (gain s : ℝ) : EucSpace h :=
  (s / gain) • query + Real.sqrt (1 - (s / gain) ^ 2) • transverse

/-- The transverse coordinate stays away from the square root's singularity.
Source context: the derived key chart for §2.5 of arXiv:1602.02068v2. -/
theorem qkKeyChart_radical_pos (gain s : ℝ) (hg : 0 < gain)
    (hl : -gain < s) (hu : s < gain) : 0 < 1 - (s / gain) ^ 2 := by
  have hlo : -1 < s / gain := (lt_div_iff₀ hg).mpr (by linarith)
  have hup : s / gain < 1 := (div_lt_iff₀ hg).mpr (by linarith)
  have hp : 0 < (1 - s / gain) * (1 + s / gain) :=
    mul_pos (by linarith) (by linarith)
  nlinarith

/-- Finite scores inhabit the chart's strict interval.
Source context: arXiv:1602.02068v2, §2.5, derived key chart. -/
example : (0 : ℝ) < 1 - ((1 / 8 : ℝ) / 2) ^ 2 :=
  qkKeyChart_radical_pos _ _ (by norm_num) (by norm_num) (by norm_num)

/-- The constructed key has norm one, rather than only norm at most one.
Source: the derived Q/K restriction for §2.5 of arXiv:1602.02068v2,
used with `F.normalize` at `73f8a0b`. -/
theorem qkKeyChart_norm {h : ℕ} (query transverse : EucSpace h) (gain s : ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (hg : 0 < gain)
    (hl : -gain < s) (hu : s < gain) : ‖qkKeyChart query transverse gain s‖ = 1 := by
  have hs : ‖qkKeyChart query transverse gain s‖ ^ 2 = 1 := by
    rw [qkKeyChart, norm_add_sq_real, norm_smul, norm_smul,
      real_inner_smul_left, real_inner_smul_right, hq, ht, ho]
    simp only [Real.norm_eq_abs, mul_one, mul_zero, sq_abs, add_zero]
    rw [Real.sq_sqrt (qkKeyChart_radical_pos gain s hg hl hu).le]
    ring
  nlinarith [norm_nonneg (qkKeyChart query transverse gain s)]

/-- Two fixed coordinates provide a concrete admissible query frame.
Source context: the derived QKNorm key chart, head dimension two. -/
def qkQueryExample : EucSpace 2 := EuclideanSpace.single 0 1

/-- The second coordinate provides the transverse direction.
Source context: the same derived QKNorm key chart. -/
def qkTransverseExample : EucSpace 2 := EuclideanSpace.single 1 1

/-- The concrete frame satisfies every geometric premise of the chart.
Source context: arXiv:1602.02068v2, §2.5, derived score realization. -/
theorem qkFrame_example : ‖qkQueryExample‖ = 1 ∧ ‖qkTransverseExample‖ = 1 ∧
    inner (𝕜 := ℝ) qkQueryExample qkTransverseExample = 0 := by
  norm_num [qkQueryExample, qkTransverseExample, EuclideanSpace.inner_single_left]

/-- The unit-key theorem has inhabited frame, gain and score premises.
Source context: arXiv:1602.02068v2, §2.5, derived key chart. -/
example : ‖qkKeyChart qkQueryExample qkTransverseExample 2 (1 / 8)‖ = 1 :=
  qkKeyChart_norm _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num) (by norm_num)

/-- Epsilon normalization fixes a unit vector when epsilon is at most one.
Source: `F.normalize` in `QKNormScores.forward` at `73f8a0b`, expressed by
the existing `normL2`; this is also valid at epsilon zero. -/
theorem normL2_unit {h : ℕ} (eps : ℝ) (x : EucSpace h)
    (he : eps ≤ 1) (hx : ‖x‖ = 1) : normL2 eps x = x := by
  rw [normL2, hx, max_eq_left he]
  norm_num

/-- The normalization premises hold for the implementation's epsilon.
Source context: `QKNormScores.forward` at `73f8a0b`. -/
example : normL2 (1 / 1000000) qkQueryExample = qkQueryExample :=
  normL2_unit _ _ (by norm_num) qkFrame_example.1

/-- The query component of the key is exactly the prescribed scaled score.
Source: the derived orthogonal key restriction for §2.5 of
arXiv:1602.02068v2, before the normalization at `73f8a0b`. -/
theorem qkKeyChart_inner {h : ℕ} (query transverse : EucSpace h) (gain s : ℝ)
    (hq : ‖query‖ = 1) (ho : inner (𝕜 := ℝ) query transverse = 0) :
    inner (𝕜 := ℝ) query (qkKeyChart query transverse gain s) = s / gain := by
  rw [qkKeyChart, inner_add_right, real_inner_smul_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, hq, ho]
  ring

/-- The query-component premises have the same concrete unit frame.
Source context: arXiv:1602.02068v2, §2.5, derived key chart. -/
example : inner (𝕜 := ℝ) qkQueryExample
    (qkKeyChart qkQueryExample qkTransverseExample 2 (1 / 8)) = (1 / 8 : ℝ) / 2 :=
  qkKeyChart_inner _ _ _ _ qkFrame_example.1 qkFrame_example.2.2

/-- Evaluating the existing QKNorm score recovers the prescribed score.
Source: normalized dot product at `73f8a0b`, using the derived key
restriction for §2.5 of arXiv:1602.02068v2. -/
theorem qkKeyChart_score {h : ℕ} (query transverse : EucSpace h) (gain eps s : ℝ)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (hg : 0 < gain)
    (he : eps ≤ 1) (hl : -gain < s) (hu : s < gain) :
    score (Real.log gain) eps query (qkKeyChart query transverse gain s) = s := by
  rw [score, Real.exp_log hg, normL2_unit eps query he hq,
    normL2_unit eps _ he (qkKeyChart_norm query transverse gain s hq ht ho hg hl hu)]
  rw [qkKeyChart_inner query transverse gain s hq ho]
  field_simp

/-- The actual score identity has finite parameters and standard epsilon.
Source context: §2.5's derived chart and the score at `73f8a0b`. -/
example : score (Real.log 2) (1 / 1000000) qkQueryExample
    (qkKeyChart qkQueryExample qkTransverseExample 2 (1 / 8)) = 1 / 8 :=
  qkKeyChart_score _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Small score changes are differentiable changes of the actual key vector.
Source: the derived upstream restriction for §2.5 of arXiv:1602.02068v2;
the square root is differentiated only at a positive argument. -/
theorem qkKeyChart_differentiableAt {h : ℕ} (query transverse : EucSpace h) (gain s : ℝ)
    (hg : 0 < gain) (hl : -gain < s) (hu : s < gain) :
    DifferentiableAt ℝ (qkKeyChart query transverse gain) s := by
  have hd := (hasDerivAt_id s).div_const gain
  have hr : DifferentiableAt ℝ (fun z : ℝ => 1 - (z / gain) ^ 2) s := by
    simpa only [Pi.pow_def, id_eq] using
      (HasDerivAt.const_sub 1 (hd.pow 2)).differentiableAt
  have hs := (Real.hasDerivAt_sqrt
    (ne_of_gt (qkKeyChart_radical_pos gain s hg hl hu))).differentiableAt.comp s hr
  unfold qkKeyChart
  exact hd.differentiableAt.smul_const query |>.add (hs.smul_const transverse)

/-- The differentiability hypotheses are inhabited at a finite key.
Source context: arXiv:1602.02068v2, §2.5, derived key chart. -/
example : DifferentiableAt ℝ (qkKeyChart qkQueryExample qkTransverseExample 2) (1 / 8) :=
  qkKeyChart_differentiableAt _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
