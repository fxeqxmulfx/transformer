/-
# The number of modes of a Gaussian KDE — a degenerate mode

§2.1 of arXiv:2412.09080v3, the "i.e." after `eq:Fn`, read in the other
direction.

**What the source says and what is carried here.**

* The source passes from "`t` is an upcrossing of `F_n` at level `0`" to "`t` is
  a mode of `P̂_n`" and back, as if the two were the same set of points.  One
  direction is `upcrossingSet_subset_modeSet` of `Section2_Field.lean` and is
  proved.  The other is **false as a pathwise statement**, and
  `not_forall_modeSet_subset_upcrossingSet` refutes it: for the two-point sample
  `X = (-1, 1)` at bandwidth `β = 1`, `t = 0` is a global maximum of `P̂_2` —
  the sum `e^{-(t+1)²/2} + e^{-(t-1)²/2} = 2e^{-1/2}e^{-t²/2}\cosh t` is
  maximal at `t = 0` because `\cosh t ≤ e^{t²/2}` — while `P̂_2''(0) = 0`, so
  `0` is not an upcrossing of `F_2`.

* What the argument needs, and what it in fact has, is the identity **in
  expectation**: a Gaussian sample almost surely has no degenerate critical
  point.  `expectedUpcrossings_le_expectedModes` is the half that follows from
  the inclusion and is proved; the equality
  `expectedModes_eq_expectedUpcrossings` is proved below using the actual
  Gaussian sample law and the nondegeneracy theorem of `Section2_Morse.lean`.

**The probability-zero step.** The derivative of the original score
`s exp(-βs²/2)` vanishes only at the two displacements `±1/sqrt(β)`.
If all sample derivatives vanish together, a pair of samples coincides or
has squared separation `4/β`; those events have probability zero for
independent Gaussian data. At every other zero, one sample coordinate
can be solved using one of the score's three actual inverse branches.
The resulting chart projects to the original sample space with singular
derivative at a degenerate query. Equal-dimension Sard and absolute
continuity of the actual product Gaussian law make these finitely many
critical images null. The two original sets therefore agree almost surely,
simultaneously for every query set, and their extended counts integrate equally.

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`.
-/

import Transformer.Modes.Section2_ModeUpcrossings
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Modes

/-! ### The two-point sample -/

/-- The sample `X = (-1, 1)`, at bandwidth `β = 1`. -/
def twoPoint : Idx 2 → ℝ := ![-1, 1]

/-- `e^{-(t+1)²/2} + e^{-(t-1)²/2} = 2e^{-1/2}e^{-t²/2}\cosh t ≤ 2e^{-1/2}`,
the bound that makes `0` a global maximum of the two-point KDE. -/
theorem twoPoint_sum_le (t : ℝ) :
    Real.exp (-((1 : ℝ) / 2) * (t - -1) ^ 2) + Real.exp (-((1 : ℝ) / 2) * (t - 1) ^ 2)
      ≤ 2 * Real.exp (-((1 : ℝ) / 2)) := by
  have hsplit : Real.exp (-((1 : ℝ) / 2) * (t - -1) ^ 2) + Real.exp (-((1 : ℝ) / 2) * (t - 1) ^ 2)
      = Real.exp (-((1 : ℝ) / 2) * (t ^ 2 + 1)) * (2 * Real.cosh t) := by
    rw [Real.cosh_eq, show -((1 : ℝ) / 2) * (t - -1) ^ 2 = -((1 : ℝ) / 2) * (t ^ 2 + 1) + -t from
      by ring, show -((1 : ℝ) / 2) * (t - 1) ^ 2 = -((1 : ℝ) / 2) * (t ^ 2 + 1) + t from by ring,
      Real.exp_add, Real.exp_add]
    ring
  rw [hsplit]
  calc Real.exp (-((1 : ℝ) / 2) * (t ^ 2 + 1)) * (2 * Real.cosh t)
      ≤ Real.exp (-((1 : ℝ) / 2) * (t ^ 2 + 1)) * (2 * Real.exp (t ^ 2 / 2)) := by
        gcongr
        exact Real.cosh_le_exp_half_sq t
    _ = 2 * Real.exp (-((1 : ℝ) / 2)) := by
        rw [show Real.exp (-((1 : ℝ) / 2) * (t ^ 2 + 1)) * (2 * Real.exp (t ^ 2 / 2))
            = 2 * (Real.exp (-((1 : ℝ) / 2) * (t ^ 2 + 1)) * Real.exp (t ^ 2 / 2)) from by ring,
          ← Real.exp_add,
          show -((1 : ℝ) / 2) * (t ^ 2 + 1) + t ^ 2 / 2 = -((1 : ℝ) / 2) from by ring]

/-- `0` is a global — hence a local — maximum of the two-point KDE. -/
theorem isLocalMax_kde_twoPoint : IsLocalMax (kde 1 twoPoint) 0 := by
  refine Filter.Eventually.of_forall fun t => ?_
  simp only [kde, Fin.sum_univ_two, twoPoint, Matrix.cons_val_zero, Matrix.cons_val_one]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  have h0 : Real.exp (-((1 : ℝ) / 2) * ((0 : ℝ) - -1) ^ 2)
      + Real.exp (-((1 : ℝ) / 2) * ((0 : ℝ) - 1) ^ 2) = 2 * Real.exp (-((1 : ℝ) / 2)) := by
    norm_num
    ring
  rw [h0]
  exact twoPoint_sum_le t

/-- The mode at `0` is degenerate: `P̂_2''(0) = 0`, because `β²(0-X_i)² - β = 0`
for both sample points. -/
theorem deriv_deriv_kde_twoPoint : deriv (deriv (kde 1 twoPoint)) 0 = 0 := by
  rw [(hasDerivAt_deriv_kde 1 twoPoint 0).deriv]
  simp [Fin.sum_univ_two, twoPoint]

/-! ### The pathwise identity is false -/

/-- **A mode need not be an upcrossing.**  The source's "i.e." after `eq:Fn`
reads the equivalence `P̂_n'(t) = 0 ∧ P̂_n''(t) < 0 ⟺ t` is a mode as an
identity of sets; it is an inclusion only.  The two-point sample `(-1, 1)` at
`β = 1` has a global maximum at `0` with `P̂_2''(0) = 0`.

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`. -/
theorem not_forall_modeSet_subset_upcrossingSet :
    ¬ ∀ (n : ℕ) (β : ℝ) (X : Idx n → ℝ) (T : Set ℝ), 0 < β → 0 < n →
      modeSet (kde β X) T ⊆ upcrossingSet (fieldF β X) 0 T := by
  intro h
  have hmem : (0 : ℝ) ∈ modeSet (kde 1 twoPoint) Set.univ :=
    ⟨Set.mem_univ _, isLocalMax_kde_twoPoint⟩
  have hup := h 2 1 twoPoint Set.univ one_pos (by norm_num) hmem
  have := ((isUpcrossing_fieldF_iff (n := 2) one_pos (by norm_num) twoPoint 0).mp hup.2).2
  rw [deriv_deriv_kde_twoPoint] at this
  exact lt_irrefl 0 this

/-! ### What holds is the identity in expectation -/

/-- The inclusion integrates: the expected number of upcrossings of `F_n` never
exceeds the expected number of modes of `P̂_n`.

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`. -/
theorem expectedUpcrossings_le_expectedModes {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (T : Set ℝ) :
    expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 T ≤ expectedModes β n T :=
  lintegral_mono fun X => upcrossingCount_le_modeCount hβ hn X T

/-- **The identity the proof of `thm:main-result` uses.**  A Gaussian sample
almost surely has no degenerate critical point, so the two counts agree in
expectation even though they differ pathwise.

The actual Gaussian sample has no degenerate critical point almost surely,
as proved by the sample-coordinate charts and critical-image measure argument.
No extra genericity, regularity, finiteness, or measurability assumption on `T` is needed.

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`. -/
theorem expectedModes_eq_expectedUpcrossings {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (T : Set ℝ) :
    expectedModes β n T = expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 T := by
  exact lintegral_congr_ae (modeCount_ae_eq_upcrossingCount hβ hn T)

/-- The original real-valued expected mode count has the same upcrossing representation.
Source: arXiv:2412.09080v3, `sec:kac-rice`, the expectation identity after `eq:Fn`.
This keeps the formalization's existing `toReal` convention, including at an infinite extended count. -/
theorem expectedModesReal_eq_expectedUpcrossings_toReal {β : ℝ} (hβ : 0 < β)
    {n : ℕ} (hn : 0 < n) (T : Set ℝ) : expectedModesReal β n T =
      (expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 T).toReal := by
  exact congrArg ENNReal.toReal (expectedModes_eq_expectedUpcrossings hβ hn T)

/-- The hypotheses of the two results above are satisfiable. -/
example : (0 : ℝ) < 1 ∧ 0 < 1 := ⟨one_pos, one_pos⟩

end Modes
end Transformer
