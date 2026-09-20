/-
# The number of modes of a Gaussian KDE — the estimator and its modes

§1.1 of arXiv:2412.09080v3, `sec:setup`: the Gaussian kernel density estimator
`eq:gkde`, the sample it is built from, and the counting of modes that every
statement of the paper is about.

**What the source says and what is carried here.**

* `eq:gkde` is `kde`:  `P̂_n(t) = √β/(n√(2π)) Σ_i e^{-β(t-X_i)²/2}`, the KDE
  with bandwidth `h = β^{-1/2}`.  The sample `X : Idx n → ℝ` is an explicit
  argument; the law `X_1, …, X_n ~iid N(0,1)` enters only through the measure
  `gaussianSample n` that expectations are taken against.

* A *mode* is a local maximum — "the expected number of modes (local maxima)".
  `modeCount f S` counts them in `S` with values in `ℝ≥0∞`, so that a sample
  with infinitely many modes is counted as infinite; `Set.ncard` would count it
  as zero.

* `expectedModes` is the expectation over the sample.  Its real shadow
  `expectedModesReal`, in which the asymptotic statements are written, also
  asserts finiteness whenever it is bounded below: `(∞ : ℝ≥0∞).toReal = 0`.

Source: arXiv:2412.09080v3, `sec:setup`, `eq:gkde`.
-/

import Transformer.Basic
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Analysis.Calculus.DerivativeTest
import Mathlib.Data.Set.Card

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

variable {n : ℕ}

/-! ### The Gaussian kernel density estimator -/

/-- **Equation (eq:gkde).**  The Gaussian kernel density estimator with
bandwidth `h = β^{-1/2}` of the sample `X_1, …, X_n`:

  `P̂_n(t) = (1/n) Σ_i K_h ∗ δ_{X_i}(t) = √β/(n√(2π)) Σ_i e^{-β(t-X_i)²/2}`.

Source: arXiv:2412.09080v3, `eq:gkde`. -/
noncomputable def kde (β : ℝ) (X : Idx n → ℝ) (t : ℝ) : ℝ :=
  Real.sqrt β / (n * Real.sqrt (2 * π)) *
    ∑ i : Idx n, Real.exp (-(β / 2) * (t - X i) ^ 2)

/-- The KDE is smooth: a finite sum of Gaussian bumps. -/
theorem contDiff_kde {k : WithTop ℕ∞} (β : ℝ) (X : Idx n → ℝ) :
    ContDiff ℝ k (kde β X) := by
  refine contDiff_const.mul (ContDiff.sum fun i _ => Real.contDiff_exp.comp ?_)
  exact contDiff_const.mul ((contDiff_id.sub contDiff_const).pow 2)

/-- The KDE is non-negative. -/
theorem kde_nonneg (β : ℝ) (X : Idx n → ℝ) (t : ℝ) : 0 ≤ kde β X t := by
  refine mul_nonneg (div_nonneg (Real.sqrt_nonneg _) ?_) (Finset.sum_nonneg fun i _ => (Real.exp_pos _).le)
  positivity

/-- The KDE is positive as soon as there is a sample and a bandwidth. -/
theorem kde_pos {β : ℝ} (hβ : 0 < β) (hn : 0 < n) (X : Idx n → ℝ) (t : ℝ) : 0 < kde β X t := by
  have hsum : 0 < ∑ i : Idx n, Real.exp (-(β / 2) * (t - X i) ^ 2) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _)
      (Finset.univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hn))
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have : 0 < Real.sqrt β / (n * Real.sqrt (2 * π)) := by
    have : 0 < Real.sqrt (2 * π) := Real.sqrt_pos.mpr (by positivity)
    positivity
  exact mul_pos this hsum

/-! ### The sample and the expectation over it -/

/-- The law of the sample: `X_1, …, X_n` independent standard Gaussians. -/
noncomputable def gaussianSample (n : ℕ) : Measure (Idx n → ℝ) :=
  Measure.pi fun _ => gaussianReal 0 1

instance instIsProbabilityMeasureGaussianSample (n : ℕ) :
    IsProbabilityMeasure (gaussianSample n) := by
  unfold gaussianSample
  infer_instance

/-- Each coordinate of the sample is a standard Gaussian. -/
theorem map_gaussianSample_eval (n : ℕ) (i : Idx n) :
    Measure.map (fun X : Idx n → ℝ => X i) (gaussianSample n) = gaussianReal 0 1 :=
  (measurePreserving_eval (fun _ => gaussianReal 0 1) i).map_eq

/-! ### Modes -/

/-- The modes of `f` in `S`: the points of `S` at which `f` has a local
maximum.

Source: arXiv:2412.09080v3, `sec:setup`, "the expected number of modes (local
maxima)". -/
def modeSet (f : ℝ → ℝ) (S : Set ℝ) : Set ℝ := {t ∈ S | IsLocalMax f t}

/-- The number of modes of `f` in `S`, infinite if there are infinitely
many. -/
noncomputable def modeCount (f : ℝ → ℝ) (S : Set ℝ) : ℝ≥0∞ := ((modeSet f S).encard : ℝ≥0∞)

theorem modeSet_mono (f : ℝ → ℝ) {S S' : Set ℝ} (h : S ⊆ S') : modeSet f S ⊆ modeSet f S' :=
  fun _ ht => ⟨h ht.1, ht.2⟩

theorem modeCount_mono (f : ℝ → ℝ) {S S' : Set ℝ} (h : S ⊆ S') :
    modeCount f S ≤ modeCount f S' := by
  simpa [modeCount] using
    (ENat.toENNReal_le.mpr (Set.encard_le_encard (modeSet_mono f h)))

/-- **The expected number of modes of `P̂_n` in `S`**, the expectation being
over `X_1, …, X_n ~iid N(0,1)`.

Source: arXiv:2412.09080v3, `sec:setup`. -/
noncomputable def expectedModes (β : ℝ) (n : ℕ) (S : Set ℝ) : ℝ≥0∞ :=
  ∫⁻ X, modeCount (kde β X) S ∂(gaussianSample n)

/-- The expected number of modes as a real number.  A lower bound on it — the
`Θ` of `thm:main-result`, say — also asserts that the expectation is finite,
since `(∞ : ℝ≥0∞).toReal = 0`. -/
noncomputable def expectedModesReal (β : ℝ) (n : ℕ) (S : Set ℝ) : ℝ :=
  (expectedModes β n S).toReal

theorem expectedModes_mono (β : ℝ) (n : ℕ) {S S' : Set ℝ} (h : S ⊆ S') :
    expectedModes β n S ≤ expectedModes β n S' :=
  lintegral_mono fun X => modeCount_mono (kde β X) h

theorem expectedModesReal_nonneg (β : ℝ) (n : ℕ) (S : Set ℝ) :
    0 ≤ expectedModesReal β n S := ENNReal.toReal_nonneg

/-! ### The count is not empty: one sample, one mode -/

/-- With a single sample the KDE is one Gaussian bump, whose peak sits at the
sample: `X 0` is a mode of `P̂_1`. -/
theorem isLocalMax_kde_one {β : ℝ} (hβ : 0 ≤ β) (X : Idx 1 → ℝ) :
    IsLocalMax (kde β X) (X 0) := by
  refine Filter.Eventually.of_forall fun t => ?_
  have hc : (0 : ℝ) ≤ Real.sqrt β / ((1 : ℕ) * Real.sqrt (2 * π)) := by positivity
  simp only [kde, Finset.univ_unique, Finset.sum_singleton, Nat.cast_one]
  refine mul_le_mul_of_nonneg_left ?_ (by simpa using hc)
  simp only [Fin.default_eq_zero, sub_self]
  exact Real.exp_le_exp.mpr (by nlinarith [sq_nonneg (t - X 0)])

/-- Hence the mode count of a one-sample KDE is at least one: the definitions
above count something. -/
theorem one_le_modeCount_kde_one {β : ℝ} (hβ : 0 ≤ β) (X : Idx 1 → ℝ) :
    1 ≤ modeCount (kde β X) Set.univ := by
  have hmem : X 0 ∈ modeSet (kde β X) Set.univ := ⟨Set.mem_univ _, isLocalMax_kde_one hβ X⟩
  have : (1 : ℕ∞) ≤ (modeSet (kde β X) Set.univ).encard :=
    Set.one_le_encard_iff_nonempty.mpr ⟨X 0, hmem⟩
  simpa [modeCount] using (ENat.toENNReal_le.mpr this)

/-- The hypotheses of `isLocalMax_kde_one` are satisfiable. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

end Modes
end Transformer
