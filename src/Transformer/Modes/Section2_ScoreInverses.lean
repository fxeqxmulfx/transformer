/-
# Inverse branches of the original Gaussian KDE score

ArXiv:2412.09080v3, §2.1, the original summand in `eq:Fn`.
The two critical score displacements split the real line into
three open intervals. The actual score is injective on each,
and its derivative is nonzero there. An inverse on its actual
image therefore has the reciprocal derivative. The proof uses
the one-dimensional inverse function theorem and the genuine
left inverse on an open neighborhood of every branch point.

The inverse is defined by `invFunOn`; no equation is imposed
outside the branch image. Every inverse identity or derivative
below includes the branch or image membership it needs.
-/

import Transformer.Modes.Section2_ScoreBranches
import Mathlib.Analysis.Calculus.InverseFunctionTheorem.Deriv

open Real Set Filter
open scoped Topology

namespace Transformer.Modes

/-- The left, middle, and right open branches of the original score in `eq:Fn`.
Source: arXiv:2412.09080v3, §2.1; these are the intervals cut out by its derivative. -/
def kdeScoreBranch (β : ℝ) (b : Fin 3) : Set ℝ :=
  ![Iio (-kdeScoreRadius β), Ioo (-kdeScoreRadius β) (kdeScoreRadius β),
    Ioi (kdeScoreRadius β)] b

/-- The original score inverse restricted to its selected injective open branch.
Source: arXiv:2412.09080v3, §2.1, solving one summand in `eq:Fn`.
Values outside the branch image have no inverse interpretation. -/
noncomputable def kdeScoreInverse (β : ℝ) (b : Fin 3) : ℝ → ℝ :=
  Function.invFunOn (kdeScore β) (kdeScoreBranch β b)

/-- Every selected original score branch is open.
Source: arXiv:2412.09080v3, §2.1, the critical displacements of `eq:Fn`. -/
theorem isOpen_kdeScoreBranch (β : ℝ) (b : Fin 3) : IsOpen (kdeScoreBranch β b) := by
  fin_cases b
  · exact isOpen_Iio
  · exact isOpen_Ioo
  · exact isOpen_Ioi

/-- Positive original bandwidth gives injectivity on every selected score branch.
Source: arXiv:2412.09080v3, §2.1, the derivative of the summands in `eq:Fn`. -/
theorem injOn_kdeScoreBranch {β : ℝ} (hβ : 0 < β) (b : Fin 3) :
    InjOn (kdeScore β) (kdeScoreBranch β b) := by
  fin_cases b
  · exact (kdeScore_branch_injective hβ).1
  · exact (kdeScore_branch_injective hβ).2.1
  · exact (kdeScore_branch_injective hβ).2.2

/-- The original score derivative is nonzero at each point of any selected branch.
Source: arXiv:2412.09080v3, §2.1, the two critical displacements in `eq:Fn`. -/
theorem kdeScoreBranch_deriv_ne_zero {β : ℝ} (hβ : 0 < β) (b : Fin 3)
    (s : ℝ) (hs : s ∈ kdeScoreBranch β b) : deriv (kdeScore β) s ≠ 0 := by
  fin_cases b
  · exact (kdeScore_deriv_neg_left hβ s hs).ne
  · exact (kdeScore_deriv_pos_middle hβ s hs).ne'
  · exact (kdeScore_deriv_neg_right hβ s hs).ne

/-- Every regular original score displacement belongs to one of the three branches.
Source: arXiv:2412.09080v3, §2.1, the derivative of the summands in `eq:Fn`. -/
theorem exists_kdeScoreBranch_of_deriv_ne_zero {β : ℝ} (hβ : 0 < β) (s : ℝ)
    (hs : deriv (kdeScore β) s ≠ 0) : ∃ b : Fin 3, s ∈ kdeScoreBranch β b := by
  rcases kdeScore_regular_branch_cover hβ s hs with hl | hm | hr
  · exact ⟨0, hl⟩
  · exact ⟨1, hm⟩
  · exact ⟨2, hr⟩

/-- The selected inverse returns a point in the original branch at every point of its image.
Source: arXiv:2412.09080v3, §2.1, inversion of the original `eq:Fn` summand. -/
theorem kdeScoreInverse_mem (β : ℝ) (b : Fin 3) (y : ℝ)
    (hy : y ∈ kdeScore β '' kdeScoreBranch β b) :
    kdeScoreInverse β b y ∈ kdeScoreBranch β b :=
  Function.invFunOn_mem hy

/-- The original score composed with its branch inverse is the image coordinate.
Source: arXiv:2412.09080v3, §2.1, inversion of the original `eq:Fn` summand. -/
theorem kdeScore_kdeScoreInverse (β : ℝ) (b : Fin 3) (y : ℝ)
    (hy : y ∈ kdeScore β '' kdeScoreBranch β b) :
    kdeScore β (kdeScoreInverse β b y) = y :=
  Function.invFunOn_eq hy

/-- At positive bandwidth the branch inverse recovers every original displacement in that branch.
Source: arXiv:2412.09080v3, §2.1, inversion of the original `eq:Fn` summand. -/
theorem kdeScoreInverse_kdeScore {β : ℝ} (hβ : 0 < β) (b : Fin 3) (s : ℝ)
    (hs : s ∈ kdeScoreBranch β b) : kdeScoreInverse β b (kdeScore β s) = s :=
  (injOn_kdeScoreBranch hβ b).leftInvOn_invFunOn hs

/-- The actual branch inverse has reciprocal score derivative on its image.
Source: arXiv:2412.09080v3, §2.1, the original score in `eq:Fn`.
The inverse function theorem applies because the branch is open and the original
derivative does not vanish there; no global inverse or extra chart regularity is assumed. -/
theorem hasDerivAt_kdeScoreInverse {β : ℝ} (hβ : 0 < β) (b : Fin 3) (y : ℝ)
    (hy : y ∈ kdeScore β '' kdeScoreBranch β b) :
    HasDerivAt (kdeScoreInverse β b)
      (deriv (kdeScore β) (kdeScoreInverse β b y))⁻¹ y := by
  rcases hy with ⟨s, hs, rfl⟩
  rw [kdeScoreInverse_kdeScore hβ b s hs]
  have hstrict := ((contDiff_kdeScore (k := 1) β).contDiffAt (x := s)).hasStrictDerivAt
    (by norm_num)
  have hleft : ∀ᶠ u in 𝓝 s, kdeScoreInverse β b (kdeScore β u) = u := by
    filter_upwards [(isOpen_kdeScoreBranch β b).mem_nhds hs] with u hu
    exact kdeScoreInverse_kdeScore hβ b u hu
  exact (hstrict.to_local_left_inverse (kdeScoreBranch_deriv_ne_zero hβ b s hs) hleft).hasDerivAt

/-- The derivative of every actual original branch inverse is nonzero on its image.
Source: arXiv:2412.09080v3, §2.1, reciprocal differentiation of the original `eq:Fn` score. -/
theorem kdeScoreInverse_deriv_ne_zero {β : ℝ} (hβ : 0 < β) (b : Fin 3) (y : ℝ)
    (hy : y ∈ kdeScore β '' kdeScoreBranch β b) : deriv (kdeScoreInverse β b) y ≠ 0 := by
  rw [(hasDerivAt_kdeScoreInverse hβ b y hy).deriv]
  exact inv_ne_zero (kdeScoreBranch_deriv_ne_zero hβ b _ (kdeScoreInverse_mem β b y hy))

/-- The actual branch inverse is continuous at every point of its score image.
Source: arXiv:2412.09080v3, §2.1, inverse coordinates for the original `eq:Fn` field. -/
theorem continuousAt_kdeScoreInverse {β : ℝ} (hβ : 0 < β) (b : Fin 3) (y : ℝ)
    (hy : y ∈ kdeScore β '' kdeScoreBranch β b) : ContinuousAt (kdeScoreInverse β b) y :=
  (hasDerivAt_kdeScoreInverse hβ b y hy).continuousAt

/-- At an original displacement the reciprocal derivative retains that same displacement.
Source: arXiv:2412.09080v3, §2.1, differentiation of the actual `eq:Fn` score inverse. -/
theorem hasDerivAt_kdeScoreInverse_image {β : ℝ} (hβ : 0 < β) (b : Fin 3) (s : ℝ)
    (hs : s ∈ kdeScoreBranch β b) :
    HasDerivAt (kdeScoreInverse β b) (deriv (kdeScore β) s)⁻¹ (kdeScore β s) := by
  have h := hasDerivAt_kdeScoreInverse hβ b (kdeScore β s) ⟨s, hs, rfl⟩
  rw [kdeScoreInverse_kdeScore hβ b s hs] at h
  exact h

/-- Positive bandwidth one realizes all three original branches and their inverse-image premises. -/
example : (0 : ℝ) < 1 ∧
    (-2 : ℝ) ∈ kdeScoreBranch 1 0 ∧ (0 : ℝ) ∈ kdeScoreBranch 1 1 ∧
    (2 : ℝ) ∈ kdeScoreBranch 1 2 ∧
    kdeScore 1 (-2) ∈ kdeScore 1 '' kdeScoreBranch 1 0 ∧
    kdeScore 1 0 ∈ kdeScore 1 '' kdeScoreBranch 1 1 ∧
    kdeScore 1 2 ∈ kdeScore 1 '' kdeScoreBranch 1 2 := by
  have hl : (-2 : ℝ) ∈ kdeScoreBranch 1 0 := by norm_num [kdeScoreBranch, kdeScoreRadius]
  have hm : (0 : ℝ) ∈ kdeScoreBranch 1 1 := by norm_num [kdeScoreBranch, kdeScoreRadius]
  have hr : (2 : ℝ) ∈ kdeScoreBranch 1 2 := by norm_num [kdeScoreBranch, kdeScoreRadius]
  exact ⟨by norm_num, hl, hm, hr, ⟨-2, hl, rfl⟩, ⟨0, hm, rfl⟩, ⟨2, hr, rfl⟩⟩

/-- An actual nonzero original middle-branch derivative supplies the regular-cover premise. -/
example : deriv (kdeScore 1) 0 ≠ 0 ∧
    kdeScoreInverse 1 1 (kdeScore 1 0) = 0 := by
  constructor
  · norm_num [(hasDerivAt_kdeScore 1 0).deriv]
  · exact kdeScoreInverse_kdeScore (by norm_num) 1 0
      (by norm_num [kdeScoreBranch, kdeScoreRadius])

end Transformer.Modes
