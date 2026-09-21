/-
# The emergence of clusters in self-attention dynamics — the attention weights
  of §10

§10 of arXiv:2305.05465v6, the proof of Step 2': the splitting of the
exponent `⟨Ae^{tV}z_i, Ae^{tV}(z_j - z_i)⟩` into a leading part `a_j(t)e^{2λ₁t}`
and a remainder `r_j(t)`, the bound `e:boundrj` on the remainder (refuted), and the
softmax estimate the third bullet of the proof runs on.

**What the source says and what is carried here.**

* `a_j(t) = ⟨π_ℱ(Az_i(t)), π_ℱ(A(z_j(t) - z_i(t)))⟩` and
  `r_j(t) = ⟨Ae^{tV}z_i(t), Ae^{tV}(z_j(t)-z_i(t))⟩ - a_j(t)e^{2λ₁t}` are
  `projScore` and `scoreRemainder`, verbatim.

* `e:boundrj`, `|r_j(t)| ≤ Ce^{t(λ₁+|λ₂|)}`, is false as written: the
  source derives it through the three cross terms `P₁, P₂, P₃`, which uses
  `π_ℱ(Ae^{tV}z_j(t)) = e^{λ₁t}π_ℱ(Az_j(t))`, true only when `A` respects the
  splitting.  The counterexample is `not_exists_bound_scoreRemainder`, in
  `Clusters.Section10_RemainderFalse`.

* The third bullet of the proof of Step 2' bounds a softmax weight of a
  non-positive exponent by `e^{-s_{j₀}}`, `j₀` being an index of largest
  exponent.  That is `softmax_le_exp_neg`, and it is proved.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`,
`e:derivativenorm123`, `e:boundrj`.
-/

import Transformer.Clusters.Section10_ProjHull

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### `a_j(t)` and `r_j(t)` -/

/-- **`a_j(t) = ⟨π_ℱ(Az_i(t)), π_ℱ(A(z_j(t) - z_i(t)))⟩`.**

Source: arXiv:2305.05465v6, the proof of Step 2'. -/
noncomputable def projScore (A proj : ParamMatrix d) (Z : Idx n → EucSpace d)
    (i j : Idx n) : ℝ :=
  inner (𝕜 := ℝ) (proj (A (Z i))) (proj (A (Z j - Z i)))

/-- A token scores `0` against itself. -/
theorem projScore_self (A proj : ParamMatrix d) (Z : Idx n → EucSpace d) (i : Idx n) :
    projScore A proj Z i i = 0 := by
  simp [projScore]

/-- **`r_j(t) = ⟨Ae^{tV}z_i(t), Ae^{tV}(z_j(t)-z_i(t))⟩ - a_j(t)e^{2λ₁t}`.**

Source: arXiv:2305.05465v6, the proof of Step 2'. -/
noncomputable def scoreRemainder (A V proj : ParamMatrix d) (lam : ℝ)
    (Z : ℝ → Idx n → EucSpace d) (t : ℝ) (i j : Idx n) : ℝ :=
  inner (𝕜 := ℝ) (A (expTime V t (Z t i))) (A (expTime V t (Z t j - Z t i)))
    - projScore A proj (Z t) i j * Real.exp (2 * lam * t)

/-- The remainder against the token itself vanishes. -/
theorem scoreRemainder_self (A V proj : ParamMatrix d) (lam : ℝ)
    (Z : ℝ → Idx n → EucSpace d) (t : ℝ) (i : Idx n) :
    scoreRemainder A V proj lam Z t i i = 0 := by
  simp [scoreRemainder, projScore_self]

/-! ### The softmax bound of the third bullet -/

/-- **The third bullet of the proof of Step 2'.**  A softmax weight whose
exponent is non-positive is bounded by `e^{-s_{j₀}}` for any `j₀`:

  `e^{s_j} / Σ_k e^{s_k} ≤ 1 / Σ_k e^{s_k} ≤ 1 / e^{s_{j₀}}`.

Source: arXiv:2305.05465v6, the proof of Step 2', third bullet. -/
theorem softmax_le_exp_neg (s : Idx n → ℝ) (j j₀ : Idx n) (hj : s j ≤ 0) :
    Real.exp (s j) / ∑ k : Idx n, Real.exp (s k) ≤ Real.exp (-s j₀) := by
  have hge : Real.exp (s j₀) ≤ ∑ k : Idx n, Real.exp (s k) :=
    Finset.single_le_sum (fun k _ => (Real.exp_pos (s k)).le) (Finset.mem_univ j₀)
  have hS : 0 < ∑ k : Idx n, Real.exp (s k) := lt_of_lt_of_le (Real.exp_pos (s j₀)) hge
  rw [div_le_iff₀ hS]
  have h1 : Real.exp (s j) ≤ 1 := Real.exp_le_one_iff.mpr hj
  have h2 : Real.exp (-s j₀) * Real.exp (s j₀) = 1 := by
    rw [← Real.exp_add]
    simp
  have h3 : Real.exp (-s j₀) * Real.exp (s j₀) ≤ Real.exp (-s j₀) * ∑ k : Idx n, Real.exp (s k) :=
    mul_le_mul_of_nonneg_left hge (Real.exp_pos _).le
  linarith

/-- The hypothesis of `softmax_le_exp_neg` is satisfiable. -/
example : ∀ j : Idx 1, (fun _ : Idx 1 => (0 : ℝ)) j ≤ 0 := fun _ => le_rfl

end Clusters
end Transformer
