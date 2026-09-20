/-
# The emergence of clusters in self-attention dynamics — the codimension
  conjecture

`c:codim` of arXiv:2305.05465v6, §4: the conjectural generalization of
`l:3hyperplanes11` from one leading direction to the whole unstable subspace.

**What the source says and what is carried here.**

* "Let `k ≥ 1` be the number of eigenvalues of `V` with positive real part."
  That count is the dimension of the unstable subspace, and it is carried by
  the `V`-invariant splitting it produces: `ℝ^d = F ⊕ G` with `dim F = k`, the
  flow `e^{tV}` growing at a definite exponential rate on `F` and at most
  polynomially on `G`.  Over ℂ, `F` is the sum of the generalized eigenspaces
  of the eigenvalues with `Re λ > 0` and `G` that of the rest; the polynomial
  factor on `G` is what the Jordan blocks of the eigenvalues on the imaginary
  axis contribute.  This needs no complexification.

* The source states the conjecture with **no hypothesis on `(Q,K)`** — unlike
  `l:3hyperplanes11`, which assumes `⟨Qφ₁, Kφ₁⟩ > 0`.  It is carried as
  written, with none.

* `k ≥ 1` is the source's own restriction, and it is genuine: for `k = 0` the
  conclusion would ask for a codimension-`0` subspace, that is for `ℝ^d`
  itself, and say nothing.

Source: arXiv:2305.05465v6, `c:codim`.
-/

import Transformer.Clusters.Section4_Hyperplanes

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **The unstable subspace and its dimension `k`.**  A `V`-invariant
splitting `ℝ^d = F ⊕ G` with `dim F = k`, on which `e^{tV}` grows at least
like `e^{εt}`, against at most polynomial growth on `G`.  This is the eigenvalue
count of `c:codim`: `k` is the number of eigenvalues of `V`, with multiplicity,
whose real part is positive.

Source: arXiv:2305.05465v6, `c:codim`. -/
def IsUnstableSplitting (V : ParamMatrix d) (k : ℕ) : Prop :=
  ∃ F G : Submodule ℝ (EucSpace d),
    (∀ w ∈ F, V w ∈ F) ∧ (∀ w ∈ G, V w ∈ G) ∧ IsCompl F G ∧
      Module.finrank ℝ F = k ∧
      (∃ c ε : ℝ, 0 < c ∧ 0 < ε ∧
        ∀ w ∈ F, ∀ t : ℝ, 0 ≤ t → c * Real.exp (ε * t) * ‖w‖ ≤ ‖expTime V t w‖) ∧
      (∃ (C : ℝ) (m : ℕ), 0 < C ∧
        ∀ w ∈ G, ∀ t : ℝ, 0 ≤ t → ‖expTime V t w‖ ≤ C * (1 + t) ^ m * ‖w‖)

/-- **The identity on `ℝ` has a one-dimensional unstable subspace.**  Its only
eigenvalue is `1 > 0`, so `k = 1`, `F = ℝ` and `G = 0`; the flow is `e^t`
exactly.  This witnesses `c:codim`'s hypothesis. -/
theorem isUnstableSplitting_one : IsUnstableSplitting (1 : ParamMatrix 1) 1 := by
  refine ⟨⊤, ⊥, fun w _ => Submodule.mem_top, fun w hw => by simpa using hw,
    isCompl_top_bot, by simp,
    ⟨1, 1, one_pos, one_pos, ?_⟩, ⟨1, 0, one_pos, ?_⟩⟩
  · intro w _ t _
    rw [expTime_one_apply, norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos t)]
    simp
  · intro w hw t _
    have hw0 : w = 0 := by simpa using hw
    rw [hw0]
    simp

/-- **Conjecture (c:codim), the codimension conjecture.**  If `V` has `k ≥ 1`
eigenvalues of positive real part, then every rescaled token approaches one of
at most three parallel affine subspaces of codimension `k`.

Open; not proved here.

Source: arXiv:2305.05465v6, `c:codim`. -/
theorem codim_dist_tendsto_zero (Q K V : ParamMatrix d) (k : ℕ) (hk : 1 ≤ k)
    (hV : IsUnstableSplitting V k) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ (W : Submodule ℝ (EucSpace d)) (S : Finset (EucSpace d)),
      Module.finrank ℝ W + k = d ∧ S.card ≤ 3 ∧
        ∀ i : Idx n, ∃ p ∈ S,
          Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  sorry

/-- The hypotheses of `codim_dist_tendsto_zero` are satisfiable at `d = 1`,
`k = 1`, `V = 1`. -/
example (z : EucSpace 1) :
    (1 : ℕ) ≤ 1 ∧ IsUnstableSplitting (1 : ParamMatrix 1) 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨le_rfl, isUnstableSplitting_one, rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer
