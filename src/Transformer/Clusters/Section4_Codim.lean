/-
# The emergence of clusters in self-attention dynamics — the codimension
  conjecture

`c:codim` of arXiv:2305.05465v6, §4 proposes a generalization of
`l:3hyperplanes11` from one leading direction to the whole unstable subspace.
Its stated numerical bound is false; a genuine four-token trajectory refutes it.

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
  `l:3hyperplanes11`, which assumes `⟨Qφ₁, Kφ₁⟩ > 0`. The counterexample
  uses the positive definite choice `Q = K = I₂` anyway.

* `k ≥ 1` is the source's own restriction, and it is genuine: for `k = 0` the
  conclusion would ask for a codimension-`0` subspace, that is for `ℝ^d`
  itself, and say nothing.

* The source concludes that at most three parallel affine subspaces of
  codimension `k` suffice for all tokens. For `V = I₂`, both eigenvalues
  are positive, so `k = 2`, and those subspaces in the plane are points.
  Four tokens started at `(±2,±2)` converge after rescaling to `(±q,±q)`
  for `q ≥ 1`. These are four distinct points, so any candidate family
  satisfying the source's distance conclusion needs at least four members.

* `Section4_SquareScalar` constructs the global scalar solution,
  `Section4_SquareAmplitude` proves its positive rescaled limit,
  `Section4_SquareGeometry` computes the corner scores, and
  `Section4_SquareDynamics` verifies the original and rescaled equations.
  `Section4_SquareCounterexample` proves the cardinality obstruction.
  The counterexample and the negation of the universal claim are carried
  below. The false claim is not retained as an unproved theorem.

Source: arXiv:2305.05465v6, `c:codim`.
-/

import Transformer.Clusters.Section4_SquareCounterexample

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

/-- The identity on `ℝ^d` has exactly `d` exponentially growing directions.
Source: arXiv:2305.05465v6, §4 `c:codim`, its positive-eigenvalue count.
The invariant splitting is the whole space against zero and the flow is `e^t`. -/
theorem isUnstableSplitting_id (d : ℕ) : IsUnstableSplitting (1 : ParamMatrix d) d := by
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

/-- **The identity on `ℝ` has a one-dimensional unstable subspace.** Its only
eigenvalue is `1 > 0`. Source: arXiv:2305.05465v6, §4 `c:codim`. -/
theorem isUnstableSplitting_one : IsUnstableSplitting (1 : ParamMatrix 1) 1 :=
  isUnstableSplitting_id 1

/-- **The planar identity has two unstable directions.** Both eigenvalues equal one.
Source: arXiv:2305.05465v6, §4 `c:codim`, the square counterexample's eigenvalue count. -/
theorem isUnstableSplitting_two : IsUnstableSplitting (1 : ParamMatrix 2) 2 :=
  isUnstableSplitting_id 2

/-- A counterexample to the codimension conjecture's distance conclusion.
Source: arXiv:2305.05465v6, §4, Conjecture `c:codim`, which proposes
at most three parallel affine subspaces of codimension `k`.
With `Q = K = V = I₂`, `k = 2`, and initial corners `(±2,±2)`,
the actual rescaled trajectories converge to four distinct points.
The original universal claim is false; this theorem refutes it without
weakening its conclusion or adding hypotheses to the source's name. -/
theorem codim_dist_tendsto_zero_counterexample :
    ∃ Z : ℝ → Idx 4 → EucSpace 2,
      (1 : ℕ) ≤ 2 ∧ IsUnstableSplitting (1 : ParamMatrix 2) 2 ∧
        RescaledDynamics (1 : ParamMatrix 2) 1 1 Z ∧ Z 0 = squareCorner 2 ∧
          ¬∃ (W : Submodule ℝ (EucSpace 2)) (S : Finset (EucSpace 2)),
            Module.finrank ℝ W + 2 = 2 ∧ S.card ≤ 3 ∧ ∀ i : Idx 4, ∃ p ∈ S,
              Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  obtain ⟨Z, hZ, h0, hbad⟩ := exists_square_counterexample
  exact ⟨Z, by norm_num, isUnstableSplitting_two, hZ, h0, hbad⟩

/-- The exact universal distance claim of the codimension conjecture is false.
Source: arXiv:2305.05465v6, §4, Conjecture `c:codim`.
The source asserts this bound for arbitrary dimension, token count, and
parameters with `k ≥ 1` positive-real-part eigenvalues. The square is a
counterexample even with all three parameter matrices equal to the identity. -/
theorem not_codim_dist_tendsto_zero :
    ¬∀ (d n : ℕ) (Q K V : ParamMatrix d) (k : ℕ),
      1 ≤ k → IsUnstableSplitting V k → ∀ Z : ℝ → Idx n → EucSpace d,
        RescaledDynamics Q K V Z →
          ∃ (W : Submodule ℝ (EucSpace d)) (S : Finset (EucSpace d)),
            Module.finrank ℝ W + k = d ∧ S.card ≤ 3 ∧ ∀ i : Idx n, ∃ p ∈ S,
              Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  intro hall
  obtain ⟨Z, hk, hV, hZ, _, hbad⟩ := codim_dist_tendsto_zero_counterexample
  exact hbad (hall 2 4 1 1 1 2 hk hV Z hZ)

/-- The refutation uses an actual solution satisfying all of the conjecture's hypotheses.
Source: arXiv:2305.05465v6, §4 `c:codim`, at `d = k = 2` with four tokens.
In particular, the initial corners and the positive-eigenvalue count occur together. -/
example : ∃ Z : ℝ → Idx 4 → EucSpace 2,
    (1 : ℕ) ≤ 2 ∧ IsUnstableSplitting (1 : ParamMatrix 2) 2 ∧
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z ∧ Z 0 = squareCorner 2 := by
  obtain ⟨Z, hk, hV, hZ, h0, _⟩ := codim_dist_tendsto_zero_counterexample
  exact ⟨Z, hk, hV, hZ, h0⟩

/-- The codimension conjecture's hypotheses are also satisfiable at `d = 1`,
`k = 1`, `V = 1`. Source: arXiv:2305.05465v6, §4 `c:codim`. -/
example (z : EucSpace 1) :
    (1 : ℕ) ≤ 1 ∧ IsUnstableSplitting (1 : ParamMatrix 1) 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨le_rfl, isUnstableSplitting_one, rescaledDynamics_one_const _ _ z⟩

end Clusters
end Transformer
