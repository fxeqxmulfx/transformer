/-
# The emergence of clusters in self-attention dynamics — how fast a
  coordinate may grow

§9 of arXiv:2305.05465v6, continued: the growth bound `l:nottoofassst` on the
eigencoordinates of the unrescaled tokens, and its replacement `e:notdi` when
`V` is not diagonalizable (`r:notdiagonalizable`).

**What the source says and what is carried here.**

* Both statements are about `x_i(t) = e^{tV}z_i(t)`, the tokens of
  `eq:trans_dyn`; that is how they are stated here, and
  `transformerDynamics_iff_rescaled` is what reads them back on `z_i`.

* `l:nottoofassst` is stated for any `k ∈ [d]`, so its `φ*_k` and `λ_k` may be
  complex and `|λ_k|` is a modulus.  `IsEigenFunctional` is stated over
  `RCLike`, so the complex case is `𝕜 = ℂ`: an `ℝ`-linear `f : ℝ^d → ℂ` with
  `f(Vz) = λ f(z)` is exactly a dual eigenvector of the complexification
  restricted to real vectors, and `|λ_k|` is `‖lam‖`.

* `r:notdiagonalizable` recalls a standard normal form — a Jordan basis whose
  superdiagonal entries are an arbitrary `ε ≠ 0` (Horn–Johnson, Cor. 3.1.21) —
  and derives from it the bound `‖π_F(Vx)‖ ≤ (|λ_k| + δ)‖π_F(x)‖`, which is
  what the modified proof runs on.  That bound is `IsProjContraction`, carried
  as the hypothesis of `e:notdi`; the normal form itself is quoted linear
  algebra, not a claim of the paper, and is not restated.  Only linearity of
  `proj` is used, not that it is an orthogonal projection.

Source: arXiv:2305.05465v6, `l:nottoofassst`, `r:notdiagonalizable`,
`e:notdi`.
-/

import Transformer.Clusters.Section9_Limits

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### `l:nottoofassst` -/

/-- At `V = 0` every functional is an eigenfunctional for `λ = 0`; together
with the stationary configuration of `transformerDynamics_const` this
witnesses the hypotheses below. -/
theorem isEigenFunctional_zero {𝕜 : Type*} [RCLike 𝕜] (f : EucSpace d →L[ℝ] 𝕜) :
    IsEigenFunctional (0 : ParamMatrix d) f 0 := by
  intro z
  simp

/-- **Lemma (l:nottoofassst).**  For every `k ∈ [d]` and `i ∈ [n]` there is a
constant `C > 0` with

  `|φ*_k(e^{tV}z_i(t))| ≤ C e^{|λ_k| t}`  for all `t ≥ 0`.

Not proved here.  The source's argument: `d/dt |φ*_k(x_i)|² ≤ 2|λ_k|
max_j |φ*_k(x_j)|²` by `eq:trans_dyn` and the row sum of `eq:P`, and then
Grönwall applied to `max_j |φ*_k(x_j)|²`.

Source: arXiv:2305.05465v6, `l:nottoofassst`. -/
theorem norm_eigenFunctional_le (Q K V : ParamMatrix d) {𝕜 : Type*} [RCLike 𝕜]
    (f : EucSpace d →L[ℝ] 𝕜) (lam : 𝕜) (hf : IsEigenFunctional V f lam)
    (X : ℝ → Idx n → EucSpace d) (hX : TransformerDynamics Q K V X) (i : Idx n) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ‖f (X t i)‖ ≤ C * Real.exp (‖lam‖ * t) := by
  sorry

/-- The hypotheses of `norm_eigenFunctional_le` are satisfiable: at `V = 0`
every functional is an eigenfunctional for `λ = 0`, and no token moves. -/
example (Q K : ParamMatrix d) (f : EucSpace d →L[ℝ] ℂ) (X : Idx n → EucSpace d) :
    IsEigenFunctional (0 : ParamMatrix d) f 0 ∧
      TransformerDynamics Q K 0 (fun _ => X) :=
  ⟨isEigenFunctional_zero f, transformerDynamics_const Q K X⟩

/-! ### `r:notdiagonalizable` -/

/-- **The key observation of `r:notdiagonalizable`.**  On a Jordan subspace
`F`, with the superdiagonal of the block taken small enough,

  `‖π_F(Vx)‖ ≤ (|λ_k| + δ) ‖π_F(x)‖`.

Source: arXiv:2305.05465v6, `r:notdiagonalizable`. -/
def IsProjContraction (V proj : ParamMatrix d) (μ : ℝ) : Prop :=
  ∀ x : EucSpace d, ‖proj (V x)‖ ≤ μ * ‖proj x‖

/-- At `V = 0` the bound holds for any `proj` and any `μ ≥ 0`. -/
theorem isProjContraction_zero (proj : ParamMatrix d) {μ : ℝ} (hμ : 0 ≤ μ) :
    IsProjContraction (0 : ParamMatrix d) proj μ := by
  intro x
  simpa using mul_nonneg hμ (norm_nonneg (proj x))

/-- **Equation (e:notdi).**  The replacement for `l:nottoofassst` when `V` is
not diagonalizable: on each Jordan subspace,

  `∃ C > 0, ∀ t ≥ 0, ∀ i ∈ [n],  ‖π_F(e^{tV}z_i(t))‖ ≤ C e^{(|λ_k| + δ)t}`.

Not proved here.  The source's argument is that of `l:nottoofassst` with
`d/dt ‖π_F(x_i(t))‖²` in place of `d/dt |φ*_k(x_i(t))|²`.

Source: arXiv:2305.05465v6, `e:notdi`. -/
theorem norm_proj_le (Q K V proj : ParamMatrix d) (μ : ℝ) (hμ : 0 ≤ μ)
    (hπ : IsProjContraction V proj μ) (X : ℝ → Idx n → EucSpace d)
    (hX : TransformerDynamics Q K V X) :
    ∃ C : ℝ, 0 < C ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖proj (X t i)‖ ≤ C * Real.exp (μ * t) := by
  sorry

/-- The hypotheses of `norm_proj_le` are satisfiable at `V = 0`. -/
example (Q K proj : ParamMatrix d) (X : Idx n → EucSpace d) :
    (0 : ℝ) ≤ 0 ∧ IsProjContraction (0 : ParamMatrix d) proj 0 ∧
      TransformerDynamics Q K 0 (fun _ => X) :=
  ⟨le_rfl, isProjContraction_zero proj le_rfl, transformerDynamics_const Q K X⟩

end Clusters
end Transformer
