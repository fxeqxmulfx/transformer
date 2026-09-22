/-
# The emergence of clusters in self-attention dynamics — a cluster at the
  origin

§8 of arXiv:2305.05465v6, `s:c<0`: the case `V = -I_d`, `QᵀK = I_d`, where the
whole configuration collapses to a single cluster at the origin.

**What the source says and what is carried here.**

* `QᵀK = I_d` is `IsIdentityQK`: the form `(u,v) ↦ ⟨Qu, Kv⟩` is the standard
  inner product.  It is what `e:-Iddyn` uses `Q` and `K` through, and `Q = K =
  I_d` witnesses it.

* `e:-Iddyn` is `eq:trans_dyn` at `V = -I_d`, and
  `transformerDynamics_neg_one_iff` proves that literally.

* `l:cas1circle` is in `Section8_Bounded`.

* `l:stationary` is in `Section8_Stationary`, `e:finiteinegral` in
  `Section8_Energy`.

* `t:cas-Idintro` is proved in `Section8_Convergence`, as
  `negId_tendsto_zero`.

* The remark following `t:cas-Idintro` is proved here, taking the theorem's
  conclusion as an explicit hypothesis: once every token tends to `0` the
  attention matrix tends to the uniform one by
  `tendsto_attentionMatrix_of_tendsto_common`, which needs no dynamics at all.

Source: arXiv:2305.05465v6, `s:c<0`, `e:-Iddyn`, `l:cas1circle`,
`l:stationary`, `e:finiteinegral`, `t:cas-Idintro`.
-/

import Transformer.Clusters.Section7_HigherDim
import Mathlib.MeasureTheory.Integral.IntegrableOn
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

open scoped BigOperators
open Real Filter

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### `e:-Iddyn` -/

/-- **The hypothesis `QᵀK = I_d`.**  The form `(u,v) ↦ ⟨Qu, Kv⟩` is the
standard inner product.

Source: arXiv:2305.05465v6, `t:cas-Idintro`. -/
def IsIdentityQK (Q K : ParamMatrix d) : Prop :=
  ∀ u v : EucSpace d, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) u v

/-- `Q = K = I_d` witnesses `QᵀK = I_d`. -/
theorem isIdentityQK_one (d : ℕ) : IsIdentityQK (1 : ParamMatrix d) 1 := by
  intro u v
  simp

/-- The right-hand side of `e:-Iddyn`: `-Σ_j P_ij x_j`. -/
noncomputable def negIdDrift (Q K : ParamMatrix d) (X : Idx n → EucSpace d)
    (i : Idx n) : EucSpace d :=
  -∑ j : Idx n, attentionMatrix Q K X i j • X j

/-- **Equation (e:-Iddyn).**  The transformer dynamics at `V = -I_d`:

  `ẋ_i(t) = -Σ_j ( e^{⟨x_i,x_j⟩} / Σ_k e^{⟨x_i,x_k⟩} ) x_j(t)`.

Source: arXiv:2305.05465v6, `e:-Iddyn`. -/
def NegIdDynamics (Q K : ParamMatrix d) (X : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => X s i) (negIdDrift Q K (X t) i) t

/-- **`e:-Iddyn` is `eq:trans_dyn` at `V = -I_d`.** -/
theorem transformerDynamics_neg_one_iff (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) :
    TransformerDynamics Q K (-1) X ↔ NegIdDynamics Q K X := by
  have key : ∀ (Y : Idx n → EucSpace d) (i : Idx n),
      (∑ j : Idx n, attentionMatrix Q K Y i j • (-1 : ParamMatrix d) (Y j))
        = negIdDrift Q K Y i := by
    intro Y i
    simp [negIdDrift, Finset.sum_neg_distrib]
  constructor
  · intro H t i
    rw [← key (X t) i]
    exact H t i
  · intro H t i
    rw [key (X t) i]
    exact H t i

/-- **The configuration at the origin does not move.**  Every drift vanishes
there, whatever `V` is; this is the solution in closed form that witnesses the
hypotheses below. -/
theorem transformerDynamics_zero (Q K V : ParamMatrix d) :
    TransformerDynamics (n := n) Q K V (fun _ _ => (0 : EucSpace d)) := by
  intro t i
  simpa using hasDerivAt_const t (0 : EucSpace d)

/-- **The remark following (t:cas-Idintro).**  In the setting of
`t:cas-Idintro` the self-attention matrix `P(t)` of `eq:P` converges, as
`t → +∞`, to the `n × n` matrix all of whose entries are `1/n`.

The conclusion of `t:cas-Idintro` is taken as an explicit hypothesis: the
convergence of `P(t)` follows from it by continuity alone.

Source: arXiv:2305.05465v6, the remark after `t:cas-Idintro`. -/
theorem attentionMatrix_tendsto_uniform (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d)
    (h : ∀ i : Idx n, Tendsto (fun t => ‖X t i‖) atTop (nhds 0)) (i j : Idx n) :
    Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds ((n : ℝ)⁻¹)) :=
  tendsto_attentionMatrix_of_tendsto_common Q K X 0
    (fun l => tendsto_zero_iff_norm_tendsto_zero.mpr (h l)) i j

/-- The hypothesis of `attentionMatrix_tendsto_uniform` is satisfiable. -/
example : ∀ i : Idx n,
    Tendsto (fun t : ℝ => ‖(fun _ _ => (0 : EucSpace d)) t i‖) atTop (nhds 0) := by
  intro i
  simp

end Clusters
end Transformer
