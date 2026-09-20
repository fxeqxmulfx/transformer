/-
# The emergence of clusters in self-attention dynamics — adding a
  feed-forward layer

§12 of arXiv:2305.05465v6, `sec:conclusion`: the dynamics `eq: mlp.transfo`,
self-attention composed with a two-layer feed-forward network, and the open
problem it carries.

**What the source says and what is carried here.**

* `eq: mlp.transfo` is `e:Rres` with the drift passed through a componentwise
  activation `σ` and then through a weight matrix `W`:

    `ż_i = Wσ( V Σ_j P_ij(t) (z_j - z_i) )`.

  `mlpRescaledDynamics_id_iff` proves that at `W = I_d` and `σ = id` it is
  `e:Rres` itself, the `V` inside the sum of `e:Rres` and the `V` in front of
  the sum here being the same by linearity.

* "A bias vector `b ∈ ℝ^d` (whether inside or outside the activation function)
  can also be included to allow for translations."  The display carries no
  bias and no statement of the source depends on one, so none is carried.

* The open problem is the one the source names: "we […] illustrate a possible
  generalization of Theorem `l:3hyperplanes11` to this setup […] The
  clustering property appears to persist […] We leave this problem open to
  further investigation."  It is carried at `σ = ReLU` and `W = I_d`, the top
  row of Figure `fig: last.pdf`, where the caption is definite: "every
  particle eventually follows one of three hyperplanes determined by the
  spectrum of `V` and the projection onto `(ℝ_{>0})^d`" — that is, the
  conclusion of `l:3hyperplanes11` verbatim.

* The other two rows of the figure ("all particles appear to collapse to `0`",
  for `σ = tanh` with `W = I_d` and for `σ = ReLU` with `W` random) are
  observations about the particular draws shown, not claims about every good
  triple, and are not carried.

Source: arXiv:2305.05465v6, `sec:conclusion`, `eq: mlp.transfo`, Figure
`fig: last.pdf`; `l:3hyperplanes11`.
-/

import Transformer.Clusters.Section4_Hyperplanes

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### Componentwise activation -/

/-- **A componentwise nonlinearity.**  `σ` applied to each coordinate of a
vector, the `σ(·)` of `eq: mlp.transfo`.

Source: arXiv:2305.05465v6, `eq: mlp.transfo`. -/
noncomputable def actPointwise (σ : ℝ → ℝ) (x : EucSpace d) : EucSpace d :=
  (EuclideanSpace.equiv (Idx d) ℝ).symm (fun i => σ ((EuclideanSpace.equiv (Idx d) ℝ x) i))

/-- Componentwise is componentwise. -/
theorem actPointwise_apply (σ : ℝ → ℝ) (x : EucSpace d) (i : Idx d) :
    actPointwise σ x i = σ (x i) := rfl

/-- An activation vanishing at `0` fixes the origin. -/
theorem actPointwise_zero {σ : ℝ → ℝ} (hσ : σ 0 = 0) : actPointwise (d := d) σ 0 = 0 := by
  ext i
  rw [actPointwise_apply]
  simpa using hσ

/-- The identity activation does nothing. -/
theorem actPointwise_id (x : EucSpace d) : actPointwise (fun r => r) x = x := by
  ext i
  rw [actPointwise_apply]

/-- **ReLU**, one of the two activations of `eq: mlp.transfo`.

Source: arXiv:2305.05465v6, `sec:conclusion`. -/
noncomputable def relu (r : ℝ) : ℝ := max r 0

/-- ReLU vanishes at `0`. -/
theorem relu_zero : relu 0 = 0 := max_self 0

/-! ### `eq: mlp.transfo` -/

/-- **Equation (eq: mlp.transfo).**  Self-attention with a two-layer
feed-forward network appended: a componentwise activation `σ` and a weight
matrix `W`,

  `ż_i(t) = Wσ( V Σ_j ( e^{⟨Qe^{tV}z_i, Ke^{tV}z_j⟩} / Σ_k e^{⟨Qe^{tV}z_i, Ke^{tV}z_k⟩} )
              (z_j(t) - z_i(t)) )`.

Source: arXiv:2305.05465v6, `eq: mlp.transfo`. -/
def MlpRescaledDynamics (W : ParamMatrix d) (σ : ℝ → ℝ) (Q K V : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    HasDerivAt (fun s => Z s i)
      (W (actPointwise σ (V (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • (Z t j - Z t i))))) t

/-- **`eq: mlp.transfo` is `e:Rres` with no network.**  At `W = I_d` and
`σ = id` the feed-forward layer is the identity and the dynamics is the
rescaled dynamics of §3. -/
theorem mlpRescaledDynamics_id_iff (Q K V : ParamMatrix d) (Z : ℝ → Idx n → EucSpace d) :
    MlpRescaledDynamics 1 (fun r => r) Q K V Z ↔ RescaledDynamics Q K V Z := by
  have key : ∀ (t : ℝ) (i : Idx n),
      (1 : ParamMatrix d) (actPointwise (fun r => r) (V (∑ j : Idx n,
          attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • (Z t j - Z t i))))
        = ∑ j : Idx n,
            attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i) := by
    intro t i
    rw [one_apply_eq_self, actPointwise_id, map_sum]
    exact Finset.sum_congr rfl fun j _ => map_smul V _ _
  constructor <;> intro H t i
  · rw [← key t i]
    exact H t i
  · rw [key t i]
    exact H t i

/-- **A configuration of equal tokens is stationary**, for any activation
fixing `0`: the drift of `e:Rres` vanishes there and `σ` and `W` keep it at
`0`.  This is the solution in closed form that witnesses the hypotheses
below. -/
theorem mlpRescaledDynamics_const (W : ParamMatrix d) {σ : ℝ → ℝ} (hσ : σ 0 = 0)
    (Q K V : ParamMatrix d) (z : EucSpace d) :
    MlpRescaledDynamics (n := n) W σ Q K V (fun _ _ => z) := by
  intro t i
  simpa [actPointwise_zero hσ] using hasDerivAt_const t z

/-! ### The open problem -/

/-- **Open problem (`sec:conclusion`), `l:3hyperplanes11` with a feed-forward
layer.**  For a good triple and the dynamics `eq: mlp.transfo` with
`σ = ReLU` and `W = I_d`, are there still at most three parallel hyperplanes
such that the distance of every rescaled token to one of them tends to `0`?

Open: "The clustering property appears to persist, the pattern depending on
the weight matrix `W` and on the activation function `σ`.  We leave this
problem open to further investigation."  The conclusion is
`l:3hyperplanes11`'s own; Figure `fig: last.pdf` names the hyperplanes as
"determined by the spectrum of `V` and the projection onto `(ℝ_{>0})^d`",
which is a description of the limit, not a further assertion, and is not
carried.

Source: arXiv:2305.05465v6, `sec:conclusion`, `eq: mlp.transfo`, Figure
`fig: last.pdf`. -/
theorem mlpRelu_dist_tendsto_zero (Q K V : ParamMatrix d) (hQKV : IsGoodTriple Q K V)
    (Z : ℝ → Idx n → EucSpace d) (hZ : MlpRescaledDynamics 1 relu Q K V Z) :
    ∃ (U : Submodule ℝ (EucSpace d)) (S : Finset (EucSpace d)),
      Module.finrank ℝ U + 1 = d ∧ S.card ≤ 3 ∧
        ∀ i : Idx n, ∃ p ∈ S,
          Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0) := by
  sorry

/-- The hypotheses of `mlpRelu_dist_tendsto_zero` are satisfiable: the good
triple `(I₁, I₁, I₁)` together with a configuration of equal tokens, which
`eq: mlp.transfo` keeps fixed because `ReLU(0) = 0`. -/
example (z : EucSpace 1) :
    IsGoodTriple (1 : ParamMatrix 1) 1 1 ∧
      MlpRescaledDynamics (n := n) 1 relu (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨isGoodTriple_one, mlpRescaledDynamics_const 1 relu_zero 1 1 1 z⟩

end Clusters
end Transformer
