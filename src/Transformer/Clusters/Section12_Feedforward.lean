module

public import Transformer.Clusters.Section4_Hyperplanes
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
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

* The source proposes an extension of `l:3hyperplanes11` and leaves it
  open. For `σ = ReLU` and `W = I_d`, the top row of Figure `fig: last.pdf`
  describes three hyperplanes. Its universal extension to every good
  triple is false: `Section12_ReluCounterexample` proves an actual
  dimension-one trajectory with four distinct limiting points. The
  dynamics and good-triple hypotheses are unchanged. No false universal
  assertion is retained as an unproved theorem.

* The other two rows of the figure ("all particles appear to collapse to `0`",
  for `σ = tanh` with `W = I_d` and for `σ = ReLU` with `W` random) are
  observations about the particular draws shown, not claims about every good
  triple, and are not carried.

Source: arXiv:2305.05465v6, `sec:conclusion`, `eq: mlp.transfo`, Figure
`fig: last.pdf`; `l:3hyperplanes11`.
-/

@[expose] public section

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

example : (fun r : ℝ => r) 0 = 0 := rfl

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
`0`. This is the solution in closed form that witnesses the hypotheses
below. -/
theorem mlpRescaledDynamics_const (W : ParamMatrix d) {σ : ℝ → ℝ} (hσ : σ 0 = 0)
    (Q K V : ParamMatrix d) (z : EucSpace d) :
    MlpRescaledDynamics (n := n) W σ Q K V (fun _ _ => z) := by
  intro t i
  simpa [actPointwise_zero hσ] using hasDerivAt_const t z

/-! ### What ReLU guarantees -/

/-- With identity output weights, each coordinate is nondecreasing.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`,
the nonnegative ReLU velocity. The caption of Figure `fig: last.pdf`
observes motion into the positive orthant in its simulation; nonnegative
velocity alone does not require a negative coordinate to reach zero. -/
theorem mlpRescaledDynamics_relu_coords_monotone (Q K V : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : MlpRescaledDynamics 1 relu Q K V Z)
    (i : Idx n) (k : Idx d) : Monotone (fun t => Z t i k) := by
  let f := fun t => (V (∑ j : Idx n,
    attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • (Z t j - Z t i))) k
  have hD (t : ℝ) : HasDerivAt (fun s => Z s i k) (relu (f t)) t := by
    have h := (EuclideanSpace.proj k).hasFDerivAt.comp_hasDerivAt t (hZ t i)
    let v := V (∑ j : Idx n,
      attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • (Z t j - Z t i))
    change HasDerivAt (fun s => Z s i k) (actPointwise relu v k) t at h
    simpa only [actPointwise_apply] using h
  exact monotone_of_hasDerivAt_nonneg hD (fun t => le_max_right _ _)

/-- The ReLU monotonicity hypotheses hold for an actual stationary solution. -/
example (z : EucSpace d) :
    MlpRescaledDynamics (n := n) 1 relu (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  mlpRescaledDynamics_const 1 relu_zero 1 1 1 z

/-- The proposed extension's hypotheses are satisfiable: the good
triple `(I₁, I₁, I₁)` together with a configuration of equal tokens, which
`eq: mlp.transfo` keeps fixed because `ReLU(0) = 0`. -/
example (z : EucSpace 1) :
    IsGoodTriple (1 : ParamMatrix 1) 1 1 ∧
      MlpRescaledDynamics (n := n) 1 relu (1 : ParamMatrix 1) 1 1 (fun _ _ => z) :=
  ⟨isGoodTriple_one, mlpRescaledDynamics_const 1 relu_zero 1 1 1 z⟩

end Clusters
end Transformer
