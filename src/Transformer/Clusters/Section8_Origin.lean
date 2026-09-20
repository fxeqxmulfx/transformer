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

* `l:cas1circle` is stated as the source states it, as the existence of a
  radius.  Its proof produces the explicit bound `max{‖x_i(0)‖, √(2n)}`, which
  is not carried: the statement is the existential.

* `l:stationary` is split in two.  The substantive direction — a stationary
  configuration is the zero one — is `eq_zero_of_isStationaryConfig`; the
  converse is proved, and it is what witnesses the hypotheses.

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

/-! ### `l:cas1circle` and `l:stationary` -/

/-- **Lemma (l:cas1circle).**  The trajectories of `e:-Iddyn` are uniformly
bounded in time: there is `R > 0`, depending only on `n` and the initial
configuration, with `‖x_i(t)‖ ≤ R` for every `i ∈ [n]` and every `t ≥ 0`.

Not proved here.  The source's proof gives `R = max{‖x_i(0)‖, √(2n)}`.

Source: arXiv:2305.05465v6, `l:cas1circle`. -/
theorem exists_bound_negIdDynamics (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) :
    ∃ R : ℝ, 0 < R ∧ ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖X t i‖ ≤ R := by
  sorry

/-- A stationary configuration of `e:-Iddyn`: one at which every drift
vanishes.

Source: arXiv:2305.05465v6, `l:stationary`. -/
def IsStationaryConfig (Q K : ParamMatrix d) (X : Idx n → EucSpace d) : Prop :=
  ∀ i : Idx n, ∑ j : Idx n, attentionMatrix Q K X i j • X j = 0

/-- The configuration at the origin is stationary — the converse half of
`l:stationary`, and the witness for its hypotheses. -/
theorem isStationaryConfig_zero (Q K : ParamMatrix d) :
    IsStationaryConfig (n := n) Q K (fun _ => (0 : EucSpace d)) := by
  intro i
  simp

/-- **Lemma (l:stationary).**  The only stationary configuration of `e:-Iddyn`
is `x̄_1 = … = x̄_n = 0`.

Not proved here.

Source: arXiv:2305.05465v6, `l:stationary`. -/
theorem eq_zero_of_isStationaryConfig (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : Idx n → EucSpace d) (hX : IsStationaryConfig Q K X) (i : Idx n) :
    X i = 0 := by
  sorry

/-- The hypotheses of `eq_zero_of_isStationaryConfig` are satisfiable. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    IsStationaryConfig (n := n) (1 : ParamMatrix d) 1 (fun _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, isStationaryConfig_zero 1 1⟩

/-! ### `e:finiteinegral` and `t:cas-Idintro` -/

/-- **Lemma (e:finiteinegral).**  The trajectories of `e:-Iddyn` satisfy
`∫_0^{+∞} ‖ẋ_i(t)‖² dt < +∞` for every `i ∈ [n]`.

Not proved here.

Source: arXiv:2305.05465v6, `e:finiteinegral`. -/
theorem integrableOn_sq_norm_negIdDrift (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) (i : Idx n) :
    MeasureTheory.IntegrableOn (fun t => ‖negIdDrift Q K (X t) i‖ ^ 2) (Set.Ici 0) := by
  sorry

/-- **Theorem (t:cas-Idintro).**  Let `V = -I_d` and `QᵀK = I_d`.  Then for
any initial sequence of tokens and any `i ∈ [n]`, `‖x_i(t)‖ → 0` as
`t → +∞`.

Not proved here.

Source: arXiv:2305.05465v6, `t:cas-Idintro`. -/
theorem negId_tendsto_zero (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) (i : Idx n) :
    Tendsto (fun t => ‖X t i‖) atTop (nhds 0) := by
  sorry

/-- The hypotheses shared by `exists_bound_negIdDynamics`,
`integrableOn_sq_norm_negIdDrift` and `negId_tendsto_zero` are satisfiable:
`Q = K = I_d` and the configuration sitting at the origin. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    NegIdDynamics (n := n) (1 : ParamMatrix d) 1 (fun _ _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, (transformerDynamics_neg_one_iff _ _ _).mp (transformerDynamics_zero 1 1 _)⟩

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
