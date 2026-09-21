/-
# The emergence of clusters in self-attention dynamics — the rescaled tokens

§3 of arXiv:2305.05465v6: the change of variables `z_i(t) = e^{-tV} x_i(t)`,
and the equation `e:Rres` it turns `eq:trans_dyn` into.  The discrete analogue
of `r:discreterescaling` is in `Section3_Discrete`.

**What the source says and what is carried here.**

* `e^{tV}` is `expTime V t := NormedSpace.exp (t • V)`, the operator
  exponential of Mathlib.  The two facts the source uses it through are
  `expTime V 0 = 1` and, for `V = I_d`, `e^{tI}x = e^t x`
  (`expTime_one_apply`), both proved below.

* The self-attention coefficients of `e:Rres` are literally those of
  `eq:trans_dyn` evaluated at the *unrescaled* tokens — this is the source's
  "the coefficients of the self-attention matrix for the rescaled tokens are
  the same as those for the original tokens", and it is why
  `attentionMatrix` is reused rather than restated.

* The source states the change of variables as a derivation: `z_i := e^{-tV}x_i`
  "solve" `e:Rres`, and conversely each result on `z_i` transfers to `x_i` "by
  virtue of the relation `x_i(t) = e^{tV}z_i(t)`".  Both directions are carried
  as one `iff` under that relation, in `transformerDynamics_iff_rescaled`.

Source: arXiv:2305.05465v6, `e:Rres`.
-/

import Transformer.Clusters.Section1_Dynamics
import Mathlib.Analysis.SpecialFunctions.Exponential

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The flow `e^{tV}` -/

/-- **`e^{tV}`**, the solution operator of `ẏ = Vy` that the rescaling is built
on.

Source: arXiv:2305.05465v6, §3. -/
noncomputable def expTime (V : ParamMatrix d) (t : ℝ) : ParamMatrix d :=
  NormedSpace.exp (t • V)

/-- At `t = 0` the flow is the identity, so `z_i(0) = x_i(0)`: the source's
"the initial condition remains the same". -/
theorem expTime_zero (V : ParamMatrix d) : expTime V 0 = 1 := by
  rw [expTime, zero_smul ℝ V, NormedSpace.exp_zero]

/-- With no drift the flow is the identity at every time. -/
theorem expTime_zero_map (t : ℝ) : expTime (0 : ParamMatrix d) t = 1 := by
  have h : t • (0 : ParamMatrix d) = 0 := by ext x; simp
  rw [expTime, h, NormedSpace.exp_zero]

/-- **The case `V = I_d`**: `e^{tI}x = e^t x`.  This is what turns the exponent
`⟨Qe^{tV}z_i, Ke^{tV}z_j⟩` of `e:Rres` into the `e^{2t}⟨Az_i,Az_j⟩` of
`e:dynIdzi`. -/
theorem expTime_one_apply (t : ℝ) (x : EucSpace d) :
    expTime (1 : ParamMatrix d) t x = Real.exp t • x := by
  rw [expTime, ← Algebra.algebraMap_eq_smul_one, ← NormedSpace.algebraMap_exp_comm,
    ← Real.exp_eq_exp_ℝ, Algebra.algebraMap_eq_smul_one]
  simp

/-! ### `e:Rres` -/

/-- **Equation (e:Rres).**  The rescaled dynamics: the tokens
`z_i(t) = e^{-tV}x_i(t)` obey

  `ż_i(t) = Σ_j P_ij(e^{tV}z(t)) · V(z_j(t) - z_i(t))`,

where the attention coefficients are read off the unrescaled configuration
`e^{tV}z(t)`.

Source: arXiv:2305.05465v6, `e:Rres`. -/
def RescaledDynamics (Q K V : ParamMatrix d) (Z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    HasDerivAt (fun s => Z s i)
      (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i)) t

/-- With `V = 0` no token moves in the rescaled variables either: the constant
curve solves `e:Rres`, which is what witnesses the hypotheses below. -/
theorem rescaledDynamics_const (Q K : ParamMatrix d) (Z : Idx n → EucSpace d) :
    RescaledDynamics Q K 0 (fun _ => Z) := by
  intro t i
  simpa using (hasDerivAt_const t (Z i))

/-- **The change of variables `x_i(t) = e^{tV}z_i(t)`.**  Under that relation,
`x` solves `eq:trans_dyn` exactly when `z` solves `e:Rres`.  The forward
direction is the source's "which solve (e:Rres)"; the backward one is how
every conclusion about `z_i` is read back on `x_i`.

Not proved here.

Source: arXiv:2305.05465v6, §3, `e:Rres`. -/
theorem transformerDynamics_iff_rescaled (Q K V : ParamMatrix d)
    (X Z : ℝ → Idx n → EucSpace d) (hXZ : ∀ (t : ℝ) (i : Idx n), X t i = expTime V t (Z t i)) :
    TransformerDynamics Q K V X ↔ RescaledDynamics Q K V Z := by
  sorry

/-- The hypothesis of `transformerDynamics_iff_rescaled` is satisfiable: at
`V = 0` the flow is the identity, so any curve is its own rescaling. -/
example (Z : ℝ → Idx n → EucSpace d) :
    ∀ (t : ℝ) (i : Idx n), Z t i = expTime (0 : ParamMatrix d) t (Z t i) := by
  intro t i
  rw [expTime_zero_map]
  rfl

end Clusters
end Transformer
