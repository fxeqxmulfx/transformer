/-
# Appendix D — the Grönwall stability estimate `eq: stability.4ortho`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The constant `c(β) = e^{10 max(1, β)}` of Appendix D, the stability estimate

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} ‖x_i(0) - y_i(0)‖`

for two solutions of `eq: SA`, and its specialization `e:shortdist` to the
orthogonal approximation.  The estimate is Grönwall's lemma applied to the
vector field `Transformer.Perspective.saField`, whose Lipschitz constant
`10 n max(1, β)` is `Perspective.lipschitzOnWith_saField`.

The rest of the first half of Appendix D — `eq: almost.ortho.vec`,
`e:ineqfirstpart`, `eq: d.large` — is in
`Perspective.AppendixD_PhaseTransition`.
-/

import Transformer.Perspective.SALipschitz
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Calculus.Deriv.Prod

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)


/-- The Lipschitz constant `c(β) = e^{10 max(1, β)}` of the flow, as it appears
throughout Appendix D.

Source: arXiv:2312.10794v5, Appendix D, `eq: lip.3`. -/
noncomputable def cBeta (β : ℝ) : ℝ := Real.exp (10 * max 1 β)

/-- `c(β) ≥ 1`: the bound it multiplies never shrinks. -/
theorem one_le_cBeta (β : ℝ) : 1 ≤ cBeta β := by
  have : (0 : ℝ) ≤ 10 * max 1 β := by
    have : (1 : ℝ) ≤ max 1 β := le_max_left _ _
    linarith
  simp [cBeta, Real.one_le_exp this]

/-- **Equation (eq: stability.4ortho).** *Grönwall stability of the flow.*

Any two solutions of `SA` with the same `β` separate at most at the rate
`c(β)^{n t}`: if every pair of initial particles is within `M`, then

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} · M`   for all `t ≥ 0` and all `i`.

The survey writes this with `max_j` on both sides; taking an arbitrary upper
bound `M` of the initial distances says the same thing and avoids carrying a
nonemptiness proof of `[n]` inside the statement.

Proved here, by Grönwall's inequality for two trajectories of one vector field:
`Perspective.lipschitzOnWith_saField` gives the drift of `SA` Lipschitz
constant `10 n max{1, β}` on tuples of unit vectors, which is exactly
`log c(β)^n`, so the Grönwall factor `e^{K t}` *is* `c(β)^{n t}`.

**What the source says and what is changed here.**  The sign condition
`0 ≤ β` is added, as it is in `lipschitzOnWith_saField`.  The survey's `β` is
an inverse temperature and is positive throughout; the constant
`c(β) = e^{10 max{1, β}}` does not see `β < 0` at all, while the drift's actual
Lipschitz constant grows like `8 |β|`, so the statement is false for `β`
negative and large.

Source: arXiv:2312.10794v5, Appendix D, `eq: stability.4ortho`. -/
theorem stability_orthogonal
    (β : ℝ) (hβ : 0 ≤ β) (X Y : ℝ → SphereTuple d n) (M : ℝ)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n, ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ M) :
    ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖ ≤ cBeta β ^ ((n : ℝ) * t) * M := by
  intro t ht i
  have hn : 0 < n := i.pos
  have hM0 : (0 : ℝ) ≤ M := le_trans (norm_nonneg _) (hM i)
  have hKnn : (0 : ℝ) ≤ 10 * (n : ℝ) * max 1 β := by
    have : (1 : ℝ) ≤ max 1 β := le_max_left _ _
    positivity
  have hK : ((Real.toNNReal (10 * (n : ℝ) * max 1 β) : NNReal) : ℝ)
      = 10 * (n : ℝ) * max 1 β := Real.coe_toNNReal _ hKnn
  have hderf : ∀ s : ℝ,
      HasDerivAt (fun r : ℝ => fun j : Idx n => ((X r j : EucSpace d)))
        (saField d n β (fun j : Idx n => ((X s j : EucSpace d)))) s :=
    fun s => hasDerivAt_pi.mpr fun j => (SA_iff d n β X).mp hX s j
  have hderg : ∀ s : ℝ,
      HasDerivAt (fun r : ℝ => fun j : Idx n => ((Y r j : EucSpace d)))
        (saField d n β (fun j : Idx n => ((Y s j : EucSpace d)))) s :=
    fun s => hasDerivAt_pi.mpr fun j => (SA_iff d n β Y).mp hY s j
  have hinit : dist (fun j : Idx n => ((X 0 j : EucSpace d)))
      (fun j : Idx n => ((Y 0 j : EucSpace d))) ≤ M := by
    rw [dist_pi_le_iff hM0]
    intro j
    rw [dist_eq_norm]
    exact hM j
  have key := dist_le_of_trajectories_ODE_of_mem
    (v := fun _ : ℝ => saField d n β) (s := fun _ : ℝ => unitTuples d n)
    (K := Real.toNNReal (10 * (n : ℝ) * max 1 β))
    (f := fun r : ℝ => fun j : Idx n => ((X r j : EucSpace d)))
    (g := fun r : ℝ => fun j : Idx n => ((Y r j : EucSpace d)))
    (δ := M) (a := 0) (b := t)
    (hv := fun _ _ => lipschitzOnWith_saField d n β hβ hn)
    (hf := fun s _ => (hderf s).continuousAt.continuousWithinAt)
    (hf' := fun s _ => (hderf s).hasDerivWithinAt)
    (hfs := fun s _ j => mem_sphere_zero_iff_norm.mp (X s j).2)
    (hg := fun s _ => (hderg s).continuousAt.continuousWithinAt)
    (hg' := fun s _ => (hderg s).hasDerivWithinAt)
    (hgs := fun s _ j => mem_sphere_zero_iff_norm.mp (Y s j).2)
    (ha := hinit) t (Set.right_mem_Icc.mpr ht)
  rw [hK] at key
  have hcoord : ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
      ≤ M * Real.exp (10 * (n : ℝ) * max 1 β * (t - 0)) := by
    calc ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
        = dist ((fun j : Idx n => ((X t j : EucSpace d))) i)
            ((fun j : Idx n => ((Y t j : EucSpace d))) i) := (dist_eq_norm _ _).symm
      _ ≤ dist (fun j : Idx n => ((X t j : EucSpace d)))
            (fun j : Idx n => ((Y t j : EucSpace d))) :=
          dist_le_pi_dist (fun j : Idx n => ((X t j : EucSpace d)))
            (fun j : Idx n => ((Y t j : EucSpace d))) i
      _ ≤ _ := key
  refine hcoord.trans (le_of_eq ?_)
  rw [cBeta, ← Real.exp_mul]
  ring_nf

/-- The hypotheses of `stability_orthogonal` are satisfiable: one consensus
solution compared with itself, at `β = 0` and initial distance `M = 0`. -/
example :
    (0 : ℝ) ≤ 0 ∧ SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖ ≤ (0 : ℝ) :=
  ⟨le_rfl, SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp⟩

/-- **Equation (e:shortdist).**

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} √(log d / d)`,

where `y` is the solution started from the orthogonal approximation supplied by
`eq: almost.ortho.vec`.  It is `eq: stability.4ortho` at `M = √(log d / d)`.

It is `stability_orthogonal` at `M = √(log d / d)`, and carries that theorem's
added hypothesis `0 ≤ β`.

Source: arXiv:2312.10794v5, Appendix D, `e:shortdist`. -/
theorem shortdist_bound
    (β : ℝ) (hβ : 0 ≤ β) (X Y : ℝ → SphereTuple d n)
    (hX : SA d n β X) (hY : SA d n β Y)
    (hM : ∀ j : Idx n,
      ‖(X 0 j : EucSpace d) - (Y 0 j : EucSpace d)‖ ≤ Real.sqrt (Real.log d / d)) :
    ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖(X t i : EucSpace d) - (Y t i : EucSpace d)‖
        ≤ cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d) :=
  stability_orthogonal d n β hβ X Y _ hX hY hM

/-- The hypotheses of `shortdist_bound` are satisfiable: at `d = 1` the bound
`√(log d / d)` is `0`, and a consensus solution is at distance `0` from
itself. -/
example :
    (0 : ℝ) ≤ 0 ∧ SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ‖((basePoint 0 : SSphere 1) : EucSpace 1) -
        ((basePoint 0 : SSphere 1) : EucSpace 1)‖
        ≤ Real.sqrt (Real.log 1 / 1) :=
  ⟨le_rfl, SA_const_consensus 1 1 one_pos 0 (basePoint 0), by simp⟩

end Perspective
end Transformer
