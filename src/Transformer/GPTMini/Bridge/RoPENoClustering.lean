/-
# RoPE attention does not cluster every initial sequence

The claim this file refutes was stated here as `rope_clustering`: for
`d_head ≥ 3`, `n ≥ 2` particles on the sphere and `β ≥ 0`, the
`eq: transformerSd.QKV` dynamics driven by RoPE-rotated `Q, K` and by
`V = I_d` send every particle to one common point.

It is false, and for a reason that has nothing to do with RoPE.  The antipodal
pair `(x, -x)` of `Perspective.antipodalPair` has mean `0`; every linear `V`
sends it to `0`; and at `β = 0` all attention weights are `exp 0 = 1`, so the
velocity is `Proj_{x_i} 0 = 0` whatever `Q, K` and `θ` are.  The constant path
is a solution, the two particles stay at distance `2` forever, and no `x⋆`
attracts both.

This is the exceptional set of `Perspective.boumal_clustering` reappearing in
the RoPE parametrization.  The survey quantifies over *Lebesgue-almost every*
initial sequence, never every one, and `Perspective.antipodalPair_not_exponential`
already makes the same point for the simplified model `Q = K = V = I_d`.

The almost-everywhere statement — the time-varying analogue of
`thm: main.d.geq.3` — is not stated here.  `Perspective.boumal_clustering`
covers `Q = K = V = I_d` only and is itself unproved, its argument does not
carry over to time-varying `Q, K`, and one more unprovable statement is worth
less than the refutation of the one that was wrong.

Source: arXiv:2312.10794v5, §2 (`eq: transformerSd.QKV`), §4 (`p:beta0`) and
§6.1 (`thm: boumal`).
-/

import Transformer.GPTMini.Bridge.RoPEAsTimeVarying
import Transformer.Perspective.Section3_SmallBeta

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- **The antipodal pair is stationary for `eq: transformerSd.QKV` at `β = 0`,
whatever `Q, K, V` are.**

At `β = 0` every attention weight is `exp 0 = 1`, so the velocity is the
projection of `V(t) x + V(t) (-x)`, and `V(t)` is linear: the sum is `0` and
the constant path is a solution.  Nothing about `Q` or `K` is used, which is
why the RoPE rotation does not help.

Source: arXiv:2312.10794v5, §2, `eq: transformerSd.QKV`; the configuration is
`Perspective.antipodalPair` of §4. -/
theorem transformerODE_const_antipodalPair
    (d : ℕ) (Q K V : TimeParam d) (x : SSphere d) :
    Perspective.transformerODE d 2 0 Q K V (fun _ => Perspective.antipodalPair d x) := by
  intro t i
  have h0 : ((Perspective.antipodalPair d x 0 : SSphere d) : EucSpace d)
      = (x : EucSpace d) := rfl
  have h1 : ((Perspective.antipodalPair d x 1 : SSphere d) : EucSpace d)
      = -(x : EucSpace d) := rfl
  simp only [zero_mul, Real.exp_zero, one_smul, Fin.sum_univ_two, h0, h1, map_neg,
    add_neg_cancel, smul_zero, proj, inner_zero_right, zero_smul, sub_self]
  exact hasDerivAt_const t _

/-- **The antipodal pair does not cluster**, whatever `Q, K, V` are.

The constant path of `transformerODE_const_antipodalPair` keeps the two
particles at `x` and `-x`, so a common limit `x⋆` would be both, forcing
`x = -x` and `‖x‖ = 0` against `‖x‖ = 1`.

Source: arXiv:2312.10794v5, §4, the "almost every" of `p:beta0`; the same
configuration refutes the `∀ X₀` reading of `thm: boumal`
(`Perspective.antipodalPair_not_exponential`). -/
theorem not_rope_clustering_antipodalPair
    (d : ℕ) (Q K V : TimeParam d) (x : SSphere d) :
    ¬ ∃ x_star : SSphere d,
        ∀ X : ℝ → SphereTuple d 2, X 0 = Perspective.antipodalPair d x →
          Perspective.transformerODE d 2 0 Q K V X →
          ∀ i : Idx 2,
            Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - (x_star : EucSpace d)))
              Filter.atTop (nhds 0) := by
  rintro ⟨x_star, h⟩
  have hconst := h (fun _ => Perspective.antipodalPair d x) rfl
    (transformerODE_const_antipodalPair d Q K V x)
  have e0 := tendsto_const_nhds_iff.mp (hconst 0)
  have e1 := tendsto_const_nhds_iff.mp (hconst 1)
  have h0 : ((Perspective.antipodalPair d x 0 : SSphere d) : EucSpace d)
      = (x : EucSpace d) := rfl
  have h1 : ((Perspective.antipodalPair d x 1 : SSphere d) : EucSpace d)
      = -(x : EucSpace d) := rfl
  rw [h0, sub_eq_zero] at e0
  rw [h1, sub_eq_zero] at e1
  have hneg : (x : EucSpace d) = -(x : EucSpace d) := e0.trans e1.symm
  have h2 : (2 : ℝ) • (x : EucSpace d) = 0 := by
    rw [two_smul]
    nth_rewrite 2 [hneg]
    exact add_neg_cancel _
  have hx0 : (x : EucSpace d) = 0 := by
    rcases smul_eq_zero.mp h2 with h' | h'
    · norm_num at h'
    · exact h'
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [hx0] at hx
  norm_num at hx

/-- **The `∀ X₀` reading of RoPE clustering is false.**

For every `d_head ≥ 3`, every `n ≥ 2` and every `β ≥ 0`, RoPE-rotated `Q, K`
with `V = I_d` do *not* send every initial sequence to a single point: the
`gpt-mini` head dimension `64`, two particles at `(e₀, -e₀)` and `β = 0` are a
counterexample.  This kills the claim the file used to state as
`rope_clustering`.

The survey's theorem is about Lebesgue-almost every initial sequence
(`Perspective.boumal_clustering`), and the exceptional set is exactly this
one, non-empty at every `β` and every `d`.

Source: arXiv:2312.10794v5, §6.1, `thm: boumal` — read for every `X₀` rather
than almost every. -/
theorem not_forall_rope_clustering :
    ¬ ∀ (cfg : Config) (n : ℕ) (Q K : ParamMatrix cfg.head_dim) (theta beta : ℝ),
        3 ≤ cfg.head_dim → 2 ≤ n → 0 ≤ beta →
        ∀ X₀ : SphereTuple cfg.head_dim n,
        ∃ x_star : SSphere cfg.head_dim,
          ∀ X : ℝ → SphereTuple cfg.head_dim n, X 0 = X₀ →
            Perspective.transformerODE cfg.head_dim n beta
              (rope_timeParam Q theta) (rope_timeParam K theta)
              (fun _ => ContinuousLinearMap.id ℝ (EucSpace cfg.head_dim)) X →
            ∀ i : Idx n,
              Filter.Tendsto
                (fun t : ℝ => ((X t i : EucSpace cfg.head_dim) - (x_star : EucSpace cfg.head_dim)))
                Filter.atTop (nhds 0) := by
  intro h
  have hdim : Config.default.head_dim = 64 := by
    norm_num [Config.head_dim, Config.default]
  let x : SSphere Config.default.head_dim :=
    cast (congrArg SSphere hdim.symm) (basePoint 63)
  refine not_rope_clustering_antipodalPair Config.default.head_dim
    (rope_timeParam (ContinuousLinearMap.id ℝ (EucSpace Config.default.head_dim)) 0)
    (rope_timeParam (ContinuousLinearMap.id ℝ (EucSpace Config.default.head_dim)) 0)
    (fun _ => ContinuousLinearMap.id ℝ (EucSpace Config.default.head_dim)) x ?_
  exact h Config.default 2 (ContinuousLinearMap.id ℝ (EucSpace Config.default.head_dim))
    (ContinuousLinearMap.id ℝ (EucSpace Config.default.head_dim)) 0 0
    (by rw [hdim]; norm_num) le_rfl le_rfl (Perspective.antipodalPair _ x)

end Bridge
end GPTMini
end Transformer
