/-
# Causal attention — Single-token dynamics (§3 of 2411.04990v2)

`Lemma lemma1` of the survey: for almost any initial condition the solution of
the single-token ODE `ẋ = Proj_x (V x)` approaches `L'(V) ∩ 𝕊^{d-1}`
exponentially and `L(V) ∩ 𝕊^{d-1}` at rate `1/t`, where `L'(V)` is spanned by
the generalized eigenvectors of `V` whose eigenvalue has maximal real part and
`L(V) ⊆ L'(V)` by those sitting in the largest Jordan block.

**What the source says and what is changed here.**  The form this development
gave the lemma existentially quantified the two subspaces and the exceptional
set — `∃ L, L', E`, with `L'` and `L` asked only to be nested and `V`-invariant
and `E` only to be proper.  Nothing in that pins them to the spectrum of `V`,
and the result is not the survey's lemma:

* it is **false** as written, and `not_forall_single_token_convergence` refutes
  it — at `d = 0` every submodule of the zero space is `⊤`, so no proper `E`
  exists;
* and for `d ≥ 1` it is **empty**: `single_token_convergence_trivial` proves it
  outright with `L = L' = ⊤` and `E = ⊥`, because the flow keeps `‖x(t)‖ = 1`
  and the distance to `⊤ ∩ 𝕊^{d-1}` is therefore `0` at every time.

What survives is `norm_eq_one_of_singleTokenODE`, the invariance of the sphere,
which the survey uses without comment and which the two statements above rest
on.  Stating `lemma1` itself needs the real Jordan decomposition of `V` — that
is what names `L'(V)` and `L(V)` — and neither Mathlib nor this development
has it; a statement that does not name them is not the lemma, so none is made
here.

Source: arXiv:2411.04990v2, §3, `lemma1`.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

variable {d : ℕ}

/-- **The unit sphere is invariant under the single-token flow.**

Along `ẋ = Proj_x (V x)` the squared norm obeys the linear equation
`(‖x‖² - 1)' = -2⟨x, V x⟩ (‖x‖² - 1)`, since `⟨x, Proj_x y⟩ = ⟨x, y⟩(1 - ‖x‖²)`.
A solution of a linear scalar equation vanishing at one time vanishes at every
time, so `‖x(0)‖ = 1` forces `‖x(t)‖ = 1`.

Source: arXiv:2411.04990v2, §3, used without comment in `lemma1`. -/
theorem norm_eq_one_of_singleTokenODE (V : ParamMatrix d) {x : ℝ → EucSpace d}
    (hx : Causal.singleTokenODE d V x) (h0 : ‖x 0‖ = 1) (t : ℝ) : ‖x t‖ = 1 := by
  have hxc : Continuous x := continuous_iff_continuousAt.mpr fun s => (hx s).continuousAt
  have hVc : Continuous fun s => V (x s) := V.continuous.comp hxc
  set g : ℝ → ℝ := fun s => 2 * inner (𝕜 := ℝ) (x s) (V (x s)) with hgdef
  set u : ℝ → ℝ := fun s => inner (𝕜 := ℝ) (x s) (x s) - 1 with hudef
  have hgc : Continuous g := continuous_const.mul (hxc.inner hVc)
  have hu' : ∀ s : ℝ, HasDerivAt u (-(g s * u s)) s := by
    intro s
    have hP : inner (𝕜 := ℝ) (x s) (proj d (x s) (V (x s)))
        = inner (𝕜 := ℝ) (x s) (V (x s)) * (1 - inner (𝕜 := ℝ) (x s) (x s)) := by
      rw [proj, inner_sub_right, real_inner_smul_right]
      ring
    have hcomm : inner (𝕜 := ℝ) (proj d (x s) (V (x s))) (x s)
        = inner (𝕜 := ℝ) (x s) (proj d (x s) (V (x s))) := real_inner_comm _ _
    have hd := ((hx s).inner ℝ (hx s)).sub_const 1
    rw [hcomm, hP] at hd
    convert hd using 1
    simp only [hgdef, hudef]
    ring
  set G : ℝ → ℝ := fun s => ∫ r in (0 : ℝ)..s, g r with hGdef
  have hG : ∀ s : ℝ, HasDerivAt G (g s) s := fun s =>
    intervalIntegral.integral_hasDerivAt_right (hgc.intervalIntegrable _ _)
      (hgc.stronglyMeasurableAtFilter _ _) hgc.continuousAt
  have hh : ∀ s : ℝ, HasDerivAt (fun r => u r * Real.exp (G r)) 0 s := by
    intro s
    have hmul := (hu' s).mul ((hG s).exp)
    convert hmul using 1
    ring
  have hconst : u t * Real.exp (G t) = u 0 * Real.exp (G 0) :=
    is_const_of_deriv_eq_zero (fun r => (hh r).differentiableAt)
      (fun r => (hh r).deriv) t 0
  have hu0 : u 0 = 0 := by
    simp only [hudef]
    rw [real_inner_self_eq_norm_mul_norm, h0]
    ring
  have hut : u t = 0 := by
    rw [hu0, zero_mul] at hconst
    exact (mul_eq_zero.mp hconst).resolve_right (Real.exp_ne_zero _)
  have hsq : ‖x t‖ * ‖x t‖ = 1 := by
    have : inner (𝕜 := ℝ) (x t) (x t) = (1 : ℝ) := by
      simpa [hudef, sub_eq_zero] using hut
    rwa [real_inner_self_eq_norm_mul_norm] at this
  nlinarith [norm_nonneg (x t)]

/-- The hypotheses of `norm_eq_one_of_singleTokenODE` are satisfiable: a token
standing still at the base point of `𝕊^0`, which solves the single-token ODE
for `V = 0`. -/
example : Causal.singleTokenODE 1 0 (fun _ => (Transformer.basePoint 0 : EucSpace 1)) ∧
    ‖((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 := by
  refine ⟨fun t => ?_, mem_sphere_zero_iff_norm.mp (Transformer.basePoint 0).2⟩
  have : proj 1 ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)
      ((0 : ParamMatrix 1) ((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)) = 0 := by
    simp [proj]
  rw [this]
  exact hasDerivAt_const _ _

/-- **The existential form of `lemma1` is false.**

Asking only for *some* nested `V`-invariant pair `L ≤ L'` and *some* proper
`E`, as this development did, fails at `d = 0`: the zero space has exactly one
submodule, `⊥ = ⊤`, so there is no proper `E` to exclude an exceptional set
with.  The survey's `lemma1` names `L'(V)` and `L(V)` through the spectrum of
`V` and is not this statement.

Source: arXiv:2411.04990v2, §3, `lemma1`. -/
theorem not_forall_single_token_convergence :
    ¬ ∀ (d : ℕ) (V : ParamMatrix d),
      ∃ (Lprime Lmain E : Submodule ℝ (EucSpace d)) (C c : ℝ),
        Lmain ≤ Lprime ∧ E ≠ ⊤ ∧ 0 < C ∧ 0 < c ∧
        (∀ y ∈ Lprime, V y ∈ Lprime) ∧ (∀ y ∈ Lmain, V y ∈ Lmain) ∧
        ∀ x : ℝ → EucSpace d, Causal.singleTokenODE d V x → ‖x 0‖ = 1 → x 0 ∉ E →
          (∀ t : ℝ, 0 ≤ t →
            Metric.infDist (x t)
                ((Lprime : Set (EucSpace d)) ∩ Metric.sphere (0 : EucSpace d) 1)
              ≤ C * Real.exp (-(c * t))) ∧
          (∀ t : ℝ, 0 < t →
            Metric.infDist (x t)
                ((Lmain : Set (EucSpace d)) ∩ Metric.sphere (0 : EucSpace d) 1)
              ≤ C / t) := by
  intro h
  obtain ⟨Lprime, Lmain, E, C, c, -, hE, -⟩ := h 0 0
  refine hE (Submodule.eq_top_iff'.mpr fun y => ?_)
  have hy : y = 0 := by
    ext i
    exact i.elim0
  rw [hy]
  exact E.zero_mem

/-- **And for `d ≥ 1` it is empty.**

`L = L' = ⊤` and `E = ⊥` satisfy it: the flow keeps `‖x(t)‖ = 1`
(`norm_eq_one_of_singleTokenODE`), so `x(t)` lies in `⊤ ∩ 𝕊^{d-1}` and both
distances are `0`, below every positive bound.  The statement therefore says
nothing about `V`; what makes the survey's `lemma1` a theorem is that `L'(V)`
and `L(V)` are the spectral subspaces of `V`, which no hypothesis here records.

Source: arXiv:2411.04990v2, §3, `lemma1`. -/
theorem single_token_convergence_trivial (hd : 1 ≤ d) (V : ParamMatrix d) :
    ∃ (Lprime Lmain E : Submodule ℝ (EucSpace d)) (C c : ℝ),
      Lmain ≤ Lprime ∧ E ≠ ⊤ ∧ 0 < C ∧ 0 < c ∧
      (∀ y ∈ Lprime, V y ∈ Lprime) ∧ (∀ y ∈ Lmain, V y ∈ Lmain) ∧
      ∀ x : ℝ → EucSpace d, Causal.singleTokenODE d V x → ‖x 0‖ = 1 → x 0 ∉ E →
        (∀ t : ℝ, 0 ≤ t →
          Metric.infDist (x t)
              ((Lprime : Set (EucSpace d)) ∩ Metric.sphere (0 : EucSpace d) 1)
            ≤ C * Real.exp (-(c * t))) ∧
        (∀ t : ℝ, 0 < t →
          Metric.infDist (x t)
              ((Lmain : Set (EucSpace d)) ∩ Metric.sphere (0 : EucSpace d) 1)
            ≤ C / t) := by
  classical
  have hne : Nonempty (Fin d) := ⟨⟨0, hd⟩⟩
  have hbot : (⊥ : Submodule ℝ (EucSpace d)) ≠ ⊤ := bot_ne_top
  refine ⟨⊤, ⊤, ⊥, 1, 1, le_rfl, hbot, one_pos, one_pos,
    fun y _ => Submodule.mem_top, fun y _ => Submodule.mem_top, ?_⟩
  intro x hxode hx0 _
  have hmem : ∀ t : ℝ, x t ∈ ((⊤ : Submodule ℝ (EucSpace d)) : Set (EucSpace d))
      ∩ Metric.sphere (0 : EucSpace d) 1 := by
    intro t
    refine ⟨Submodule.mem_top, ?_⟩
    rw [mem_sphere_zero_iff_norm]
    exact norm_eq_one_of_singleTokenODE V hxode hx0 t
  constructor
  · intro t _
    rw [Metric.infDist_zero_of_mem (hmem t)]
    positivity
  · intro t ht
    rw [Metric.infDist_zero_of_mem (hmem t)]
    exact div_nonneg zero_le_one ht.le

/-- The hypothesis of `single_token_convergence_trivial` is satisfiable:
`d = 1`. -/
example : 1 ≤ 1 := le_rfl

end Causal
end Transformer
