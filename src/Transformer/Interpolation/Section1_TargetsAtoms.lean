/-
# Interpolation to point-mass targets

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, proved in §5: distinct
input laws with a common support hole are sent within `ε` of distinct
point masses at every horizon, piecewise constantly with `O(d · N)`
switches and parameter norm `O(d · N / T + log(1/ε))`. The hole of
`eq: first.assp` and `eq: assumption.hole` is `IsHole`; a constant
Dirac family has the antipode as a hole. The controls are those of
`Interpolation.Section1_AtomicTargetsNorm`.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Wasserstein
import Transformer.Interpolation.Section1_AtomicTargetsNorm

open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d N : ℕ)

/-- `w` misses every input support — the "hole" of `eq: assumption.hole`. -/
def IsHole (μ : Idx N → ProbSphere d) (w : SSphere d) : Prop :=
  ∀ i : Idx N, w ∉ (μ i : Measure (SSphere d)).support

/-- A constant family of Dirac masses at `x` has a hole, namely the antipode of
`x`: every support is `{x}`, and `-x ≠ x`.  This is the witness the
satisfiability examples of `targets_atoms` and `main_result` use for
`eq: assumption.hole`. -/
theorem isHole_antipode_diracProb (x : SSphere d) :
    IsHole d N (fun _ => diracProb d x) (antipode d x) := by
  intro i hmem
  have hmem' : antipode d x ∈ (Measure.dirac x).support := hmem
  exact antipode_ne d x (eq_of_mem_support_dirac hmem')

/-- **Theorem (thm: targets.atoms).**  *Interpolation to point-mass targets.*

For `d ≥ 3` and data `(μ_0^i, δ_{x^i})_{i=1}^N` with a hole
`w_0 ∈ 𝕊^{d-1} \ ⋃_i supp μ_0^i`, and for any `T, ε > 0`, there is
`θ ∈ L^∞((0, T); Θ)` such that the solution `μ^i` of `eq: cauchy.pb` with data
`μ_0^i` and parameters `θ` satisfies `W_2(μ^i(T), δ_{x^i}) ≤ ε`.  Moreover `θ`
can be chosen piecewise constant with `O(d · N)` switches and

  `‖θ‖_{L^∞((0,T); Θ)} = O((d · N) / T + log(1/ε))`.

The `L^∞` norm of `θ` is measured coordinate by coordinate, as the sum of the
operator norms of the four matrices and of `‖b‖`; `Params d` carries no norm of
its own.

**What the source says and what is changed here.**  Six repairs, none of
them a weakening of what the paper proves.

*`W_2` is `Interpolation.W2`.*  It was a free function of two measures, of
which nothing was assumed — see `not_forall_monge` for what that costs.

*"The unique solution" is read as in `two_balls`:* a solution from each input
exists, and every solution ends within `ε` of its target.  The statement asked
only for one solution reaching the target, which says less unless uniqueness is
known.

*The switch constant is quantified before the data, the norm constant after
it.*  `O(d · N)` switches means one constant serving every dimension, every `N`
and every datum, so `C` stands outside every other quantifier and `d` and `N`
are bound here rather than taken from the section.  The norm constant is not
uniform: the proof in §5 says of its first stage that "the implicit constant
depends only on the supports of the initial measures", and its matching stage
depends on the targets.  Quantified before the data, as it was, the estimate is
false, even for one solution per input — `not_uniform_targets_atoms_norm`: close
Dirac inputs sent to antipodal targets in unit time need unbounded parameters.  So
`Cnorm` follows the data and precedes `T` and `ε`; the factor `d · N` of `1/T` is
then kept as the source writes it.

*The standing assumption of §1 is written in.*  Before the theorem the source
fixes `μ_0^i ≢ μ_0^j` and `μ_1^i ≢ μ_1^j` for `i ≠ j`, "for simplicity"; the
targets `δ_{x^i}` are pairwise distinct exactly when the `x^i` are.  The
statement had neither.  Without distinct inputs it fails: the Cauchy problem has
a unique solution, so equal inputs `μ_0^1 = μ_0^2` end at the same measure and
cannot both come within `ε` of different targets.  The footnote which says the
assumption can be removed (`appendix: technical`) concerns the targets and the
hole, not the inputs.

*The norm bound is asked for `ε ≤ 1/2`.*  The `O(·)` in `log(1/ε)` is an
asymptotic as `ε ↓ 0`.  As an inequality for every `ε > 0` it cannot hold: at
`ε > e^{d N / T}` the right-hand side `Cnorm · ((d N)/T + log(1/ε))` is negative
for `Cnorm > 0`, and a norm is not.  With a constant depending on the data, every
threshold `t < 1` gives the same statement: the controls for accuracy `1/2` serve
every `ε ∈ [1/2, t]` at the price of the factor `log 2 / log(1/t)`.  The former
threshold `1` asked more than the source's proof gives: as `T → ∞` and `ε ↑ 1`
together the right-hand side tends to `0`, while the matching stage of the
source is a perceptron, `𝐕 ≡ 0`, whose field moves a point at speed at most
`‖𝐖‖ (‖𝐔‖ + ‖b‖) ≤ ‖θ‖² / 4`, so that moving an atom a distance `D` in time `T`
needs `‖θ‖ ≥ 2 √(D / T)`.  The other conclusions hold for every `ε > 0`.

*`N ≥ 1`.*  The index set of the source is `⟦1, N⟧`.  `PiecewiseConstant` counts
pieces, and `[0, T]` has at least one, so at `N = 0` the bound `K ≤ C · d · N = 0`
was false whatever `C`.

Proved by `exists_atomic_target_norm_controls`, with at most `7 · d · N` pieces.
Its clustering stage is the ridge neuron of Step 2 in the proof of
`lem: induction.barycenter`, one per law, rather than the attention clustering of
`prop: targets.atoms`.

Source: arXiv:2411.04551v3, §1, `thm: targets.atoms`, and its proof in §5. -/
theorem targets_atoms :
    ∃ C : ℝ, ∀ (d N : ℕ) (μ₀ : Idx N → ProbSphere d) (xtarget : Idx N → SSphere d),
      3 ≤ d → 1 ≤ N → Function.Injective μ₀ → Function.Injective xtarget →
      (∃ w₀ : SSphere d, IsHole d N μ₀ w₀) →
      ∃ Cnorm : ℝ, ∀ T ε : ℝ, 0 < T → 0 < ε →
        ∃ (θ : TimeParams d) (K : ℕ),
          (K : ℝ) ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
          (ε ≤ 1 / 2 → ∀ s ∈ Set.Icc (0 : ℝ) T,
            ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
              ≤ Cnorm * ((d * N : ℝ) / T + Real.log (1 / ε))) ∧
          (∀ i : Idx N, ∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
          ∀ (i : Idx N) (μ : ℝ → ProbSphere d), μ 0 = μ₀ i → cauchyPB d θ μ →
            W2 d (μ T : Measure (SSphere d)) (Measure.dirac (xtarget i)) ≤ ε := by
  refine ⟨7, fun d N μ₀ xtarget hd hN hinj htinj hhole => ?_⟩
  have hunion : (⋃ k, (μ₀ k : Measure (SSphere d)).support) ≠ Set.univ := by
    obtain ⟨w₀, hw₀⟩ := hhole
    intro he
    obtain ⟨k, hk⟩ := Set.mem_iUnion.mp (he ▸ Set.mem_univ w₀)
    exact hw₀ k hk
  obtain ⟨Cnorm, hcon⟩ := exists_atomic_target_norm_controls hd hN μ₀ hinj hunion xtarget htinj
  refine ⟨Cnorm, fun T ε hT hε => ?_⟩
  obtain ⟨θ, K, hK, hθ, _, hnorm, hμ, hacc⟩ := hcon T ε hT hε
  refine ⟨θ, K, ?_, hθ, fun hε2 s _ => hnorm hε2 s, hμ, hacc⟩
  have hK' : (K : ℝ) ≤ 7 * d * N := by exact_mod_cast hK
  exact hK'.trans_eq (by ring)

/-- The conditions inside `targets_atoms` are satisfiable, so its conclusion is
asked of a nonempty class of data: `d = 3`, `T = ε = 1`, and a one-element
family of Dirac masses on `𝕊^2`, whose hole is the antipode. -/
example :
    3 ≤ 3 ∧ 1 ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      Function.Injective (fun _ : Idx 1 => diracProb 3 (basePoint 2)) ∧
      Function.Injective (fun _ : Idx 1 => basePoint 2 : Idx 1 → SSphere 3) ∧
      ∃ w₀ : SSphere 3, IsHole 3 1 (fun _ : Idx 1 => diracProb 3 (basePoint 2)) w₀ :=
  ⟨le_rfl, le_rfl, one_pos, one_pos, Function.injective_of_subsingleton _,
    Function.injective_of_subsingleton _, _, isHole_antipode_diracProb 3 1 (basePoint 2)⟩

end Interpolation
end Transformer
