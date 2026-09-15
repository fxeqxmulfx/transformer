/-
# Normalization — The convergence machinery of the proof (App. D of 2510.22026v2)

Three statements the proof of `thm: convergence` rests on:

* `Lemma lem: loj` — the Łojasiewicz convergence theorem generalized to a
  modulated gradient flow `ẋ = -M(t) ∇E(x)`;
* `Proposition sprop: time_change` — a monotone time change turns `eq: NA`
  with speed factors `s_j` into `eq: NA` with `s_j(t(τ)) / t'(τ)`;
* `Lemma lem: matrix` — the product of a symmetric positive-definite matrix
  and a symmetric unstable one is unstable.

"Unstable" is read as "has a positive real eigenvalue": that is what the
proof of `lem: matrix` produces from Sylvester's law of inertia, and what its
use in Step 4 of Appendix D needs of the energy Hessian.

None of the three is proved here.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.Order.Star.Real
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

/-- **Lemma (lem: loj).** *Łojasiewicz convergence for a modulated gradient
flow.*

Let `M(t)` be symmetric with `C λ(t) I ≻ M(t) ≻ λ(t) I`, `λ(t) > 0` and
`∫_0^∞ λ = ∞`, and let `E` be analytic on an open `U ⊆ ℝ^N`.  Then any
trajectory of

  `ẋ = -M(t) ∇_x E(x)`

that stays in a compact subset of `U` converges to a critical point of `E`.

The two matrix inequalities are written out on vectors, `λ(t) ‖v‖² < ⟨M(t) v, v⟩
< C λ(t) ‖v‖²` for `v ≠ 0`, rather than through a positive-definiteness order
on operators; the divergence of `∫ λ` is the divergence of its primitive.
The compact path is a compact `K ⊆ U` the trajectory never leaves.

Not proved here.

Source: arXiv:2510.22026v2, Appendix D.1, `lem: loj`. -/
theorem lojasiewicz_modulated
    (N : ℕ) (U : Set (EucSpace N)) (hU : IsOpen U)
    (E : EucSpace N → ℝ) (hE : AnalyticOnNhd ℝ E U)
    (C : ℝ) (lam : ℝ → ℝ) (M : ℝ → ParamMatrix N)
    (hlam : ∀ t : ℝ, 0 ≤ t → 0 < lam t)
    (hMsymm : ∀ t : ℝ, ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (M t v) w = inner (𝕜 := ℝ) v (M t w))
    (hMlower : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      lam t * ‖v‖ ^ 2 < inner (𝕜 := ℝ) (M t v) v)
    (hMupper : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      inner (𝕜 := ℝ) (M t v) v < C * lam t * ‖v‖ ^ 2)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (x : ℝ → EucSpace N) (K : Set (EucSpace N)) (hK : IsCompact K) (hKU : K ⊆ U)
    (hxK : ∀ t : ℝ, 0 ≤ t → x t ∈ K)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t) :
    ∃ xstar ∈ U, Filter.Tendsto x Filter.atTop (nhds xstar) ∧
      gradient E xstar = 0 := by
  sorry

/-- The hypotheses of `lojasiewicz_modulated` are satisfiable: a constant
energy on the whole space, `M(t) = I`, `λ(t) = 1/2` and `C = 4` meet both
matrix inequalities strictly, `∫_0^T 1/2 = T/2 → ∞`, and the trajectory
`x ≡ 0` stays in the compact `{0}`. -/
example :
    IsOpen (Set.univ : Set (EucSpace 1)) ∧
      AnalyticOnNhd ℝ (fun _ : EucSpace 1 => (0 : ℝ)) Set.univ ∧
      (∀ t : ℝ, 0 ≤ t → (0 : ℝ) < 1 / 2) ∧
      (∀ v : EucSpace 1, v ≠ 0 →
        (1 / 2 : ℝ) * ‖v‖ ^ 2 < inner (𝕜 := ℝ) v v) ∧
      (∀ v : EucSpace 1, v ≠ 0 →
        inner (𝕜 := ℝ) v v < 4 * (1 / 2 : ℝ) * ‖v‖ ^ 2) ∧
      Filter.Tendsto (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 / 2 : ℝ))
        Filter.atTop Filter.atTop ∧
      IsCompact ({0} : Set (EucSpace 1)) := by
  refine ⟨isOpen_univ, analyticOnNhd_const, fun t _ => by norm_num, fun v hv => ?_,
    fun v hv => ?_, ?_, isCompact_singleton⟩
  · have hpos : (0 : ℝ) < ‖v‖ ^ 2 := by positivity
    rw [real_inner_self_eq_norm_sq]
    linarith
  · have hpos : (0 : ℝ) < ‖v‖ ^ 2 := by positivity
    rw [real_inner_self_eq_norm_sq]
    linarith
  · have h : (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 / 2 : ℝ)) = fun T : ℝ => T / 2 := by
      funext T
      simp
      ring
    rw [h]
    exact Filter.tendsto_id.atTop_div_const two_pos

variable (d n : ℕ)

/-- **Proposition (sprop: time_change).** *A monotone time change rescales the
speed factors.*

For a strictly increasing `t(τ)` with positive derivative, the normalized
attention dynamics with speed regulation factors `s_j(t)` read in the time `τ`
is the normalized attention dynamics with speed regulation factors

  `s̃_j(τ) = s_j(t(τ)) / t'(τ)`,

the time-dependent parameters being reparametrized along with it.  Growth of
`t(τ)` to infinity is not needed for the equivalence itself, only for the use
the paper makes of it, so it is not among the hypotheses.

Not proved here.

Source: arXiv:2510.22026v2, Appendix D, `sprop: time_change`. -/
theorem na_time_change
    (β : ℝ) (Q K V : ℝ → ParamMatrix d) (s : ℝ → Idx n → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (tOf tOf' : ℝ → ℝ)
    (hmono : StrictMono tOf) (hpos : ∀ u : ℝ, 0 < tOf' u)
    (hderiv : ∀ u : ℝ, HasDerivAt tOf (tOf' u) u)
    (hNA : NA d n β Q K V s θ) :
    NA d n β (fun u => Q (tOf u)) (fun u => K (tOf u)) (fun u => V (tOf u))
      (fun u j => s (tOf u) j / tOf' u) (fun u => θ (tOf u)) := by
  sorry

/-- The hypotheses of `na_time_change` are satisfiable: the identity time
change is strictly increasing with derivative `1`, and the frozen dynamics
`s ≡ 0`, `θ ≡ 0` solves `eq: NA` — at `s_j = 0` the prescribed velocity is
`0⁻¹ • _ = 0`, which is the derivative of a constant. -/
example :
    StrictMono (id : ℝ → ℝ) ∧ (0 : ℝ) < 1 ∧
      (∀ u : ℝ, HasDerivAt (id : ℝ → ℝ) 1 u) ∧
      NA 1 1 0 (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ _ => 0) (fun _ _ => 0) := by
  refine ⟨strictMono_id, one_pos, fun u => hasDerivAt_id u, fun t j => ?_⟩
  simpa using hasDerivAt_const t (0 : EucSpace 1)

/-- **Lemma (lem: matrix).** *A positive-definite factor preserves
instability.*

For a symmetric unstable `A` and a symmetric positive-definite `D`, the
product `D A` is unstable.  The proof conjugates by the square root `P` of `D`:
`P⁻¹ (D A) P = P A P` is symmetric and has the inertia of `A`, so it keeps a
positive eigenvalue, and similar matrices share their spectrum.

Instability is spelled out as the existence of a real eigenvalue `μ > 0` with
an eigenvector; the symmetry of `A` is what makes that the right reading, and
the lemma is false without it.

Not proved here.

Source: arXiv:2510.22026v2, Appendix D, Step 4, `lem: matrix`. -/
theorem unstable_mul_of_posDef
    (N : ℕ) (A D : Matrix (Fin N) (Fin N) ℝ)
    (hA : A.IsHermitian) (hD : D.PosDef)
    (hAunstable : ∃ (μ : ℝ) (v : Fin N → ℝ),
      v ≠ 0 ∧ A.mulVec v = μ • v ∧ 0 < μ) :
    ∃ (μ : ℝ) (v : Fin N → ℝ), v ≠ 0 ∧ (D * A).mulVec v = μ • v ∧ 0 < μ := by
  sorry

/-- The hypotheses of `unstable_mul_of_posDef` are satisfiable: the identity
matrix is symmetric, positive-definite, and unstable — its only eigenvalue is
`1 > 0`. -/
example :
    (1 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian ∧
      (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef ∧
      ∃ (μ : ℝ) (v : Fin 1 → ℝ),
        v ≠ 0 ∧ (1 : Matrix (Fin 1) (Fin 1) ℝ).mulVec v = μ • v ∧ 0 < μ := by
  refine ⟨Matrix.isHermitian_one, Matrix.PosDef.one, 1, fun _ => 1, ?_, ?_, one_pos⟩
  · intro h
    simpa using congrFun h 0
  · simp

end Normalization
end Transformer
