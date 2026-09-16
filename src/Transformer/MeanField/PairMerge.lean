/-
# Mean-Field Dynamics — what the pairing dynamics does (§5 of 2512.01868v4)

The limiting dynamics `hardmaxPair` of `thm: agazzi_merge`,

  `ẏ_ī = Proj_{y_ī}(y_j̄)`,  `ẏ_j̄ = Proj_{y_j̄}(y_ī)`,  `ẏ_k = 0` otherwise,

is read off here:

* every cluster outside the pair stands still — `hardmaxPair_stationary`,
* the cosine `ρ(s) = ⟨y_ī(s), y_j̄(s)⟩` solves `ρ̇ = 2(1 - ρ²)` —
  `hardmaxPair_inner_hasDerivAt`,
* so the pair is together at some time only if it was together to begin with —
  `hardmaxPair_eq_of_eq`.

The survey says the pair "merges in finite rescaled time"; for this vector
field the merge is asymptotic, `ρ(s) = tanh(2 s + artanh ρ(0)) < 1`, and the
last theorem is the exact statement that no finite time will do.  The proof
needs no solution formula: `(1 - ρ(s)) e^{4 s}` has derivative
`2 e^{4 s} (1 - ρ(s))² ≥ 0`, so it cannot come down to `0`.
-/

import Transformer.MeanField.Merging
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- The constant configuration solves `hardmaxPair`: each of the two particles
attends to the other, which sits on top of it, and `Proj_x x = 0`.  It is the
witness that the hypotheses of this file are satisfiable. -/
theorem hardmaxPair_const (K : ℕ) (ibar jbar : Idx K) (x₀ : SSphere d) :
    hardmaxPair d K ibar jbar (fun _ _ => x₀) := by
  intro s k
  have hx : ‖(x₀ : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x₀.2
  have hzero : proj d ((x₀ : EucSpace d)) ((x₀ : EucSpace d)) = 0 := by
    rw [proj, real_inner_self_eq_norm_mul_norm, hx, one_mul, one_smul, sub_self]
  simp only [hzero, ite_self]
  exact hasDerivAt_const s _

/-- **Every cluster outside the merging pair stands still.**

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge` ("all clusters remain
stationary except for the closest pair"). -/
theorem hardmaxPair_stationary
    (K : ℕ) (ibar jbar : Idx K) (Y : ℝ → SphereTuple d K)
    (hY : hardmaxPair d K ibar jbar Y)
    (k : Idx K) (hki : k ≠ ibar) (hkj : k ≠ jbar) (s : ℝ) :
    Y s k = Y 0 k := by
  have hderiv : ∀ r : ℝ, HasDerivAt (fun r => (Y r k : EucSpace d)) 0 r := by
    intro r
    have h := hY r k
    rwa [ite_eq_right hki, ite_eq_right hkj] at h
  exact Subtype.ext (is_const_of_deriv_eq_zero
    (fun r => (hderiv r).differentiableAt) (fun r => (hderiv r).deriv) s 0)

/-- The hypotheses of `hardmaxPair_stationary` are satisfiable: the constant
configuration of three clusters on `𝕊^0`, the third of which is outside the
pair. -/
example (s : ℝ) :
    (fun (_ : ℝ) (_ : Idx 3) => (basePoint 0 : SSphere 1)) s 2
      = (fun (_ : ℝ) (_ : Idx 3) => (basePoint 0 : SSphere 1)) 0 2 :=
  hardmaxPair_stationary 1 3 0 1 _ (hardmaxPair_const 1 3 0 1 (basePoint 0)) 2
    (by decide) (by decide) s

/-- **The cosine of the pair solves `ρ̇ = 2(1 - ρ²)`.**

Both particles are unit vectors, so `⟨y_ī, Proj_{y_j̄} y_ī⟩ = 1 - ρ²` and
likewise for the other term.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`. -/
theorem hardmaxPair_inner_hasDerivAt
    (K : ℕ) (ibar jbar : Idx K) (hij : ibar ≠ jbar)
    (Y : ℝ → SphereTuple d K) (hY : hardmaxPair d K ibar jbar Y) (s : ℝ) :
    HasDerivAt
      (fun r => inner (𝕜 := ℝ) ((Y r ibar : EucSpace d)) ((Y r jbar : EucSpace d)))
      (2 * (1 - (inner (𝕜 := ℝ) ((Y s ibar : EucSpace d))
                    ((Y s jbar : EucSpace d))) ^ 2)) s := by
  have hi := hY s ibar
  rw [ite_eq_left rfl] at hi
  have hj := hY s jbar
  rw [ite_eq_right hij.symm, ite_eq_left rfl] at hj
  have hii : inner (𝕜 := ℝ) ((Y s ibar : EucSpace d)) ((Y s ibar : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (Y s ibar).2]
    ring
  have hjj : inner (𝕜 := ℝ) ((Y s jbar : EucSpace d)) ((Y s jbar : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (Y s jbar).2]
    ring
  have hsymm : inner (𝕜 := ℝ) ((Y s jbar : EucSpace d)) ((Y s ibar : EucSpace d))
      = inner (𝕜 := ℝ) ((Y s ibar : EucSpace d)) ((Y s jbar : EucSpace d)) :=
    real_inner_comm _ _
  convert hi.inner ℝ hj using 1
  simp only [proj, inner_sub_right, inner_sub_left, real_inner_smul_right,
    real_inner_smul_left, hii, hjj, hsymm]
  ring

/-- The hypotheses of `hardmaxPair_inner_hasDerivAt` are satisfiable: at the
constant configuration `ρ ≡ 1` and both sides are `0`. -/
example (s : ℝ) :
    HasDerivAt
      (fun r => inner (𝕜 := ℝ)
        (((fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) r 0 : EucSpace 1))
        (((fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) r 1 : EucSpace 1)))
      (2 * (1 - (inner (𝕜 := ℝ)
        (((fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) s 0 : EucSpace 1))
        (((fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) s 1 : EucSpace 1))) ^ 2)) s :=
  hardmaxPair_inner_hasDerivAt 1 2 0 1 (by decide) _
    (hardmaxPair_const 1 2 0 1 (basePoint 0)) s

/-- **The pair merges in finite time only if it starts merged.**

If `y_ī(S) = y_j̄(S)` for some `S ≥ 0` then `y_ī(0) = y_j̄(0)`: the survey's
"merge in finite rescaled time" is asymptotic for this vector field.  The
witness is `F(s) = (1 - ρ(s)) e^{4 s}`, whose derivative is
`2 e^{4 s} (1 - ρ(s))² ≥ 0`, so `F(0) ≤ F(S) = 0` while `F(0) = 1 - ρ(0) ≥ 0`
by Cauchy–Schwarz.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`, corrected. -/
theorem hardmaxPair_eq_of_eq
    (K : ℕ) (ibar jbar : Idx K) (hij : ibar ≠ jbar)
    (Y : ℝ → SphereTuple d K) (hY : hardmaxPair d K ibar jbar Y)
    (S : ℝ) (hS : 0 ≤ S) (hmerge : Y S ibar = Y S jbar) :
    Y 0 ibar = Y 0 jbar := by
  set ρ : ℝ → ℝ :=
    fun r => inner (𝕜 := ℝ) ((Y r ibar : EucSpace d)) ((Y r jbar : EucSpace d)) with hρdef
  have hρ : ∀ r : ℝ, HasDerivAt ρ (2 * (1 - ρ r ^ 2)) r := fun r =>
    hardmaxPair_inner_hasDerivAt d K ibar jbar hij Y hY r
  have hexp : ∀ r : ℝ, HasDerivAt (fun r : ℝ => Real.exp (4 * r)) (Real.exp (4 * r) * 4) r := by
    intro r
    simpa using ((hasDerivAt_id r).const_mul (4 : ℝ)).exp
  have hF : ∀ r : ℝ, HasDerivAt (fun r => (1 - ρ r) * Real.exp (4 * r))
      (-(2 * (1 - ρ r ^ 2)) * Real.exp (4 * r) + (1 - ρ r) * (Real.exp (4 * r) * 4)) r :=
    fun r => ((hρ r).const_sub 1).mul (hexp r)
  have hmono : Monotone (fun r => (1 - ρ r) * Real.exp (4 * r)) := by
    refine monotone_of_deriv_nonneg (fun r => (hF r).differentiableAt) fun r => ?_
    rw [(hF r).deriv]
    nlinarith [Real.exp_pos (4 * r), sq_nonneg (1 - ρ r)]
  have hρS : ρ S = 1 := by
    simp only [hρdef, hmerge]
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (Y S jbar).2]
    ring
  have hle := hmono hS
  simp only [hρS, sub_self, zero_mul, mul_zero, Real.exp_zero, mul_one] at hle
  have hcs : ρ 0 ≤ 1 := by
    have h := real_inner_le_norm ((Y 0 ibar : EucSpace d)) ((Y 0 jbar : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (Y 0 ibar).2,
      mem_sphere_zero_iff_norm.mp (Y 0 jbar).2, one_mul] at h
  have hρ0 : ρ 0 = 1 := le_antisymm hcs (by linarith)
  have hnorm : ‖(Y 0 ibar : EucSpace d) - (Y 0 jbar : EucSpace d)‖ ^ 2 = 0 := by
    rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp (Y 0 ibar).2,
      mem_sphere_zero_iff_norm.mp (Y 0 jbar).2]
    have : inner (𝕜 := ℝ) ((Y 0 ibar : EucSpace d)) ((Y 0 jbar : EucSpace d)) = 1 := hρ0
    rw [this]
    ring
  refine Subtype.ext (sub_eq_zero.mp (norm_eq_zero.mp ?_))
  nlinarith [norm_nonneg ((Y 0 ibar : EucSpace d) - (Y 0 jbar : EucSpace d))]

/-- The hypotheses of `hardmaxPair_eq_of_eq` are satisfiable: the constant
configuration is together at every time, including `S = 1`. -/
example :
    (fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) 0 0
      = (fun (_ : ℝ) (_ : Idx 2) => (basePoint 0 : SSphere 1)) 0 1 :=
  hardmaxPair_eq_of_eq 1 2 0 1 (by decide) _ (hardmaxPair_const 1 2 0 1 (basePoint 0))
    1 zero_le_one rfl

end MeanField
end Transformer
