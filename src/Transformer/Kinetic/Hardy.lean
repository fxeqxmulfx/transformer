/-
# Kinetic theory for Transformers — the Volterra-Hardy profile

Formalization of `prop:Bessel` and `thm:U-shape` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The linearized limiting correlation diagonalizes in Fourier, and each mode
solves a Volterra-Hardy equation whose solution is explicit in modified Bessel
functions.  Summing the modes gives the leading correction `𝒮_t` to the soft
accuracy, and that correction is U-shaped: primacy at one end, recency at the
other, one interior minimum.
-/

import Transformer.Kinetic.Defs
import Transformer.Kinetic.FourierDecay
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Kinetic

/-- The modified Bessel function of the first kind,

  `I_n(x) = Σ_{k≥0} (x/2)^{2k+n} / (k! (k+n)!)`.

Source: arXiv:2605.09213v1, §1, `eq:Bessel-full` (standard). -/
noncomputable def besselI (n : ℕ) (x : ℝ) : ℝ :=
  ∑' k : ℕ, (x / 2) ^ (2 * k + n) / ((Nat.factorial k : ℝ) * (Nat.factorial (k + n) : ℝ))

@[simp]
theorem besselI_succ_zero (n : ℕ) : besselI (n + 1) 0 = 0 := by
  simp [besselI]

/-- `eq:Ylambda`: `Y(σ; σ₀) = log((e^{λσ} - 1)/(e^{λσ₀} - 1))`, the change of
variable that turns `eq:Hardy-hom` into a Goursat problem.  At `λ = 0` it is
the limit `log(σ/σ₀)`, which is the value taken on that branch.

Source: arXiv:2605.09213v1, `eq:Ylambda` and `eq:Bessel-output`. -/
noncomputable def Yfun (lam σ σ₀ : ℝ) : ℝ :=
  if lam = 0 then Real.log (σ / σ₀)
  else Real.log ((Real.exp (lam * σ) - 1) / (Real.exp (lam * σ₀) - 1))

/-- The right-hand side of `eq:Bessel-full`,

  `g_a(t,σ;σ₀) = k_λ(σ,σ₀) √(at/Y(σ;σ₀)) I₁(2√(a t Y(σ;σ₀)))`.

The prefactor `λ e^{-λ(σ-σ₀)}/(1 - e^{-λσ})` of `eq:Bessel-full` is exactly
`graphon lam σ σ₀` for `σ₀ < σ`, and the `λ = 0` branch of `graphon` is the
`1/σ` of `eq:Bessel-output`, so one formula carries both cases of
`prop:Bessel`.

Source: arXiv:2605.09213v1, `eq:Bessel-full`, `eq:Bessel-output`. -/
noncomputable def hardyProfile (lam a t σ σ₀ : ℝ) : ℝ :=
  graphon lam σ σ₀ * Real.sqrt (a * t / Yfun lam σ σ₀) *
    besselI 1 (2 * Real.sqrt (a * t * Yfun lam σ σ₀))

/-- The profile satisfies the initial condition of `eq:Hardy-hom`. -/
@[simp]
theorem hardyProfile_zero (lam a σ σ₀ : ℝ) : hardyProfile lam a 0 σ σ₀ = 0 := by
  simp [hardyProfile]

/-- **Equation (eq:Hardy-hom).**  The Volterra-Hardy equation for the profile
`g_a(·, ·; σ₀)` of the mode with eigenvalue `a`:

  `∂_t g_a(t,σ;σ₀) = a (k_λ(σ,σ₀) + ∫_{σ₀}^{σ} k_λ(σ,σ') g_a(t,σ';σ₀) dσ')`,
  `g_a(0,σ;σ₀) = 0`.

Source: arXiv:2605.09213v1, `eq:Hardy-hom`. -/
def IsVolterraHardySolution (lam a σ₀ : ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  (∀ σ ∈ Set.Ioc σ₀ 1, g 0 σ = 0) ∧
  ∀ σ ∈ Set.Ioc σ₀ 1, ∀ t ∈ Set.Ici (0 : ℝ),
    HasDerivWithinAt (fun s => g s σ)
      (a * (graphon lam σ σ₀ + ∫ σ' in σ₀..σ, graphon lam σ σ' * g t σ'))
      (Set.Ici 0) t

/-- **Proposition (prop:Bessel).**  For `a > 0`, `λ ∈ ℝ` and `0 < σ₀ < σ ≤ 1`,
the Volterra-Hardy equation `eq:Hardy-hom` has `hardyProfile` as its unique
solution: `eq:Bessel-full` for `λ ≠ 0`, `eq:Bessel-output` for `λ = 0`.

Stated as existence together with uniqueness — the source's "the unique
solution of `eq:Hardy-hom` is given by" — so that nothing is assumed: the
formula is produced, not hypothesized.

Not proved here.

Source: arXiv:2605.09213v1, `prop:Bessel`. -/
theorem hardy_bessel (lam a σ₀ : ℝ) (ha : 0 < a) (hσ₀ : 0 < σ₀) (hσ₁ : σ₀ < 1) :
    IsVolterraHardySolution lam a σ₀ (fun t σ => hardyProfile lam a t σ σ₀) ∧
      ∀ g : ℝ → ℝ → ℝ, IsVolterraHardySolution lam a σ₀ g →
        ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ ∈ Set.Ioc σ₀ 1,
          g t σ = hardyProfile lam a t σ σ₀ := by
  sorry

/-- The hypotheses of `hardy_bessel` are satisfiable. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 := by norm_num

/-- `eq:Sb-def`: the leading correction to the soft accuracy,

  `𝒮_t(σ₀) = Σ_{n≥1} e^{-π²n²/(2M²)} g_{a_n}(t, 1; σ₀)`.

The sum over `n ≥ 1` is indexed here by `n : ℕ` through `n + 1`, and
`g_{a_n}(t,1;σ₀)` is written as `hardyProfile`, which `hardy_bessel` identifies
with the solution of `eq:Hardy-hom` that the source's definition names.

Source: arXiv:2605.09213v1, `eq:Sb-def`. -/
noncomputable def softCorrection (β lam M t σ₀ : ℝ) : ℝ :=
  ∑' n : ℕ, Real.exp (-(π ^ 2 / (2 * M ^ 2)) * ((n : ℝ) + 1) ^ 2) *
    hardyProfile lam (aCoeff β (n + 1)) t 1 σ₀

/-- **Theorem (thm:U-shape), Lost in the middle.**  Let `λ > 0` and assume
`eq:affine-smallness`,

  `t sup_{n≥1} a_n ≤ min{3 - √3, 2(1 - e^{-λ})}`.

Then

* (i) *primacy*: `𝒮_t(σ₀) → +∞` as `σ₀ ↓ 0`;
* (ii) *recency*: `𝒮_t'(1⁻) > 0`;
* (iii) *U-shape*: `σ₀ ↦ 𝒮_t(σ₀)` has a unique global minimum in `(0,1)`.

**What the source says and what is changed here.**  Two standing assumptions of
the manuscript are written into the statement, because without them the
conclusions are false rather than merely unproved.  `β > 0`: at `β = 0` the
interaction `w_β ≡ 1` has `a_n = 0` for every `n ≥ 1`, so `𝒮_t ≡ 0` and neither
(i) nor (ii) holds.  `t > 0`: at `t = 0` the profile vanishes identically, by
`hardyProfile_zero`, with the same consequence; the source's own footnote reads
the smallness condition as a condition "for `t > 0`".

The supremum of `eq:affine-smallness` is written pointwise, as `t a_n ≤ …` for
every `n ≥ 1`.  For `t ≥ 0` that is equivalent to `t sup_{n≥1} a_n ≤ …`, and it
does not lean on the family being bounded, which `aCoeff_tendsto_zero` proves.

Not proved here.

Source: arXiv:2605.09213v1, `thm:U-shape`, `eq:affine-smallness`. -/
theorem u_shape (β lam M t : ℝ) (hβ : 0 < β) (hlam : 0 < lam) (hM : 0 < M)
    (ht : 0 < t)
    (hsmall : ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))) :
    Filter.Tendsto (fun σ₀ => softCorrection β lam M t σ₀)
        (nhdsWithin 0 (Set.Ioi 0)) Filter.atTop ∧
      (∃ D : ℝ, 0 < D ∧
        HasDerivWithinAt (fun σ₀ => softCorrection β lam M t σ₀) D (Set.Iic 1) 1) ∧
      (∃! s : ℝ, s ∈ Set.Ioo (0 : ℝ) 1 ∧
        ∀ σ₀ ∈ Set.Ioo (0 : ℝ) 1, softCorrection β lam M t s ≤ softCorrection β lam M t σ₀) := by
  sorry

/-- The hypotheses of `u_shape` are satisfiable, at `β = λ = M = 1`: the
family `a_n` tends to `0` (`aCoeff_tendsto_zero`), so it is bounded above by
some `B`, and `t = m / (|B| + 1)` meets `eq:affine-smallness`, `m` being its
positive threshold. -/
example : ∃ t : ℝ, 0 < t ∧ ∀ n : ℕ, 1 ≤ n →
    t * aCoeff 1 n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-(1 : ℝ)))) := by
  set m := min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-(1 : ℝ))))
  have hm : 0 < m := by
    refine lt_min ?_ ?_
    · nlinarith [Real.sq_sqrt (by norm_num : (3 : ℝ) ≥ 0), Real.sqrt_nonneg 3]
    · have : Real.exp (-(1 : ℝ)) < 1 := by simp
      linarith
  obtain ⟨B, hB⟩ := (aCoeff_tendsto_zero 1).bddAbove_range
  have hB1 : 0 < |B| + 1 := by positivity
  refine ⟨m / (|B| + 1), div_pos hm hB1, fun n _ => ?_⟩
  have ha : aCoeff 1 n ≤ |B| + 1 := (hB ⟨n, rfl⟩).trans ((le_abs_self B).trans (by linarith))
  calc m / (|B| + 1) * aCoeff 1 n ≤ m / (|B| + 1) * (|B| + 1) :=
        mul_le_mul_of_nonneg_left ha (div_pos hm hB1).le
    _ = m := div_mul_cancel₀ m hB1.ne'

end Kinetic
end Transformer
