module

public import Transformer.Kinetic.Defs
public import Transformer.Kinetic.FourierDecay
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-!
# Kinetic theory for Transformers — the Volterra-Hardy profile

Formalization of `prop:Bessel` and `thm:U-shape` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The linearized limiting correlation diagonalizes in Fourier, and each mode
solves a Volterra-Hardy equation whose solution is explicit in modified Bessel
functions. Summing the modes gives the leading correction `𝒮_t` to the soft
accuracy, and that correction is U-shaped: primacy at one end, recency at the
other, one interior minimum.
-/

@[expose] public section

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
variable that turns `eq:Hardy-hom` into a Goursat problem. At `λ = 0` it is
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

/-- `eq:Sb-def`: the leading correction to the soft accuracy,

  `𝒮_t(σ₀) = Σ_{n≥1} e^{-π²n²/(2M²)} g_{a_n}(t, 1; σ₀)`.

The sum over `n ≥ 1` is indexed here by `n : ℕ` through `n + 1`, and
`g_{a_n}(t,1;σ₀)` is written as `hardyProfile`, which `hardy_bessel` identifies
with the solution of `eq:Hardy-hom` that the source's definition names.

Source: arXiv:2605.09213v1, `eq:Sb-def`. -/
noncomputable def softCorrection (β lam M t σ₀ : ℝ) : ℝ :=
  ∑' n : ℕ, Real.exp (-(π ^ 2 / (2 * M ^ 2)) * ((n : ℝ) + 1) ^ 2) *
    hardyProfile lam (aCoeff β (n + 1)) t 1 σ₀

/-- The smallness condition `eq:affine-smallness` is met by some `t > 0`, at every `β` and every
`λ > 0`: the family `a_n` tends to `0` (`aCoeff_tendsto_zero`), so it is bounded above by some
`B`, and `t = m / (|B| + 1)` works, `m` being the positive threshold. This is what makes the
hypotheses of `u_shape` satisfiable. -/
theorem exists_affine_smallness (β lam : ℝ) (hlam : 0 < lam) :
    ∃ t : ℝ, 0 < t ∧ ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam))) := by
  set m := min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))
  have hm : 0 < m := by
    refine lt_min ?_ ?_
    · nlinarith [Real.sq_sqrt (by norm_num : (3 : ℝ) ≥ 0), Real.sqrt_nonneg 3]
    · have : Real.exp (-lam) < 1 := Real.exp_lt_one_iff.2 (by linarith)
      linarith
  obtain ⟨B, hB⟩ := (aCoeff_tendsto_zero β).bddAbove_range
  have hB1 : 0 < |B| + 1 := by positivity
  refine ⟨m / (|B| + 1), div_pos hm hB1, fun n _ => ?_⟩
  have ha : aCoeff β n ≤ |B| + 1 := (hB ⟨n, rfl⟩).trans ((le_abs_self B).trans (by linarith))
  calc m / (|B| + 1) * aCoeff β n ≤ m / (|B| + 1) * (|B| + 1) :=
        mul_le_mul_of_nonneg_left ha (div_pos hm hB1).le
    _ = m := div_mul_cancel₀ m hB1.ne'

/-- The smallness bound is witnessed at the original positive bias.
Source: arXiv:2605.09213v1, §1, `eq:affine-smallness`. -/
example := exists_affine_smallness 1 1 one_pos

/-- The actual constant interaction has zero Fourier coefficient at
every positive frequency. Source: arXiv:2605.09213v1, §1, `eq:wbeta` and the coefficients `ŵ_β`. -/
theorem wHat_zero_succ (n : ℕ) : wHat 0 (n + 1) = 0 := by
  have hn : ((n + 1 : ℕ) : ℝ) ≠ 0 := by positivity
  have hs : Real.sin (((n + 1 : ℕ) : ℝ) * (2 * π)) = 0 := by
    rw [show ((n + 1 : ℕ) : ℝ) * (2 * π) = ((2 * (n + 1) : ℕ) : ℝ) * π by
      push_cast; ring]
    exact Real.sin_nat_mul_pi _
  simp only [wHat, wBeta, zero_mul, Real.exp_zero, mul_one]
  rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) hn, integral_cos, hs]
  simp

/-- Every actual eigenvalue vanishes at zero interaction strength.
Source: arXiv:2605.09213v1, §1, `a_n=n² ŵ_β(n)`. -/
theorem aCoeff_zero (n : ℕ) : aCoeff 0 n = 0 := by
  cases n with
  | zero => simp [aCoeff]
  | succ n => rw [aCoeff, wHat_zero_succ, mul_zero]

/-- The actual Bessel profile vanishes at a zero mode parameter.
Source: arXiv:2605.09213v1, §1, `eq:Bessel-full`, `eq:Bessel-output`. -/
theorem hardyProfile_zero_mode (lam t σ σ₀ : ℝ) : hardyProfile lam 0 t σ σ₀ = 0 := by
  simp [hardyProfile]

/-- The original Fourier correction vanishes at zero interaction strength.
This explains the explicit positive-`β` standing assumption in `u_shape`.
Source: arXiv:2605.09213v1, §1, `eq:Sb-def` and `thm:U-shape`. -/
theorem softCorrection_zero_beta (lam M t σ₀ : ℝ) : softCorrection 0 lam M t σ₀ = 0 := by
  simp [softCorrection, aCoeff_zero, hardyProfile_zero_mode]

/-- The original correction vanishes at zero time, so the source's
positive-time footnote is essential for `thm:U-shape`.
Source: arXiv:2605.09213v1, §1, `eq:Sb-def`, `eq:affine-smallness`. -/
theorem softCorrection_zero_time (β lam M σ₀ : ℝ) : softCorrection β lam M 0 σ₀ = 0 := by
  simp [softCorrection, hardyProfile_zero]

/-- At zero time every actual interior position minimizes the original
correction. Thus the unique-minimum conclusion of `thm:U-shape` needs
the source's positive-time assumption.
Source: arXiv:2605.09213v1, §1, `thm:U-shape` (iii), `eq:affine-smallness` footnote. -/
theorem every_position_minimizes_at_zero_time (β lam M s : ℝ) :
    ∀ σ₀ : ℝ, softCorrection β lam M 0 s ≤ softCorrection β lam M 0 σ₀ := by
  intro σ₀
  rw [softCorrection_zero_time, softCorrection_zero_time]

end Kinetic
end Transformer
