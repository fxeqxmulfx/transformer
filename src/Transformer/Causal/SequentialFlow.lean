/-
# Causal attention — Sequential gradient flows (§A of 2411.04990v2)

The causal dynamics is not a gradient flow, but it is a *sequential* one:
particle `k` descends an energy `E_k` that only the first `k` particles enter,

  `φ̇_k = -(1 / Z_k(φ_1,…,φ_k)) ∂_{φ_k} E_k(φ_1,…,φ_k)`.

`Lemma lemma:convergence` is the substitute for Łojasiewicz's theorem for such
systems on `𝕊^1`: under isolated critical points and hyperbolicity of every
truncation, almost every trajectory converges to a strongly stable critical
point.  It is stated here and not proved.

The reference measure is Lebesgue on the angles, so it is `volume` and not a
parameter; "strongly stable" and "strongly unstable" are read off the
characteristic polynomial of the Jacobian of the truncated system, over `ℂ`,
which is where the eigenvalues of a real matrix live.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Mathlib.LinearAlgebra.Matrix.Charpoly.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (n : ℕ)

/-- **The velocity of particle `i` in a sequential gradient flow:**

  `-(1 / Z_i(φ)) ∂_{φ_i} E_i(φ)`.

Source: arXiv:2411.04990v2, §A, `lemma:convergence`. -/
noncomputable def seqVelocity
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (i : Idx n) (ψ : Idx n → ℝ) : ℝ :=
  -(Z i ψ)⁻¹ * deriv (fun u => E i (Function.update ψ i u)) (ψ i)

/-- **The sequential gradient flow itself:** every angle follows its own
velocity.

Source: arXiv:2411.04990v2, §A, `lemma:convergence`. -/
def SequentialFlow
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (φ : ℝ → Idx n → ℝ) : Prop :=
  ∀ t : ℝ, ∀ k : Idx n,
    HasDerivAt (fun s => φ s k) (seqVelocity n E Z k (φ t)) t

/-- `F` depends only on the first `k` angles, for every `k`: this is what
makes the flow *sequential* rather than a gradient flow. -/
def DependsOnPrefix (F : Idx n → (Idx n → ℝ) → ℝ) : Prop :=
  ∀ k : Idx n, ∀ ψ ψ' : Idx n → ℝ,
    (∀ i : Idx n, (i : ℕ) ≤ (k : ℕ) → ψ i = ψ' i) → F k ψ = F k ψ'

/-- `F` is `2π`-periodic in every angle: it is a function on `(𝕊^1)^n`, as
the source's `φ_k ∈ 2π ℝ/ℤ` requires, read through the angles' lifts to `ℝ`. -/
def PeriodicAngles (F : Idx n → (Idx n → ℝ) → ℝ) : Prop :=
  ∀ k i : Idx n, ∀ ψ : Idx n → ℝ, F k (Function.update ψ i (ψ i + 2 * π)) = F k ψ

/-- The critical points of `E_k` in its own variable are isolated, the other
angles being held fixed. -/
def IsolatedCritical (E : Idx n → (Idx n → ℝ) → ℝ) : Prop :=
  ∀ k : Idx n, ∀ ψ : Idx n → ℝ, ∀ u : ℝ,
    deriv (fun v => E k (Function.update ψ k v)) u = 0 →
      ∃ ε : ℝ, 0 < ε ∧ ∀ v : ℝ, |v - u| < ε → v ≠ u →
        deriv (fun w => E k (Function.update ψ k w)) v ≠ 0

/-- The index set of the system truncated to the first `k` particles. -/
abbrev Prefix (k : Idx n) : Type := { i : Idx n // (i : ℕ) ≤ (k : ℕ) }

/-- **The Jacobian of the system truncated to the first `k` particles**, at a
configuration `ψ`: the partial derivatives of the velocities `seqVelocity … i`
in the angles `φ_j`, for `i, j ≤ k`.

Source: arXiv:2411.04990v2, §A, `lemma:convergence`, assumption 2. -/
noncomputable def seqJacobian
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (k : Idx n) (ψ : Idx n → ℝ) :
    Matrix (Prefix n k) (Prefix n k) ℝ :=
  fun i j => fderiv ℝ (seqVelocity n E Z i.1) ψ (Pi.single j.1 1)

/-- A configuration is a **critical point of the `k`-th truncation** when the
first `k` velocities vanish there. -/
def IsCriticalPrefix
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (k : Idx n) (ψ : Idx n → ℝ) : Prop :=
  ∀ i : Prefix n k, seqVelocity n E Z i.1 ψ = 0

/-- **Strongly stable:** every eigenvalue of the truncated Jacobian has
strictly negative real part.  The eigenvalues are the complex roots of the
characteristic polynomial of the matrix pushed forward to `ℂ`. -/
def StronglyStablePrefix
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (k : Idx n) (ψ : Idx n → ℝ) : Prop :=
  ∀ μ : ℂ,
    (((seqJacobian n E Z k ψ).map (fun r : ℝ => (r : ℂ))).charpoly).eval μ = 0 →
      μ.re < 0

/-- **Strongly unstable:** some eigenvalue of the truncated Jacobian has
strictly positive real part. -/
def StronglyUnstablePrefix
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (k : Idx n) (ψ : Idx n → ℝ) : Prop :=
  ∃ μ : ℂ,
    (((seqJacobian n E Z k ψ).map (fun r : ℝ => (r : ℂ))).charpoly).eval μ = 0 ∧
      0 < μ.re

/-- **Lemma (lemma:convergence).** *Convergence of a sequential gradient flow
on `𝕊^1`.*

For `C^1` energies `E_k` and normalizations `Z_k` with `0 < c < Z_k < C`, such
that each `E_k` has isolated critical points in its own variable and every
critical point of every truncation is either strongly stable or strongly
unstable, almost every trajectory of

  `φ̇_k = -(1 / Z_k) ∂_{φ_k} E_k`

converges to a strongly stable critical point.

"Almost every" is with respect to Lebesgue measure on the angles (the product measure); smoothness
is `ContDiff ℝ 1`, the `C^1` of the statement; the two-sided bound on `Z_k` is
carried as a hypothesis on every configuration.

The angles are lifted to `ℝ`, and `E_k`, `Z_k` are required to be
`2π`-periodic in each of them (`PeriodicAngles`): the source's particles live
on `𝕊^1`.  An earlier version dropped the periodicity and was false as
written — `E_k(ψ) = ψ_k`, `Z_k ≡ 1` has no critical point, and every angle
drifts to `-∞`.  Convergence of the lift in `ℝ` is the same as convergence on
`𝕊^1`, since a continuous lift of a convergent path converges.

Not proved here: the paper's proof runs through a stable-manifold argument that
is not available in this development.

Source: arXiv:2411.04990v2, §A, `lemma:convergence`. -/
theorem sequentialFlow_converges
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (c C : ℝ) (hc : 0 < c)
    (hsmooth : ∀ k : Idx n, ContDiff ℝ 1 (E k) ∧ ContDiff ℝ 1 (Z k))
    (hZ : ∀ k : Idx n, ∀ ψ : Idx n → ℝ, c < Z k ψ ∧ Z k ψ < C)
    (hEper : PeriodicAngles n E) (hZper : PeriodicAngles n Z)
    (hEpre : DependsOnPrefix n E) (hZpre : DependsOnPrefix n Z)
    (hiso : IsolatedCritical n E)
    (hhyp : ∀ k : Idx n, ∀ ψ : Idx n → ℝ, IsCriticalPrefix n E Z k ψ →
      StronglyStablePrefix n E Z k ψ ∨ StronglyUnstablePrefix n E Z k ψ) :
    ∀ᵐ φ₀ : Idx n → ℝ ∂(Measure.pi fun _ : Idx n => (volume : Measure ℝ)),
      ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ → SequentialFlow n E Z φ →
        ∃ φstar : Idx n → ℝ,
          Filter.Tendsto φ Filter.atTop (nhds φstar) ∧
          (∀ k : Idx n, IsCriticalPrefix n E Z k φstar ∧
            StronglyStablePrefix n E Z k φstar) := by
  sorry

/-- The hypotheses of `sequentialFlow_converges` are satisfiable, and not
vacuously so: one particle with `E_1(ψ) = cos ψ_1` and `Z_1 ≡ 1`, `c = 1/2`,
`C = 2`.  The velocity is `sin φ_1`; its critical points are the multiples
`mπ`, strongly unstable for even `m` (Jacobian `cos = 1`) and strongly stable
for odd `m` (Jacobian `cos = -1`). -/
example :
    (0 : ℝ) < 1 / 2 ∧
      (∀ k : Idx 1, ContDiff ℝ 1 (fun ψ : Idx 1 → ℝ => Real.cos (ψ k)) ∧
        ContDiff ℝ 1 (fun _ : Idx 1 → ℝ => (1 : ℝ))) ∧
      (∀ _k : Idx 1, ∀ _ψ : Idx 1 → ℝ, (1 / 2 : ℝ) < 1 ∧ (1 : ℝ) < 2) ∧
      PeriodicAngles 1 (fun k ψ => Real.cos (ψ k)) ∧
      PeriodicAngles 1 (fun _ _ => (1 : ℝ)) ∧
      DependsOnPrefix 1 (fun k ψ => Real.cos (ψ k)) ∧
      DependsOnPrefix 1 (fun _ _ => (1 : ℝ)) ∧
      IsolatedCritical 1 (fun k ψ => Real.cos (ψ k)) ∧
      (∀ k : Idx 1, ∀ ψ : Idx 1 → ℝ,
        IsCriticalPrefix 1 (fun k ψ => Real.cos (ψ k)) (fun _ _ => (1 : ℝ)) k ψ →
          StronglyStablePrefix 1 (fun k ψ => Real.cos (ψ k)) (fun _ _ => (1 : ℝ)) k ψ ∨
            StronglyUnstablePrefix 1 (fun k ψ => Real.cos (ψ k))
              (fun _ _ => (1 : ℝ)) k ψ) := by
  refine ⟨by norm_num, fun k => ⟨by fun_prop, by fun_prop⟩, fun _ _ => by norm_num,
    ?_, fun _ _ _ => rfl, fun k ψ ψ' h => by simp only [h k le_rfl],
    fun _ _ _ _ => rfl, ?_, ?_⟩
  · intro k i ψ
    have hi : i = k := Subsingleton.elim _ _
    subst hi
    simp [Function.update_self]
  · intro k ψ u hu
    simp only [Function.update_self, Real.deriv_cos', neg_eq_zero] at hu
    refine ⟨π, Real.pi_pos, fun v hv hvu => ?_⟩
    simp only [Function.update_self, Real.deriv_cos', neg_ne_zero]
    intro hsv
    obtain ⟨m, rfl⟩ := Real.sin_eq_zero_iff.mp hu
    obtain ⟨m', rfl⟩ := Real.sin_eq_zero_iff.mp hsv
    rw [← sub_mul, abs_mul, abs_of_pos Real.pi_pos] at hv
    have h1 : |((m' : ℝ) - m)| < 1 := by
      have := Real.pi_pos; nlinarith [abs_nonneg ((m' : ℝ) - m)]
    have h2 : m' = m := by
      have : |m' - m| < 1 := by exact_mod_cast h1
      rw [abs_lt] at this; omega
    exact hvu (by rw [h2])
  · intro k ψ hcrit
    have hv : ∀ i : Idx 1, seqVelocity 1 (fun k ψ => Real.cos (ψ k)) (fun _ _ => (1 : ℝ)) i
        = fun ψ => Real.sin (ψ i) := by
      intro i; funext ψ
      simp [seqVelocity, Function.update_self]
    have hJ : ∀ i j : Prefix 1 k,
        seqJacobian 1 (fun k ψ => Real.cos (ψ k)) (fun _ _ => (1 : ℝ)) k ψ i j
          = Real.cos (ψ k) := by
      intro i j
      have hi : i.1 = k := Subsingleton.elim _ _
      have hj : j.1 = k := Subsingleton.elim _ _
      simp only [seqJacobian, hv, hi, hj]
      rw [((hasFDerivAt_apply k ψ).sin).fderiv]; simp
    have : Unique (Prefix 1 k) :=
      ⟨⟨⟨k, le_rfl⟩⟩, fun i => Subtype.ext (Subsingleton.elim _ _)⟩
    have hcp : ∀ μ : ℂ, (((seqJacobian 1 (fun k ψ => Real.cos (ψ k))
        (fun _ _ => (1 : ℝ)) k ψ).map (fun r : ℝ => (r : ℂ))).charpoly).eval μ
          = μ - Real.cos (ψ k) := by
      intro μ
      rw [Matrix.charpoly, Matrix.det_unique]
      simp [hJ]
    have hs : Real.sin (ψ k) = 0 := by
      have := hcrit ⟨k, le_rfl⟩; rwa [hv] at this
    have hc2 : Real.cos (ψ k) ^ 2 = 1 := by nlinarith [Real.sin_sq_add_cos_sq (ψ k)]
    rcases sq_eq_one_iff.mp hc2 with h1 | h1
    · right
      refine ⟨1, ?_, by norm_num⟩
      rw [hcp, h1]; simp
    · left
      intro μ hμ
      rw [hcp, h1, sub_eq_zero] at hμ
      rw [hμ]; norm_num

end Causal
end Transformer
