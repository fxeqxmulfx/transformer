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

Not proved here: the paper's proof runs through a stable-manifold argument that
is not available in this development.

Source: arXiv:2411.04990v2, §A, `lemma:convergence`. -/
theorem sequentialFlow_converges
    (E Z : Idx n → (Idx n → ℝ) → ℝ) (c C : ℝ) (hc : 0 < c)
    (hsmooth : ∀ k : Idx n, ContDiff ℝ 1 (E k) ∧ ContDiff ℝ 1 (Z k))
    (hZ : ∀ k : Idx n, ∀ ψ : Idx n → ℝ, c < Z k ψ ∧ Z k ψ < C)
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
vacuously so: take `E_k(ψ) = ψ_k` and `Z_k ≡ 1` on one particle, with
`c = 1/2` and `C = 2`.  Every angle then turns at the constant speed `-1`, so
there is no critical point at all — which is what makes the last two
hypotheses hold, the first of them by an isolation condition with no premise
to satisfy. -/
example :
    (0 : ℝ) < 1 / 2 ∧
      (∀ k : Idx 1, ContDiff ℝ 1 (fun ψ : Idx 1 → ℝ => ψ k) ∧
        ContDiff ℝ 1 (fun _ : Idx 1 → ℝ => (1 : ℝ))) ∧
      (∀ _k : Idx 1, ∀ _ψ : Idx 1 → ℝ, (1 / 2 : ℝ) < 1 ∧ (1 : ℝ) < 2) ∧
      DependsOnPrefix 1 (fun k ψ => ψ k) ∧
      DependsOnPrefix 1 (fun _ _ => (1 : ℝ)) ∧
      IsolatedCritical 1 (fun k ψ => ψ k) ∧
      (∀ k : Idx 1, ∀ ψ : Idx 1 → ℝ,
        IsCriticalPrefix 1 (fun k ψ => ψ k) (fun _ _ => (1 : ℝ)) k ψ →
          StronglyStablePrefix 1 (fun k ψ => ψ k) (fun _ _ => (1 : ℝ)) k ψ ∨
            StronglyUnstablePrefix 1 (fun k ψ => ψ k) (fun _ _ => (1 : ℝ)) k ψ) := by
  refine ⟨by norm_num, fun k => ⟨by fun_prop, by fun_prop⟩, fun _ _ => by norm_num,
    fun k ψ ψ' h => h k le_rfl, fun _ _ _ _ => rfl, ?_, ?_⟩
  · intro k ψ u hu
    simp [Function.update_self] at hu
  · intro k ψ hcrit
    have := hcrit ⟨k, le_rfl⟩
    simp [seqVelocity, Function.update_self] at this

end Causal
end Transformer
