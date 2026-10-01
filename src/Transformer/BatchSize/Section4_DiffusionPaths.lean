/-
# The paper's diffusion law on genuine continuous paths

arXiv:2506.12543v1, Section 4.3, Theorem 1 and equations (2)--(3).
The generator martingale problem specifies the continuous law independently
of any claim comparing it with the discrete optimizer.
-/

import Transformer.BatchSize.Section4_Regularity

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Continuous sample paths for the SDEs in Section 4.3. -/
abbrev DiffusionPath (d : ℕ) := C(ℝ, EucSpace d)

/-- A compensated test observable over [s,t], Section 4.3, equations (2)--(3).
For the claimed SDE it is a martingale increment. -/
def compensatedIncrement {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (φ : EucSpace d → ℝ) (s t : ℝ) (ω : DiffusionPath d) : ℝ :=
  φ (ω t) - φ (ω s) - ∫ u in s..t, diffusionGenerator method η B f σ φ (ω u)

/-- Law of the SDE in Section 4.3, characterized by its generator
martingale problem. The bounded cylinder observable depends on finitely
many past times; quantifying over all such observables encodes the natural
filtration without assuming any optimizer approximation. -/
def IsDiffusionLaw {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x₀ : EucSpace d) (P : Measure (DiffusionPath d)) : Prop :=
  IsProbabilityMeasure P ∧ (∀ᵐ ω ∂P, ω 0 = x₀) ∧
    ∀ φ, BoundedSmoothTest 2 φ → ∀ s t : ℝ, 0 ≤ s → s ≤ t →
      ∀ n : ℕ, ∀ times : Fin n → ℝ, (∀ i, 0 ≤ times i ∧ times i ≤ s) →
        ∀ F : (Fin n → EucSpace d) → ℝ, Measurable F → (∀ y, |F y| ≤ 1) →
          Integrable (fun ω => F (fun i => ω (times i)) *
            compensatedIncrement method η B f σ φ s t ω) P ∧
          (∫ ω, F (fun i => ω (times i)) *
            compensatedIncrement method η B f σ φ s t ω ∂P) = 0

end Transformer.BatchSize
