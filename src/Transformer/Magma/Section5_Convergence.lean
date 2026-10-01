/-
# Corrected finite-horizon stationarity

Replacement for arXiv:2602.15322v1, Section 5, theorem:main_result,
eq:magma:conv, and Appendix A.4. The source theorem is false (see
Section5_MainRefutation). This version uses a global smooth upper model,
explicit unbiasedness, uniform positive damping, and finite minibatch laws.
It analyzes the normalized SGD recurrence, with dense auxiliary states.
-/

import Transformer.Magma.Section5_StochasticDescent
import Transformer.Magma.Section5_Transition

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

variable {Z ι E Aux : Type*} [Fintype Z] [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Corrected stationarity guarantee for the actual sampled trajectory.
The expectation is defined by repeated normalizedTransition, rather than
supplied as a sequence satisfying descent. The bound is coarser than the
paper's effective block-curvature claim, and requires global smoothness
and explicit unbiasedness. It does not assert convergence of arbitrary
adaptive base optimizers. Source: arXiv:2602.15322v1, Section 5,
theorem:main_result and Appendix A.4, corrected hypotheses and constants. -/
theorem finite_horizon_stationarity_corrected (loss : E → ℝ) (L rate p a b σ star : ℝ)
    (hf : SmoothObjective loss L) (hL : 0 < L) (hrate : 0 < rate)
    (hp : 0 < p) (hp' : p ≤ 1) (ha : 0 < a)
    (hstep : L * rate * b ^ 2 ≤ a * p / 2) (hlower : ∀ x, star ≤ loss x)
    (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z) (hw : ∑ z, w z = 1)
    (g : E × Aux → Z → E)
    (hmean : ∀ state, ∑ z, w z • g state z = gradient loss state.1)
    (hraw : ∀ state, finiteExpectation w (fun z => ‖g state z‖ ^ 2) ≤
      ‖gradient loss state.1‖ ^ 2 + σ ^ 2)
    (S : E × Aux → Z → E →L[ℝ] E)
    (hS : ∀ state z, DampingBounds a b (S state z))
    (candidate : E × Aux → Z → Aux × (ι → E))
    (hsum : ∀ state z, ∑ j, (candidate state z).2 j = S state z (g state z))
    (horth : ∀ state z j k, j ≠ k →
      ⟪(candidate state z).2 j, (candidate state z).2 k⟫_ℝ = 0)
    (initial : E × Aux) (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T,
      pathExpectation (jointMass w p) (normalizedTransition rate p candidate) initial t
        (fun state => ‖gradient loss state.1‖ ^ 2)) / T ≤
      4 * (loss initial.1 - star) / (rate * a * T) +
        2 * b ^ 2 * σ ^ 2 / a ^ 2 + 2 * L * rate * b ^ 2 * σ ^ 2 / (p * a) := by
  let next := normalizedTransition rate p candidate
  let weight := jointMass (ι := ι) w p
  let c := rate * a / 4
  let noise := (rate * b ^ 2 / (2 * a) + L * rate ^ 2 * b ^ 2 / (2 * p)) * σ ^ 2
  let objective := fun state : E × Aux => loss state.1
  let energy := fun state : E × Aux => ‖gradient loss state.1‖ ^ 2
  have hwj : ∑ sample, weight sample = 1 := jointMass_sum w hw p
  have hwj0 : ∀ sample, 0 ≤ weight sample := jointMass_nonneg w hw0 p hp.le hp'
  have hmodel (state : E × Aux) : finiteExpectation weight
      (fun sample => objective (next state sample)) ≤
        objective state - c * energy state + noise := by
    change finiteExpectation (jointMass w p)
      (fun sample => loss (normalizedTransition rate p candidate state sample).1) ≤ _
    rw [normalizedTransition_expectation]
    exact stochastic_descent_corrected loss L rate p a b σ state.1 hf hL hrate hp hp'
      ha hstep w hw0 hw (g state) (hmean state) (hraw state) (S state) (hS state)
      (fun z => (candidate state z).2) (hsum state) (horth state)
  have hprogress (t : ℕ) : pathExpectation weight next initial (t + 1) objective ≤
      pathExpectation weight next initial t objective -
        c * pathExpectation weight next initial t energy + noise := by
    have h := pathExpectation_mono weight hwj0 next initial t _ _ hmodel
    simpa only [pathExpectation, pathExpectation_add, pathExpectation_sub,
      pathExpectation_mul, pathExpectation_const weight hwj] using h
  have htel (n : ℕ) : c * (∑ t ∈ Finset.range n,
      pathExpectation weight next initial t energy) ≤
        loss initial.1 - pathExpectation weight next initial n objective + n * noise := by
    induction n with
    | zero => simp [pathExpectation, objective]
    | succ n ih =>
      rw [Finset.sum_range_succ, Nat.cast_add, Nat.cast_one]
      linarith [hprogress n]
  have hfloor : star ≤ pathExpectation weight next initial T objective := by
    have h := pathExpectation_mono weight hwj0 next initial T
      (fun _ => star) objective (fun state => hlower state.1)
    rwa [pathExpectation_const weight hwj] at h
  have hc : 0 < c := by dsimp [c]; positivity
  have hs : (∑ t ∈ Finset.range T, pathExpectation weight next initial t energy) ≤
      (loss initial.1 - star + T * noise) / c := by
    apply (le_div_iff₀ hc).mpr
    nlinarith [htel T]
  have hTr : 0 < (T : ℝ) := Nat.cast_pos.mpr hT
  change (∑ t ∈ Finset.range T, pathExpectation weight next initial t energy) / (T : ℝ) ≤ _
  calc
    _ ≤ ((loss initial.1 - star + T * noise) / c) / T :=
      div_le_div_of_nonneg_right hs hTr.le
    _ = _ := by
      dsimp [c, noise]
      field_simp
      ring

end Transformer.Magma
