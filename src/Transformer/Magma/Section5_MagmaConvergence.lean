/-
# Corrected convergence of the actual dense-state Magma SGD model

Replacement for arXiv:2602.15322v1, Section 5, theorem:main_result and
Appendix A.4. The proposal uses the real first-moment recurrence and
cosine/sigmoid scale EMA. Uniform operator bounds are derived, not assumed.
Global smoothness and unbiased finite sampling are explicit corrections.
-/

import Transformer.Magma.Section5_MagmaTransition
import Transformer.Magma.Section5_Convergence

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

variable {Z ι E : Type*} [Fintype Z] [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Corrected stationarity for actual dense-state Magma with an SGD base
direction. Each transition updates momentum and the alignment EMA densely,
then samples independent normalized masks. With a=sigmoid(-1/tau), the
positive-temperature EMA invariant supplies the needed damping bound.
The theorem requires global smoothness and explicit unbiasedness; these
replace the source's insufficient coordinate smoothness and raw moments.
It does not assert this bound for arbitrary Adam/Muon base updates.
Source: arXiv:2602.15322v1, Section 5, theorem:main_result, corrected. -/
theorem magma_finite_horizon_corrected (loss : E → ℝ) (L rate p τ β σ star : ℝ)
    (hτ : 0 < τ) (hf : SmoothObjective loss L) (hL : 0 < L) (hrate : 0 < rate)
    (hp : 0 < p) (hp' : p ≤ 1)
    (hstep : L * rate ≤ Real.sigmoid (-1 / τ) * p / 2) (hlower : ∀ x, star ≤ loss x)
    (P : ι → E →L[ℝ] E) (hP : OrthogonalBlocks P)
    (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z) (hw : ∑ z, w z = 1)
    (g : E × DenseBlockState ι E τ → Z → E)
    (hmean : ∀ state, ∑ z, w z • g state z = gradient loss state.1)
    (hraw : ∀ state, finiteExpectation w (fun z => ‖g state z‖ ^ 2) ≤
      ‖gradient loss state.1‖ ^ 2 + σ ^ 2)
    (initial : E × DenseBlockState ι E τ) (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T,
      pathExpectation (jointMass w p)
        (normalizedTransition rate p (fun state z => magmaProposal P τ hτ β state (g state z)))
          initial t (fun state => ‖gradient loss state.1‖ ^ 2)) / T ≤
      4 * (loss initial.1 - star) / (rate * Real.sigmoid (-1 / τ) * T) +
        2 * σ ^ 2 / (Real.sigmoid (-1 / τ)) ^ 2 +
        2 * L * rate * σ ^ 2 / (p * Real.sigmoid (-1 / τ)) := by
  have h := finite_horizon_stationarity_corrected loss L rate p
    (Real.sigmoid (-1 / τ)) 1 σ star hf hL hrate hp hp'
    (Real.sigmoid_pos _) (by simpa using hstep) hlower w hw0 hw g hmean hraw
    (fun state z => magmaDampingOperator P τ hτ β state (g state z))
    (fun state z => magmaDampingOperator_bounds P hP τ hτ β state (g state z))
    (fun state z => magmaProposal P τ hτ β state (g state z))
    (fun state z => magmaProposal_sum P τ hτ β state (g state z))
    (fun state z => magmaProposal_orthogonal P hP τ hτ β state (g state z))
    initial T hT
  simpa using h

omit [Fintype Z] [Fintype ι] [DecidableEq ι] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] in
/-- The actual EMA invariant has a scalar initial state with zero
momentum and scale 1/2. Source: arXiv:2602.15322v1, Sections 2--3,
explicit nonempty corrected initialization domain. -/
def quadraticMagmaInitial : ℝ × DenseBlockState Unit ℝ 1 :=
  (1, (fun _ => 0, fun _ => ⟨1 / 2, by
    constructor
    · simpa using Real.sigmoid_le (by norm_num : (-1 : ℝ) ≤ 0)
    · norm_num⟩))

omit [Fintype Z] [Fintype ι] [DecidableEq ι] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] in
/-- Every hypothesis of the corrected actual-Magma theorem is satisfied
by the nonconstant quadratic, exact gradients, dense momentum beta=0.9,
initial scale 1/2, survival 1/2, and positive step a/10. This is an actual
stateful cosine/EMA model, not constant damping supplied as a premise.
Source: arXiv:2602.15322v1, Section 5, corrected theorem domain. -/
theorem actual_magma_quadratic_witness (T : ℕ) (hT : 0 < T) :
    let a := Real.sigmoid (-1)
    (∑ t ∈ Finset.range T,
      pathExpectation (jointMass (fun _ : Unit => (1 : ℝ)) (1 / 2))
        (normalizedTransition (a / 10) (1 / 2)
          (fun state (_ : Unit) => magmaProposal (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ)
            1 (by norm_num) (9 / 10) state (gradient quadratic state.1)))
          quadraticMagmaInitial t (fun state => ‖gradient quadratic state.1‖ ^ 2)) / T ≤
      20 / (a ^ 2 * T) := by
  dsimp only
  have ha := Real.sigmoid_pos (-1)
  have hP : OrthogonalBlocks (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) := by
    constructor
    · simp
    · intro u j k h
      exact (h (Subsingleton.elim j k)).elim
  have h := magma_finite_horizon_corrected quadratic 1 (Real.sigmoid (-1) / 10)
    (1 / 2) 1 (9 / 10) 0 0 (by norm_num) quadratic_smooth (by norm_num)
    (by positivity) (by norm_num) (by norm_num) (by norm_num; linarith)
    (by intro x; unfold quadratic; positivity)
    (fun _ : Unit => ContinuousLinearMap.id ℝ ℝ) hP
    (fun _ : Unit => (1 : ℝ)) (by simp) (by simp)
    (fun state _ => gradient quadratic state.1) (by simp)
    (by intro state; simp [finiteExpectation]) quadraticMagmaInitial T hT
  norm_num [quadraticMagmaInitial, quadratic] at h ⊢
  convert h using 1
  field_simp
  ring

/-- The actual-Magma witness's horizon condition has an instance.
Source: arXiv:2602.15322v1, Section 5, T>=1. -/
example : 0 < (1 : ℕ) := by decide

end Transformer.Magma
