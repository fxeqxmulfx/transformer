module

public import Mathlib.Algebra.BigOperators.Module
public import Mathlib.Tactic

/-!
# A Schedule-Free wrapper for an explicit stateful base optimizer

arXiv:2405.15682v4, §2 and §2.1: gradients are evaluated at the mixed
query, while the base optimizer changes `z` and evaluation uses the mean
of the base sequence. This wrapper retains the initial point with weight
one, matching the paper's equal-weighted sequence and the existing SGD.
The base optimizer is an explicit function; no regret or convergence
guarantee is assumed merely by wrapping it.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.ScheduleFree

variable {E S : Type*} [AddCommGroup E] [Module ℝ E]

/-- Stored evaluation/base points, native inner state, and completed steps.
Source: arXiv:2405.15682v4, §2; `inner` is the chosen base optimizer's state. -/
structure WrapperState (E S : Type*) where
  evaluation : E
  base : E
  inner : S
  updates : ℕ

/-- The query uses the same interpolation for every parameter block.
Source: arXiv:2405.15682v4, §2, `y_t = beta x_t + (1-beta) z_t`. -/
noncomputable def wrapperQuery (β : ℝ) (s : WrapperState E S) : E :=
  β • s.evaluation + (1 - β) • s.base

/-- After `k` completed steps, include the new base with weight `1/(k+2)`.
The initial base already has weight one. Source: arXiv:2405.15682v4, §2. -/
noncomputable def wrapperCoefficient (k : ℕ) : ℝ := 1 / ((k : ℝ) + 2)

/-- The base optimizer receives its old state, old base, and a query-point
gradient, and returns its new state and base. Source: arXiv:2405.15682v4,
§2.1, arbitrary online/base sequence, specialized to equal weights. -/
noncomputable def wrapperStep (fast : S → E → E → S × E)
    (s : WrapperState E S) (g : E) : WrapperState E S :=
  let next := fast s.inner s.base g
  let c := wrapperCoefficient s.updates
  ⟨(1 - c) • s.evaluation + c • next.2, next.2, next.1, s.updates + 1⟩

/-- Actual oracle locations are the mixed queries, while native optimizer
updates act on the base points. Rates and other time dependence can be
supplied in `fast t`. Source: arXiv:2405.15682v4, §§2–2.1. -/
noncomputable def wrapperRun (β : ℝ) (fast : ℕ → S → E → E → S × E)
    (oracle : ℕ → E → E) (initial : E) (initialInner : S) : ℕ → WrapperState E S
  | 0 => ⟨initial, initial, initialInner, 0⟩
  | t + 1 =>
    let s := wrapperRun β fast oracle initial initialInner t
    wrapperStep (fast t) s (oracle t (wrapperQuery β s))

/-- The online averaging coefficient is strictly between zero and one.
Source: arXiv:2405.15682v4, §2, positive equal prefix weights. -/
theorem wrapperCoefficient_bounds (k : ℕ) :
    0 < wrapperCoefficient k ∧ wrapperCoefficient k < 1 := by
  unfold wrapperCoefficient
  constructor
  · positivity
  · rw [div_lt_one (by positivity)]
    have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    linarith

/-- Both interpolation endpoints have their intended query locations.
Source: arXiv:2405.15682v4, §2, PR averaging and primal averaging. -/
theorem wrapperQuery_endpoints (s : WrapperState E S) :
    wrapperQuery 0 s = s.base ∧ wrapperQuery 1 s = s.evaluation := by
  simp [wrapperQuery]

/-- The displacement from the evaluation point is scaled by `1-beta`.
Source: arXiv:2405.15682v4, §2, interpolation identity. -/
theorem wrapperQuery_displacement (β : ℝ) (s : WrapperState E S) :
    wrapperQuery β s - s.evaluation = (1 - β) • (s.base - s.evaluation) := by
  unfold wrapperQuery
  module

/-- A complete step obeys the exact balance of the old prefix and the new
base point, independently of the inner update. Source: arXiv:2405.15682v4,
§2.1, weighted-average recurrence at unit weights. -/
theorem wrapperStep_balance (fast : S → E → E → S × E)
    (s : WrapperState E S) (g : E) :
    ((s.updates : ℝ) + 2) • (wrapperStep fast s g).evaluation =
      ((s.updates : ℝ) + 1) • s.evaluation + (wrapperStep fast s g).base := by
  have hc : ((s.updates : ℝ) + 2) * wrapperCoefficient s.updates = 1 := by
    unfold wrapperCoefficient
    field_simp
  have hp : ((s.updates : ℝ) + 2) * (1 - wrapperCoefficient s.updates) =
      (s.updates : ℝ) + 1 := by
    nlinarith
  change ((s.updates : ℝ) + 2) •
      ((1 - wrapperCoefficient s.updates) • s.evaluation +
        wrapperCoefficient s.updates • (fast s.inner s.base g).2) = _
  rw [smul_add, ← mul_smul, ← mul_smul, hp, hc, one_smul]
  rfl

/-- The stored clock equals the number of native base updates.
Source: the sequential indexing of arXiv:2405.15682v4, §2. -/
theorem wrapperRun_updates (β : ℝ) (fast : ℕ → S → E → E → S × E)
    (oracle : ℕ → E → E) (initial : E) (initialInner : S) (t : ℕ) :
    (wrapperRun β fast oracle initial initialInner t).updates = t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change (wrapperRun β fast oracle initial initialInner t).updates + 1 = t + 1
    rw [ih]

/-- The evaluation sequence is the mean of the actual base iterates,
including their initial point. Source: arXiv:2405.15682v4, §2.
This invariant does not require the native optimizer to be SGD. -/
theorem wrapperRun_average_balance (β : ℝ) (fast : ℕ → S → E → E → S × E)
    (oracle : ℕ → E → E) (initial : E) (initialInner : S) (t : ℕ) :
    ((t : ℝ) + 1) • (wrapperRun β fast oracle initial initialInner t).evaluation =
      ∑ k ∈ Finset.range (t + 1), (wrapperRun β fast oracle initial initialInner k).base := by
  induction t with
  | zero => simp [wrapperRun]
  | succ t ih =>
    let s := wrapperRun β fast oracle initial initialInner t
    have h := wrapperStep_balance (fast t) s (oracle t (wrapperQuery β s))
    have ht : s.updates = t := wrapperRun_updates β fast oracle initial initialInner t
    rw [ht] at h
    calc
      _ = ((t : ℝ) + 1) • s.evaluation +
          (wrapperRun β fast oracle initial initialInner (t + 1)).base := by
        rw [show ((t + 1 : ℕ) : ℝ) + 1 = (t : ℝ) + 2 by push_cast; ring]
        exact h
      _ = _ := by rw [Finset.sum_range_succ, ← ih]

/-- Solving the prefix balance gives the arithmetic mean of the actual
base trajectory, including its initial point. Source: arXiv:2405.15682v4,
§2.1, arbitrary online sequence with all weights equal to one. -/
theorem wrapperRun_evaluation (β : ℝ) (fast : ℕ → S → E → E → S × E)
    (oracle : ℕ → E → E) (initial : E) (initialInner : S) (t : ℕ) :
    (wrapperRun β fast oracle initial initialInner t).evaluation =
      ((t : ℝ) + 1)⁻¹ •
        ∑ k ∈ Finset.range (t + 1), (wrapperRun β fast oracle initial initialInner k).base := by
  rw [← wrapperRun_average_balance, smul_smul,
    inv_mul_cancel₀ (by positivity : (t : ℝ) + 1 ≠ 0), one_smul]

/-- At initialization all choices of the interpolation query the same
initial point. Source: arXiv:2405.15682v4, §2, `x_1=z_1`. -/
theorem wrapper_initial_query (β : ℝ) (initial : E) (initialInner : S) :
    wrapperQuery β (WrapperState.mk initial initial initialInner 0) = initial := by
  unfold wrapperQuery
  module

end Transformer.ScheduleFree
