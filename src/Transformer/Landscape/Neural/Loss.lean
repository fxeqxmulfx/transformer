module

public import Transformer.Landscape.Neural.Basic

/-!
# Finite classification and regression losses

Sources: `landscape_grokking_plan.md`, Model and protocol and Mechanisms to
test; the Rotate90 lab uses mean cross-entropy while its scalar control uses
mean squared error. The class sum includes all answers, including zero.

Analyticity of a loss follows from analyticity of its actual predictions.
This closure result makes no claim about which minima generalize. Finite
logits with a competing answer always have strictly positive cross-entropy.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Landscape.Neural

variable {P V X : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P] [Fintype V]

/-- One row's cross-entropy, with the full softmax competition of the lab.
Source: Model and protocol, row-loss equation in the research plan. -/
noncomputable def crossEntropy (logits : V → ℝ) (target : V) : ℝ :=
  Real.log (∑ q, Real.exp (logits q)) - logits target

/-- The mean loss on a specified set of equations. Empty sets have loss
zero by real division convention; experiments use nonempty splits. -/
noncomputable def meanCrossEntropy (rows : Finset X) (logits : X → V → ℝ)
    (target : X → V) : ℝ :=
  (∑ r ∈ rows, crossEntropy (logits r) (target r)) / rows.card

/-- The scalar baseline's mean squared error in its specified output units.
Source: Model and protocol, scalar MLP comparison in the research plan. -/
noncomputable def meanSquaredError (rows : Finset X) (prediction target : X → ℝ) : ℝ :=
  (∑ r ∈ rows, (prediction r - target r) ^ 2) / rows.card

/-- A target index witnesses that the softmax denominator is positive;
there is no nonzero-logit assumption (Model and protocol). -/
theorem sum_exp_pos (logits : V → ℝ) (target : V) :
    0 < ∑ q, Real.exp (logits q) := by
  have h := Finset.single_le_sum
    (fun q (_ : q ∈ (Finset.univ : Finset V)) => (Real.exp_pos (logits q)).le)
    (Finset.mem_univ target)
  exact (Real.exp_pos (logits target)).trans_le h

/-- Cross-entropy is nonnegative, retaining the competing answer logits
omitted in the clock reduction (Model and protocol and M5.4). -/
theorem crossEntropy_nonneg (logits : V → ℝ) (target : V) :
    0 ≤ crossEntropy logits target := by
  have h := Finset.single_le_sum
    (fun q (_ : q ∈ (Finset.univ : Finset V)) => (Real.exp_pos (logits q)).le)
    (Finset.mem_univ target)
  have hlog := Real.log_le_log (Real.exp_pos (logits target)) h
  rw [Real.log_exp] at hlog
  exact sub_nonneg.mpr hlog

/-- The lab's empirical cross-entropy is nonnegative for every parameter
value and every split (Model and protocol, mean row loss). -/
theorem meanCrossEntropy_nonneg (rows : Finset X) (logits : X → V → ℝ)
    (target : X → V) : 0 ≤ meanCrossEntropy rows logits target := by
  exact div_nonneg
    (Finset.sum_nonneg fun r _ => crossEntropy_nonneg (logits r) (target r))
    (Nat.cast_nonneg _)

/-- The scalar control's loss has the same lower bound, although its
prediction units and optimization geometry differ (Model and protocol). -/
theorem meanSquaredError_nonneg (rows : Finset X) (prediction target : X → ℝ) :
    0 ≤ meanSquaredError rows prediction target := by
  exact div_nonneg (Finset.sum_nonneg fun r _ => sq_nonneg (prediction r - target r))
    (Nat.cast_nonneg _)

/-- With at least one competing answer, no finite real logits attain zero
cross-entropy (Mechanisms to test, possible escape at separable losses). -/
theorem crossEntropy_pos (logits : V → ℝ) (target other : V) (hne : other ≠ target) :
    0 < crossEntropy logits target := by
  have h := Finset.single_lt_sum hne (Finset.mem_univ target) (Finset.mem_univ other)
    (Real.exp_pos (logits other))
    (fun q (_ : q ∈ (Finset.univ : Finset V)) _ => (Real.exp_pos (logits q)).le)
  have hlog := Real.log_lt_log (Real.exp_pos (logits target)) h
  rw [Real.log_exp] at hlog
  exact sub_pos.mpr hlog

/-- Cross-entropy is analytic in parameters when every logit is analytic;
the full positive class sum justifies the logarithm (Model and protocol). -/
theorem analyticAt_crossEntropy {logits : P → V → ℝ} {x : P}
    (hl : ∀ q, AnalyticAt ℝ (fun y => logits y q) x) (target : V) :
    AnalyticAt ℝ (fun y => crossEntropy (logits y) target) x := by
  have hs : AnalyticAt ℝ (fun y => ∑ q, Real.exp (logits y q)) x :=
    Finset.analyticAt_fun_sum _ fun q _ => (hl q).rexp'
  have hlog : AnalyticAt ℝ (fun y => Real.log (∑ q, Real.exp (logits y q))) x :=
    (analyticAt_log (sum_exp_pos (logits x) target)).comp
      (f := fun y : P => ∑ q, Real.exp (logits y q)) hs
  exact hlog.fun_sub (hl target)

/-- Averaging analytic row losses preserves analyticity. Only rows actually
in the split are required (Model and protocol, seeded train/test split). -/
theorem analyticAt_meanCrossEntropy (rows : Finset X) {logits : P → X → V → ℝ}
    {x : P} (hl : ∀ r ∈ rows, ∀ q, AnalyticAt ℝ (fun y => logits y r q) x)
    (target : X → V) :
    AnalyticAt ℝ (fun y => meanCrossEntropy rows (logits y) target) x := by
  exact (Finset.analyticAt_fun_sum rows fun r hr =>
    analyticAt_crossEntropy (hl r hr) (target r)).div_const

/-- A finite scalar MSE is analytic in the predicted values, with no
assumption that its parameters fit the table (Model and protocol). -/
theorem analyticAt_meanSquaredError (rows : Finset X) {prediction : P → X → ℝ}
    {x : P} (hp : ∀ r ∈ rows, AnalyticAt ℝ (fun y => prediction y r) x) (target : X → ℝ) :
    AnalyticAt ℝ (fun y => meanSquaredError rows (prediction y) target) x := by
  exact (Finset.analyticAt_fun_sum rows fun r hr =>
    ((hp r hr).fun_sub analyticAt_const).fun_pow 2).div_const

/- Witnesses include nonconstant logits, both answer classes, and nonempty
row sets; they do not obtain analyticity by choosing an empty training set. -/

example : 0 ≤ meanSquaredError (Finset.univ : Finset (Fin 2))
    (fun r => (r.val : ℝ) + 1) (fun r => r.val) := by
  exact meanSquaredError_nonneg _ _ _

example : (1 : Fin 2) ≠ 0 ∧
    0 < crossEntropy (fun q : Fin 2 => if q = 0 then 1 else 0) 0 := by
  have h : (1 : Fin 2) ≠ 0 := by decide
  exact ⟨h, crossEntropy_pos _ 0 1 h⟩

example : (∀ q : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (q.val : ℝ)) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => crossEntropy (fun q : Fin 2 => t * (q.val : ℝ)) 0) 0 := by
  have h : ∀ q : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (q.val : ℝ)) 0 := by
    intro q
    fun_prop
  exact ⟨h, analyticAt_crossEntropy h 0⟩

example : (∀ r ∈ (Finset.univ : Finset (Fin 2)), ∀ q : Fin 2,
      AnalyticAt ℝ (fun t : ℝ => t * (q.val : ℝ) + r.val) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => meanCrossEntropy (Finset.univ : Finset (Fin 2))
      (fun r q : Fin 2 => t * (q.val : ℝ) + r.val) id) 0 := by
  have h : ∀ r ∈ (Finset.univ : Finset (Fin 2)), ∀ q : Fin 2,
      AnalyticAt ℝ (fun t : ℝ => t * (q.val : ℝ) + r.val) 0 := by
    intro r hr q
    fun_prop
  exact ⟨h, analyticAt_meanCrossEntropy _ h id⟩

example : (∀ r ∈ (Finset.univ : Finset (Fin 2)),
      AnalyticAt ℝ (fun t : ℝ => t * (r.val : ℝ)) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => meanSquaredError (Finset.univ : Finset (Fin 2))
      (fun r : Fin 2 => t * (r.val : ℝ)) (fun r => r.val)) 0 := by
  have h : ∀ r ∈ (Finset.univ : Finset (Fin 2)),
      AnalyticAt ℝ (fun t : ℝ => t * (r.val : ℝ)) 0 := by
    intro r hr
    fun_prop
  exact ⟨h, analyticAt_meanSquaredError _ h (fun r => r.val)⟩

end Transformer.Landscape.Neural
