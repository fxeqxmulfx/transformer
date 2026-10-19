module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Algebra.BigOperators.Module
public import Mathlib.Analysis.Convex.Function
public import Mathlib.Tactic

/-!
# Schedule-Free averaging

Defazio et al., "The Road Less Scheduled", arXiv:2405.15682v4, §2.
Indices here start at zero: `z t` is the paper's `z_{t+1}` and
`average w z (t+1)` is its `x_{t+1}`. The empty average is zero and is
always multiplied by the empty weight sum. Positive nonempty prefix
sums make the displayed divisions meaningful; zero later weights are allowed.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- §2: total weight of the first `n` base points. -/
def weightSum (w : ℕ → ℝ) (n : ℕ) : ℝ := ∑ t ∈ Finset.range n, w t

/-- §2: the weighted evaluation sequence, with a zero empty average. -/
def average (w : ℕ → ℝ) (z : ℕ → E) (n : ℕ) : E :=
  (weightSum w n)⁻¹ • ∑ t ∈ Finset.range n, w t • z t

/-- §2 and Appendix C: `α = 1-β` is the weight of the base point. -/
def mix (α : ℝ) (x z : E) : E := (1 - α) • x + α • z

/-- §1.3: a supporting subgradient, without a differentiability assumption. -/
def IsSubgradient (F : E → ℝ) (x g : E) : Prop :=
  ∀ u, F x + inner ℝ g (u - x) ≤ F u

/-- §2: gradient query point associated with the weighted base sequence. -/
def query (w : ℕ → ℝ) (α : ℕ → ℝ) (z : ℕ → E) (t : ℕ) : E :=
  mix (α t) (average w z (t + 1)) (z t)

theorem weightSum_zero (w : ℕ → ℝ) : weightSum w 0 = 0 := by
  simp [weightSum]

theorem weightSum_succ (w : ℕ → ℝ) (n : ℕ) :
    weightSum w (n + 1) = weightSum w n + w n := by
  exact Finset.sum_range_succ (f := w) n

theorem weightSum_nonneg {w : ℕ → ℝ} (hw : ∀ t, 0 ≤ w t) (n : ℕ) :
    0 ≤ weightSum w n := Finset.sum_nonneg fun t _ => hw t

example (n : ℕ) : 0 ≤ weightSum (fun _ => 1) n :=
  weightSum_nonneg (fun _ => zero_le_one) n

theorem weightSum_one (n : ℕ) : weightSum (fun _ => 1) n = n := by
  simp [weightSum]

theorem weightSum_pos_succ {w : ℕ → ℝ} (hw : ∀ t, 0 ≤ w t) (hfirst : 0 < w 0)
    (n : ℕ) : 0 < weightSum w (n + 1) := by
  induction n with
  | zero => simpa [weightSum] using hfirst
  | succ n ih => rw [weightSum_succ]; exact add_pos_of_pos_of_nonneg ih (hw _)

example (n : ℕ) : 0 < weightSum (fun _ => 1) (n + 1) :=
  weightSum_pos_succ (fun _ => zero_le_one) zero_lt_one n

/-- §2: multiplying a nonempty average by its total weight recovers its sum. -/
theorem weightSum_smul_average (w : ℕ → ℝ) (z : ℕ → E) (n : ℕ)
    (hW : weightSum w n ≠ 0) :
    weightSum w n • average w z n = ∑ t ∈ Finset.range n, w t • z t := by
  rw [average, smul_smul, mul_inv_cancel₀ hW, one_smul]

example (z : ℕ → E) :
    weightSum (fun _ => 1) 1 • average (fun _ => 1) z 1 =
      ∑ t ∈ Finset.range 1, (1 : ℝ) • z t :=
  weightSum_smul_average _ _ _ (by simp [weightSum])

/-- Appendix A, eq. (8): division-free weighted-average recursion. -/
theorem average_balance (w : ℕ → ℝ) (z : ℕ → E) (n : ℕ)
    (hW : weightSum w (n + 1) ≠ 0) (hprev : n = 0 ∨ weightSum w n ≠ 0) :
    weightSum w (n + 1) • average w z (n + 1) =
      weightSum w n • average w z n + w n • z n := by
  rw [weightSum_smul_average w z (n + 1) hW, Finset.sum_range_succ]
  rcases hprev with rfl | hprev
  · simp [weightSum]
  · rw [weightSum_smul_average w z n hprev]

example (z : ℕ → E) (n : ℕ) :
    weightSum (fun _ => 1) (n + 1) • average (fun _ => 1) z (n + 1) =
      weightSum (fun _ => 1) n • average (fun _ => 1) z n + (1 : ℝ) • z n := by
  apply average_balance
  · exact (weightSum_pos_succ (fun _ => zero_le_one) zero_lt_one n).ne'
  · by_cases hn : n = 0
    · exact Or.inl hn
    · exact Or.inr (by simpa [weightSum_one] using hn)

/-- Appendix A: the displacement identity used by the conversion proof. -/
theorem balance_displacement {U w : ℝ} {p x z : E}
    (h : (U + w) • x = U • p + w • z) : U • (x - p) = w • (z - x) := by
  rw [smul_sub, smul_sub]
  rw [add_smul] at h
  exact sub_eq_sub_iff_add_eq_add.mpr (h.trans (add_comm _ _))

example (x : E) : (0 : ℝ) • (x - x) = (1 : ℝ) • (x - x) :=
  balance_displacement (by simp)

/-- §2: the two endpoint choices recover PR averaging and primal averaging. -/
theorem mix_endpoints (x z : E) : mix 1 x z = z ∧ mix 0 x z = x := by
  simp [mix]

/-- Appendix A: the displacement to the average is scaled by `α`. -/
theorem mix_sub_average (α : ℝ) (x z : E) : mix α x z - x = α • (z - x) := by
  unfold mix
  module

/-- Appendix A: the displacement from the query to the base uses `1-α`. -/
theorem base_sub_mix (α : ℝ) (x z : E) : z - mix α x z = (1 - α) • (z - x) := by
  unfold mix
  module

/-- Supporting subgradients are monotone; Appendix A's final inequality. -/
theorem subgradient_monotone {F : E → ℝ} {x y gx gy : E}
    (hx : IsSubgradient F x gx) (hy : IsSubgradient F y gy) :
    0 ≤ inner ℝ (gx - gy) (x - y) := by
  have hxy := hx y
  have hyx := hy x
  simp only [inner_sub_left, inner_sub_right] at *
  linarith

example : 0 ≤ inner ℝ ((0 : ℝ) - 0) ((0 : ℝ) - 0) :=
  subgradient_monotone (F := fun _ : ℝ => 0) (by intro u; simp)
    (by intro u; simp)

/-- §1.3: a global supporting-subgradient selection enforces ordinary convexity. -/
theorem convex_of_subgradients {F : E → ℝ} {grad : E → E}
    (hgrad : ∀ x, IsSubgradient F x (grad x)) : ConvexOn ℝ Set.univ F := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  let q := a • x + b • y
  have h1 := mul_le_mul_of_nonneg_left (hgrad q x) ha
  have h2 := mul_le_mul_of_nonneg_left (hgrad q y) hb
  have hv : a • (x - q) + b • (y - q) = 0 := by
    calc
      _ = (a • x + b • y) - (a + b) • q := by module
      _ = 0 := by rw [hab, one_smul]; exact sub_self q
  have hi := congrArg (fun v => inner ℝ (grad q) v) hv
  simp only [inner_add_right, real_inner_smul_right, inner_zero_right] at hi
  have hc := congrArg (fun s : ℝ => s * F q) hab
  change F q ≤ a * F x + b * F y
  nlinarith

example : ConvexOn ℝ Set.univ (fun _ : ℝ => (0 : ℝ)) :=
  convex_of_subgradients (grad := fun _ => 0) (by intro x u; simp)

end Transformer.ScheduleFree
