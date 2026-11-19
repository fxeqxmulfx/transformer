module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Domain
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Queries

/-!
# Actual feasible class-bias CE queries in the learned rotary family

Source: `review/convex/moments.md`, CM98. Zero-frequency identity components
remain ordinary parameter points in this family. They supply finite loss
queries at the same capacity-reserving centre, for every original class.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary

variable {O : Type*} [DecidableEq O]

/-- A feasible constant-class probe in the same learned rotary family,
CM98. It supplies a loss query, not a task-specific answer rule. -/
noncomputable def bias (target : O) (s : ℝ) : Parameters O :=
  ⟨(0, 0), LearnedRecurrent.biasParameters (1, 0) (Pair.row (1, 0)) target s⟩

/-- The actual full probe parameters satisfy every capacity constraint,
CM98. The state/readout pair is explicitly nonzero. -/
theorem bias_unit (target : O) (s : ℝ) (hs : |s| ≤ 1) : Unit (bias target s) :=
  LearnedRecurrent.bias_unit _ _ target s (by simp [Prod.norm_def])
    (Pair.row_norm_bound _ (by simp)) hs

/-- The actual probe changes precisely the chosen original class score
at every input sequence, CM98. -/
theorem bias_score (target : O) (s : ℝ) (xs : List O) (c : O) :
    score (bias target s) xs c = if c = target then s else 0 := by
  rw [bias, score_zero]
  exact LearnedRecurrent.bias_score _ _ (by simp [Pair.row_apply]) target s xs c

/-- The actual rotary-domain query point, CM98. -/
noncomputable def query (B eta : ℝ) (r : List O → O → ℝ) (target : O) (s : ℝ) :
    List O → O → ℝ := (1 - eta) • r + eta • (B • score (bias target s))

/-- Capacity-reserving convex steps keep both signed probes feasible,
CM98. No derivative or exact component optimizer is assumed. -/
theorem query_feasible (B eta : ℝ) (r : List O → O → ℝ) (target : O) (s : ℝ)
    (hr : r ∈ domain O B) (hs : |s| ≤ 1) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    query B eta r target s ∈ domain O B :=
  step_feasible B eta r _ hr (bias_unit target s hs) he he1

/-- Full raw logits of the actual feasible query, CM98. -/
theorem query_logits (B eta : ℝ) (r : List O → O → ℝ) (target : O) (s : ℝ)
    (xs : List O) (c : O) : query B eta r target s xs c =
      (1 - eta) * r xs c + if c = target then eta * B * s else 0 := by
  simp only [query, Pi.add_apply, Pi.smul_apply, smul_eq_mul, bias_score]
  split_ifs <;> ring

/-- Finite CE differences are evaluations at two actual feasible rotary
models, CM98. They are not identified with analytic derivatives. -/
theorem feasible_difference [Fintype O] (B eta : ℝ) (r : List O → O → ℝ)
    (xs : List O) (target c : O) :
    (Landscape.Neural.crossEntropy (query B eta r c 1 xs) target -
      Landscape.Neural.crossEntropy (query B eta r c (-1) xs) target) / (2 * (eta * B)) =
    (Landscape.Neural.crossEntropy (fun d => (1 - eta) * r xs d +
      if d = c then eta * B else 0) target -
      Landscape.Neural.crossEntropy (fun d => (1 - eta) * r xs d +
      if d = c then -(eta * B) else 0) target) / (2 * (eta * B)) := by
  have hp : query B eta r c 1 xs =
      (fun d => (1 - eta) * r xs d + if d = c then eta * B else 0) := by
    funext d
    simpa only [mul_one] using query_logits B eta r c 1 xs d
  have hn : query B eta r c (-1) xs =
      (fun d => (1 - eta) * r xs d + if d = c then -(eta * B) else 0) := by
    funext d
    simpa only [mul_neg_one] using query_logits B eta r c (-1) xs d
  rw [hp, hn]

example : Unit (bias true (-1)) := bias_unit _ _ (by norm_num)

example : query 2 (1 / 2) (0 : List Bool → Bool → ℝ) true (-1) ∈ domain Bool 2 :=
  query_feasible _ _ _ _ _ (zero_mem 2) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary
