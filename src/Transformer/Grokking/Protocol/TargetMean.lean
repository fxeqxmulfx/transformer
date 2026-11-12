module

public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Transformer.Grokking.Protocol.Gradient

/-!
# Means over the targets of a batch

Source: openai/grok at 3d64b1d8, `grok/training.py`, `TrainableTransformer._step`:
`F.cross_entropy(y_hat_rhs, y_rhs, reduction="mean")`; `experiments/grok_division/PROTOCOL.md`,
*The loss and the accuracy*: "a mean over `2 · batch` terms".

The source's loss is a mean over all the targets of a batch, and so is every coordinate of its
gradient. `targetMean B d` is the mean of a quantity `d` of an example and one of its targets;
the loss is the mean of the cross-entropies (`batchLoss_eq_targetMean`). A mean lies between
the bounds of its terms (`targetMean_le`, `le_targetMean`, `abs_targetMean_le`), a combination
of means is the mean of the combinations (`sum_mul_targetMean`), and the derivative of a mean
is the mean of the derivatives (`hasDerivAt_targetMean`).

A batch without targets has the mean `0`, by the convention of division by zero that
`batchLoss` has too; the bounds that need a target say so.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy)

variable {S : Sizes}

/-- **The mean over the targets of a batch** of a quantity `d` of an example and a target. -/
noncomputable def targetMean (B : List (Example S))
    (d : Example S → Fin S.context × Fin S.vocab → ℝ) : ℝ :=
  (B.map fun e => (e.targets.map (d e)).sum).sum / targetCount B

/-- **The loss of a batch is the mean of the cross-entropies of its targets.** -/
theorem batchLoss_eq_targetMean (θ : Param S → ℝ) (B : List (Example S)) :
    batchLoss θ B =
      targetMean B fun e pc => crossEntropy (logits θ (relu S) e.tokens pc.1) pc.2 := rfl

variable {B : List (Example S)} {d : Example S → Fin S.context × Fin S.vocab → ℝ} {F : ℝ}

lemma sum_targets_le (h : ∀ e ∈ B, ∀ pc ∈ e.targets, d e pc ≤ F) :
    (B.map fun e => (e.targets.map (d e)).sum).sum ≤ targetCount B * F := by
  induction B with
  | nil => simp [targetCount]
  | cons e B ih =>
    have h1 : (e.targets.map (d e)).sum ≤ e.targets.length * F := by
      have h0 := List.sum_le_length_nsmul (e.targets.map (d e)) F fun x hx => by
        obtain ⟨pc, hpc, rfl⟩ := List.mem_map.1 hx
        exact h e List.mem_cons_self pc hpc
      simpa using h0
    have h2 := ih fun e' he' => h e' (List.mem_cons_of_mem e he')
    simp only [targetCount, List.map_cons, List.sum_cons, Nat.cast_add, add_mul] at h2 ⊢
    exact add_le_add h1 h2

lemma le_sum_targets (h : ∀ e ∈ B, ∀ pc ∈ e.targets, F ≤ d e pc) :
    targetCount B * F ≤ (B.map fun e => (e.targets.map (d e)).sum).sum := by
  induction B with
  | nil => simp [targetCount]
  | cons e B ih =>
    have h1 : e.targets.length * F ≤ (e.targets.map (d e)).sum := by
      have h0 := List.length_nsmul_le_sum (e.targets.map (d e)) F fun x hx => by
        obtain ⟨pc, hpc, rfl⟩ := List.mem_map.1 hx
        exact h e List.mem_cons_self pc hpc
      simpa using h0
    have h2 := ih fun e' he' => h e' (List.mem_cons_of_mem e he')
    simp only [targetCount, List.map_cons, List.sum_cons, Nat.cast_add, add_mul] at h2 ⊢
    exact add_le_add h1 h2

/-- **A mean is at most a bound of its terms**, in a batch with a target. -/
theorem targetMean_le (hN : 0 < targetCount B) (h : ∀ e ∈ B, ∀ pc ∈ e.targets, d e pc ≤ F) :
    targetMean B d ≤ F :=
  (div_le_iff₀ (Nat.cast_pos.2 hN)).2 ((sum_targets_le h).trans_eq (mul_comm _ _))

/-- **A mean is at least a bound of its terms**, in a batch with a target. -/
theorem le_targetMean (hN : 0 < targetCount B) (h : ∀ e ∈ B, ∀ pc ∈ e.targets, F ≤ d e pc) :
    F ≤ targetMean B d :=
  (le_div_iff₀ (Nat.cast_pos.2 hN)).2 ((mul_comm _ _).trans_le (le_sum_targets h))

/-- **A mean of terms bounded in absolute value is so bounded**, in every batch. -/
theorem abs_targetMean_le (hF : 0 ≤ F) (h : ∀ e ∈ B, ∀ pc ∈ e.targets, |d e pc| ≤ F) :
    |targetMean B d| ≤ F := by
  rcases Nat.eq_zero_or_pos (targetCount B) with h0 | hN
  · rw [targetMean, h0, Nat.cast_zero, div_zero, abs_zero]
    exact hF
  · exact abs_le.2 ⟨le_targetMean hN fun e he pc hpc => neg_le_of_abs_le (h e he pc hpc),
      targetMean_le hN fun e he pc hpc => le_of_abs_le (h e he pc hpc)⟩

lemma sum_mul_list_sum {κ α : Type*} (s : Finset κ) (a : κ → ℝ) (l : List α)
    (f : κ → α → ℝ) :
    ∑ c ∈ s, a c * (l.map (f c)).sum = (l.map fun x => ∑ c ∈ s, a c * f c x).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp only [List.map_cons, List.sum_cons, mul_add, Finset.sum_add_distrib, ih]

/-- **A combination of means is the mean of the combinations.** -/
theorem sum_mul_targetMean {κ : Type*} (s : Finset κ) (a : κ → ℝ) (B : List (Example S))
    (d : κ → Example S → Fin S.context × Fin S.vocab → ℝ) :
    ∑ c ∈ s, a c * targetMean B (d c) = targetMean B fun e pc => ∑ c ∈ s, a c * d c e pc := by
  simp only [targetMean, mul_div_assoc', ← Finset.sum_div, sum_mul_list_sum]

/-- A sum of means is the mean of the sums. -/
theorem sum_targetMean {κ : Type*} (s : Finset κ) (B : List (Example S))
    (d : κ → Example S → Fin S.context × Fin S.vocab → ℝ) :
    ∑ c ∈ s, targetMean B (d c) = targetMean B fun e pc => ∑ c ∈ s, d c e pc := by
  simpa using sum_mul_targetMean s (fun _ => 1) B d

lemma hasDerivAt_list_sum_map {α : Type*} (l : List α) {f : α → ℝ → ℝ} {f' : α → ℝ} {x : ℝ}
    (h : ∀ a ∈ l, HasDerivAt (f a) (f' a) x) :
    HasDerivAt (fun s => (l.map fun a => f a s).sum) (l.map f').sum x := by
  induction l with
  | nil => simpa using hasDerivAt_const x (0 : ℝ)
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact (h a List.mem_cons_self).fun_add (ih fun b hb => h b (List.mem_cons_of_mem a hb))

/-- **The derivative of a mean is the mean of the derivatives.** -/
theorem hasDerivAt_targetMean (B : List (Example S))
    {f : Example S → Fin S.context × Fin S.vocab → ℝ → ℝ}
    {f' : Example S → Fin S.context × Fin S.vocab → ℝ} {x : ℝ}
    (h : ∀ e ∈ B, ∀ pc ∈ e.targets, HasDerivAt (f e pc) (f' e pc) x) :
    HasDerivAt (fun s => targetMean B fun e pc => f e pc s) (targetMean B f') x :=
  HasDerivAt.div_const (hasDerivAt_list_sum_map B fun e he =>
    hasDerivAt_list_sum_map e.targets fun pc hpc => h e he pc hpc) _

/-! The witnesses: a batch of one example with the two targets that the quantity sends to `1`
and `3`; the mean is `2`. -/

/-- An example with two targets. -/
def twoTargets : Example ⟨2, 1, 1, 1, 1, 1, 2⟩ := ⟨fun _ => 0, [(0, 0), (1, 1)]⟩

example :
    targetMean [twoTargets] (fun _ pc => 2 * (pc.2 : ℕ) + 1) = 2 := by
  norm_num [targetMean, targetCount, twoTargets]

example : targetMean [twoTargets] (fun _ pc => 2 * (pc.2 : ℕ) + 1) ≤ 3 :=
  targetMean_le (by decide) fun e he pc hpc => by
    obtain rfl := List.mem_singleton.1 he
    fin_cases hpc <;> norm_num

example : (1 : ℝ) ≤ targetMean [twoTargets] (fun _ pc => 2 * (pc.2 : ℕ) + 1) :=
  le_targetMean (by decide) fun e he pc hpc => by
    obtain rfl := List.mem_singleton.1 he
    fin_cases hpc <;> norm_num

example : |targetMean [twoTargets] (fun _ pc => 2 * (pc.2 : ℕ) + 1)| ≤ 3 :=
  abs_targetMean_le (by norm_num) fun e he pc hpc => by
    obtain rfl := List.mem_singleton.1 he
    fin_cases hpc <;> norm_num

example : HasDerivAt (fun s : ℝ => targetMean [twoTargets] fun _ pc => (pc.2 : ℕ) * s)
    (targetMean [twoTargets] fun _ pc => ((pc.2 : ℕ) : ℝ)) 5 :=
  hasDerivAt_targetMean [twoTargets] fun _ _ pc _ => by
    simpa using (hasDerivAt_id' (5 : ℝ)).const_mul ((pc.2 : ℕ) : ℝ)

end Transformer.Grokking.Protocol
