module

public import Transformer.SequenceModel.Information.Basic

/-!
# A fixed unseen target for a distribution of starts

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, N2.
The target below is chosen independently of the initialization seed. The
distribution has arbitrary nonnegative finite support; samples may depend on
the seed as long as the designated example is never observed. This concerns
unconstrained label extensions, not a known algebraic family of tasks.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Information

open ScheduleFree
open scoped BigOperators Classical

variable {S V : Type*} [Fintype S] [Fintype V]

omit [Fintype V] in
/-- N2: strict classification has at most one winning label. -/
theorem winner_unique (l : V → ℝ) (v u : V)
    (hv : ∀ q ≠ v, l q < l v) (hu : ∀ q ≠ u, l q < l u) : v = u := by
  by_contra h
  exact lt_asymm (hv u (Ne.symm h)) (hu v h)

/-- Probability mass assigned to strict correctness for one candidate label. -/
def winnerMass (μ : S → ℝ) (logits : S → V → ℝ) (v : V) : ℝ :=
  ∑ s, if ∀ q ≠ v, logits s q < logits s v then μ s else 0

/-- N2: ties may lose mass, but a seed cannot count as two strict winners. -/
theorem winner_sum_le (l : V → ℝ) (m : ℝ) (hm : 0 ≤ m) :
    (∑ v, if ∀ q ≠ v, l q < l v then m else 0) ≤ m := by
  classical
  by_cases h : ∃ v, ∀ q ≠ v, l q < l v
  · obtain ⟨v, hv⟩ := h
    rw [Finset.sum_eq_single v]
    · split_ifs; linarith
    · intro u _ huv
      have hu : ¬ ∀ q ≠ u, l q < l u := fun hu => huv (winner_unique l u v hu hv)
      simp [hu]
    · simp
  · have hn : ∀ v, ¬ ∀ q ≠ v, l q < l v := by simpa only [not_exists] using h
    simpa [hn] using hm

/-- N2: averaging over random seeds cannot make all unconstrained labels
likely; the sum of their strict-success probabilities is at most one. -/
theorem sum_winnerMass_le (μ : S → ℝ) (logits : S → V → ℝ)
    (hμ : ∀ s, 0 ≤ μ s) (hprob : ∑ s, μ s = 1) :
    ∑ v, winnerMass μ logits v ≤ 1 := by
  unfold winnerMass
  rw [Finset.sum_comm, ← hprob]
  exact Finset.sum_le_sum (fun s _ => winner_sum_le (logits s) (μ s) (hμ s))

/-- N2: one fixed label, independent of the seed, has success probability
at most the reciprocal of the number of labels. Multiplication avoids a
division convention for empty vocabularies, which are explicitly excluded. -/
theorem exists_low_winnerMass [Nonempty V] (μ : S → ℝ) (logits : S → V → ℝ)
    (hμ : ∀ s, 0 ≤ μ s) (hprob : ∑ s, μ s = 1) :
    ∃ v, (Fintype.card V : ℝ) * winnerMass μ logits v ≤ 1 := by
  classical
  by_contra h
  push Not at h
  have hs := Finset.sum_lt_sum_of_nonempty (s := Finset.univ) Finset.univ_nonempty
    (fun v _ => h v)
  have hc : (0 : ℝ) < Fintype.card V := by
    exact_mod_cast (Fintype.card_pos_iff.mpr inferInstance : 0 < Fintype.card V)
  have ht := sum_winnerMass_le μ logits hμ hprob
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one,
    ← Finset.mul_sum] at hs
  nlinarith

variable {I X : Type*} [Fintype I] [DecidableEq X]

/-- N2: the actual source AdamW learner has a single indistinguishable
target extension whose success at an unseen row is at most `1 / card V`,
even when initialization and the finite sampling plan vary with the seed.
This is an information limit, not an optimizer convergence assertion. -/
theorem exists_unseen_low_success [Nonempty V] (forward : Weights I → X → V → ℝ)
    (samples : S → ℕ → Finset X) (f : X → V) (a : X) (p : AdamW.Parameters)
    (offset n : ℕ) (initial : S → AdamW.TrainingState I) (μ : S → ℝ)
    (hμ : ∀ s, 0 ≤ μ s) (hprob : ∑ s, μ s = 1)
    (hunseen : ∀ s t, t < n → a ∉ samples s (offset + t)) :
    ∃ v, (Fintype.card V : ℝ) *
      (∑ s, if Correct forward (Rotate90.Learning.evaluated p
        (trajectory forward (samples s) (Function.update f a v) p offset (initial s) n))
          (Function.update f a v) a then μ s else 0) ≤ 1 := by
  classical
  let logits := fun s => forward (Rotate90.Learning.evaluated p
    (trajectory forward (samples s) f p offset (initial s) n)) a
  obtain ⟨v, hv⟩ := exists_low_winnerMass μ logits hμ hprob
  refine ⟨v, ?_⟩
  simpa only [winnerMass, Correct, logits, Function.update_self,
    unseen_trajectory_eq forward (samples _) f a v p offset (initial _) n (hunseen _)] using hv

private def coin : Fin 2 → ℝ := fun _ => 1 / 2
private def scores (s v : Fin 2) : ℝ := if s = v then 1 else 0

example := winner_unique (scores 0) 0 0
  (by intro q h; fin_cases q <;> simp_all [scores])
  (by intro q h; fin_cases q <;> simp_all [scores])
example := winner_sum_le (scores 0) (1 / 2) (by norm_num)
example := sum_winnerMass_le coin scores (by intro s; norm_num [coin])
  (by norm_num [coin, Fin.sum_univ_two])
example := exists_low_winnerMass coin scores (by intro s; norm_num [coin])
  (by norm_num [coin, Fin.sum_univ_two])

/-- N2: the probability bound is sharp, with both labels occurring as
strict winners under equally weighted initialization seeds. -/
example (v : Fin 2) : 2 * winnerMass coin scores v = 1 := by
  fin_cases v <;> norm_num [winnerMass, coin, scores, Fin.sum_univ_two, Fin.forall_fin_succ]

example := exists_unseen_low_success
  (fun (w : Weights (Fin 2)) (a : Fin 3) (v : Fin 2) => w v * (a.val + 1))
  (fun (_ : Fin 2) (_ : ℕ) => ({0, 1} : Finset (Fin 3))) (fun _ => 0) 2
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 0 100
  (fun s => ⟨fun i => scores s i + 1, fun i => scores s i + 1, ⟨0, 0, 0⟩⟩) coin
  (by intro s; norm_num [coin]) (by norm_num [coin, Fin.sum_univ_two])
  (by intro s t ht; decide)

end Transformer.SequenceModel.Information
