module

public import Transformer.SequenceModel.Basic

/-!
# What sampled training can distinguish

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, N1.
The actual CE function, its full gradient and all Schedule-Free AdamW state
are identical when two tasks agree on every label sampled so far. This is
independent of architecture, initialization, width and depth. It does not
replace the gradient by an assumed oracle or omit optimizer history.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Information

open ScheduleFree

variable {I X V : Type*} [Fintype I] [Fintype V]

omit [Fintype I] in
/-- N1: sampled CE is the same function of all weights for observationally
identical tasks, including the multiplicity of independently indexed rows. -/
theorem loss_eq_of_labels (forward : Weights I → X → V → ℝ) (rows : Finset X)
    (f g : X → V) (h : ∀ x ∈ rows, f x = g x) :
    loss forward rows f = loss forward rows g := by
  funext w
  unfold loss Landscape.Neural.meanCrossEntropy
  congr 1
  exact Finset.sum_congr rfl (fun x hx => by rw [h x hx])

/-- N1: equality of the actual loss functions forces equality of every
sampled parameter derivative, without a differentiability assumption. -/
theorem oracle_eq_of_labels (forward : Weights I → X → V → ℝ) (samples : ℕ → Finset X)
    (f g : X → V) (t : ℕ) (h : ∀ x ∈ samples t, f x = g x) (y : I → ℝ) :
    sampledOracle forward samples f t y = sampledOracle forward samples g t y := by
  unfold sampledOracle
  rw [loss_eq_of_labels forward (samples t) f g h]

/-- N1: the entire source optimizer state agrees after any finite number
of steps whose observed labels agree, not only the final prediction. -/
theorem trajectory_eq_of_labels (forward : Weights I → X → V → ℝ) (samples : ℕ → Finset X)
    (f g : X → V) (p : AdamW.Parameters) (offset : ℕ) (initial : AdamW.TrainingState I)
    (n : ℕ) (h : ∀ t < n, ∀ x ∈ samples (offset + t), f x = g x) :
    trajectory forward samples f p offset initial n = trajectory forward samples g p offset initial n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have hn := ih (fun t ht => h t (by omega))
    rw [trajectory, trajectory, hn]
    rw [oracle_eq_of_labels forward samples f g (offset + n) (h n (by omega))]

variable [DecidableEq X]

/-- N1: changing only a still-unobserved target leaves the actual complete
trajectory unchanged, for any replacement label and any initialization. -/
theorem unseen_trajectory_eq (forward : Weights I → X → V → ℝ) (samples : ℕ → Finset X)
    (f : X → V) (a : X) (v : V) (p : AdamW.Parameters) (offset : ℕ)
    (initial : AdamW.TrainingState I) (n : ℕ)
    (hunseen : ∀ t < n, a ∉ samples (offset + t)) :
    trajectory forward samples (Function.update f a v) p offset initial n =
      trajectory forward samples f p offset initial n := by
  apply trajectory_eq_of_labels
  intro t ht x hx
  have hxa : x ≠ a := by intro he; subst x; exact hunseen t ht hx
  exact Function.update_of_ne hxa v f

/-- N1: two different unobserved labels cannot both be strictly predicted
by their indistinguishable trained models. -/
theorem unseen_correct_incompatible (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (f : X → V) (a : X) (v u : V) (hne : v ≠ u)
    (p : AdamW.Parameters) (offset : ℕ) (initial : AdamW.TrainingState I) (n : ℕ)
    (hunseen : ∀ t < n, a ∉ samples (offset + t)) :
    ¬ (Correct forward (Rotate90.Learning.evaluated p
        (trajectory forward samples (Function.update f a v) p offset initial n)) (Function.update f a v) a ∧
      Correct forward (Rotate90.Learning.evaluated p
        (trajectory forward samples (Function.update f a u) p offset initial n)) (Function.update f a u) a) := by
  rw [unseen_trajectory_eq forward samples f a v p offset initial n hunseen,
    unseen_trajectory_eq forward samples f a u p offset initial n hunseen]
  simp only [Correct, Function.update_self]
  rintro ⟨hv, hu⟩
  exact lt_asymm (hv u hne.symm) (hu v hne)

/-- N1: a model-independent finite-horizon obstruction. For each particular
start there is an indistinguishable target extension it fails; the next
module strengthens this to one fixed extension for a distribution of starts. -/
theorem exists_unseen_failure [Nontrivial V] (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (f : X → V) (a : X) (p : AdamW.Parameters) (offset : ℕ)
    (initial : AdamW.TrainingState I) (n : ℕ) (hunseen : ∀ t < n, a ∉ samples (offset + t)) :
    ∃ v : V, ¬ Correct forward (Rotate90.Learning.evaluated p
      (trajectory forward samples (Function.update f a v) p offset initial n)) (Function.update f a v) a := by
  classical
  obtain ⟨v, u, hne⟩ := exists_pair_ne V
  by_cases hv : Correct forward (Rotate90.Learning.evaluated p
      (trajectory forward samples (Function.update f a v) p offset initial n)) (Function.update f a v) a
  · exact ⟨u, fun hu => unseen_correct_incompatible forward samples f a v u hne p offset initial n hunseen ⟨hv, hu⟩⟩
  · exact ⟨v, hv⟩

private def forward (w : Weights (Fin 2)) (a : Fin 3) (v : Fin 2) : ℝ :=
  w v * (a.val + 1)
private def samples : ℕ → Finset (Fin 3) := fun _ => {0, 1}
private def p₀ : AdamW.Parameters := ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩
private def initial : AdamW.TrainingState (Fin 2) := ⟨fun i => i.val + 1, fun i => i.val + 1, ⟨0, 0, 0⟩⟩

example := loss_eq_of_labels forward {0, 1} (fun _ => 0) (fun a => if a = 2 then 1 else 0)
  (by intro a ha; simp only [Finset.mem_insert, Finset.mem_singleton] at ha; rcases ha with rfl | rfl <;> decide)
example := oracle_eq_of_labels forward samples (fun _ => 0) (fun a => if a = 2 then 1 else 0)
  5 (by intro a ha; fin_cases a <;> simp_all [samples]) (fun i => i.val + 1)
example := trajectory_eq_of_labels forward samples (fun _ => 0) (fun a => if a = 2 then 1 else 0)
  p₀ 0 initial 100 (by intro t ht a ha; fin_cases a <;> simp_all [samples])
example := unseen_trajectory_eq forward samples (fun _ => 0) 2 1 p₀ 0 initial 100
  (by intro t ht; simp [samples])
example := unseen_correct_incompatible forward samples (fun _ => 0) 2 0 1 (by decide)
  p₀ 0 initial 100 (by intro t ht; simp [samples])
example := exists_unseen_failure forward samples (fun _ => 0) 2 p₀ 0 initial 100
  (by intro t ht; simp [samples])

end Transformer.SequenceModel.Information
