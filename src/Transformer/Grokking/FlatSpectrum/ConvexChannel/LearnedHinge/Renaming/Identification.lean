module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Convexity
public import Transformer.Grokking.Protocol.Symmetry
public import Transformer.Rotate90.Learning.Basic

/-!
# Conditional identification at genuinely renamed unseen prefixes

Source: `review/flat_spectrum.md`, CM78. The architecture imposes equality
of aligned logits within each renaming pair. Observed CE below log 2 then
identifies a unique renamed next object. Correctness for an actual task needs
that task's answer rule to respect the same renaming. Empty or fixed inputs
expose a necessary restriction on moved output labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming

variable {O J : Type*} [Fintype O] [Fintype J] {T : ℕ}

/-- Ordinary answer CE is exactly equal at a consistently renamed input
and target, CM78. Every original output class participates in the sum. -/
theorem pairedLoss_renamed (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (target : O) :
    pairedLoss σ (renamed σ tokens) (σ target) p = pairedLoss σ tokens target p := by
  have hz : pairedLogits σ p (renamed σ tokens) = pairedLogits σ p tokens ∘ σ := by
    funext c
    have h := pairedLogits_equivariant σ hσ p tokens (σ c)
    rw [hσ c] at h
    exact h
  unfold pairedLoss
  rw [hz, Protocol.crossEntropy_comp_perm, hσ target]

/-- Low ordinary CE excludes every other original object and every tie,
CM78. This is a strict next-object decision, not a hidden-path prediction. -/
theorem low_loss_unique (σ : Equiv.Perm O) (p : Parameters J O (ObjectWeights O T))
    (tokens : Prefix O T) (target : O) (hfit : pairedLoss σ tokens target p < Real.log 2) :
    ∀ c, c ≠ target → pairedLogits σ p tokens c < pairedLogits σ p tokens target := by
  intro c hc
  by_contra h
  have hbad := Rotate90.Learning.log_two_le_crossEntropy (pairedLogits σ p tokens)
    target c hc.symm (le_of_not_gt h)
  exact (not_le_of_gt hfit) hbad

/-- The observed fit fixes the aligned object at the renamed prefix, CM78.
The prefix need not appear among the observed training records. -/
theorem renamed_unique (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (target : O)
    (hfit : pairedLoss σ tokens target p < Real.log 2) :
    ∀ c, c ≠ σ target →
      pairedLogits σ p (renamed σ tokens) c < pairedLogits σ p (renamed σ tokens) (σ target) := by
  apply low_loss_unique
  rw [pairedLoss_renamed σ hσ]
  exact hfit

/-- Transfer is correct for a task only under its explicit answer symmetry,
CM78. The label function is a hypothesis about data, never a model input. -/
theorem task_transfer (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (answer : Prefix O T → O) (hanswer : ∀ x, answer (renamed σ x) = σ (answer x))
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T)
    (hfit : pairedLoss σ tokens (answer tokens) p < Real.log 2) :
    ∀ c, c ≠ answer (renamed σ tokens) → pairedLogits σ p (renamed σ tokens) c <
      pairedLogits σ p (renamed σ tokens) (answer (renamed σ tokens)) := by
  rw [hanswer]
  exact renamed_unique σ hσ p tokens (answer tokens) hfit

/-- Equal observed logits force equal logits at the entire renamed prefix,
CM78. The independent unseen perturbation from CM77 is excluded on this pair. -/
theorem paired_identification (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p q : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T)
    (hsame : pairedLogits σ p tokens = pairedLogits σ q tokens) :
    pairedLogits σ p (renamed σ tokens) = pairedLogits σ q (renamed σ tokens) := by
  funext c
  have hp := pairedLogits_equivariant σ hσ p tokens (σ c)
  have hq := pairedLogits_equivariant σ hσ q tokens (σ c)
  rw [hσ c] at hp hq
  rw [hp, hq, hsame]

/-- A fixed input ties every moved target with its rename, CM78. This
condition cannot be imposed on tasks needing a unique moved answer there. -/
theorem fixed_input_loss_bound (σ : Equiv.Perm O) (hσ : Function.Involutive σ)
    (p : Parameters J O (ObjectWeights O T)) (tokens : Prefix O T) (target : O)
    (hx : renamed σ tokens = tokens) (hy : σ target ≠ target) :
    Real.log 2 ≤ pairedLoss σ tokens target p := by
  have h := pairedLogits_equivariant σ hσ p tokens target
  rw [hx] at h
  exact Rotate90.Learning.log_two_le_crossEntropy _ target (σ target) hy.symm h.symm.le

example := pairedLoss_renamed (J := Unit) (T := 2) (Equiv.refl Bool)
  (fun _ => rfl) 0 ⟨[true, false], by decide⟩ true
example := paired_identification (J := Unit) (T := 2) (Equiv.refl Bool)
  (fun _ => rfl) 0 0 ⟨[true, false], by decide⟩ rfl
example := fixed_input_loss_bound (J := Unit) (T := 2)
  (Function.Involutive.toPerm Bool.not (by intro b; cases b <;> rfl))
  (by intro b; cases b <;> rfl) 0 ⟨[], by decide⟩ false rfl (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming
