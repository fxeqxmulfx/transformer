module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Separation
public import Transformer.Grokking.FlatSpectrum.Recall.Basic

/-!
# Convex answer training recovers a flat kernel

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR6;
AR2-AR3 and Theorem A. Applying any fixed scalar readout to the learned
categorical logits recovers the recall kernel. For a centered readout it
is flat and has full row rank. With no zero readout values it is still
not a nuclear-norm minimum from any proper set of observed cells.
Both facts hold: structured answer training and unstructured nuclear-norm
completion are different optimization problems.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators

variable {κ ν S : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν] [DecidableEq ν]
  [Fintype S] [Nontrivial ν]

/-- Fixed scalar evaluation of the categorical logits across all queries
and dictionaries, used only to measure their spectrum (CR6). -/
noncomputable def outputKernel (φ : ν → ℂ) (A : Matrix κ κ ℝ) : Matrix κ (κ → ν) ℂ :=
  fun q D => ∑ c, (logits A q D c : ℂ) * φ c

/-- The scalar output of every global answer-CE optimizer is the actual
recall kernel, including all unseen dictionaries (CR6). -/
theorem trained_outputKernel (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A) (φ : ν → ℂ) :
    outputKernel φ A = Recall.recallKernel φ := by
  funext q D
  simp only [outputKernel, trained_logits query data hsep hA hmin, Recall.recallKernel]
  rw [Finset.sum_eq_single (D q)]
  · simp
  · intro c _ hc
    simp [Ne.symm hc]
  · simp

/-- **Ordinary convex CE training can recover a flat spectrum.** The
squared singular value is the same as AR2; the model retains every key
direction rather than imposing a low-rank bottleneck (CR6). -/
theorem trained_isFlat (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (φ : ν → ℂ) (hφ : ∑ v, φ v = 0) :
    IsFlat (outputKernel φ A)
      ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2) := by
  rw [trained_outputKernel query data hsep hA hmin]
  exact Recall.isFlat_recallKernel hφ

/-- A nonzero centered readout yields full key rank after training (CR6). -/
theorem trained_rank (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (φ : ν → ℂ) (hφ : ∑ v, φ v = 0) {v : ν} (hv : φ v ≠ 0) :
    (outputKernel φ A).rank = Fintype.card κ := by
  rw [trained_outputKernel query data hsep hA hmin]
  exact Recall.rank_recallKernel hφ hv

/-- The trained kernel simultaneously fails nuclear-norm completion from
any proper observation set when the readout has no zeros (CR6).
Thus this construction does not contradict the flat-spectrum obstruction. -/
theorem trained_not_nuclear_minimum (query : S → κ) (data : S → κ → ν)
    (hsep : Separates query data) {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (hmin : IsMinOn (answerLoss query data) {B | IsStochastic B} A)
    (φ : ν → ℂ) (hφ : ∑ v, φ v = 0) (hv : ∀ v, φ v ≠ 0)
    {Ω : Set (κ × (κ → ν))} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = outputKernel φ A ij.1 ij.2}
      (outputKernel φ A) := by
  rw [trained_outputKernel query data hsep hA hmin]
  exact Recall.not_isMinOn_recallKernel hφ hv hΩ

/-- Boolean signed readout: centered, nonzero, and all training conditions
are realized by the two queried bindings of the identity dictionary. -/
example :
    IsFlat (outputKernel (fun v : Bool => if v then (1 : ℂ) else -1)
      (1 : Matrix Bool Bool ℝ)) 4 := by
  have h := trained_isFlat (id : Bool → Bool) (fun _ => id) separates_bool_id
    (stochastic_one (κ := Bool)) (isMinOn_answerLoss_one _ _)
    (fun v : Bool => if v then (1 : ℂ) else -1) (by simp)
  norm_num at h ⊢
  exact h

example := trained_outputKernel (id : Bool → Bool) (fun _ => id) separates_bool_id
  (stochastic_one (κ := Bool)) (isMinOn_answerLoss_one _ _) (fun v => if v then (1 : ℂ) else -1)

example := trained_rank (id : Bool → Bool) (fun _ => id) separates_bool_id
  (stochastic_one (κ := Bool)) (isMinOn_answerLoss_one _ _)
  (fun v => if v then (1 : ℂ) else -1) (by simp) (v := true) (by simp)

example := trained_not_nuclear_minimum (id : Bool → Bool) (fun _ => id) separates_bool_id
  (stochastic_one (κ := Bool)) (isMinOn_answerLoss_one _ _)
  (fun v => if v then (1 : ℂ) else -1) (by simp) (by intro v; cases v <;> norm_num)
  (Ω := ∅) (by
    intro h
    have hmem : (true, (id : Bool → Bool)) ∈ (∅ : Set (Bool × (Bool → Bool))) :=
      h.symm ▸ Set.mem_univ _
    exact hmem)

end Transformer.Grokking.FlatSpectrum.ConvexRecall
