module

public import Transformer.Grokking.FiniteRecovery.Argand
public import Transformer.Grokking.FiniteRecovery.FlatSpectrum

/-! The bridge from Argand's final token probabilities to the phase tables
of review/flat_spectrum.md. The loss bound concerns the full forward pass at
each input, and applies to any sequence of trained weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Filter
open scoped BigOperators Topology

/-- A phase prediction is the expectation of the token code at the output. -/
def phaseExpectation {A : Type*} [Fintype A] (code : A → ℂ) (prob : A → ℝ) : ℂ :=
  ∑ a, (prob a : ℂ) * code a

/-- The remaining token mass bounds the phase error for any code in the closed unit disk. -/
theorem phaseError_le_wrongMass {A : Type*} [Fintype A] [DecidableEq A]
    (prob : A → ℝ) (code : A → ℂ) (target : A)
    (hnonneg : ∀ a, 0 ≤ prob a) (hmass : ∑ a, prob a = 1)
    (hcode : ∀ a, ‖code a‖ ≤ 1) :
    ‖phaseExpectation code prob - code target‖ ≤ 2 * (1 - prob target) := by
  have hmassC : ∑ a, (prob a : ℂ) = 1 := by exact_mod_cast hmass
  have hcenter : (∑ a, (prob a : ℂ) * (code a - code target)) =
      phaseExpectation code prob - code target := by
    simp only [phaseExpectation, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul,
      hmassC, one_mul]
  have herase : (∑ a ∈ Finset.univ.erase target, (prob a : ℂ) * (code a - code target)) =
      ∑ a, (prob a : ℂ) * (code a - code target) := by
    simpa only [sub_self, mul_zero, add_zero] using
      Finset.sum_erase_add Finset.univ (fun a => (prob a : ℂ) * (code a - code target))
        (Finset.mem_univ target)
  have hremaining := Finset.sum_erase_add Finset.univ prob (Finset.mem_univ target)
  calc
    ‖phaseExpectation code prob - code target‖
        = ‖∑ a ∈ Finset.univ.erase target, (prob a : ℂ) * (code a - code target)‖ := by
            rw [herase, hcenter]
    _ ≤ ∑ a ∈ Finset.univ.erase target, ‖(prob a : ℂ) * (code a - code target)‖ :=
          norm_sum_le _ _
    _ ≤ ∑ a ∈ Finset.univ.erase target, prob a * 2 := by
          apply Finset.sum_le_sum
          intro a _
          rw [norm_mul, Complex.norm_real, Real.norm_of_nonneg (hnonneg a)]
          apply mul_le_mul_of_nonneg_left _ (hnonneg a)
          calc
            ‖code a - code target‖ ≤ ‖code a‖ + ‖code target‖ := norm_sub_le _ _
            _ ≤ 2 := by linarith only [hcode a, hcode target]
    _ = 2 * (1 - prob target) := by
          rw [← Finset.sum_mul]
          linarith

/-- Cross-entropy bounds phase error at the same labeled input. -/
theorem phaseError_le_crossEntropy {A : Type*} [Fintype A] [DecidableEq A]
    (prob : A → ℝ) (code : A → ℂ) (target : A)
    (hnonneg : ∀ a, 0 ≤ prob a) (hmass : ∑ a, prob a = 1)
    (hunit : ∀ a, ‖code a‖ = 1) (htarget : 0 < prob target) :
    ‖phaseExpectation code prob - code target‖ ≤ -2 * Real.log (prob target) := by
  apply (phaseError_le_wrongMass prob code target hnonneg hmass (fun a => (hunit a).le)).trans
  linarith [Real.log_le_sub_one_of_pos htarget]

namespace Argand

variable {A U V : Type*} {T H d layers : ℕ} [Fintype A]

/-- Index the last-token distribution of the full Argand model by a word cut. -/
noncomputable def Model.phaseKernel (model : Model A H d layers) (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) : Matrix U V ℂ :=
  Matrix.of fun u v =>
    phaseExpectation code (model.probability repeats (cut (u, v)) (Fin.last T))

/-- Every entry of the complete model's phase table is controlled by its
next-token loss on the corresponding input. -/
theorem Model.phaseError_le_labelLoss [DecidableEq A] [Nonempty A]
    (model : Model A H d layers) (repeats : ℕ) (cut : U × V ≃ Word A (T + 1))
    (code : A → ℂ) (hunit : ∀ a, ‖code a‖ = 1)
    (target : Word A (T + 1) → A) (u : U) (v : V) :
    ‖model.phaseKernel repeats cut code u v - targetKernel cut code target u v‖ ≤
      2 * model.labelLoss repeats (cut (u, v)) (target (cut (u, v))) := by
  let word := cut (u, v)
  let prob := model.probability repeats word (Fin.last T)
  have hbound := phaseError_le_crossEntropy prob code (target word)
    (fun token => (model.probability_pos repeats word (Fin.last T) token).le)
    (model.sum_probability repeats word (Fin.last T)) hunit
    (model.probability_pos repeats word (Fin.last T) (target word))
  simpa only [Model.phaseKernel, targetKernel, Matrix.of_apply, Model.labelLoss, mul_neg, neg_mul]
    using hbound

/-- When the loss on a particular pair tends to zero, its Argand phase
prediction tends to the target phase. The same statement applies to hidden
and observed cells. -/
theorem phaseKernel_tendsto_of_labelLoss [DecidableEq A] [Nonempty A]
    (models : ℕ → Model A H d layers) (repeats : ℕ) (cut : U × V ≃ Word A (T + 1))
    (code : A → ℂ) (hunit : ∀ a, ‖code a‖ = 1)
    (target : Word A (T + 1) → A) (u : U) (v : V)
    (hloss : Tendsto (fun step =>
      (models step).labelLoss repeats (cut (u, v)) (target (cut (u, v)))) atTop (𝓝 0)) :
    Tendsto (fun step => (models step).phaseKernel repeats cut code u v) atTop
      (𝓝 (targetKernel cut code target u v)) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  apply squeeze_zero (fun _ => norm_nonneg _)
    (fun step => (models step).phaseError_le_labelLoss repeats cut code hunit target u v)
  simpa only [mul_zero] using (tendsto_const_nhds (x := (2 : ℝ))).mul hloss

/-- A fixed labeled pair has vanishing loss if the mean observed loss vanishes. -/
theorem labelLoss_tendsto_of_observedLoss [Nonempty A]
    (models : ℕ → Model A H d layers) (repeats : ℕ)
    {part : Finset (Word A (T + 1) × A)} {pair : Word A (T + 1) × A} (hp : pair ∈ part)
    (hfit : Tendsto (fun step => (models step).observedLoss repeats part) atTop (𝓝 0)) :
    Tendsto (fun step => (models step).labelLoss repeats pair.1 pair.2) atTop (𝓝 0) := by
  have hupper := (tendsto_const_nhds (x := (part.card : ℝ))).mul hfit
  rw [mul_zero] at hupper
  exact squeeze_zero (fun step => (models step).labelLoss_nonneg repeats pair.1 pair.2)
    (fun step => (models step).labelLoss_le_card_mul_observedLoss repeats part hp) hupper

/-- Fitting the observed labels and controlling the nuclear norm identifies
the full Argand phase limit whenever Flat Spectrum gives unique completion. -/
theorem phaseLimit_eq_target_of_nuclearRecovers [DecidableEq A] [Nonempty A]
    [Fintype U] [Fintype V] [DecidableEq V]
    (models : ℕ → Model A H d layers) (repeats : ℕ) (cut : U × V ≃ Word A (T + 1))
    (code : A → ℂ) (hunit : ∀ a, ‖code a‖ = 1)
    (target : Word A (T + 1) → A) (inputs : Finset (Word A (T + 1)))
    (limit : Matrix U V ℂ)
    (hlimit : ∀ u v, Tendsto (fun step => (models step).phaseKernel repeats cut code u v)
      atTop (𝓝 (limit u v)))
    (hfit : Tendsto (fun step => (models step).observedLoss repeats (observe inputs target)) atTop
      (𝓝 0))
    (hnorm : FlatSpectrum.nuclearNorm limit ≤
      FlatSpectrum.nuclearNorm (targetKernel cut code target))
    (hunique : NuclearRecovers (observedCells cut inputs) (targetKernel cut code target)) :
    limit = targetKernel cut code target := by
  have hfeasible : limit ∈ completions (observedCells cut inputs) (targetKernel cut code target) := by
    rintro ⟨u, v⟩ hcell
    have hp : (cut (u, v), target (cut (u, v))) ∈ observe inputs target :=
      mem_observe.mpr ⟨hcell, rfl⟩
    have hloss := labelLoss_tendsto_of_observedLoss models repeats hp hfit
    exact tendsto_nhds_unique (hlimit u v)
      (phaseKernel_tendsto_of_labelLoss models repeats cut code hunit target u v hloss)
  have hmin := (hunique _ (fun _ _ => rfl)).mpr rfl
  apply (hunique limit hfeasible).mp
  intro candidate hcand
  exact hnorm.trans (hmin hcand)

example (model : Model (Fin 2) 4 16 1) (target : Word (Fin 2) 2 → Fin 2)
    (left right : Word (Fin 2) 1) :=
  model.phaseError_le_labelLoss 1 (appendCut (Fin 2) 1 1) (fun _ => 1)
    (fun _ => norm_one) target left right

private theorem singleton_labelLoss (model : Model (Fin 1) 1 2 1) (word : Word (Fin 1) 2) :
    model.labelLoss 1 word 0 = 0 := by
  have hprob : model.probability 1 word (Fin.last 1) 0 = 1 := by
    simpa only [Fin.sum_univ_one] using model.sum_probability 1 word (Fin.last 1)
  rw [Model.labelLoss, hprob, Real.log_one, neg_zero]

example (model : Model (Fin 1) 1 2 1) (left right : Word (Fin 1) 1) :=
  phaseKernel_tendsto_of_labelLoss (fun _ => model) 1 (appendCut (Fin 1) 1 1)
    (fun _ => 1) (fun _ => norm_one) (fun _ => 0) left right (by
      simpa only [singleton_labelLoss] using (tendsto_const_nhds (x := (0 : ℝ))))

example (model : Model (Fin 1) 1 2 1) :
    let cut := appendCut (Fin 1) 1 1
    let truth := targetKernel cut (fun _ => 1) (fun _ => 0)
    model.phaseKernel 1 cut (fun _ => 1) = truth := by
  dsimp only
  let cut := appendCut (Fin 1) 1 1
  let truth := targetKernel cut (fun _ => 1) (fun _ => 0)
  have heq : model.phaseKernel 1 cut (fun _ => 1) = truth := by
    ext u v
    have h := model.phaseError_le_labelLoss 1 cut (fun _ => 1) (fun _ => norm_one) (fun _ => 0) u v
    rw [singleton_labelLoss, mul_zero, norm_le_zero_iff, sub_eq_zero] at h
    exact h
  apply phaseLimit_eq_target_of_nuclearRecovers (fun _ => model) 1 cut (fun _ => 1)
    (fun _ => norm_one) (fun _ => 0) Finset.univ _ (fun _ _ => tendsto_const_nhds)
  · have hzero : ∀ a : Fin 1, a = 0 := fun a => Subsingleton.elim a _
    simpa only [Model.observedLoss, hzero, singleton_labelLoss, Finset.sum_const_zero, zero_div]
      using (tendsto_const_nhds (x := (0 : ℝ)))
  · rw [heq]
  · rw [observedCells_univ]
    exact nuclearRecovers_univ _

end Argand
end Transformer.Grokking.FiniteRecovery
