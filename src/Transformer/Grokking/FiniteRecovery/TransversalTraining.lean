module

public import Transformer.Grokking.FiniteRecovery.RecoveryStability
public import Transformer.Grokking.FiniteRecovery.FactorConvergence

/-! Training on a minimum transversal. The factor solver receives the
observed phase values with zeros elsewhere. A common outer index gives
strict nearest-token decisions for all words and all targets of the
finite class. Inner descent is the stacked-identity run in
Flat Spectrum, Training/SquaredRun. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set Filter Matrix FlatSpectrum FlatSpectrum.Training
open scoped Topology

variable {U V A : Type*} {T : ℕ} [Fintype U] [Fintype V] [DecidableEq U] [DecidableEq V]
  [Fintype A]

omit [Fintype U] [Fintype V] [DecidableEq U] [DecidableEq V] in
private theorem strict_phase_eventually (cut : U × V ≃ Word A T) (code : A → ℂ)
    (hc : Function.Injective code) (f : Word A T → A) (X : ℕ → Matrix U V ℂ)
    (h : Tendsto X atTop (𝓝 (targetKernel cut code f))) :
    ∀ᶠ j in atTop, ∀ w b, b ≠ f w →
      ‖X j (cut.symm w).1 (cut.symm w).2 - code (f w)‖ <
        ‖X j (cut.symm w).1 (cut.symm w).2 - code b‖ := by
  apply eventually_all.mpr
  intro w
  apply eventually_all.mpr
  intro b
  by_cases he : b = f w
  · exact Eventually.of_forall fun _ hn => (hn he).elim
  have ht := tendsto_pi_nhds.mp (tendsto_pi_nhds.mp h (cut.symm w).1) (cut.symm w).2
  have hz : targetKernel cut code f (cut.symm w).1 (cut.symm w).2 = code (f w) := by
    simp [targetKernel]
  rw [hz] at ht
  have hd := (ht.sub_const (code (f w))).norm
  rw [sub_self, norm_zero] at hd
  have hp := norm_pos_iff.mpr (sub_ne_zero.mpr (fun heq => he (hc heq).symm))
  exact (hd.eventually_lt (ht.sub_const (code b)).norm hp).mono fun _ hlt _ => hlt

/-- There is one minimum mask and one solver on the observed data for
the entire family. Each outer stage is a convergent factor descent at
positive decay. At all sufficiently late stages each correct token
is strictly closer than every competing token on every word. -/
theorem exists_minimum_transversal_factor_training (cut : U × V ≃ Word A T)
    (code : A → ℂ) (hc : Function.Injective code) (family : Set (Word A T → A)) :
    ∃ inputs : Finset (Word A T),
      inputs.card = transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm) ∧
      ∃ (us : Matrix U V ℂ → ℕ → Matrix U (U ⊕ V) ℂ) (vs : Matrix U V ℂ → ℕ → Matrix V (U ⊕ V) ℂ),
        let Ω := observedCells cut inputs
        let K := targetKernel cut code
        let data f := mask Ω (K f)
        let P f j := us (data f) j * (vs (data f) j)ᴴ
        (∀ f ∈ family, ObservedFactorRun Ω (K f) decaySchedule (us (data f)) (vs (data f)) ∧
          Tendsto (P f) atTop (𝓝 (K f))) ∧
        ∀ᶠ j in atTop, ∀ f ∈ family, ∀ w b, b ≠ f w →
          ‖P f j (cut.symm w).1 (cut.symm w).2 - code (f w)‖ <
            ‖P f j (cut.symm w).1 (cut.symm w).2 - code b‖ := by
  classical
  obtain ⟨inputs, hcard, hh⟩ := (isLeast_phase_observations cut code family).1
  let Ω := observedCells cut inputs
  let K := targetKernel cut code
  let data f := mask Ω (K f)
  choose us vs hrun using fun obs : Matrix U V ℂ =>
    exists_observedFactorRun Ω obs decaySchedule decaySchedule_pos
  refine ⟨inputs, hcard, us, vs, ?_, ?_⟩
  · intro f hf
    have hmask : mask Ω (data f) = data f := by
      ext u v
      by_cases h : (u,v) ∈ Ω <;> simp [data, mask, h]
    have hr : ObservedFactorRun Ω (K f) decaySchedule (us (data f)) (vs (data f)) := by
      intro j
      obtain ⟨hmin, η, hη, hs⟩ := hrun (data f) j
      rw [hmask] at hs
      exact ⟨fitLoss_mask_labels Ω (K f) ▸ hmin, η, hη, hs⟩
    exact ⟨hr, tendsto_mul_conjTranspose_of_forall_isMinOn (hh f hf) decaySchedule_pos
      decaySchedule_tendsto (fun j => (hr j).1)⟩
  · have he : ∀ᶠ j in atTop, ∀ f : family, ∀ w b, b ≠ f.1 w →
        let x := (us (data f.1) j * (vs (data f.1) j)ᴴ) (cut.symm w).1 (cut.symm w).2
        ‖x - code (f.1 w)‖ < ‖x - code b‖ := by
      apply eventually_all.mpr
      intro f
      apply strict_phase_eventually cut code hc f.1 _
      apply tendsto_mul_conjTranspose_of_forall_isMinOn (hh f.1 f.2) decaySchedule_pos decaySchedule_tendsto
      intro j
      exact fitLoss_mask_labels Ω (K f.1) ▸ (hrun (data f.1) j).1
    exact he.mono fun j hj f hf => hj ⟨f,hf⟩

example := exists_minimum_transversal_factor_training (appendCut (ZMod 2) 1 1)
  ZMod.stdAddChar ZMod.injective_stdAddChar (univ : Set (Word (ZMod 2) 2 → ZMod 2))

end Transformer.Grokking.FiniteRecovery
