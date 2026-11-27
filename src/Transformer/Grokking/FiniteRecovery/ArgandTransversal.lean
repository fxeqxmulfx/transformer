module

public import Transformer.Grokking.FiniteRecovery.ArgandNuclearSearch

/-! End-to-end recovery with full Argand weights from a native tensor
draw. The search uses observed graph pairs only. At the minimum sample
size for nuclear completion, native strict representability is the exact
remaining condition for the search to stabilize at correct weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set Filter FlatSpectrum
open scoped Topology

variable {A U V : Type*} [Fintype A] [Nonempty A] [Fintype U] [Fintype V] [DecidableEq V]
  {T H d layers : ℕ}

/-- The weights are constant from N onward, converge to that same vector,
and every correct token has a logit gap of at least γ there. -/
def SettlesWithMargin (weights : ℕ → Weights A H d layers) (repeats : ℕ)
    (f : Word A (T + 1) → A) (N : ℕ) (γ : ℝ) : Prop :=
  ∃ w : Weights A H d layers, (∀ t ≥ N, weights t = w) ∧ Tendsto weights atTop (𝓝 w) ∧
    ∀ x a, a ≠ f x → γ ≤ (Model.fromWeights w).logits repeats x (Fin.last T) (f x) -
      (Model.fromWeights w).logits repeats x (Fin.last T) a

/-- One finite stopping bound and one positive logit margin suffice for
every function in the finite class at this draw. The converse witnesses
strict native representability at the actual final weights. -/
theorem transversalSearch_settles_iff (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A (T + 1) → A)) (inputs : Finset (Word A (T + 1)))
    (candidates : Set (Matrix U V ℂ)) (hcover : nativePhaseTables A H d layers repeats cut code ⊆ candidates)
    (hm : IsTransversal (recoveryHypergraph (targetKernel cut code '' family) candidates nuclearNorm)
      (observedCells cut inputs)) (draw : ComplexParameter A H d layers → ℂ) :
    (∃ N γ, 0 < γ ∧ ∀ f ∈ family, SettlesWithMargin
      (transversalSearch realScoreInit repeats cut code (observe inputs f) lam rate draw) repeats f N γ) ↔
    ∀ f ∈ family, ∃ w : Weights A H d layers, (Model.fromWeights w).realizes repeats f := by
  constructor
  · rintro ⟨N,γ,hγ,hs⟩ f hf
    obtain ⟨w,_,_,hgap⟩ := hs f hf
    exact ⟨w, fun x a ha => sub_pos.mp (hγ.trans_le (hgap x a ha))⟩
  · intro hrep
    let run (f : family) := transversalSearch realScoreInit repeats cut code (observe inputs f.1) lam rate draw
    have hfit (f : family) : ∃ w : Weights A H d layers,
        (∀ᶠ t in atTop, run f t = w) ∧ (Model.fromWeights w).realizes repeats f.1 := by
      apply nuclearSearch_settles _ _ repeats cut code hc inputs f.1 _ (hrep f.1 f.2)
      intro X hX hfit hn
      exact (isTransversal_iff_unique_completion _ _ _ _).mp hm _ (mem_image_of_mem _ f.2)
        X (hcover hX) hfit hn
    choose star hstop hcorrect using hfit
    choose margin hpos hgap using fun f : family => exists_uniform_logit_margin (star f) repeats f.1 (hcorrect f)
    obtain ⟨γ,hγ,hsmall⟩ := exists_pos_le_all margin hpos
    obtain ⟨N,hN⟩ := eventually_atTop.mp (eventually_all.mpr hstop)
    refine ⟨N,γ,hγ,fun f hf => ?_⟩
    let i : family := ⟨f,hf⟩
    exact ⟨star i, fun t ht => hN t ht i, tendsto_const_nhds.congr' ((hstop i).mono fun _ h => h.symm),
      fun x a ha => (hsmall i).trans (hgap i x a ha)⟩

/-- The class-wide minimum number of Flat Spectrum observations suffices
for an explicitly specified full Argand search from every native draw.
Widths, depth and repeat count are fixed before the target and mask. -/
theorem exists_minimum_transversal_argand (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A (T + 1) → A))
    (hrep : ∀ f ∈ family, ∃ w : Weights A H d layers, (Model.fromWeights w).realizes repeats f) :
    ∃ inputs : Finset (Word A (T + 1)),
      inputs.card = transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm) ∧
      (∀ f ∈ family, NuclearRecovers (observedCells cut inputs) (targetKernel cut code f)) ∧
      ∀ draw : ComplexParameter A H d layers → ℂ, ∃ N γ, 0 < γ ∧ ∀ f ∈ family, SettlesWithMargin
        (transversalSearch realScoreInit repeats cut code (observe inputs f) lam rate draw) repeats f N γ := by
  obtain ⟨inputs, hcard, hh⟩ := (isLeast_phase_observations cut code family).1
  have hm : IsTransversal (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm)
      (observedCells cut inputs) :=
    (isTransversal_iff_nuclearRecovers _ _).2 (by rintro _ ⟨f,hf,rfl⟩; exact hh f hf)
  exact ⟨inputs, hcard, hh, fun draw => (transversalSearch_settles_iff realScoreInit repeats lam rate
    cut code hc family inputs univ (subset_univ _) hm draw).2 hrep⟩

/-- Restricting competitors to strictly representable native tables gives
a second exact minimum for unique nuclear selection over this architecture.
Every smaller mask admits a target with another consistent native table
of equal or lower cost. -/
theorem minimum_native_transversal_argand (realScoreInit : Bool) (repeats : ℕ) (lam : ℝ) (rate : ℕ → ℝ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A (T + 1) → A))
    (hrep : ∀ f ∈ family, ∃ w : Weights A H d layers, (Model.fromWeights w).realizes repeats f) :
    let G := recoveryHypergraph (targetKernel cut code '' family) (nativePhaseTables A H d layers repeats cut code) nuclearNorm
    transversalNumber G ≤ transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm) ∧
    ∃ cells : Finset (U × V), cells.card = transversalNumber G ∧ IsTransversal G cells ∧
      ∀ draw : ComplexParameter A H d layers → ℂ, ∃ N γ, 0 < γ ∧ ∀ f ∈ family, SettlesWithMargin
        (transversalSearch realScoreInit repeats cut code (observe (inputsAtCells cut cells) f) lam rate draw) repeats f N γ := by
  classical
  let targets := targetKernel cut code '' family
  let native := nativePhaseTables A H d layers repeats cut code
  have least := isLeast_recovery_transversal targets native nuclearNorm
  obtain ⟨cells, hn, hm⟩ := least.1
  constructor
  · obtain ⟨full, hcard, hh⟩ := (isLeast_nuclear_observations targets).1
    refine least.2 ⟨full,hcard,(isTransversal_iff_unique_completion _ _ _ _).2 ?_⟩
    exact fun K hK X _ hfit hle => nuclearRecovers_iff_sublevel.mp (hh K hK) X hfit hle
  · refine ⟨cells,hn,hm,fun draw => ?_⟩
    apply (transversalSearch_settles_iff realScoreInit repeats lam rate cut code hc family _
      native (Subset.rfl) _ draw).2 hrep
    simpa only [observedCells_inputsAtCells] using hm

end Transformer.Grokking.FiniteRecovery.Argand
