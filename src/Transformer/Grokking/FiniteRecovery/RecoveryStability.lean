module

public import Transformer.Grokking.FiniteRecovery.RecoveryHypergraph
public import Transformer.Grokking.FiniteRecovery.CompletionStability

/-! Uniform stability on a minimum observation mask. The target may
vary with the approximation index. Finiteness of the target family
makes the tolerance independent of the individual function. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set Filter FlatSpectrum
open scoped Topology

variable {U V I A : Type*} {T : ℕ} [Fintype U] [Fintype V] [DecidableEq V]

/-- A common positive lower bound, including for an empty index type. -/
theorem exists_pos_le_all [Finite I] (a : I → ℝ) (ha : ∀ i, 0 < a i) :
    ∃ b > 0, ∀ i, b ≤ a i := by
  classical
  let := Fintype.ofFinite I
  let S := insert 1 (Finset.univ.image a)
  have hS : S.Nonempty := Finset.insert_nonempty _ _
  refine ⟨S.min' hS, ?_, fun i => Finset.min'_le S _ (Finset.mem_insert_of_mem
    (Finset.mem_image_of_mem a (Finset.mem_univ i)))⟩
  have hm := Finset.min'_mem S hS
  rcases Finset.mem_insert.mp hm with h | h
  · rw [h]; exact zero_lt_one
  · obtain ⟨i, _, hi⟩ := Finset.mem_image.mp h
    exact hi ▸ ha i

/-- For a finite indexed class the transversal condition is equivalent
to uniform stability for approximate observations and approximate nuclear
minimization. All entries, including unobserved entries, are controlled. -/
theorem isTransversal_iff_uniform_stability [Finite I] (K : I → Matrix U V ℂ)
    (mask : Set (U × V)) :
    IsTransversal (recoveryHypergraph (range K) univ nuclearNorm) mask ↔
      ∀ ε > (0 : ℝ), ∃ δ > 0, ∀ i (X : Matrix U V ℂ),
        (∀ c ∈ mask, ‖X c.1 c.2 - K i c.1 c.2‖ ≤ δ) →
        nuclearNorm X ≤ nuclearNorm (K i) + δ → ∀ u v, ‖X u v - K i u v‖ < ε := by
  rw [isTransversal_iff_nuclearRecovers]
  constructor
  · intro hr ε hε
    choose δ hδ hclose using fun i => (hr _ (mem_range_self i)).exists_error_threshold hε
    obtain ⟨b, hb, hbδ⟩ := exists_pos_le_all δ hδ
    exact ⟨b, hb, fun i X ho hn => hclose i X (fun c hc => (ho c hc).trans (hbδ i))
      (hn.trans (add_le_add_right (hbδ i) _))⟩
  · rintro hs _ ⟨i,rfl⟩
    apply nuclearRecovers_iff_sublevel.mpr
    intro X hX hnorm
    ext u v
    by_contra hne
    have hp := norm_pos_iff.mpr (sub_ne_zero.mpr hne)
    obtain ⟨δ, hδ, hclose⟩ := hs _ hp
    exact (lt_irrefl _ (hclose i X (fun c hc => by rw [hX c hc, sub_self, norm_zero]; exact hδ.le)
      (hnorm.trans (le_add_of_nonneg_right hδ.le)) u v))

/-- Vanishing observed error and vanishing upper nuclear excess imply
uniform convergence over all cells, even if the target changes each step. -/
theorem uniform_completion_of_transversal [Finite I] (K : I → Matrix U V ℂ)
    {mask : Set (U × V)} (hm : IsTransversal (recoveryHypergraph (range K) univ nuclearNorm) mask)
    (target : ℕ → I) (X : ℕ → Matrix U V ℂ) (err : ℕ → ℝ) (he : Tendsto err atTop (𝓝 0))
    (ho : ∀ t c, c ∈ mask → ‖X t c.1 c.2 - K (target t) c.1 c.2‖ ≤ err t)
    (hn : ∀ t, nuclearNorm (X t) ≤ nuclearNorm (K (target t)) + err t) :
    ∀ ε > (0 : ℝ), ∀ᶠ t in atTop, ∀ u v, ‖X t u v - K (target t) u v‖ < ε := by
  intro ε hε
  obtain ⟨δ, hδ, hc⟩ := (isTransversal_iff_uniform_stability K mask).mp hm ε hε
  exact (he.eventually (gt_mem_nhds hδ)).mono fun t ht => hc (target t) (X t)
    (fun c h => (ho t c h).trans ht.le) ((hn t).trans (add_le_add_right ht.le _))

/-- Passing from cells to observed graph pairs preserves the exact
minimum. The measure of sample size is the number of distinct words
revealed. The same mask must serve every function of the class. -/
theorem isLeast_phase_observations [Fintype A] (cut : U × V ≃ Word A T) (code : A → ℂ)
    (family : Set (Word A T → A)) :
    IsLeast {m : ℕ | ∃ inputs : Finset (Word A T), inputs.card = m ∧
      ∀ f ∈ family, NuclearRecovers (observedCells cut inputs) (targetKernel cut code f)}
      (transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm)) := by
  classical
  let targets := targetKernel cut code '' family
  have h := isLeast_nuclear_observations targets
  constructor
  · obtain ⟨cells, hc, hh⟩ := h.1
    refine ⟨cells.map cut.toEmbedding, (Finset.card_map _).trans hc, fun f hf => ?_⟩
    have heq : observedCells cut (cells.map cut.toEmbedding) = (cells : Set (U × V)) := by
      ext c
      simp [observedCells]
    exact heq ▸ hh _ (mem_image_of_mem _ hf)
  · rintro _ ⟨inputs,rfl,hh⟩
    apply h.2
    refine ⟨inputs.map cut.symm.toEmbedding, Finset.card_map _, ?_⟩
    rintro _ ⟨f,hf,rfl⟩
    have heq : (↑(inputs.map cut.symm.toEmbedding) : Set (U × V)) = observedCells cut inputs := by
      ext c
      simp [observedCells]
    exact heq ▸ hh f hf

/-- Injective label codes convert a transversal into exact recovery of
the token function by a single observation-based learner for the class. -/
theorem exists_learner_of_transversal [Nonempty A] (cut : U × V ≃ Word A T) (code : A → ℂ)
    (hc : Function.Injective code) (family : Set (Word A T → A)) (inputs : Finset (Word A T))
    (hm : IsTransversal (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm)
      (observedCells cut inputs)) : ∃ learner, Recovers inputs family learner :=
  exists_learner_of_nuclear_recovers cut code hc inputs family fun f hf =>
    (isTransversal_iff_nuclearRecovers _ _).mp hm (targetKernel cut code f) (mem_image_of_mem _ hf)

/-- A minimum nuclear-recovery mask for an injectively coded class must
also distinguish the functions as label vectors. This bounds class size
by the alphabet size to the transversal number. -/
theorem card_family_le_pow_transversalNumber [Fintype A] [Nonempty A]
    (cut : U × V ≃ Word A T) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A T → A)) :
    family.ncard ≤ Fintype.card A ^
      transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ nuclearNorm) := by
  classical
  obtain ⟨inputs, hcard, hh⟩ := (isLeast_phase_observations cut code family).1
  have hinj := (exists_learner_iff_injOn inputs family).mp
    (exists_learner_of_nuclear_recovers cut code hc inputs family hh)
  let data (f : family) (w : inputs) := f.1 w.1
  have hd : Function.Injective data := by
    intro f g he
    exact Subtype.ext (hinj f.2 g.2 (observe_eq_iff_eqOn.mpr fun w hw => congrFun he ⟨w,hw⟩))
  rw [← Nat.card_coe_set_eq, Nat.card_eq_fintype_card, ← hcard]
  simpa only [Fintype.card_fun, Fintype.card_coe] using Fintype.card_le_of_injective data hd

/-- For the class of all functions on a nontrivial alphabet the whole
word domain is necessary and sufficient. Structural restrictions enter
through the choice of the family and its phase tables. -/
theorem unrestricted_transversal_iff [Fintype A] [Nontrivial A] (cut : U × V ≃ Word A T)
    (code : A → ℂ) (hc : Function.Injective code) (inputs : Finset (Word A T)) :
    IsTransversal (recoveryHypergraph (targetKernel cut code '' univ) univ nuclearNorm)
      (observedCells cut inputs) ↔ inputs = Finset.univ := by
  constructor
  · intro hm
    apply Finset.eq_univ_of_forall
    intro w
    by_contra hw
    exact no_unrestricted_recovery inputs (Fintype.one_lt_card_iff_nontrivial.mpr inferInstance)
      ⟨w,hw⟩ (exists_learner_of_transversal cut code hc univ inputs hm)
  · rintro rfl
    rw [observedCells_univ, isTransversal_iff_nuclearRecovers]
    exact fun K _ => nuclearRecovers_univ K

example (cut : Fin 2 × Fin 2 ≃ Word (Fin 2) 2) (code : Fin 2 → ℂ) :=
  isLeast_phase_observations cut code (univ : Set (Word (Fin 2) 2 → Fin 2))

end Transformer.Grokking.FiniteRecovery
