module

public import Transformer.Grokking.FiniteRecovery.RecoveryStability
public import Transformer.Grokking.FiniteRecovery.ArgandMomentumLimits
public import Mathlib.Topology.Algebra.Order.Archimedean

/-! Rational displacements of the full Argand weight vector. Strict
realization is open because the logits are continuous and the word
domain is finite. This gives a finite-index realizing proposal from
every native initialization whenever the architecture admits one. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Set
open scoped Topology Classical

variable {A : Type*} [Fintype A] {T H d layers : ℕ}

/-- A fixed enumeration of rational changes to all trainable coordinates,
translated by the starting weights. The enumeration is independent of labels. -/
noncomputable def rationalCandidate (start : Weights A H d layers) (i : ℕ) : Weights A H d layers :=
  let q := (exists_surjective_nat (Parameter A H d layers → ℚ)).choose i
  fun p => start p + (q p : ℝ)

theorem isOpen_realizingWeights (repeats : ℕ) (f : Word A (T + 1) → A) :
    IsOpen {w : Weights A H d layers | (Model.fromWeights w).realizes repeats f} := by
  simp only [Model.realizes, ofPred_forall]
  refine isOpen_iInter_of_finite fun word => isOpen_iInter_of_finite fun a => ?_
  by_cases h : a = f word
  · simp [h]
  · simp only [h, ne_eq, not_false_eq_true, iInter_true]
    exact isOpen_lt (continuous_logits_fromWeights repeats word _ a)
      (continuous_logits_fromWeights repeats word _ (f word))

/-- Strict realizability by weights of the specified native architecture
is equivalent to success of some finite-index rational perturbation of
any specified start. -/
theorem exists_rationalCandidate_iff (start : Weights A H d layers) (repeats : ℕ)
    (f : Word A (T + 1) → A) :
    (∃ i, (Model.fromWeights (rationalCandidate start i)).realizes repeats f) ↔
    ∃ w : Weights A H d layers, (Model.fromWeights w).realizes repeats f := by
  refine ⟨fun ⟨i,hi⟩ => ⟨_,hi⟩, ?_⟩
  rintro ⟨w, hw⟩
  let G := {v : Weights A H d layers | (Model.fromWeights (start + v)).realizes repeats f}
  have ho : IsOpen G := (isOpen_realizingWeights repeats f).preimage (continuous_const.add continuous_id)
  have hn : G.Nonempty := ⟨w - start, by simpa [G] using hw⟩
  let cast : (Parameter A H d layers → ℚ) → Weights A H d layers := fun q p => (q p : ℝ)
  have hd : DenseRange cast := DenseRange.piMap (fun _ : Parameter A H d layers => Rat.denseRange_cast)
  obtain ⟨_, ⟨q,rfl⟩, hq⟩ := Dense.exists_mem_open hd ho hn
  obtain ⟨i,hi⟩ := (exists_surjective_nat (Parameter A H d layers → ℚ)).choose_spec q
  change (Model.fromWeights (fun p => start p + (q p : ℝ))).realizes repeats f at hq
  exact ⟨i, by simpa only [rationalCandidate, hi] using hq⟩

/-- A single positive logit margin works on the full word domain at any
strictly realizing finite weight vector. -/
theorem exists_uniform_logit_margin (w : Weights A H d layers) (repeats : ℕ)
    (f : Word A (T + 1) → A) (hw : (Model.fromWeights w).realizes repeats f) :
    ∃ γ > (0 : ℝ), ∀ x a, a ≠ f x → γ ≤ (Model.fromWeights w).logits repeats x (Fin.last T) (f x) -
      (Model.fromWeights w).logits repeats x (Fin.last T) a := by
  let P := {p : Word A (T + 1) × A // p.2 ≠ f p.1}
  let gap (p : P) := (Model.fromWeights w).logits repeats p.1.1 (Fin.last T) (f p.1.1) -
    (Model.fromWeights w).logits repeats p.1.1 (Fin.last T) p.1.2
  obtain ⟨γ, hp, hγ⟩ := exists_pos_le_all gap (fun p => sub_pos.mpr (hw _ _ p.2))
  exact ⟨γ,hp,fun x a ha => hγ ⟨(x,a),ha⟩⟩

end Transformer.Grokking.FiniteRecovery.Argand
