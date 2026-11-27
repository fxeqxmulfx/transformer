module

public import Transformer.Grokking.FiniteRecovery.ArgandRationalSearch

/-! Selection of complete Argand weight vectors using only observed labels
and the nuclear norm of the candidate's own token predictions. Ties
in cost retain the earlier weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set FlatSpectrum
open scoped Classical

variable {W A U V : Type*} {T H d layers : ℕ}

/-- An online search keeps a feasible candidate of least cost among the
initial vector and all proposals seen so far. -/
noncomputable def keepBest (start : W) (candidates : ℕ → W) (fits : W → Prop) (cost : W → ℝ) : ℕ → W
  | 0 => start
  | t + 1 =>
    let w := keepBest start candidates fits cost t
    let v := candidates t
    if fits v ∧ (¬ fits w ∨ cost v < cost w) then v else w

/-- Once a global optimum occurs in the proposal sequence, the retained
weights are constant and have optimal cost from that iteration onward. -/
theorem keepBest_settles (start : W) (candidates : ℕ → W) (fits : W → Prop) (cost : W → ℝ)
    (i : ℕ) (hi : fits (candidates i)) (hmin : ∀ v, fits v → cost (candidates i) ≤ cost v) :
    ∃ w, fits w ∧ cost w = cost (candidates i) ∧ ∀ t ≥ i + 1, keepBest start candidates fits cost t = w := by
  let run := keepBest start candidates fits cost
  have step (t) : run (t+1) = if fits (candidates t) ∧ (¬ fits (run t) ∨ cost (candidates t) < cost (run t))
      then candidates t else run t := by rfl
  have ho : fits (run (i+1)) ∧ cost (run (i+1)) = cost (candidates i) := by
    rw [step]; split
    · exact ⟨hi, rfl⟩
    · rename_i h
      have hp : fits (run i) := by by_contra hn; exact h ⟨hi, Or.inl hn⟩
      exact ⟨hp, le_antisymm (le_of_not_gt (fun hh => h ⟨hi,Or.inr hh⟩)) (hmin _ hp)⟩
  refine ⟨run (i+1), ho.1, ho.2, ?_⟩
  intro t ht
  change run t = _
  induction t, ht using Nat.le_induction with
  | base => rfl
  | succ t _ ih =>
    rw [step, ih]
    apply ite_eq_right
    rintro ⟨hv, h | h⟩
    · exact (h ho.1).elim
    · rw [ho.2] at h
      exact (hmin _ hv).not_gt h

variable [Fintype A] [Nonempty A]

/-- The token of greatest logit with a fixed selection at ties. -/
noncomputable def Model.predictedToken (model : Model A H d layers) (repeats : ℕ)
    (word : Word A (T + 1)) : A :=
  (Finset.exists_max_image Finset.univ (model.logits repeats word (Fin.last T)) Finset.univ_nonempty).choose

theorem Model.predictedToken_eq_of_realizes (model : Model A H d layers) (repeats : ℕ)
    {f : Word A (T + 1) → A} (h : model.realizes repeats f) : model.predictedToken repeats = f := by
  funext x
  by_contra hn
  have hmax := (Finset.exists_max_image Finset.univ (model.logits repeats x (Fin.last T))
    Finset.univ_nonempty).choose_spec.2 (f x) (Finset.mem_univ _)
  exact (h x _ hn).not_ge hmax

/-- Feasibility checks strictness on predictions and matches the given
graph pairs. Only supplied observations are read as labels. -/
def fitsObserved (repeats : ℕ) (part : Finset (Word A (T + 1) × A)) (w : Weights A H d layers) : Prop :=
  let m := Model.fromWeights w
  m.realizes (T := T) repeats (m.predictedToken repeats) ∧ ∀ p ∈ part, m.predictedToken repeats p.1 = p.2

/-- The penalty is the nuclear norm of the candidate's decoded phase
table, evaluated on words without looking up hidden labels. -/
noncomputable def predictedNuclearNorm [Fintype U] [Fintype V] [DecidableEq V] (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (w : Weights A H d layers) : ℝ :=
  nuclearNorm (targetKernel cut code ((Model.fromWeights w).predictedToken repeats))

/-- Phase tables of functions strictly represented by native weights at
the fixed widths, layer count and repeat count. -/
def nativePhaseTables (A : Type*) [Fintype A] (H d layers repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) : Set (Matrix U V ℂ) :=
  {X | ∃ (w : Weights A H d layers) (g : Word A (T+1) → A),
    (Model.fromWeights w).realizes repeats g ∧ X = targetKernel cut code g}

/-- For feasible weights, achieving the optimal target cost is equivalent
to strict realization. Every feasible proposal has cost at least that of
the truth, including proposals whose token functions lie outside the class. -/
theorem fitsObserved_nuclear_optimum [Fintype U] [Fintype V] [DecidableEq V] (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hc : Function.Injective code)
    (inputs : Finset (Word A (T + 1))) (f : Word A (T + 1) → A)
    (hr : ∀ X ∈ nativePhaseTables A H d layers repeats cut code,
      X ∈ completions (observedCells cut inputs) (targetKernel cut code f) →
      nuclearNorm X ≤ nuclearNorm (targetKernel cut code f) → X = targetKernel cut code f)
    (w : Weights A H d layers) (hw : fitsObserved repeats (observe inputs f) w) :
    nuclearNorm (targetKernel cut code f) ≤ predictedNuclearNorm repeats cut code w ∧
      (predictedNuclearNorm repeats cut code w ≤ nuclearNorm (targetKernel cut code f) ↔
        (Model.fromWeights w).realizes repeats f) := by
  let g := (Model.fromWeights w).predictedToken (T := T) repeats
  have hfit : targetKernel cut code g ∈ completions (observedCells cut inputs) (targetKernel cut code f) := by
    rintro ⟨u,v⟩ h
    exact congrArg code (hw.2 (cut (u,v), f (cut (u,v))) (mem_observe.mpr ⟨h,rfl⟩))
  have htable : targetKernel cut code g ∈ nativePhaseTables A H d layers repeats cut code := ⟨w,g,hw.1,rfl⟩
  have hlo : nuclearNorm (targetKernel cut code f) ≤ nuclearNorm (targetKernel cut code g) := by
    by_contra! hlt
    exact hlt.ne (congrArg nuclearNorm (hr _ htable hfit hlt.le))
  refine ⟨hlo, ?_, ?_⟩
  · intro hle
    have he := hr _ htable hfit hle
    have hgf : g = f := by
      funext w
      obtain ⟨⟨u,v⟩,rfl⟩ := cut.surjective w
      exact hc (congrArg (fun X : Matrix U V ℂ => X u v) he)
    exact hgf ▸ hw.1
  · intro h
    exact le_of_eq (congrArg (fun g => nuclearNorm (targetKernel cut code g))
      ((Model.fromWeights w).predictedToken_eq_of_realizes repeats h))

end Transformer.Grokking.FiniteRecovery.Argand
