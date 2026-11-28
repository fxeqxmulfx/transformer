module

public import Transformer.Grokking.FiniteRecovery.ArgandEqualMoment
public import Transformer.Grokking.FiniteRecovery.AdamWRelabeling

/-! Alphabet invariance of the ordinary equal-decay optimizer. Permuting
the tied embedding and readout permutes every iterate and preserves the
entire distribution of first strict hit times on any finite evaluation grid. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.EqualMoment

variable {I J : Type*} [Fintype I] [Fintype J]

/-- The two moving averages, global clipping and decoupled shrinkage
commute with every reindexing of trainable coordinates. -/
theorem uncorrectedRun_reindex (e : I ≃ J) (β γ ε decay : ℝ) (rate : ℕ → ℝ)
    (g : ℕ → (I → ℝ) → I → ℝ) (h : ℕ → (J → ℝ) → J → ℝ)
    (hg : ∀ t w, h t (w ∘ e.symm) = g t w ∘ e.symm) (w : I → ℝ) (n : ℕ) :
    uncorrectedRun β γ ε decay rate h (w ∘ e.symm) n = (uncorrectedRun β γ ε decay rate g w n).reindex e := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [uncorrectedRun,ih,AdamW.State.reindex,hg,AdamW.clip_reindex,Function.comp_apply]
    rfl

theorem run_reindex (e : I ≃ J) (β ε decay : ℝ) (rate : ℕ → ℝ)
    (g : ℕ → (I → ℝ) → I → ℝ) (h : ℕ → (J → ℝ) → J → ℝ)
    (hg : ∀ t w, h t (w ∘ e.symm) = g t w ∘ e.symm) (w : I → ℝ) (n : ℕ) :
    run β ε decay rate h (w ∘ e.symm) n = (run β ε decay rate g w n).reindex e :=
  uncorrectedRun_reindex e β β ε decay rate g h hg w n

example (e : Fin 7 ≃ Fin 7) (w : Fin 7 → ℝ) (t : ℕ) :=
  run_reindex e (9/10) (1/100000000) (1/10) (fun _ => 3/1000)
    (fun _ x => x) (fun _ x => x) (fun _ _ => rfl) w t

end Transformer.Grokking.FiniteRecovery.EqualMoment

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set
open scoped NNReal

variable {A B X : Type*} [Fintype A] [Fintype B] {H d L T : ℕ}

/-- The predicted score is unchanged when the prompt, answer and batches
are transported with the native initialization, at every position and update. -/
theorem uncorrectedAdam_logits_relabel (e : A ≃ B) (r t : ℕ) (first second ε decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (z : ComplexParameter A H d L → ℂ)
    (x : Word A T) (i : Fin T) (a : A) :
    (Model.fromWeights (uncorrectedAdamRun r first second ε decay (fun k => (batch k).map (relabelPosition e))
      rate (relabelDraw e z) t).weights).logits r (e ∘ x) i (e a) =
      (Model.fromWeights (uncorrectedAdamRun r first second ε decay batch rate z t).weights).logits r x i a := by
  unfold uncorrectedAdamRun
  rw [initialWeights_relabel]
  unfold relabelWeights
  rw [EqualMoment.uncorrectedRun_reindex (parameterTokenEquiv e) first second ε decay rate
    (fun k => batchGradient r (batch k)) _ (fun k => batchGradient_relabel e r (batch k))]
  change (Model.fromWeights (relabelWeights e _)).logits _ _ _ _ = _
  rw [fromWeights_relabel,Model.logits_relabel]

theorem equalMoment_logits_relabel (e : A ≃ B) (r t : ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (z : ComplexParameter A H d L → ℂ)
    (x : Word A T) (i : Fin T) (a : A) :
    (Model.fromWeights (equalMomentRun r decay (fun k => (batch k).map (relabelPosition e))
      rate (relabelDraw e z) t).weights).logits r (e ∘ x) i (e a) =
      (Model.fromWeights (equalMomentRun r decay batch rate z t).weights).logits r x i a :=
  uncorrectedAdam_logits_relabel e r t (9/10) (9/10) (1/100000000) decay rate batch z x i a

noncomputable def equalMomentHit (r : ℕ) (grid : Finset ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (z : ComplexParameter A H d L → ℂ) : Option ℕ :=
  firstPerfect grid fun t => StrictOn (fun x => (Model.fromWeights (equalMomentRun r decay batch rate z t).weights).logits
    r (prompt x) (Fin.last T)) f

/-- On coupled draws the first successful checkpoint agrees exactly,
including simultaneous failure at all steps on the grid. -/
theorem equalMomentHit_relabel (e : A ≃ B) (r : ℕ) (grid : Finset ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (z : ComplexParameter A H d L → ℂ) :
    equalMomentHit r grid decay rate (fun k => (batch k).map (relabelPosition e))
      (fun x => e ∘ prompt x) (e ∘ f) (relabelDraw e z) = equalMomentHit r grid decay rate batch prompt f z := by
  apply congrArg (firstPerfect grid)
  funext t
  apply propext
  constructor
  · intro h x a ha
    simpa only [Function.comp_apply,equalMoment_logits_relabel] using h x (e a) (fun he => ha (e.injective he))
  · intro h x b hb
    obtain ⟨a,rfl⟩ := e.surjective b
    simpa only [Function.comp_apply,equalMoment_logits_relabel] using h x a (fun ha => hb (congrArg e ha))

/-- For any interval of update counts, or failure by the cutoff, the
probability is the same in every encoding of the original task. The batch
stream may depend on the draw if it is transported together with the data. -/
theorem equalMoment_hit_probability_relabel (e : A ≃ B) (r : ℕ) (grid : Finset ℕ)
    (decay : ℝ) (rate : ℕ → ℝ) (batch : (ComplexParameter A H d L → ℂ) → ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (variance : ComplexParameter A H d L → ℝ≥0)
    (times : Set (Option ℕ)) :
    independentGaussianDraws (variance ∘ (tensorTokenEquiv e).symm)
      {y | equalMomentHit r grid decay rate (fun k => (batch (relabelDraw e.symm y) k).map (relabelPosition e))
        (fun x => e ∘ prompt x) (e ∘ f) y ∈ times} =
    independentGaussianDraws variance {z | equalMomentHit r grid decay rate (batch z) prompt f z ∈ times} := by
  rw [gaussian_relabel_preimage]
  apply congrArg (independentGaussianDraws variance)
  ext z
  simp only [mem_preimage,mem_ofPred_eq,relabelDraw_inverse,equalMomentHit_relabel]

open Division.Factors

example (e : Equiv.Perm (Fin 239)) (S : Set DivisionInput)
    (batch : (ComplexParameter (Fin 239) 4 16 6 → ℂ) → ℕ → List (LabeledPosition (Fin 239))) :=
  equalMoment_hit_probability_relabel e 1 (evaluationGrid 256 20000) (1/10) (fun t => 3*wsdRate 20000 t)
    batch (fun x : (Sᶜ : Set DivisionInput) => divisionPrompt x) (fun x => divisionAnswer x)
    (fun _ => (1/50 : ℝ≥0)^2) {none}

end Transformer.Grokking.FiniteRecovery.Argand
