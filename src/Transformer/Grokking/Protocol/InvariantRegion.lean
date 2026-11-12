module

public import Transformer.Grokking.Protocol.CompleteState

/-!
# An all-correct region of the complete state that every shuffle keeps

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 (left), whose caption says that "it takes
close to `10⁶` steps for validation accuracy to reach that level", the level of the training
accuracy; the paper states no invariant set and nothing of the steps after. openai/grok at
3d64b1d8, `grok/training.py` and `grok/data.py`; `experiments/grok_division/PROTOCOL.md`, *The
optimizer* and *Randomness*; `review/grok_division_convergence.md`, section 5.

A proof that training reaches and keeps the held-out answers could exhibit a region of the
complete state (parameters, both moments, the count of updates) in which every held-out
equation is answered and which a run never leaves, whatever its later shuffles. Here is what
such a region is and what it gives; `EventualStay` shows that a region kept almost surely is
such a region (`stayRight_eq_one_iff`).

The parameters answer the held-out set of a split (`AnswersHeldout`). A family of sets of
states, one for the end of each epoch, is invariant and right for a split (`InvariantRight`)
if from each of its states every shuffle of the next epoch answers the held-out set at every
update of the epoch and ends in the family. A state is robust (`Robust`) if every continuation
answers the held-out set at every update. Then:

* the robust states form an invariant right family (`invariantRight_robust`), and every
  invariant right family consists of robust states (`InvariantRight.robust`): the robust
  states are the largest such family;
* a sample whose state at the end of some epoch is robust answers the held-out set at every
  later update (`Robust.answersHeldout`, `Robust.eventually_answersHeldout`), so the
  probability that a run eventually answers every held-out equation is at least the
  probability that it reaches a robust state (`law_robust_le`);
* one continuation with a wrong held-out answer shows that a state is not robust and lies in
  no invariant right family (`not_robust_of_wrong`, `InvariantRight.notMem_of_wrong`).

The model, its labels and the gradient oracle are arguments. Nothing here says that the
protocol of the paper has a robust state, nor that it has none. The witnesses are the two
models of `Witness`: every state of the one is robust, no state of the other.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter MeasureTheory

variable {ι X C : Type*} {N k : ℕ} [Fintype C] [LinearOrder C] [Nonempty C] [DecidableEq X]

/-- The parameters answer every held-out equation of the split. -/
def AnswersHeldout (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (eqs : Fin N → X) (k : ℕ)
    (π : Equiv.Perm (Fin N)) (θ : ι → ℝ) : Prop :=
  ∀ x ∈ heldout eqs k π, predict (logits θ x) = label x

variable (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (h : Hyper)
  (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X) (hk : k ≤ N) (b : ℕ)

/-- **A robust state** at the end of epoch `e`: under every sequence of later shuffles the
held-out set is answered at every update. -/
def Robust (π : Equiv.Perm (Fin N)) (e : ℕ) (s : AdamState ι) : Prop :=
  ∀ (τ : ℕ → Equiv.Perm (Fin k)) (n : ℕ),
    AnswersHeldout logits label eqs k π (continuation h lossGrad eqs hk b π e s τ n).params

/-- **An invariant right family** for the split `π`: sets of complete states, `A e` for the
end of epoch `e`, that every shuffle keeps and in which the held-out set stays answered. -/
structure InvariantRight (π : Equiv.Perm (Fin N)) (A : ℕ → Set (AdamState ι)) : Prop where
  /-- From a state of the family every shuffle of the next epoch ends in the family. -/
  step : ∀ e, ∀ s ∈ A e, ∀ τ : ℕ → Equiv.Perm (Fin k),
    continuation h lossGrad eqs hk b π e s τ (batchesPerEpoch k b) ∈ A (e + 1)
  /-- From a state of the family the held-out set is answered at every update of the epoch. -/
  right : ∀ e, ∀ s ∈ A e, ∀ (τ : ℕ → Equiv.Perm (Fin k)) (n : ℕ), n ≤ batchesPerEpoch k b →
    AnswersHeldout logits label eqs k π (continuation h lossGrad eqs hk b π e s τ n).params

variable {logits label h lossGrad eqs hk b} {π : Equiv.Perm (Fin N)}

/-- **Every state of an invariant right family is robust.** -/
theorem InvariantRight.robust (hE : 0 < batchesPerEpoch k b) {A : ℕ → Set (AdamState ι)}
    (hA : InvariantRight logits label h lossGrad eqs hk b π A) {e : ℕ} {s : AdamState ι}
    (hs : s ∈ A e) : Robust logits label h lossGrad eqs hk b π e s := fun τ n => by
  induction n using Nat.strong_induction_on generalizing e s τ with
  | _ n ih =>
    rcases le_or_gt n (batchesPerEpoch k b) with hn | hn
    · exact hA.right e s hs τ n hn
    · have hsplit : n = batchesPerEpoch k b * 1 + (n - batchesPerEpoch k b) := by omega
      rw [hsplit, continuation_mul_add hE, Nat.mul_one]
      exact ih _ (by omega) (hA.step e s hs τ) _

/-- **The robust states form an invariant right family**, the largest by
`InvariantRight.robust`. -/
theorem invariantRight_robust (hE : 0 < batchesPerEpoch k b) (π : Equiv.Perm (Fin N)) :
    InvariantRight logits label h lossGrad eqs hk b π
      fun e => {s | Robust logits label h lossGrad eqs hk b π e s} where
  right e s hs τ n _ := hs τ n
  step e s hs τ := fun τ' n => by
    have key := hs (fun j => if j = 0 then τ 0 else τ' (j - 1)) (batchesPerEpoch k b * 1 + n)
    rw [continuation_mul_add hE, Nat.mul_one,
      continuation_congr (τ' := τ) (a := 1) π e s (Nat.mul_one _).ge fun j hj => by
        rw [ite_eq_left (by omega), show j = 0 by omega]] at key
    have hshift : (fun j => if 1 + j = 0 then τ 0 else τ' (1 + j - 1)) = τ' := by
      funext j
      rw [ite_eq_right (by omega), Nat.add_sub_cancel_left]
    rwa [hshift] at key

/-- **A sample whose state at the end of an epoch is robust answers the held-out set at every
later update.** -/
theorem Robust.answersHeldout (hE : 0 < batchesPerEpoch k b) {ω : Sample N k ι} {e : ℕ}
    (hr : Robust logits label h lossGrad eqs hk b ω.1 e
      (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))) (n : ℕ) :
    AnswersHeldout logits label eqs k ω.1
      (trajectory h lossGrad eqs hk b ω (batchesPerEpoch k b * e + n)) := by
  rw [← stateAt_params, stateAt_mul_add hE]
  exact hr _ n

/-- Such a sample answers the held-out set from some update on. -/
theorem Robust.eventually_answersHeldout (hE : 0 < batchesPerEpoch k b) {ω : Sample N k ι}
    {e : ℕ} (hr : Robust logits label h lossGrad eqs hk b ω.1 e
      (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))) :
    ∀ᶠ t in atTop, AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t) :=
  eventually_atTop.2 ⟨batchesPerEpoch k b * e, fun t ht => by
    obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le ht
    exact hr.answersHeldout hE n⟩

/-- **The probability that a run eventually answers every held-out equation is at least the
probability that its state at the end of some epoch is robust.** -/
theorem law_robust_le (hE : 0 < batchesPerEpoch k b) (ν : Measure (ι → ℝ)) :
    law N k ν {ω | ∃ e, Robust logits label h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))} ≤
      law N k ν {ω | ∀ᶠ t in atTop,
        AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t)} :=
  measure_mono fun _ ⟨_, hr⟩ => hr.eventually_answersHeldout hE

/-- **A continuation with a wrong held-out answer shows that a state is not robust.** -/
theorem not_robust_of_wrong {e : ℕ} {s : AdamState ι} {τ : ℕ → Equiv.Perm (Fin k)} {n : ℕ}
    (hw : ¬AnswersHeldout logits label eqs k π
      (continuation h lossGrad eqs hk b π e s τ n).params) :
    ¬Robust logits label h lossGrad eqs hk b π e s := fun hr => hw (hr τ n)

/-- **Such a state lies in no invariant right family.** -/
theorem InvariantRight.notMem_of_wrong (hE : 0 < batchesPerEpoch k b)
    {A : ℕ → Set (AdamState ι)} (hA : InvariantRight logits label h lossGrad eqs hk b π A)
    {e : ℕ} {s : AdamState ι} {τ : ℕ → Equiv.Perm (Fin k)} {n : ℕ}
    (hw : ¬AnswersHeldout logits label eqs k π
      (continuation h lossGrad eqs hk b π e s τ n).params) : s ∉ A e :=
  fun hs => not_robust_of_wrong hw (hA.robust hE hs)

/-! The witnesses, over two equations of which one is trained on: for the model that marks the
label of its input every state is robust, so all states form an invariant right family and
every sample reaches a robust state; for the model with equal logits and the second class as
label no state is robust, as its first continuation shows. -/

/-- Every state of the model that marks the label of its input is robust. -/
lemma robust_indicator (label : Fin 2 → Bool) (π : Equiv.Perm (Fin 2)) (e : ℕ)
    (s : AdamState Unit) :
    Robust (k := 1) (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0)
      (fun _ _ _ => 0) id (by norm_num) 1 π e s :=
  fun _ _ x _ => predict_indicator_bool (label x)

example (label : Fin 2 → Bool) (π : Equiv.Perm (Fin 2)) :
    InvariantRight (ι := Unit) (k := 1) (fun _ x c => if c = label x then (1 : ℝ) else 0) label
      (Hyper.source 0) (fun _ _ _ => 0) id (by norm_num) 1 π fun _ => Set.univ :=
  ⟨fun _ _ _ _ => Set.mem_univ _, fun e s _ τ n _ => robust_indicator label π e s τ n⟩

example (label : Fin 2 → Bool) (π : Equiv.Perm (Fin 2)) (s : AdamState Unit) :=
  (invariantRight_robust (logits := fun _ x c => if c = label x then (1 : ℝ) else 0)
    (label := label) (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id)
    (hk := (by norm_num : 1 ≤ 2)) (b := 1) (by decide) π).robust (by decide)
    (robust_indicator label π 3 s)

example (label : Fin 2 → Bool) (ω : Sample 2 1 Unit) (n : ℕ) :=
  (robust_indicator label ω.1 3 _).answersHeldout (by decide) (ω := ω) n

example (label : Fin 2 → Bool) (ω : Sample 2 1 Unit) :=
  (robust_indicator label ω.1 3 _).eventually_answersHeldout (by decide) (ω := ω)

example (label : Fin 2 → Bool) := law_robust_le (k := 1)
  (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
  (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id) (hk := (by norm_num : 1 ≤ 2))
  (b := 1) (by decide) (Measure.dirac (0 : Unit → ℝ))

/-- No state of the model with equal logits is robust when the label is the second class. -/
lemma not_robust_const (π : Equiv.Perm (Fin 2)) (e : ℕ) (s : AdamState Unit) :
    ¬Robust (k := 1) (fun _ _ (_ : Bool) => (0 : ℝ)) (fun _ => true) (Hyper.source 0)
      (fun _ _ _ => 0) (id : Fin 2 → Fin 2) (by norm_num) 1 π e s :=
  not_robust_of_wrong (τ := fun _ => 1) (n := 0) fun hall => by
    obtain ⟨x, hx⟩ := heldout_nonempty (id : Fin 2 → Fin 2) (by norm_num : 1 < 2) π
    exact Bool.false_ne_true ((predict_const_bool).symm.trans (hall x hx))

example (π : Equiv.Perm (Fin 2)) (s : AdamState Unit) {A : ℕ → Set (AdamState Unit)}
    (hA : InvariantRight (k := 1) (fun _ _ (_ : Bool) => (0 : ℝ)) (fun _ => true)
      (Hyper.source 0) (fun _ _ _ => 0) (id : Fin 2 → Fin 2) (by norm_num) 1 π A) : s ∉ A 3 :=
  fun hs => not_robust_const π 3 s (hA.robust (by decide) hs)

end Transformer.Grokking.Protocol
