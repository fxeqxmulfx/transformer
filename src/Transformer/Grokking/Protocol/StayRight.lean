module

public import Transformer.Grokking.Protocol.AccuracyMeasurable
public import Transformer.Grokking.Protocol.HistoryFiltration
public import Transformer.Grokking.Protocol.InvariantRegion

/-!
# Along a run that is eventually right the probability of staying right tends to one

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 (left); the paper states no probability
and nothing of the steps after its curves end. openai/grok at 3d64b1d8,
`grok/training.py` and `grok/data.py`; `experiments/grok_division/PROTOCOL.md`, *The
optimizer* and *Randomness*; `review/grok_division_convergence.md`, section 5.

At the end of an epoch the complete state (parameters, both moments, the count of updates)
and the split fix the law of what follows, since the later shuffles are fresh
(`HistoryFiltration`). The probability of staying right (`stayRight`) is the probability, over
those shuffles, that the held-out set is answered at every later update; it is `1` at a robust
state of `InvariantRegion`. For a model whose logits and whose gradient oracle are measurable
in the parameters, and every law of the initial parameters:

    for almost every sample that from some update on answers every held-out equation, the
    probability of staying right from its state at the end of epoch `e` tends to `1`

(`ae_tendsto_stayRight`). So almost every run that is eventually right has, for every `δ < 1`,
from some epoch on a state from which it stays right with probability above `δ`. The proof is
the zero-one law of Lévy for the events `rightFrom e`, "right at every update from the end of
epoch `e` on" (`measurableSet_rightFrom`), which grow with `e`
(`rightFrom_mono`): resumed after epoch `e` with fresh shuffles, a sample is in `rightFrom e`
exactly if the continuation of its state stays right (`resume_mem_rightFrom_iff`).

Nothing here says that a run is eventually right with positive probability, nor that the
probability of staying right is positive at any state of the paper's protocol. `EventualStay`
draws the consequences for the probabilities.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

variable {ι X C : Type*} {N k : ℕ} [Fintype C] [LinearOrder C] [Nonempty C] [DecidableEq X]
  (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (h : Hyper)
  (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X) (hk : k ≤ N) (b : ℕ)

/-- The samples that answer the held-out set at every update from the end of epoch `e` on. -/
def rightFrom (e : ℕ) : Set (Sample N k ι) :=
  {ω | ∀ n, AnswersHeldout logits label eqs k ω.1
    (trajectory h lossGrad eqs hk b ω (batchesPerEpoch k b * e + n))}

/-- **The probability of staying right** from the complete state `s` at the end of epoch `e`:
the probability, over fresh shuffles of the epochs from `e` on, that the held-out set is
answered at every update. -/
noncomputable def stayRight (π : Equiv.Perm (Fin N)) (e : ℕ) (s : AdamState ι) : ENNReal :=
  Measure.infinitePi (fun _ : ℕ => uniformPerm k) {τ | ∀ n,
    AnswersHeldout logits label eqs k π (continuation h lossGrad eqs hk b π e s τ n).params}

variable {logits label h lossGrad eqs hk b}

/-- The events `rightFrom e` grow with `e`. -/
lemma rightFrom_mono {e₀ e : ℕ} (he : e₀ ≤ e) :
    rightFrom logits label h lossGrad eqs hk b e₀ ⊆ rightFrom logits label h lossGrad eqs hk b e :=
  fun ω hω n => by
    have hsum : batchesPerEpoch k b * e + n =
        batchesPerEpoch k b * e₀ + (batchesPerEpoch k b * (e - e₀) + n) := by
      rw [← Nat.add_assoc, ← Nat.mul_add, Nat.add_sub_cancel' he]
    rw [hsum]
    exact hω _

/-- The probability of staying right from a robust state is `1`. -/
lemma stayRight_eq_one_of_robust {π : Equiv.Perm (Fin N)} {e : ℕ} {s : AdamState ι}
    (hr : Robust logits label h lossGrad eqs hk b π e s) :
    stayRight logits label h lossGrad eqs hk b π e s = 1 := by
  rw [stayRight, show {τ : ℕ → Equiv.Perm (Fin k) | ∀ n, AnswersHeldout logits label eqs k π
    (continuation h lossGrad eqs hk b π e s τ n).params} = Set.univ from
      Set.eq_univ_of_forall fun τ n => hr τ n, measure_univ]

/-- **A sample resumed after epoch `e` with fresh shuffles is right from the end of that epoch
on exactly if the continuation of its state stays right.** -/
theorem resume_mem_rightFrom_iff (hE : 0 < batchesPerEpoch k b) (e : ℕ) (ω : Sample N k ι)
    (τ : ℕ → Equiv.Perm (Fin k)) :
    resume e ω τ ∈ rightFrom logits label h lossGrad eqs hk b e ↔
      ∀ n, AnswersHeldout logits label eqs k ω.1 (continuation h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e)) τ n).params := by
  have hstate : stateAt h lossGrad eqs hk b (resume e ω τ) (batchesPerEpoch k b * e) =
      stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e) :=
    stateAt_congr (a := e) le_rfl rfl rfl fun j hj => by simp [resume, hj]
  have hlater : (fun j => (resume e ω τ).2.2 (e + j)) = τ := by
    funext j
    simp [resume]
  refine forall_congr' fun n => ?_
  rw [← stateAt_params, stateAt_mul_add hE, hstate, hlater]
  rfl

/-- The probability of staying right from the state of a sample at the end of epoch `e` is the
probability that the sample resumed with fresh shuffles is right from there on. -/
lemma stayRight_stateAt (hE : 0 < batchesPerEpoch k b) (e : ℕ) (ω : Sample N k ι) :
    stayRight logits label h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e)) =
      Measure.infinitePi (fun _ : ℕ => uniformPerm k)
        {τ | resume e ω τ ∈ rightFrom logits label h lossGrad eqs hk b e} := by
  rw [stayRight]
  congr 1
  exact Set.ext fun τ => (resume_mem_rightFrom_iff hE e ω τ).symm

variable (hlogits : ∀ x c, Measurable fun θ => logits θ x c)
  (hL : ∀ B i, Measurable fun θ => lossGrad B θ i)
include hlogits hL

/-- The samples that answer the held-out set after `t` updates form an event. -/
theorem measurableSet_answersHeldout (t : ℕ) :
    MeasurableSet {ω : Sample N k ι | AnswersHeldout logits label eqs k ω.1
      (trajectory h lossGrad eqs hk b ω t)} := by
  have htraj : Measurable fun ω : Sample N k ι => trajectory h lossGrad eqs hk b ω t :=
    measurable_pi_iff.2 fun i => measurable_trajectory h hL eqs hk b t i
  have hset : {ω : Sample N k ι | AnswersHeldout logits label eqs k ω.1
      (trajectory h lossGrad eqs hk b ω t)} = ⋃ π : Equiv.Perm (Fin N), (Prod.fst ⁻¹' {π}) ∩
        ⋂ x ∈ heldout eqs k π,
          {ω | predict (logits (trajectory h lossGrad eqs hk b ω t) x) = label x} := by
    ext ω
    simp [AnswersHeldout]
  rw [hset]
  exact MeasurableSet.iUnion fun π => (measurable_fst MeasurableSpace.measurableSet_top).inter
    (Finset.measurableSet_biInter _ fun x _ =>
      measurableSet_predict_eq (fun c => (hlogits x c).comp htraj) (label x))

/-- **The samples that are right from the end of epoch `e` on form an event.** -/
theorem measurableSet_rightFrom (e : ℕ) :
    MeasurableSet (rightFrom logits label h lossGrad eqs hk b e) := by
  rw [rightFrom, Set.ofPred_forall]
  exact MeasurableSet.iInter fun n => measurableSet_answersHeldout hlogits hL _

/-- **For almost every sample that is eventually right, the probability of staying right from
its state at the end of epoch `e` tends to `1`.** -/
theorem ae_tendsto_stayRight (hE : 0 < batchesPerEpoch k b) (ν : Measure (ι → ℝ))
    [IsProbabilityMeasure ν] :
    ∀ᵐ ω ∂law N k ν, (∀ᶠ t in atTop,
        AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t)) →
      Tendsto (fun e => stayRight logits label h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))) atTop (𝓝 1) := by
  filter_upwards [ae_all_iff.2 fun e₀ => ae_tendsto_measure_resume ν
    (measurableSet_rightFrom (h := h) (eqs := eqs) (hk := hk) (b := b) hlogits hL e₀)]
    with ω hω hev
  obtain ⟨T, hT⟩ := eventually_atTop.1 hev
  have hmem : ω ∈ rightFrom logits label h lossGrad eqs hk b T := fun n =>
    hT _ ((Nat.le_mul_of_pos_left T hE).trans (Nat.le_add_right _ _))
  have hlim := hω T
  rw [Set.indicator_of_mem hmem] at hlim
  have hreal : Tendsto (fun e => (stayRight logits label h lossGrad eqs hk b ω.1 e
      (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))).toReal) atTop (𝓝 1) := by
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hlim tendsto_const_nhds ?_
      (Eventually.of_forall fun e => ENNReal.toReal_le_of_le_ofReal zero_le_one
        (by rw [ENNReal.ofReal_one, stayRight]; exact prob_le_one))
    filter_upwards [eventually_ge_atTop T] with e he
    rw [stayRight_stateAt hE]
    exact ENNReal.toReal_mono (measure_ne_top _ _)
      (measure_mono fun τ hτ => rightFrom_mono he hτ)
  have hofReal := ENNReal.tendsto_ofReal hreal
  rw [ENNReal.ofReal_one] at hofReal
  refine hofReal.congr fun e => ENNReal.ofReal_toReal ?_
  rw [stayRight]
  exact measure_ne_top _ _

omit hlogits hL

/-! The witness: the model of `Witness` that marks the label of its input, over two equations of
which one is trained on. Every sample is right at every update, every state is robust, and the
probability of staying right is `1` at every state. -/

example (label : Fin 2 → Bool) (π : Equiv.Perm (Fin 2)) (s : AdamState Unit) :
    stayRight (k := 1) (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0)
      (fun _ _ _ => 0) id (by norm_num) 1 π 3 s = 1 :=
  stayRight_eq_one_of_robust (robust_indicator label π 3 s)

example (label : Fin 2 → Bool) (ω : Sample 2 1 Unit) :
    ω ∈ rightFrom (k := 1) (fun _ x c => if c = label x then (1 : ℝ) else 0) label
      (Hyper.source 0) (fun _ _ _ => 0) id (by norm_num) 1 3 :=
  rightFrom_mono (Nat.zero_le 3) fun _ x _ => predict_indicator_bool (label x)

example (label : Fin 2 → Bool) (ω : Sample 2 1 Unit) (τ : ℕ → Equiv.Perm (Fin 1)) :=
  resume_mem_rightFrom_iff (logits := fun _ x c => if c = label x then (1 : ℝ) else 0)
    (label := label) (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id)
    (hk := (by norm_num : 1 ≤ 2)) (b := 1) (by decide) 3 ω τ

example (label : Fin 2 → Bool) := measurableSet_rightFrom (ι := Unit) (N := 2) (k := 1)
  (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
  (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id) (hk := (by norm_num : 1 ≤ 2))
  (b := 1) (fun _ _ => measurable_const) (fun _ _ => measurable_const) 3

example (label : Fin 2 → Bool) := ae_tendsto_stayRight (ι := Unit) (N := 2) (k := 1)
  (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
  (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id) (hk := (by norm_num : 1 ≤ 2))
  (b := 1) (fun _ _ => measurable_const) (fun _ _ => measurable_const) (by decide)
  (Measure.dirac (0 : Unit → ℝ))

end Transformer.Grokking.Protocol
