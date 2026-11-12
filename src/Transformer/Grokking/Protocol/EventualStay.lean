module

public import Transformer.Grokking.Protocol.StayRight
public import Transformer.Grokking.Protocol.Uncovered

/-!
# Eventual correctness has positive probability exactly if states that stay right are reached

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; the paper states no probability.
openai/grok at 3d64b1d8, `grok/training.py` and `grok/data.py`;
`experiments/grok_division/PROTOCOL.md`, *The optimizer* and *Randomness*;
`review/grok_division_convergence.md`, section 5.

`StayRight` gives every complete state at the end of an epoch its probability of staying
right, over fresh shuffles. For a model whose logits and whose gradient oracle are measurable
in the parameters, and every law of the initial parameters:

* the probability that a run is right from the end of epoch `e` on is the mean of the
  probability of staying right from its state at that end (`measureReal_rightFrom`);
* if a run is eventually right with positive probability, then for every `δ < 1` it reaches,
  with positive probability, a state from which it stays right with probability above `δ`
  (`exists_lt_stayRight`);
* if with positive probability it reaches a state from which it stays right with positive
  probability, then it is eventually right with positive probability
  (`law_eventually_ne_zero_of_stayRight`).

So "eventually right with positive probability" and "a region of the complete state that is
kept with positive probability, or with probability above any `δ < 1`, is reached with
positive probability" are one statement. The region kept by every shuffle is the smaller one
of `InvariantRegion`: the probability of staying right is `1` exactly at the robust states
(`stayRight_eq_one_iff`), since a continuation with a wrong answer is shared by a set of
shuffle sequences of positive probability.

Nothing here says that any of these probabilities is positive for the paper's protocol.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

variable {ι X C : Type*} {N k : ℕ} [Fintype C] [LinearOrder C] [Nonempty C] [DecidableEq X]
  {logits : (ι → ℝ) → X → C → ℝ} {label : X → C} {h : Hyper}
  {lossGrad : List X → (ι → ℝ) → ι → ℝ} {eqs : Fin N → X} {hk : k ≤ N} {b : ℕ}

/-- **The probability of staying right is `1` exactly at the robust states.** -/
theorem stayRight_eq_one_iff (hE : 0 < batchesPerEpoch k b) {π : Equiv.Perm (Fin N)} {e : ℕ}
    {s : AdamState ι} :
    stayRight logits label h lossGrad eqs hk b π e s = 1 ↔
      Robust logits label h lossGrad eqs hk b π e s := by
  refine ⟨fun hone => ?_, stayRight_eq_one_of_robust⟩
  by_contra hnot
  simp only [Robust, not_forall] at hnot
  obtain ⟨τ₀, n₀, hw⟩ := hnot
  set c := Measure.infinitePi fun _ : ℕ => uniformPerm k with hc
  have hcyl : c (Set.pi (Finset.range n₀) fun j => {τ₀ j}) ≠ 0 := by
    rw [hc, Measure.infinitePi_pi _ fun _ _ => MeasurableSpace.measurableSet_top]
    exact Finset.prod_ne_zero_iff.2 fun j _ => (uniformPerm_pos_of_mem (Set.mem_singleton _)).ne'
  have hsub : {τ : ℕ → Equiv.Perm (Fin k) | ∀ n, AnswersHeldout logits label eqs k π
      (continuation h lossGrad eqs hk b π e s τ n).params} ⊆
      (Set.pi (Finset.range n₀) fun j => {τ₀ j})ᶜ := fun τ hτ hmem => by
    refine hw ?_
    rw [← continuation_congr (τ := τ) (a := n₀) π e s (Nat.le_mul_of_pos_left n₀ hE)
      fun j hj => Set.mem_singleton_iff.1 (hmem j (Finset.mem_coe.2 (Finset.mem_range.2 hj)))]
    exact hτ n₀
  have hlt : c (Set.pi (Finset.range n₀) fun j => {τ₀ j})ᶜ < 1 := by
    rw [prob_compl_eq_one_sub (MeasurableSet.pi (Finset.countable_toSet _)
      fun _ _ => MeasurableSpace.measurableSet_top)]
    exact ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hcyl
  exact (((measure_mono hsub).trans_lt hlt).ne) hone

variable (hlogits : ∀ x c, Measurable fun θ => logits θ x c)
  (hL : ∀ B i, Measurable fun θ => lossGrad B θ i) (hE : 0 < batchesPerEpoch k b)
  (ν : Measure (ι → ℝ)) [IsProbabilityMeasure ν]
include hlogits hL hE

/-- **The probability that a run is right from the end of epoch `e` on is the mean of the
probability of staying right from its state at that end.** -/
theorem measureReal_rightFrom (e : ℕ) :
    (law N k ν).real (rightFrom logits label h lossGrad eqs hk b e) =
      ∫ ω, (stayRight logits label h lossGrad eqs hk b ω.1 e
        (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))).toReal ∂law N k ν := by
  simp only [stayRight_stateAt hE]
  exact (integral_measure_resume ν (measurableSet_rightFrom hlogits hL e) e).symm

/-- **If a run is eventually right with positive probability, then for every `δ < 1` it
reaches with positive probability a state from which it stays right with probability above
`δ`.** -/
theorem exists_lt_stayRight
    (hpos : law N k ν {ω | ∀ᶠ t in atTop,
      AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t)} ≠ 0)
    {δ : ENNReal} (hδ : δ < 1) :
    ∃ e, law N k ν {ω | δ < stayRight logits label h lossGrad eqs hk b ω.1 e
      (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))} ≠ 0 := by
  by_contra hnot
  push Not at hnot
  refine hpos (measure_eq_zero_iff_ae_notMem.2 ?_)
  filter_upwards [ae_tendsto_stayRight hlogits hL hE ν,
    measure_eq_zero_iff_ae_notMem.1 (measure_iUnion_null hnot)] with ω hω hmem hev
  obtain ⟨e, he⟩ := ((hω hev).eventually (lt_mem_nhds hδ)).exists
  exact hmem (Set.mem_iUnion.2 ⟨e, he⟩)

/-- **If with positive probability a run reaches a state from which it stays right with
positive probability, then it is eventually right with positive probability.** -/
theorem law_eventually_ne_zero_of_stayRight {e : ℕ}
    (hpos : law N k ν {ω | stayRight logits label h lossGrad eqs hk b ω.1 e
      (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e)) ≠ 0} ≠ 0) :
    law N k ν {ω | ∀ᶠ t in atTop,
      AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t)} ≠ 0 := by
  intro hzero
  have hG := measurableSet_rightFrom (label := label) (h := h) (eqs := eqs) (hk := hk) (b := b)
    hlogits hL e
  have hsub : rightFrom logits label h lossGrad eqs hk b e ⊆ {ω | ∀ᶠ t in atTop,
      AnswersHeldout logits label eqs k ω.1 (trajectory h lossGrad eqs hk b ω t)} :=
    fun ω hω => eventually_atTop.2 ⟨batchesPerEpoch k b * e, fun t ht => by
      obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le ht
      exact hω n⟩
  have hreal := measureReal_rightFrom (label := label) (h := h) (eqs := eqs) (hk := hk)
    hlogits hL hE ν e
  rw [measureReal_def, measure_mono_null hsub hzero, ENNReal.toReal_zero] at hreal
  have hf : StronglyMeasurable ((rightFrom logits label h lossGrad eqs hk b e).indicator
      fun _ : Sample N k ι => (1 : ℝ)) := stronglyMeasurable_const.indicator hG
  have hint : Integrable (fun ω : Sample N k ι => (stayRight logits label h lossGrad eqs hk b
      ω.1 e (stateAt h lossGrad eqs hk b ω (batchesPerEpoch k b * e))).toReal) (law N k ν) :=
    (integrable_condExp (μ := law N k ν) (m := historyFiltration N k ι e)).congr <| by
      filter_upwards [condExp_history ν e hf ((integrable_const 1).indicator hG)] with ω hω
      rw [hω, integral_indicator_resume hG, stayRight_stateAt hE]
      rfl
  have hae := (integral_eq_zero_iff_of_nonneg (fun _ => ENNReal.toReal_nonneg) hint).1
    hreal.symm
  refine hpos (measure_eq_zero_iff_ae_notMem.2 ?_)
  filter_upwards [hae] with ω hω hne
  refine hne (((ENNReal.toReal_eq_zero_iff _).1 hω).resolve_right ?_)
  rw [stayRight]
  exact measure_ne_top _ _

omit hlogits hL hE

/-! The witnesses, over two equations of which one is trained on. For the model that marks the
label of its input the probability of staying right is `1` at every state, and every run is
right from the start. For the model with equal logits and the second class as label no state
is robust, so the probability of staying right is not `1` anywhere. -/

example (π : Equiv.Perm (Fin 2)) (s : AdamState Unit) :
    stayRight (k := 1) (fun _ _ (_ : Bool) => (0 : ℝ)) (fun _ => true) (Hyper.source 0)
      (fun _ _ _ => 0) (id : Fin 2 → Fin 2) (by norm_num) 1 π 3 s ≠ 1 :=
  fun hone => not_robust_const π 3 s ((stayRight_eq_one_iff (by decide)).1 hone)

example (label : Fin 2 → Bool) := measureReal_rightFrom (ι := Unit) (N := 2) (k := 1)
  (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
  (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id) (hk := (by norm_num : 1 ≤ 2))
  (b := 1) (fun _ _ => measurable_const) (fun _ _ => measurable_const) (by decide)
  (Measure.dirac (0 : Unit → ℝ)) 3

/-- The hypothesis of `law_eventually_ne_zero_of_stayRight` holds for the model that marks the
label: every sample is at a state from which it stays right with probability `1`. -/
lemma law_stayRight_indicator_ne_zero (label : Fin 2 → Bool) :
    law 2 1 (Measure.dirac (0 : Unit → ℝ)) {ω | stayRight (k := 1)
      (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0) (fun _ _ _ => 0)
      id (by norm_num) 1 ω.1 3 (stateAt (Hyper.source 0) (fun _ _ _ => 0) id (by norm_num) 1 ω
        (batchesPerEpoch 1 1 * 3)) ≠ 0} ≠ 0 := by
  rw [show {ω : Sample 2 1 Unit | stayRight (k := 1)
      (fun _ x c => if c = label x then (1 : ℝ) else 0) label (Hyper.source 0) (fun _ _ _ => 0)
      id (by norm_num) 1 ω.1 3 (stateAt (Hyper.source 0) (fun _ _ _ => 0) id (by norm_num) 1 ω
        (batchesPerEpoch 1 1 * 3)) ≠ 0} = Set.univ from Set.eq_univ_of_forall fun ω => by
    rw [Set.mem_ofPred_eq, stayRight_eq_one_of_robust (robust_indicator label ω.1 3 _)]
    exact one_ne_zero, measure_univ]
  exact one_ne_zero

/-- So the runs of that model are eventually right with positive probability. -/
lemma law_eventually_indicator_ne_zero (label : Fin 2 → Bool) :
    law 2 1 (Measure.dirac (0 : Unit → ℝ)) {ω | ∀ᶠ t in atTop, AnswersHeldout
      (fun _ x c => if c = label x then (1 : ℝ) else 0) label id 1 ω.1
      (trajectory (Hyper.source 0) (fun _ _ _ => 0) id (by norm_num) 1 ω t)} ≠ 0 :=
  law_eventually_ne_zero_of_stayRight (ι := Unit) (N := 2) (k := 1)
    (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
    (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id)
    (hk := (by norm_num : 1 ≤ 2)) (b := 1) (fun _ _ => measurable_const)
    (fun _ _ => measurable_const) (by decide) (Measure.dirac (0 : Unit → ℝ))
    (law_stayRight_indicator_ne_zero label)

example (label : Fin 2 → Bool) :=
  exists_lt_stayRight (ι := Unit) (N := 2) (k := 1)
    (logits := fun _ x c => if c = label x then (1 : ℝ) else 0) (label := label)
    (h := Hyper.source 0) (lossGrad := fun _ _ _ => 0) (eqs := id)
    (hk := (by norm_num : 1 ≤ 2)) (b := 1) (fun _ _ => measurable_const)
    (fun _ _ => measurable_const) (by decide) (Measure.dirac (0 : Unit → ℝ))
    (law_eventually_indicator_ne_zero label) (δ := 2⁻¹) ENNReal.one_half_lt_one

end Transformer.Grokking.Protocol
