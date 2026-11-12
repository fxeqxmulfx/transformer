module

public import Transformer.Grokking.Protocol.CarriedSplit
public import Transformer.Grokking.Protocol.DivisionMeasurable

/-!
# A set of splits with a symmetry that moves a held-out answer

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and section 1, "distinct abstract
symbols"; `experiments/grok_division/PROTOCOL.md`, *The split*;
`review/grok_division_convergence.md`, section 2. The target and its modes are those of
`Transformer.Grokking.Protocol.Target`; `wrongAt` is that of `Refutation`.

`Refutation` and `LabelCoverage` refute the target on splits that hide a token from training.
The argument needs less: a renaming of the tokens that carries the split to a split
(`Carries`), and two held-out equations that the renaming identifies up to the position of
`=` while it leaves their two different answers alone (`Twisted`). The run of the carried
split from the permuted initial parameters is the permuted run, so it reads the first equation
as the run reads the second (`answerLogits_comp_paramPerm`), and the two are not both answered
(`not_both_predicted_of_twisted`): the sample or the carried sample has a wrong held-out
answer at every update (`mem_wrongAt_or_carry`).

So on such a set a held-out answer is wrong with a conditional probability of at least one
half at every update (`uniformPerm_le_of_twisted`), under Adam and under AdamW alike; and
under a condition on the split that contains such a set, not empty, the target fails almost
surely and in probability (`not_ae_eventually_of_twisted`, `not_tendsto_law_of_twisted`). The
uncovered splits are such a set, with no reordering (`twisted_uncovered`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **A set `A` of splits with a symmetry that moves a held-out answer**: the reordering `ρ`
keeps `A`; with the renaming `τ` it carries every split of `A`; the equations `x` and `y` are
held out by every split of `A`; up to the position of `=` the renaming turns `x` into `y`; and
it does not move their answers, which differ. -/
structure Twisted (τ : Equiv.Perm (Fin 239)) (ρ : Equiv.Perm (Fin (97 * 96)))
    (A : Set (Equiv.Perm (Fin (97 * 96)))) (x y : Equation) : Prop where
  /-- The reordering keeps the set. -/
  closed : ∀ π ∈ A, ρ * π ∈ A
  /-- The renaming and the reordering carry every split of the set. -/
  carries : ∀ π ∈ A, Carries τ ρ π
  /-- The first equation is held out by every split of the set. -/
  heldout_left : ∀ π ∈ A, x ∈ heldout equations 4656 π
  /-- The second equation is held out by every split of the set. -/
  heldout_right : ∀ π ∈ A, y ∈ heldout equations 4656 π
  /-- Up to the position of `=` the renaming turns the first equation into the second. -/
  read_eq : ∀ s : Fin 6, s ≤ 4 → τ (read x s) = read y s
  /-- The renaming does not move the answer of the first equation. -/
  fixed_left : τ (residueToken x.1) = residueToken x.1
  /-- The renaming does not move the answer of the second equation. -/
  fixed_right : τ (residueToken y.1) = residueToken y.1
  /-- The two answers differ. -/
  ne : x.1 ≠ y.1

variable {τ : Equiv.Perm (Fin 239)} {ρ : Equiv.Perm (Fin (97 * 96))}
  {A : Set (Equiv.Perm (Fin (97 * 96)))} {x y : Equation}

/-- **At the permuted parameters the logits after `=` of the first equation are those of the
second**, with the tokens renamed. -/
lemma answerLogits_comp_paramPerm (hread : ∀ s : Fin 6, s ≤ 4 → τ (read x s) = read y s)
    (θ : Param paperSizes → ℝ) (c : Fin 239) :
    answerLogits (θ ∘ paramPerm τ) x c = answerLogits θ y (τ c) := by
  have h := logits_comp_paramPerm (S := paperSizes) (τ := τ) (θ := θ)
    (act := relu paperSizes) (read x) (4 : Fin 6) c
  exact h.trans (congrFun (logits_congr (S := paperSizes) (t := (4 : Fin 6)) hread) (τ c))

/-- **The second equation at some parameters and the first at the permuted ones are not both
answered.** -/
theorem not_both_predicted_of_twisted (h : Twisted τ ρ A x y) (θ : Param paperSizes → ℝ) :
    ¬ (predict (answerLogits θ y) = residueToken y.1 ∧
      predict (answerLogits (θ ∘ paramPerm τ) x) = residueToken x.1) := fun ⟨h2, h1⟩ => by
  rcases lt_or_gt_of_ne (residueToken_injective.ne h.ne) with hlt | hlt
  · have a := (predict_eq_iff.1 h2).2 _ hlt
    have b := (predict_eq_iff.1 h1).1 (residueToken y.1)
    rw [answerLogits_comp_paramPerm h.read_eq, answerLogits_comp_paramPerm h.read_eq,
      h.fixed_left, h.fixed_right] at b
    exact absurd b (not_le.2 a)
  · have a := (predict_eq_iff.1 h1).2 _ hlt
    rw [answerLogits_comp_paramPerm h.read_eq, answerLogits_comp_paramPerm h.read_eq,
      h.fixed_left, h.fixed_right] at a
    exact absurd ((predict_eq_iff.1 h2).1 (residueToken x.1)) (not_le.2 a)

/-- **A sample with a split of the set or the carried sample has a wrong held-out answer**, at
every update. -/
theorem mem_wrongAt_or_carry (decay : ℝ) (t : ℕ) (h : Twisted τ ρ A x y)
    {ω : Sample (97 * 96) 4656 (Param paperSizes)} (hω : ω.1 ∈ A) :
    ω ∈ wrongAt decay t ∨ carrySample τ ρ ω ∈ wrongAt decay t := by
  by_contra hnot
  rw [not_or] at hnot
  simp only [wrongAt, Set.mem_ofPred_eq, not_exists, not_and, not_not] at hnot
  have hy := hnot.1 y (h.heldout_right _ hω)
  have hx := hnot.2 x (h.heldout_left _ (h.closed _ hω))
  rw [show carrySample τ ρ ω = (ρ * ω.1, ω.2.1 ∘ paramPerm τ, ω.2.2) from rfl,
    trajectory_carried decay (h.carries _ hω)] at hx
  exact not_both_predicted_of_twisted h _ ⟨hy, hx⟩

/-- **Given that the split is in the set, a held-out answer is wrong with a conditional
probability of at least one half, at every update.** -/
theorem uniformPerm_le_of_twisted (decay : ℝ) (t : ℕ) (h : Twisted τ ρ A x y) :
    uniformPerm (97 * 96) A ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn A ∩ wrongAt decay t) := by
  set μ := law (97 * 96) 4656 (initMeasure paperSizes) with hμ
  have hF : μ (splitIn A) = uniformPerm (97 * 96) A := by
    rw [hμ, law, Measure.prod_prod, measure_univ, mul_one]
  have hE : MeasurableSet (splitIn A ∩ wrongAt decay t) :=
    ((MeasurableSpace.measurableSet_top).prod MeasurableSet.univ).inter
      (measurableSet_wrongAt decay t)
  have hpre : μ (carrySample τ ρ ⁻¹' (splitIn A ∩ wrongAt decay t)) =
      μ (splitIn A ∩ wrongAt decay t) :=
    (measurePreserving_carrySample τ ρ).measure_preimage hE.nullMeasurableSet
  calc uniformPerm (97 * 96) A = μ (splitIn A) := hF.symm
    _ ≤ μ ((splitIn A ∩ wrongAt decay t) ∪
          carrySample τ ρ ⁻¹' (splitIn A ∩ wrongAt decay t)) :=
        measure_mono fun ω hω => by
          rcases mem_wrongAt_or_carry decay t h hω.1 with hw | hw
          · exact Or.inl ⟨hω, hw⟩
          · exact Or.inr ⟨⟨h.closed _ hω.1, Set.mem_univ _⟩, hw⟩
    _ ≤ μ (splitIn A ∩ wrongAt decay t) +
          μ (carrySample τ ρ ⁻¹' (splitIn A ∩ wrongAt decay t)) := measure_union_le _ _
    _ = 2 * μ (splitIn A ∩ wrongAt decay t) := by rw [hpre, two_mul]

/-- **Under a condition on the split that contains such a set, not empty, the target is
false**: it is not so that for almost every sample that meets the condition every held-out
equation is eventually answered. -/
theorem not_ae_eventually_of_twisted (decay : ℝ) (h : Twisted τ ρ A x y)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} (hAS : A ⊆ S) {π : Equiv.Perm (Fin (97 * 96))}
    (hπ : π ∈ A) :
    ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ S →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t := fun hS => by
  have h2 := (measurePreserving_carrySample τ ρ).quasiMeasurePreserving.ae hS
  have hnull : ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω ∉ splitIn A := by
    filter_upwards [hS, h2] with ω hω1 hω2 hmem
    obtain ⟨t, ht1, ht2⟩ :=
      ((hω1 (hAS hmem.1)).and (hω2 (hAS (h.closed _ hmem.1)))).exists
    rcases mem_wrongAt_or_carry decay t h hmem.1 with hw | hw
    exacts [ht1 hw, ht2 hw]
  have hzero := measure_eq_zero_iff_ae_notMem.2 hnull
  rw [law, Measure.prod_prod, measure_univ, mul_one] at hzero
  exact (uniformPerm_pos_of_mem hπ).ne' hzero

/-- **Under such a condition the target fails in probability too**: the probability that the
condition holds and a held-out answer is wrong does not tend to zero. -/
theorem not_tendsto_law_of_twisted (decay : ℝ) (h : Twisted τ ρ A x y)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} (hAS : A ⊆ S) {π : Equiv.Perm (Fin (97 * 96))}
    (hπ : π ∈ A) :
    ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn S ∩ wrongAt decay t)) atTop (𝓝 0) := fun hlim => by
  have hbound : ∀ t : ℕ, uniformPerm (97 * 96) A ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn S ∩ wrongAt decay t) := fun t =>
    (uniformPerm_le_of_twisted decay t h).trans (mul_le_mul_right (measure_mono
      (Set.inter_subset_inter_left _ (Set.prod_mono hAS le_rfl))) 2)
  have hzero := ge_of_tendsto' (ENNReal.Tendsto.const_mul hlim (Or.inr ENNReal.ofNat_ne_top))
    hbound
  rw [mul_zero] at hzero
  exact (uniformPerm_pos_of_mem hπ).not_ge hzero

/-! The witness: the uncovered splits, with the exchange of the two hidden tokens, no
reordering, and the held-out equations `1 / 3 = 65` and `2 / 3 = 33`. -/

/-- **The uncovered splits are a set with a symmetry that moves a held-out answer.** -/
theorem twisted_uncovered : Twisted hiddenSwap 1 uncovered hiddenOne hiddenTwo where
  closed π hπ := by rwa [one_mul]
  carries _ hπ := carries_hiddenSwap_one hπ
  heldout_left _ hπ := mem_heldout_of_not_avoids hπ (by decide)
  heldout_right _ hπ := mem_heldout_of_not_avoids hπ (by decide)
  read_eq := by decide
  fixed_left := by decide
  fixed_right := by decide
  ne := by decide

example (θ : Param paperSizes → ℝ) (c : Fin 239) :
    answerLogits (θ ∘ paramPerm hiddenSwap) hiddenOne c = answerLogits θ hiddenTwo (hiddenSwap c) :=
  answerLogits_comp_paramPerm (by decide) θ c

example (θ : Param paperSizes → ℝ) := not_both_predicted_of_twisted twisted_uncovered θ

example (decay : ℝ) (t : ℕ) :
    ∃ ω, ω ∈ wrongAt decay t ∨ carrySample hiddenSwap 1 ω ∈ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨(π, 0, fun _ => 1), mem_wrongAt_or_carry decay t twisted_uncovered hπ⟩

example (decay : ℝ) (t : ℕ) : uniformPerm (97 * 96) uncovered ≤
    2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn uncovered ∩ wrongAt decay t) :=
  uniformPerm_le_of_twisted decay t twisted_uncovered

example (decay : ℝ) : ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes),
    ω.1 ∈ uncovered → ∀ᶠ t in atTop, ω ∉ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact not_ae_eventually_of_twisted decay twisted_uncovered subset_rfl hπ

example (decay : ℝ) : ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
    (splitIn uncovered ∩ wrongAt decay t)) atTop (𝓝 0) := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact not_tendsto_law_of_twisted decay twisted_uncovered subset_rfl hπ

end Transformer.Grokking.Protocol
