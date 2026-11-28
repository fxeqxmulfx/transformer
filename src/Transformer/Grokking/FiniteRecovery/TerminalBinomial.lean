module

public import Transformer.Grokking.FiniteRecovery.TerminalValidation
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Analysis.Calculus.Deriv.Pow

/-! Exact validation probabilities at a fixed training horizon. A trial can
be a whole seed and hardware execution, and its success can require a strict
answer on every single hidden input. Disjoint sets of successful trials
give the binomial law used for the experimental one-sided confidence limits. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open MeasureTheory Set Finset
open scoped ENNReal Classical

variable {D : Type*} [MeasurableSpace D]

private def trialPattern (U : Set D) (I J : Finset ℕ) : Set (ℕ → D) :=
  Set.pi (I : Set ℕ) fun i => if i ∈ J then U else Uᶜ

omit [MeasurableSpace D] in
private theorem mem_trialPattern (U : Set D) {I J : Finset ℕ} (hJ : J ⊆ I) (z : ℕ → D) :
    z ∈ trialPattern U I J ↔ I.filter (fun i => z i ∈ U) = J := by
  constructor
  · intro hz
    ext i
    by_cases hi : i ∈ I
    · have h := hz i hi
      by_cases hj : i ∈ J <;> simpa [hi,hj] using h
    · have hj : i ∉ J := fun h => hi (hJ h)
      simp [hi,hj]
  · intro h i hi
    have he : i ∈ I.filter (fun j => z j ∈ U) ↔ i ∈ J := by rw [h]
    by_cases hj : i ∈ J
    · simpa [hj] using (mem_filter.mp (he.mpr hj)).2
    · simp only [hj,ite_false,Set.mem_compl_iff]
      exact fun hz => hj (he.mp (mem_filter.mpr ⟨hi,hz⟩))

private theorem pattern_measurable {U : Set D} (hU : MeasurableSet U) (I J : Finset ℕ) :
    MeasurableSet (trialPattern U I J) := by
  apply MeasurableSet.pi I.countable_toSet
  intro i _
  split <;> first | exact hU | exact hU.compl

private theorem pattern_measure (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) {I J : Finset ℕ} (hJ : J ⊆ I) :
    Measure.infinitePi (fun _ : ℕ => μ) (trialPattern U I J) =
      (μ U)^J.card * (1-μ U)^(I.card-J.card) := by
  rw [trialPattern, Measure.infinitePi_pi _ (fun i _ => by split <;> first | exact hU | exact hU.compl)]
  simp_rw [apply_ite μ, prob_compl_eq_one_sub hU]
  rw [prod_ite,filter_mem_eq_inter,Finset.inter_eq_right.mpr hJ,filter_notMem_eq_sdiff]
  simp [card_sdiff_of_subset hJ]

omit [MeasurableSpace D] in
private theorem count_decomposition (U : Set D) (I : Finset ℕ) (k : ℕ) :
    {z | terminalPassCount U I z = k} = ⋃ J ∈ I.powersetCard k, trialPattern U I J := by
  ext z
  constructor
  · intro hz
    let J := I.filter fun i => z i ∈ U
    have hJ : J ⊆ I := filter_subset _ _
    exact mem_iUnion₂.mpr ⟨J,mem_powersetCard.mpr ⟨hJ,hz⟩,(mem_trialPattern U hJ z).2 rfl⟩
  · intro hz
    obtain ⟨J,hJ,h⟩ := mem_iUnion₂.mp hz
    obtain ⟨hs,hc⟩ := mem_powersetCard.mp hJ
    exact (congrArg Finset.card ((mem_trialPattern U hs z).1 h)).trans hc

/-- The exact binomial mass for any specified number of strict successes. -/
theorem iid_terminal_count_mass (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) (I : Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | terminalPassCount U I z = k} =
      (I.card.choose k : ℝ≥0∞) * (μ U)^k * (1-μ U)^(I.card-k) := by
  rw [count_decomposition, measure_biUnion_finset _ (fun _ _ => pattern_measurable hU _ _)]
  · calc
      _ = ∑ _J ∈ I.powersetCard k, (μ U)^k * (1-μ U)^(I.card-k) := by
        apply sum_congr rfl
        intro J hJ
        obtain ⟨hs,hc⟩ := mem_powersetCard.mp hJ
        rw [pattern_measure μ hU hs,hc]
      _ = _ := by simp [mul_assoc]
  · intro J hJ L hL hne
    apply Set.disjoint_left.mpr
    intro z hz hz'
    exact hne (((mem_trialPattern U (mem_powersetCard.mp hJ).1 z).1 hz).symm.trans
      ((mem_trialPattern U (mem_powersetCard.mp hL).1 z).1 hz'))

/-- Every threshold counts the complete terminal outcomes in its own event. -/
theorem iid_terminal_count_tail (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) (I : Finset ℕ) (k : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | k ≤ terminalPassCount U I z} =
      ∑ j ∈ Finset.Icc k I.card, (I.card.choose j : ℝ≥0∞) * (μ U)^j * (1-μ U)^(I.card-j) := by
  have hset : {z | k ≤ terminalPassCount U I z} =
      ⋃ j ∈ Finset.Icc k I.card, {z | terminalPassCount U I z = j} := by
    ext z
    have hz : terminalPassCount U I z ≤ I.card := card_le_card (filter_subset _ _)
    simp [hz]
  rw [hset,measure_biUnion_finset]
  · exact sum_congr rfl fun j _ => iid_terminal_count_mass μ hU I j
  · intro i _ j _ hne
    exact Set.disjoint_left.mpr fun _ hi hj => hne (hi.symm.trans hj)
  · intro j _
    rw [count_decomposition]
    exact MeasurableSet.iUnion fun J => MeasurableSet.iUnion fun _ => pattern_measurable hU I J

/-- The exhaustive AND evaluation had eleven successful endpoints in twelve
fresh draws. The formula applies to its hidden-string event or to any other
fixed measurable test, separately from the primary seven-example test. -/
theorem eleven_of_twelve_exact_tail (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) :
    Measure.infinitePi (fun _ : ℕ => μ) {z | 11 ≤ terminalPassCount U (range 12) z} =
      12 * (μ U)^11 * (1-μ U) + (μ U)^12 := by
  rw [iid_terminal_count_tail μ hU,card_range]
  rw [show Finset.Icc 11 12 = ({11,12} : Finset ℕ) by decide]
  norm_num [Nat.choose]

private theorem eleven_tail_monotone :
    MonotoneOn (fun x : ℝ => 12*x^11*(1-x) + x^12) (Set.Icc 0 1) := by
  apply monotoneOn_of_deriv_nonneg (convex_Icc _ _) (by fun_prop) (by fun_prop)
  intro x hx
  have hx' : x ∈ Set.Icc (0:ℝ) 1 := interior_subset hx
  have h : HasDerivAt (fun x : ℝ => 12*x^11*(1-x) + x^12) (132*x^10*(1-x)) x := by
    have hd := ((((hasDerivAt_id x).pow 11).const_mul (12:ℝ)).mul
      ((hasDerivAt_const x (1:ℝ)).sub (hasDerivAt_id x))).add ((hasDerivAt_id x).pow 12)
    change HasDerivAt (fun y : ℝ => 12*y^11*(1-y)+y^12)
      (12*(11*x^10*1)*(1-x) + 12*x^11*(0-1) + 12*x^11*1) x at hd
    convert hd using 1
    ring
  rw [h.deriv]
  exact mul_nonneg (by positivity) (sub_nonneg.mpr hx'.2)

/-- The boundary of the composite null maximizes its upper tail. -/
theorem eleven_of_twelve_null_bound (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D}
    (hU : MeasurableSet U) {q : ℝ} (hp : μ.real U ≤ q) (hq : q ≤ 1) :
    (Measure.infinitePi (fun _ : ℕ => μ)).real {z | 11 ≤ terminalPassCount U (range 12) z} ≤
      12*q^11*(1-q) + q^12 := by
  have hle : μ U ≤ 1 := prob_le_one
  have hr : (μ U).toReal ≤ 1 := by simpa using ENNReal.toReal_mono ENNReal.one_ne_top hle
  rw [Measure.real,eleven_of_twelve_exact_tail μ hU,ENNReal.toReal_add (by finiteness) (by finiteness)]
  simp only [ENNReal.toReal_mul,ENNReal.toReal_ofNat,ENNReal.toReal_pow,
    ENNReal.toReal_sub_of_le hle ENNReal.one_ne_top,ENNReal.toReal_one]
  exact eleven_tail_monotone ⟨ENNReal.toReal_nonneg,hr⟩ ⟨ENNReal.toReal_nonneg.trans hp,hq⟩ hp

example (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D} (hU : MeasurableSet U) (hp : μ.real U ≤ 1/2) :
    (Measure.infinitePi (fun _ : ℕ => μ)).real {z | 11 ≤ terminalPassCount U (range 12) z} ≤ 13/4096 := by
  apply (eleven_of_twelve_null_bound μ hU hp (by norm_num)).trans
  norm_num

example (μ : Measure D) [IsProbabilityMeasure μ] {U : Set D} (hU : MeasurableSet U) (hp : μ.real U ≤ 3/4) :
    (Measure.infinitePi (fun _ : ℕ => μ)).real {z | 11 ≤ terminalPassCount U (range 12) z} ≤ 2657205/16777216 := by
  apply (eleven_of_twelve_null_bound μ hU hp (by norm_num)).trans
  norm_num

example := eleven_of_twelve_exact_tail (Measure.dirac (0 : ℕ)) (U := {0}) (measurableSet_singleton _)

end Transformer.Grokking.FiniteRecovery
