module

public import Transformer.Grokking.FiniteRecovery.ArgandPAGEGaussian
public import Transformer.Grokking.FiniteRecovery.StaircaseDecoder
public import Mathlib.Probability.ProductMeasure

/-! Certifying a finite PAGE checkpoint on the binary modular-sum staircase.
Independent native starts give a geometric failure bound and an almost surely
finite first accepted checkpoint. Each trial uses ordinary observed-loss PAGE. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set Filter Finset
open scoped ENNReal NNReal Topology

private theorem iid_miss {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ]
    {U : Set D} (hu : MeasurableSet U) (J m : ℕ) :
    Measure.infinitePi (fun _ : ℕ => μ)
      {z | ∀ i ∈ Finset.Ico J (J+m), z i ∉ U} = (1-μ U)^m := by
  change Measure.infinitePi _ (Set.pi (Finset.Ico J (J+m) : Set ℕ) (fun _ => Uᶜ)) = _
  rw [Measure.infinitePi_pi _ (fun _ _ => hu.compl)]
  simp [prob_compl_eq_one_sub hu]

private theorem iid_eventually_hits {D : Type*} [MeasurableSpace D] (μ : Measure D)
    [IsProbabilityMeasure μ] {U : Set D} (hu : MeasurableSet U) (hp : 0 < μ U) (J : ℕ) :
    ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j ≥ J, z j ∈ U := by
  rw [ae_iff]
  apply le_antisymm _ bot_le
  have hlt : 1-μ U < 1 := ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hp.ne'
  apply le_of_tendsto_of_tendsto' tendsto_const_nhds (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
  intro m
  rw [← iid_miss μ hu J m]
  apply measure_mono
  intro z hz i hi hiU
  exact hz ⟨i,(Finset.mem_Ico.mp hi).1,hiU⟩

variable {k s : ℕ}

abbrev StaircaseIndex (s : ℕ) := Fin (s+1) ⊕ Fin s
abbrev ParityDraw := ComplexParameter (ZMod 2) 4 16 6 → ℂ

/-- Trial j makes K observed-loss steps from its native draw, at the single
constant rate a/(j+1). No unknown success radius is an algorithm input. -/
noncomputable def staircasePAGETrial (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (labels : StaircaseIndex s → ZMod 2) (a : ℝ) (K j : ℕ) (z : ParityDraw)
    (path : ℕ → Option (StaircaseIndex s)) : Weights (ZMod 2) 4 16 6 :=
  observedPAGE (T := (k+1)+k) (staircaseWord e) (staircasePrediction e labels) 1 (a/((j:ℝ)+1)) z path K

/-- Check the strict logits of the trained model on each word against the label
decoded using only the supplied diagonal and superdiagonal. -/
def staircaseAccepts (e : Fin (s+1) ≃ Word (ZMod 2) (k+1)) (labels : StaircaseIndex s → ZMod 2)
    (w : Weights (ZMod 2) 4 16 6) : Prop :=
  (Model.fromWeights w).realizes (T := (k+1)+k) 1 (staircasePrediction e labels)

/-- The sampling guarantee holds even for adversarial within-trial PAGE paths.
Only the independent native restarts are used for the geometric tail. The
first acceptable full weight vector has a strict uniform margin on all words. -/
theorem modularSum_page_restarts (e : Fin (s+1) ≃ Word (ZMod 2) (k+1)) (hk : (k+1)+(k+1) ≤ 32)
    (variance : ComplexParameter (ZMod 2) 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i)
    {a : ℝ} (ha : 0 < a) (K : ℕ) :
    let μ := independentGaussianDraws variance
    let P := Measure.infinitePi (fun _ : ℕ => μ)
    let labels := modularSum e ∘ staircaseWord e
    let trial := staircasePAGETrial e labels a K
    let ok := staircaseAccepts e labels
    ∃ J : ℕ, ∃ q : ℝ≥0∞, 0 < q ∧ q ≤ 1 ∧
      (∀ m (paths : ℕ → ℕ → Option (StaircaseIndex s)),
        P {z | ∀ j ∈ Finset.Ico J (J+m), ¬ ok (trial j (z j) (paths j))} ≤ (1-q)^m) ∧
      ∀ᵐ z ∂P, ∀ (paths : ℕ → ℕ → Option (StaircaseIndex s)), ∃ j : ℕ,
        (∀ i < j, ¬ ok (trial i (z i) (paths i))) ∧ ok (trial j (z j) (paths j)) ∧
        ∃ γ > (0:ℝ), ∀ x b, b ≠ modularSum e x →
          γ ≤ (Model.fromWeights (trial j (z j) (paths j))).logits 1 x (Fin.last ((k+1)+k)) (modularSum e x) -
            (Model.fromWeights (trial j (z j) (paths j))).logits 1 x (Fin.last ((k+1)+k)) b := by
  classical
  dsimp only
  let μ := independentGaussianDraws variance
  let labels := modularSum e ∘ staircaseWord e
  let trial := staircasePAGETrial e labels a K
  let ok := staircaseAccepts e labels
  have hd := staircasePrediction_modularSum e
  obtain ⟨δ,hδ,h⟩ := modularSum_gaussian_page_prefix e hk variance hv (staircaseWord e) K
  obtain ⟨U,hU,hp,hs⟩ := h 0
  have hr : Tendsto (fun j : ℕ => a / ((j : ℝ)+1)) atTop (𝓝 0) := by
    simpa only [mul_one_div, mul_zero] using (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).const_mul a
  obtain ⟨J,hJ⟩ := Filter.eventually_atTop.mp (hr.eventually (gt_mem_nhds hδ))
  have hit (j : ℕ) (hj : J ≤ j) (z : ParityDraw) (hz : z ∈ U) (path : ℕ → Option (StaircaseIndex s)) :
      ok (trial j z path) := by
    have hη : 0 < a / ((j : ℝ)+1) := by positivity
    have hw := hs _ ((abs_of_pos hη).trans_lt (hJ j hj)) z hz path K le_rfl
    simp only [ok, trial, staircaseAccepts, staircasePAGETrial, labels, hd]
    simpa only [add_zero] using hw
  refine ⟨J,μ U,hp,prob_le_one,?_,?_⟩
  · intro m paths
    rw [← iid_miss μ hU.measurableSet J m]
    apply measure_mono
    intro z hz i hi hiU
    exact hz i hi (hit i (Finset.mem_Ico.mp hi).1 (z i) hiU (paths i))
  · filter_upwards [iid_eventually_hits μ hU.measurableSet hp J] with z hz
    intro paths
    obtain ⟨j,hj,hz⟩ := hz
    have existsOk : ∃ i, ok (trial i (z i) (paths i)) := ⟨j,hit j hj _ hz _⟩
    let i := Nat.find existsOk
    have hi : ok (trial i (z i) (paths i)) := Nat.find_spec existsOk
    refine ⟨i,fun t ht => Nat.find_min existsOk ht,hi,?_⟩
    apply exists_uniform_logit_margin
    simpa only [ok, staircaseAccepts, labels, hd] using hi

example := modularSum_page_restarts (s := 15) (wordEnumeration 2 4) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num) (a := 1) (by norm_num) 100
example := modularSum_page_restarts (s := 255) (wordEnumeration 2 8) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num) (a := 1) (by norm_num) 100
example := modularSum_page_restarts (s := 65535) (wordEnumeration 2 16) (by decide)
  (fun _ => (1/50 : ℝ≥0)^2) (fun _ => by norm_num) (a := 1) (by norm_num) 100

end Transformer.Grokking.FiniteRecovery.Argand
