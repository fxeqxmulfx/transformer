import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryRobustness

/-!
# Exact binary coordinate-error threshold on the original native path

Sources: Varma et al., arXiv:2309.02390v1, section 3 accuracy versus
confidence and appendix C held-out competing logits; original legal-
nonzero-beta margin/robustness and double-zero delayed accuracy at lab
commit bac6930. Refine the sufficient one-third current-margin bound.

For arbitrary current binary logits and a nonnegative coordinate-error
radius, strict target correctness survives every full logit perturbation
of that radius exactly when twice the radius is less than the target
minus competitor gap. At equality, symmetric target-down/competitor-up
errors create a tie, already defeating strict correctness.

Define the original native half-margin as a numerical observer; it may
be negative before actual selection. Initialized native selection gives
a generated tail on which this observer is positive and is the exact
strict coordinate-error threshold. Its full-clock limit is zero. The
same source-style double-zero delayed path has any prescribed wrong-test
prefix and later correct decisions with these positive shrinking radii.

No future convergence, gap, success, constant coefficient or input
stream is supplied. Native beta1=0.9/beta2=0.98, both buffers, full
correction clocks and shared clipping remain. Strong decay 100,
epsilon/cap 1, gains 3/2 and rate 0.001 are fixed. The strict radius
condition has an open endpoint and concerns actual binary logits;
it does not predict the errors of a floating-point kernel or supply
learned stochastic/numerical GPTMini transfer. Fixed gained tables and
uniform decoupled native decay differ from appendix C coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- All coordinate errors below half the current binary gap preserve
strict target correctness. Sources: appendix C binary competitors and
current native logit controls at bac6930; no future-margin premise enters. -/
theorem binary_errors_preserve_strict_below_half_gap (logits perturbed : Fin 2 → ℝ) (radius : ℝ)
    (hgap : 2 * radius < logits 0 - logits 1)
    (hbound : ∀ k, |perturbed k - logits k| ≤ radius) : StrictCorrect perturbed 0 := by
  have h0 := abs_le.mp (hbound 0)
  have h1 := abs_le.mp (hbound 1)
  intro k hk
  fin_cases k
  · exact (hk rfl).elim
  · change perturbed 1 < perturbed 0
    linarith only [hgap, h0.1, h1.2]

example :
    let logits : Fin 2 → ℝ := ![1, 0]
    2 * (1 / 3 : ℝ) < logits 0 - logits 1 ∧ ∀ k, |logits k - logits k| ≤ (1 / 3 : ℝ) := by
  dsimp only
  constructor
  · norm_num
  · intro k
    fin_cases k <;> norm_num

/-- At or above half the binary gap, bounded symmetric coordinate
errors defeat strict correctness, including a tie at equality. Sources:
appendix C distinct target/Mem logits and controls at bac6930; both
coordinates are actual class logits, without an accuracy surrogate. -/
theorem binary_symmetric_adversary_at_half_gap (logits : Fin 2 → ℝ) (radius : ℝ)
    (hr : 0 ≤ radius) (hgap : logits 0 - logits 1 ≤ 2 * radius) :
    ∃ perturbed : Fin 2 → ℝ, (∀ k, |perturbed k - logits k| ≤ radius) ∧ ¬StrictCorrect perturbed 0 := by
  let perturbed : Fin 2 → ℝ := fun k => if k = 0 then logits 0 - radius else logits 1 + radius
  refine ⟨perturbed, ?_, ?_⟩
  · intro k
    apply abs_le.mpr
    fin_cases k
    · change -radius ≤ logits 0 - radius - logits 0 ∧ logits 0 - radius - logits 0 ≤ radius
      constructor <;> linarith only [hr]
    · change -radius ≤ logits 1 + radius - logits 1 ∧ logits 1 + radius - logits 1 ≤ radius
      constructor <;> linarith only [hr]
  · intro hc
    have hh := hc 1 (by decide)
    change logits 1 + radius < logits 0 - radius at hh
    linarith only [hgap, hh]

example :
    let logits : Fin 2 → ℝ := ![1, 0]
    (0 : ℝ) ≤ 1 / 2 ∧ logits 0 - logits 1 ≤ 2 * (1 / 2 : ℝ) := by norm_num

/-- The exact strict binary coordinate-error guarantee is twice the
radius below the actual gap. Sources: appendix C binary target/Mem
competition and native current controls at bac6930; equality is excluded. -/
theorem binary_robustness_iff_twice_radius_lt_gap (logits : Fin 2 → ℝ) (radius : ℝ) (hr : 0 ≤ radius) :
    (∀ perturbed : Fin 2 → ℝ, (∀ k, |perturbed k - logits k| ≤ radius) → StrictCorrect perturbed 0) ↔
      2 * radius < logits 0 - logits 1 := by
  constructor
  · intro hrobust
    by_contra hgap
    obtain ⟨perturbed, hbound, hwrong⟩ := binary_symmetric_adversary_at_half_gap logits radius hr (le_of_not_gt hgap)
    exact hwrong (hrobust perturbed hbound)
  · exact fun hgap perturbed hbound => binary_errors_preserve_strict_below_half_gap logits perturbed radius hgap hbound

example : (0 : ℝ) ≤ 1 / 3 := by norm_num

/-- Numerical half of the original target-minus-Mem held-out margin.
Sources: appendix C binary class gap and native readout at bac6930;
every argument enters the forward, and positivity is not encoded. -/
noncomputable def fixedGainMemoryHeldoutRadius (initial : NativeSubweightState) (n : ℕ) : ℝ :=
  (1 / 2 : ℝ) * fixedGainMemoryHeldoutMargin initial n

/-- The original initialized native half-margin observer vanishes on
its full clock axis. Sources: appendix C true logits and generated
retained-beta margin collapse at bac6930; no future radius is supplied. -/
theorem fixed_gain_memory_half_margin_tendsto_zero (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    Tendsto (fun n => fixedGainMemoryHeldoutRadius initial n) atTop (nhds 0) := by
  have hm := fixed_gain_memory_heldout_margin_tendsto_zero initial hs
  have hh := hm.const_mul (1 / 2 : ℝ)
  simpa only [fixedGainMemoryHeldoutRadius, mul_zero] using hh

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Original initialized native selection generates positive half-
margin observers which exactly characterize all bounded-logit strict
correctness guarantees. Sources: appendix C true decisions and native
formation/selection at bac6930; neither a future success nor gap is supplied. -/
theorem fixed_gain_memory_eventual_exact_radius (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hg : 0 < fixedGainGenMemoryMass initial) :
    ∃ start : ℕ, ∀ n, start ≤ n → 0 < fixedGainMemoryHeldoutRadius initial n ∧
      ∀ radius : ℝ, 0 ≤ radius →
        ((∀ perturbed : Fin 2 → ℝ, (∀ k, |perturbed k - fixedGainMemoryHeldoutLogits initial n k| ≤ radius) →
          StrictCorrect perturbed 0) ↔ radius < fixedGainMemoryHeldoutRadius initial n) := by
  obtain ⟨start, htail⟩ := fixed_gain_memory_eventual_correct initial hs hg
  refine ⟨start, ?_⟩
  intro n hn
  have hgap := (htail n hn).2 1 (by decide)
  change fixedGainMemoryHeldoutLogits initial n 1 < fixedGainMemoryHeldoutLogits initial n 0 at hgap
  constructor
  · unfold fixedGainMemoryHeldoutRadius fixedGainMemoryHeldoutMargin
    linarith only [hgap]
  · intro radius hr
    have heq := binary_robustness_iff_twice_radius_lt_gap (fixedGainMemoryHeldoutLogits initial n) radius hr
    constructor
    · intro hrobust
      have hb := heq.mp hrobust
      unfold fixedGainMemoryHeldoutRadius fixedGainMemoryHeldoutMargin
      linarith only [hb]
    · intro hsmall
      apply heq.mpr
      unfold fixedGainMemoryHeldoutRadius fixedGainMemoryHeldoutMargin at hsmall
      linarith only [hsmall]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- The same source-style double-zero delayed permanent-correct native
path has positive exact coordinate-error thresholds on its successful
tail, with those numerical thresholds tending to zero. Sources: section
3 accuracy/confidence, appendix C logits and native original initialized
formation/selection and margin collapse at bac6930. -/
theorem fixed_gain_memory_double_zero_delay_with_shrinking_exact_radii (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      (∃ start : ℕ, budget + 1 < start ∧ ∀ n, start ≤ n →
        StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 ∧ 0 < fixedGainMemoryHeldoutRadius initial n ∧
          ∀ radius : ℝ, 0 ≤ radius →
            ((∀ perturbed : Fin 2 → ℝ, (∀ k, |perturbed k - fixedGainMemoryHeldoutLogits initial n k| ≤ radius) →
              StrictCorrect perturbed 0) ↔ radius < fixedGainMemoryHeldoutRadius initial n)) ∧
      Tendsto (fun n => fixedGainMemoryHeldoutRadius initial n) atTop (nhds 0) := by
  obtain ⟨seed, hseed, hunit, _, htrain, hwrong, ⟨start, hlate, htrue⟩⟩ :=
    fixed_gain_memory_double_zero_delayed_accuracy budget
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  have hs := native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  have hg : 0 < fixedGainGenMemoryMass initial := by
    simpa [fixedGainGenMemoryMass, gainNativePairMemoryMass, initial,
      seededNativeSubweights, seededScalarState, nativeFactorPartner] using hseed
  obtain ⟨radiusStart, hexact⟩ := fixed_gain_memory_eventual_exact_radius initial hs hg
  refine ⟨seed, hseed, hunit, htrain, hwrong, ?_, fixed_gain_memory_half_margin_tendsto_zero initial hs⟩
  refine ⟨start + radiusStart, by omega, ?_⟩
  intro n hn
  exact ⟨htrue n (by omega), hexact n (by omega)⟩

end Transformer.Grokking.CircuitEfficiency
