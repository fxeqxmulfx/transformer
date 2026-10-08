import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryFormedOutcomes
import Transformer.Grokking.CircuitEfficiency.SectionC_GainCollapseRobustness

/-!
# Original nonzero-memory delayed accuracy without a positive robust radius

Sources: Varma et al., arXiv:2309.02390v1, section 3 confidence versus
decisions and appendix C actual held-out logits; original native
legal-beta score limits at lab commit 2c2a85b, double-zero delayed
accuracy at fe4769b and target-only logit controls at cb1e286.

The actual held-out margin is the target logit minus the distinct wrong
Mem-class logit. It tends to zero from initialized numerical signs on
the pinned original beta1=0.9/beta2=0.98 path. Every fixed positive
radius therefore eventually admits, at every later clock, a perturbed
full logit vector with all coordinate errors bounded by that radius
which breaks strict target correctness. Only the target is changed.

No positive margin lower bound or robust radius holds on a tail. The
same source-style double-zero initialized path nevertheless has every
prescribed wrong-test prefix after formation and later permanently
correct train/held-out decisions in exact reals. Each fixed positive
radius has its own generated counterperturbation tail after success.

Neither a future margin/parameter limit nor a successful reference is
independently supplied. Actual CE, shared clipping, both buffers and
full completed clocks are retained. The perturbation is a mathematical
control, not a training intervention or proof of a floating-point kernel
error. A current positive binary margin does give a local radius equal
to one third of that margin. This preserves finite-clock robustness
while leaving a fixed positive uniform tail guarantee unavailable.
Strong decay 100, epsilon/cap 1, gains 3/2 and rate 0.001 differ from
the preserved learned GPTMini configuration and appendix C coupled-cost
GD. Weaker-decay attraction and learned stochastic/numerical transfer remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- At a current positive binary margin, coordinate errors at most one
third of that margin preserve strict target correctness. Sources:
appendix C actual binary competitors and target-only controls at cb1e286;
this local guarantee does not impose a fixed future margin or radius. -/
theorem binary_margin_controls_perturbed_correctness (logits perturbed : Fin 2 → ℝ)
    (hgap : logits 1 < logits 0)
    (hbound : ∀ k, |perturbed k - logits k| ≤ (logits 0 - logits 1) / 3) :
    StrictCorrect perturbed 0 := by
  have h0 := abs_le.mp (hbound 0)
  have h1 := abs_le.mp (hbound 1)
  intro k hk
  fin_cases k
  · exact (hk rfl).elim
  · change perturbed 1 < perturbed 0
    linarith only [hgap, h0.1, h1.2]

example :
    let logits : Fin 2 → ℝ := ![1, 0]
    logits 1 < logits 0 ∧ ∀ k, |logits k - logits k| ≤ (logits 0 - logits 1) / 3 := by
  dsimp only
  constructor
  · norm_num
  · intro k
    fin_cases k <;> norm_num

/-- Actual target-minus-Mem held-out logit margin at the current native
clock. Sources: appendix C test table and fixed readout at 917fc73;
every argument enters the physical original forward. -/
noncomputable def fixedGainMemoryHeldoutMargin (initial : NativeSubweightState) (n : ℕ) : ℝ :=
  fixedGainMemoryHeldoutLogits initial n 0 - fixedGainMemoryHeldoutLogits initial n 1

/-- Original legal-beta full held-out margins vanish from initial
numerical signs under the pinned static strong decay. Sources: appendix
C actual logits and native generated retained-beta limits at 2c2a85b;
future convergence and task success are not independent premises. -/
theorem fixed_gain_memory_heldout_margin_tendsto_zero (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    Tendsto (fun n => fixedGainMemoryHeldoutMargin initial n) atTop (nhds 0) := by
  have hm := gain_native_memory_scores_tendsto_zero 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have heq : ∀ n, fixedGainMemoryHeldoutMargin initial n =
      physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter -
        physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter (fixedGainMemoryPath initial n 3).parameter := by
    intro n
    norm_num [fixedGainMemoryHeldoutMargin, fixedGainMemoryHeldoutLogits, heldoutTableLogits]
  simpa only [heq, fixedGainMemoryPath] using hm.2.2

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Every fixed positive radius eventually admits bounded actual-logit
counterperturbations on the original legal-beta path. Sources: appendix
C true logits and target-only controls at cb1e286 with initialized native
score limits at 2c2a85b; no numerical-error stream is assumed or generated. -/
theorem fixed_gain_memory_eventually_perturbable (initial : NativeSubweightState) (radius : ℝ)
    (hs : NonnegativeNativeState initial) (hr : 0 < radius) :
    ∃ start : ℕ, ∀ n, start ≤ n → ∃ perturbed : Fin 2 → ℝ,
      (∀ k, |perturbed k - fixedGainMemoryHeldoutLogits initial n k| ≤ radius) ∧ ¬StrictCorrect perturbed 0 := by
  have hm := fixed_gain_memory_heldout_margin_tendsto_zero initial hs
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hm.eventually_lt_const hr)
  refine ⟨start, ?_⟩
  intro n hn
  have hgap : physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter
      (fixedGainMemoryPath initial n 1).parameter -
        physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter
          (fixedGainMemoryPath initial n 3).parameter < radius := by
    simpa [fixedGainMemoryHeldoutMargin, fixedGainMemoryHeldoutLogits, heldoutTableLogits] using htail n hn
  exact heldout_table_target_perturbation 0 _ _ radius hr hgap

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧ (0 : ℝ) < 1 / 100 := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Actual original native held-out margins have no fixed positive
lower bound on any tail. Sources: appendix C decision gap and retained
initialized zero-margin limit at 2c2a85b; this is distinct from strict
finite-clock target correctness and positive factor formation. -/
theorem fixed_gain_memory_no_positive_margin_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ¬∃ radius : ℝ, 0 < radius ∧ ∃ start : ℕ, ∀ n, start ≤ n → radius ≤ fixedGainMemoryHeldoutMargin initial n := by
  rintro ⟨radius, hr, start, htail⟩
  have hm := fixed_gain_memory_heldout_margin_tendsto_zero initial hs
  obtain ⟨after, hbelow⟩ := eventually_atTop.mp (hm.eventually_lt_const hr)
  have hlow := hbelow (start + after) (by omega)
  have hhigh := htail (start + after) (by omega)
  linarith only [hlow, hhigh]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- No fixed positive coordinate-error radius preserves strict target
correctness against every bounded logit perturbation on a native tail.
Sources: appendix C test forward and controls at cb1e286 with generated
nonzero-memory margin collapse at 2c2a85b; actual full logits are used. -/
theorem fixed_gain_memory_no_robust_radius_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ¬∃ radius : ℝ, 0 < radius ∧ ∃ start : ℕ, ∀ n, start ≤ n → ∀ perturbed : Fin 2 → ℝ,
      (∀ k, |perturbed k - fixedGainMemoryHeldoutLogits initial n k| ≤ radius) → StrictCorrect perturbed 0 := by
  rintro ⟨radius, hr, start, hrobust⟩
  obtain ⟨after, htail⟩ := fixed_gain_memory_eventually_perturbable initial radius hs hr
  obtain ⟨perturbed, hbound, hwrong⟩ := htail (start + after) (by omega)
  exact hwrong (hrobust (start + after) (by omega) perturbed hbound)

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Arbitrarily delayed source-style double-zero native accuracy can
be permanently correct and eventually counterperturbable at every
positive fixed radius on the same full retained path. Sources: section
3 slow formation, appendix C actual test logits and original native
initialized delayed accuracy at fe4769b, with controls at cb1e286. -/
theorem fixed_gain_memory_double_zero_delay_without_tail_robustness (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      ∃ start : ℕ, budget + 1 < start ∧
        (∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
        ∀ radius : ℝ, 0 < radius → ∃ after : ℕ, start ≤ after ∧
          ∀ n, after ≤ n → ∃ perturbed : Fin 2 → ℝ,
            (∀ k, |perturbed k - fixedGainMemoryHeldoutLogits initial n k| ≤ radius) ∧ ¬StrictCorrect perturbed 0 := by
  obtain ⟨seed, hseed, hunit, _, htrain, hwrong, ⟨start, hlate, htrue⟩⟩ :=
    fixed_gain_memory_double_zero_delayed_accuracy budget
  have hs := native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  refine ⟨seed, hseed, hunit, htrain, hwrong, start, hlate, htrue, ?_⟩
  intro radius hr
  obtain ⟨after, htail⟩ := fixed_gain_memory_eventually_perturbable _ radius hs hr
  exact ⟨start + after, by omega, fun n hn => htail n (by omega)⟩

end Transformer.Grokking.CircuitEfficiency
