import Transformer.Grokking.CircuitEfficiency.SectionC_NativeUnitPath

/-!
# Positive finite-limit consequences on actual closed native CE paths

Sources: Varma et al., arXiv:2309.02390v1, appendix C's product train
and test tables; retained native AdamW/clipping at lab commit 5168744.
Assume only finite convergence of the four current parameters to a
strictly positive point on a sequence satisfying the actual closed
update. Derive actual CE inputs and buffer limits, native balance and
equal Gen/Mem factors. No gradient limit, buffer/current-input matching
or symmetric-buffer premise is supplied.

The actual Gen-minus-Mem held-out margin tends to zero. Actual held-out
CE has its true finite-point limit, at least log two, and cannot tend
to zero. A vanishing limiting margin does not assign finite-clock
accuracy or prevent positive margins at every finite clock. Reference
buffers and clocks are not asserted limits of the retained optimizer.

Every joint-hypothesis example is a nonzero actual native unit-seed path
with beta1=0.9, beta2=0.98, growing clocks and evolving zero-initialized
buffers. Its epsilon is the explicitly chosen CE scale from UnitPath,
not the frozen GPTMini configuration. Plain CE/uniform native decay is
distinct from appendix C's asymmetric coupled penalty/GD. General seed
convergence, boundary limits, physical efficiency and stochastic/float
bridges remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Any positive finite parameter limit on actual native CE feedback
has equal factors and positive decay. Sources: appendix C products
and native AdamW at 5168744; convergence of actual gradients and
retained moments is derived, independently of initial circuit symmetry. -/
theorem native_closed_positive_limit_allocation (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = nativeSubweightStep remaining bound b1 b2 eps decay rate (state n))
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter))
    (hr : ∀ i, 0 < (reference i).parameter) :
    0 < decay ∧ ∀ i, (reference i).parameter = (reference 0).parameter := by
  exact native_positive_balanced_point_equal remaining bound eps decay reference hclip he hr
    (fun i => native_closed_limit_balance remaining bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp)

example :
    let reference := seededNativeSubweights ((1, 1), (1, 1))
    let eps := nativeCEGradientScale 111 1 reference
    let state := nativeSubweightPath 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) reference
    (∀ n, state (n + 1) = nativeSubweightStep 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) ∧
    (∀ i, 0 < (reference i).parameter) := by
  dsimp only
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_ce_gradient_scale_pos 111 1 _ (by norm_num), by norm_num, ?_, ?_⟩
  · intro i
    have hr : (seededNativeSubweights ((1, 1), (1, 1)) i).parameter = 1 := by fin_cases i <;> rfl
    rw [hr]
    exact native_unit_path_parameter_tendsto 111 1 (9 / 10) (49 / 50) (1 / 1000) i
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The actual competing held-out product margin tends to zero at
every such positive native parameter limit. Sources: appendix C test
logits and native feedback at 5168744; this does not infer finite-clock
incorrectness or a tie-breaking accuracy from a limiting tie. -/
theorem native_closed_positive_limit_margin_zero (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = nativeSubweightStep remaining bound b1 b2 eps decay rate (state n))
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter))
    (hr : ∀ i, 0 < (reference i).parameter) :
    Tendsto (fun n => (state n 0).parameter * (state n 1).parameter -
      (state n 2).parameter * (state n 3).parameter) atTop (nhds 0) := by
  have ha := (native_closed_positive_limit_allocation remaining bound b1 b2 eps decay rate
    state reference hstep hclip hb1 h1 hb2 h2 he heta hp hr).2
  have hm : (reference 0).parameter * (reference 1).parameter -
      (reference 2).parameter * (reference 3).parameter = 0 := by
    rw [ha 1, ha 2, ha 3]
    ring
  simpa only [hm] using ((hp 0).mul (hp 1)).sub ((hp 2).mul (hp 3))

example :
    let reference := seededNativeSubweights ((1, 1), (1, 1))
    let eps := nativeCEGradientScale 111 1 reference
    let state := nativeSubweightPath 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) reference
    (∀ n, state (n + 1) = nativeSubweightStep 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) ∧
    (∀ i, 0 < (reference i).parameter) := by
  dsimp only
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_ce_gradient_scale_pos 111 1 _ (by norm_num), by norm_num, ?_, ?_⟩
  · intro i
    have hr : (seededNativeSubweights ((1, 1), (1, 1)) i).parameter = 1 := by fin_cases i <;> rfl
    rw [hr]
    exact native_unit_path_parameter_tendsto 111 1 (9 / 10) (49 / 50) (1 / 1000) i
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Actual held-out CE has a positive finite limit on every closed
native path with positive finite parameter limits. Sources: appendix C
multiclass test CE and native feedback at 5168744; the input and buffer
limits are derived, while parameter convergence remains explicit. -/
theorem native_closed_positive_limit_heldout_ce (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hstep : ∀ n, state (n + 1) = nativeSubweightStep remaining bound b1 b2 eps decay rate (state n))
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter))
    (hr : ∀ i, 0 < (reference i).parameter) :
    let loss := fun s : NativeSubweightState => Transformer.Grokking.NaiveLoss.crossEntropy
      (heldoutTableLogits remaining ((s 0).parameter * (s 1).parameter)
        ((s 2).parameter * (s 3).parameter)) 0
    Tendsto (fun n => loss (state n)) atTop (nhds (loss reference)) ∧ Real.log 2 ≤ loss reference ∧
      ¬Tendsto (fun n => loss (state n)) atTop (nhds 0) := by
  dsimp only
  have hf := native_positive_balanced_point_heldout_ce_floor remaining bound eps decay reference hclip he hr
    (fun i => native_closed_limit_balance remaining bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp)
  have ht := native_heldout_ce_tendsto remaining state reference hp
  refine ⟨ht, hf, ?_⟩
  intro hz
  have heq := tendsto_nhds_unique ht hz
  rw [heq] at hf
  have hl : 0 < Real.log 2 := Real.log_pos (by norm_num)
  linarith

example :
    let reference := seededNativeSubweights ((1, 1), (1, 1))
    let eps := nativeCEGradientScale 111 1 reference
    let state := nativeSubweightPath 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) reference
    (∀ n, state (n + 1) = nativeSubweightStep 111 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) (state n)) ∧
    (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧
    (∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) ∧
    (∀ i, 0 < (reference i).parameter) := by
  dsimp only
  refine ⟨fun n => rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_ce_gradient_scale_pos 111 1 _ (by norm_num), by norm_num, ?_, ?_⟩
  · intro i
    have hr : (seededNativeSubweights ((1, 1), (1, 1)) i).parameter = 1 := by fin_cases i <;> rfl
    rw [hr]
    exact native_unit_path_parameter_tendsto 111 1 (9 / 10) (49 / 50) (1 / 1000) i
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
