module

public import Transformer.Grokking.FiniteRecovery.AdamWTrials
public import Transformer.Grokking.FiniteRecovery.ArgandPAGEGaussian

/-! Generalization of finite-horizon AdamW across an identifiable finite
class. Native strict representability yields a common positive total-rate
budget and a positive Gaussian basin for each member, independently of
the minibatch order. Independent proposals and an observed-label verifier
then attain any prescribed confidence for the entire finite class. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Filter Set Metric
open scoped NNReal ENNReal Topology

variable {A X : Type*} [Fintype A] [Finite X] {T H d L : ℕ}

/-- One positive-mass native basin and one cumulative-rate bound work
at all horizons and for all minibatch streams. -/
theorem adamW_native_rate_basin (r : ℕ) (prompt : X → Word A (T+1)) (f : X → A)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (witness : ComplexParameter A H d L → ℂ)
    (hw : StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true witness)).logits r
      (prompt x) (Fin.last T)) f) :
    ∃ δ > (0:ℝ), ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) →
        ∀ batch : ℕ → List (LabeledPosition A), ∀ n, (∑ t ∈ Finset.range n, rate t) ≤ δ →
          StrictOn (fun x => (Model.fromWeights (adamWRun r batch rate z n).weights).logits r
            (prompt x) (Fin.last T)) f := by
  let init := sgdInitialWeights (A := A) (H := H) (d := d) (layers := L) true
  let c := init witness
  obtain ⟨ε,hε,hball⟩ := Metric.isOpen_iff.mp (strict_prompt_weights_open r prompt f) c hw
  let ρ := min 1 (ε/3)
  have hρ : 0 < ρ := lt_min zero_lt_one (by positivity)
  let U := init ⁻¹' ball c ρ
  have hU : IsOpen U := isOpen_ball.preimage (continuous_sgdInitialWeights true)
  let := independentGaussianDraws_openPos variance hv
  let C := max (‖c‖+1) (120:ℝ)/10 + 12
  have hC : 0 < C := by dsimp [C]; positivity
  refine ⟨ε/(3*C),by positivity,U,hU,hU.measure_pos _ ⟨witness,mem_ball_self hρ⟩,?_⟩
  intro z hz rate hr batch n hs
  have hi : dist (init z) c < ρ := hz
  have hnorm : ‖init z‖ ≤ ‖c‖+1 := by
    have he := norm_sub_norm_le (init z) c
    rw [← dist_eq_norm] at he
    linarith [hi.trans_le (min_le_left _ _)]
  have hscale : max ‖init z‖ 120 / 10 + 12 ≤ C := by
    dsimp only [C]
    linarith [max_le_max_right (120:ℝ) hnorm]
  let fit := adamWRun r batch rate z
  have hstep (t) : dist (fit t).weights (fit (t+1)).weights ≤ C * rate t :=
    (AdamW.step_dist rate _ (init z) hr t).trans (mul_le_mul_of_nonneg_right hscale (hr t).1)
  have hd := dist_le_range_sum_of_dist_le (f := fun t => (fit t).weights) n
    (d := fun t => C * rate t) (fun {t} _ => hstep t)
  rw [← Finset.mul_sum] at hd
  have hd' : dist (init z) (fit n).weights ≤ ε/3 := by
    refine hd.trans ((mul_le_mul_of_nonneg_left hs hC.le).trans_eq ?_)
    field_simp
  apply hball
  apply Metric.mem_ball.mpr
  calc
    dist (fit n).weights c ≤ dist (fit n).weights (init z) + dist (init z) c := dist_triangle _ _ _
    _ < ε := by rw [dist_comm (fit n).weights]; linarith [hi.trans_le (min_le_right _ _)]

/-- A native realizing draw yields a positive-measure event of strictly
correct trained endpoints and earlier iterates at a fixed horizon K. -/
theorem adamW_native_basin (r K : ℕ) (prompt : X → Word A (T+1)) (f : X → A)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (witness : ComplexParameter A H d L → ℂ)
    (hw : StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true witness)).logits r
      (prompt x) (Fin.last T)) f) :
    ∃ δ > (0:ℝ), ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) →
        (∑ t ∈ Finset.range K, rate t) ≤ δ → ∀ batch : ℕ → List (LabeledPosition A), ∀ n ≤ K,
          StrictOn (fun x => (Model.fromWeights (adamWRun r batch rate z n).weights).logits r
            (prompt x) (Fin.last T)) f := by
  obtain ⟨δ,hδ,U,ho,hp,hU⟩ := adamW_native_rate_basin r prompt f variance hv witness hw
  refine ⟨δ,hδ,U,ho,hp,fun z hz rate hr hs batch n hn => hU z hz rate hr batch n ?_⟩
  exact (Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hn) (fun i _ _ => (hr i).1)).trans hs

/-- The same rate budget and number of restarts work uniformly for all
members of the class. Every trial may use different/adaptive minibatches.
Only labels on S enter the verifier. Failure by m is bounded by a geometric
tail that becomes smaller than the requested error probability. -/
theorem adamW_uniform_class_confidence (r K : ℕ) (prompt : X → Word A (T+1))
    (S : Set X) (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (variance : ComplexParameter A H d L → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (hrep : ∀ f ∈ family, ∃ z, StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true
      (z : ComplexParameter A H d L → ℂ))).logits r (prompt x) (Fin.last T)) f)
    {failure : ℝ≥0∞} (hfailure : 0 < failure) :
    let μ := independentGaussianDraws variance
    let P := Measure.infinitePi (fun _ : ℕ => μ)
    ∃ δ > (0:ℝ), ∃ N, ∀ m ≥ N, ∀ f ∈ family, ∀ rate : ℕ → ℝ,
      (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) → (∑ t ∈ Finset.range K, rate t) ≤ δ →
      ∀ batch : (ℕ → ComplexParameter A H d L → ℂ) → ℕ → ℕ → List (LabeledPosition A),
        P {z | ∀ j ∈ Finset.range m, ¬classCertificate S family (fun x => f x)
          (fun x => (Model.fromWeights (adamWRun r (batch z j) rate (z j) K).weights).logits r
            (prompt x) (Fin.last T))} < failure := by
  classical
  let μ := independentGaussianDraws variance
  let P := Measure.infinitePi (fun _ : ℕ => μ)
  choose witness hw using fun f : family => hrep f f.property
  choose δ hδ U hU hp basin using fun f : family =>
    adamW_native_basin r K prompt f.val variance hv (witness f) (hw f)
  obtain ⟨a,ha,hle⟩ := exists_pos_le_all δ hδ
  obtain ⟨N,hN⟩ := uniform_restart_confidence (fun f : family => μ (U f)) hp hfailure
  refine ⟨a,ha,N,fun m hm f hf rate hr hs batch => ?_⟩
  let i : family := ⟨f,hf⟩
  have hmiss := iid_miss_window μ (hU i).measurableSet 0 m
  simp only [zero_add, Nat.Ico_zero_eq_range] at hmiss
  apply lt_of_le_of_lt _ (hN m hm i)
  rw [← hmiss]
  apply measure_mono
  intro z hz j hj hu
  exact hz j hj ((classCertificate_iff _ _ hid f hf _).2
    (basin i _ hu rate hr (hs.trans (hle i)) (batch z j) K le_rfl))

/-- Binary modular sum strictly represented by the native initialization,
with all full Argand tensors, score, and gates updated for K minibatches. -/
theorem modularSum_adamW_native_basin {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1))
    (hk : (k+1)+(k+1) ≤ 32) (K : ℕ) (variance : ComplexParameter (ZMod 2) 4 16 6 → ℝ≥0)
    (hv : ∀ i, 0 < variance i) :
    ∃ δ > (0:ℝ), ∃ U, IsOpen U ∧ 0 < independentGaussianDraws variance U ∧
      ∀ z ∈ U, ∀ rate : ℕ → ℝ, (∀ t, 0 ≤ rate t ∧ rate t ≤ 10) →
        (∑ t ∈ Finset.range K, rate t) ≤ δ → ∀ batch : ℕ → List (LabeledPosition (ZMod 2)), ∀ n ≤ K,
          (Model.fromWeights (adamWRun 1 batch rate z n).weights).realizes (T := (k+1)+k) 1 (modularSum e) := by
  obtain ⟨z,hz⟩ := modularSum_initial_or_complement e hk 0
  simp only [add_zero] at hz
  exact adamW_native_basin (T := (k+1)+k) 1 K id (modularSum e) variance hv z hz

example := modularSum_adamW_native_basin (wordEnumeration 2 4) (by decide) 100000 (fun _ => 1) (by intro; norm_num)
example := modularSum_adamW_native_basin (wordEnumeration 2 8) (by decide) 100000 (fun _ => 1) (by intro; norm_num)
example := modularSum_adamW_native_basin (wordEnumeration 2 16) (by decide) 100000 (fun _ => 1) (by intro; norm_num)

private def twoWords : Set (Word Bool 8) := {w | w = (fun _ => false) ∨ w = (fun _ => true)}

private theorem twoWords_identify :
    InjOn (fun f : Word Bool 8 → Bool => fun x : twoWords => f x) (lastTokenFamily 7) := by
  rintro _ ⟨a,rfl⟩ _ ⟨b,rfl⟩ he
  funext x
  let w : Word Bool 8 := fun _ => x (Fin.last 7)
  have hw : w ∈ twoWords := by
    cases h : x (Fin.last 7)
    · exact Or.inl (funext fun _ => h)
    · exact Or.inr (funext fun _ => h)
  exact congrFun he ⟨w,hw⟩

private theorem three_native_draws (f : Word Bool 8 → Bool) (hf : f ∈ lastTokenFamily 7) :
    ∃ z : ComplexParameter Bool 1 1 2 → ℂ, StrictOn (fun x =>
      (Model.fromWeights (sgdInitialWeights true z)).logits 1 x (Fin.last 7)) f := by
  obtain ⟨mode,rfl⟩ := hf
  refine ⟨binaryWitnessDraw 1 mode, fun x a ha => ?_⟩
  have he := (by decide : ∀ c d : Bool, c ≠ d → c = !d) a (lastTokenTask mode x) ha
  rw [he]
  exact sub_pos.mp ((by norm_num : (0:ℝ) < 1/4).trans_le
    (binaryWitness_gap (layers := 2) le_rfl mode 1 x (Fin.last 7)))

/-- A single confidence threshold covers the two constants and last-token
copying on 256 inputs, when the two homogeneous words are observed. -/
example := adamW_uniform_class_confidence (T := 7) 1 100000 id twoWords (lastTokenFamily 7)
  twoWords_identify (fun _ => 1) (by intro; norm_num) three_native_draws (failure := 1/4) (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
