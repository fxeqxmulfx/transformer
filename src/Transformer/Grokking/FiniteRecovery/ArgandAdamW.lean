module

public import Transformer.Grokking.FiniteRecovery.AdamWTrajectory
public import Transformer.Grokking.FiniteRecovery.ArgandCandidateSelection
public import Transformer.Grokking.FiniteRecovery.ArgandBinaryInitialization
public import Transformer.Grokking.FiniteRecovery.NativeConvergence

/-! Native Argand with the ordinary decoupled AdamW update. The WSD horizon
is the length of the finite experiment. An explicitly chosen summable
continuation after that horizon has a uniformly bounded displacement for
every minibatch stream. This transfers an identifiable checkpoint to
strict recovery throughout the continuation and at its limit. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Set
open scoped Topology

variable {A : Type*} [Fintype A] {T H d L : ℕ}

/-- The native start sets the score to one, gate offsets to zero, and
keeps all their coordinates trainable. Each step uses the labeled positions
of the supplied minibatch, including multiple targets in a single context. -/
noncomputable def adamWRun (r : ℕ) (batches : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  AdamW.run rate (fun t => batchGradient r (batches t)) (sgdInitialWeights true draw)

noncomputable def adamWContinuedRate (budget : ℕ) (a : ℝ) (t : ℕ) : ℝ :=
  if t < budget then wsdRate budget t else a * (1/2 : ℝ)^(t-budget)

/-- A continuation preserves the entire original optimizer state through B. -/
theorem adamW_prefix (r B : ℕ) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) (a : ℝ) {n : ℕ} (hn : n ≤ B) :
    adamWRun r batch (adamWContinuedRate B a) draw n = adamWRun r batch (wsdRate B) draw n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have he := ih (n.le_succ.trans hn)
    change AdamW.run _ _ _ n = AdamW.run _ _ _ n at he
    unfold adamWRun
    simp only [AdamW.run, he, adamWContinuedRate, ite_eq_left (Nat.lt_of_succ_le hn)]

/-- A single bound controls every later iterate and its limit in the
sup norm of all real parameters. For the implemented continuation a = 10⁻⁸. -/
theorem adamW_tail_ball (r B : ℕ) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) {a : ℝ} (ha : 0 ≤ a) (ha' : a ≤ 10) :
    let fit := adamWRun r batch (adamWContinuedRate B a) draw
    let endW := (adamWRun r batch (wsdRate B) draw B).weights
    let C := max ‖sgdInitialWeights true draw‖ 120 / 10 + 12
    ∃ w, Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧
      dist endW w ≤ C * (2*a) ∧ ∀ k, dist endW (fit (B+k)).weights ≤ C * (2*a) := by
  let rate := adamWContinuedRate B a
  let g := fun t => batchGradient (H := H) (d := d) (layers := L) r (batch t)
  let x := sgdInitialWeights true draw
  let C := max ‖x‖ (120 : ℝ) / 10 + 12
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hr (t : ℕ) : 0 ≤ rate t ∧ rate t ≤ 10 := by
    dsimp only [rate, adamWContinuedRate]
    split
    · exact ⟨(wsdRate_bounds B t).1, (wsdRate_bounds B t).2.trans (by norm_num)⟩
    · exact ⟨by positivity,
        (mul_le_of_le_one_right ha (pow_le_one₀ (by norm_num) (by norm_num))).trans ha'⟩
  have ht (k : ℕ) : rate (B+k) = a * (1/2 : ℝ)^k := by simp [rate, adamWContinuedRate]
  have hsum : HasSum (fun k => rate (B+k)) (2*a) := by
    simpa only [ht, mul_comm 2 a] using hasSum_geometric_two.mul_left a
  have hs : Summable rate := (summable_nat_add_iff B).mp (by
    simpa only [Nat.add_comm B] using hsum.summable)
  obtain ⟨w, hw, hb⟩ := AdamW.limit_and_tail rate g x hr hs
  have he := congrArg AdamW.State.weights (adamW_prefix r B batch draw a (n := B) le_rfl)
  change (AdamW.run rate g x B).weights = _ at he
  dsimp only
  rw [← he]
  refine ⟨w, hw, (hb B).trans_eq (congrArg (C * ·) hsum.tsum_eq), fun k => ?_⟩
  let s := AdamW.run rate g x
  change dist (s B).weights (s (B+k)).weights ≤ _
  have hdist (j : ℕ) : dist (s (B+j)).weights (s (B+(j+1))).weights ≤ C * rate (B+j) :=
    AdamW.step_dist rate g x hr (B+j)
  have hfin := dist_le_range_sum_of_dist_le (f := fun j => (s (B+j)).weights)
    k (d := fun j => C * rate (B+j)) (fun {j} _ => hdist j)
  simp only [Nat.add_zero] at hfin
  exact hfin.trans ((hsum.summable.mul_left C).sum_le_tsum _ (fun j _ =>
    mul_nonneg hC (hr (B+j)).1) |>.trans_eq (hsum.mul_left C).tsum_eq)

variable [Nonempty A]

/-- On any identifying observation mask, a strict checkpoint whose decoded
function is in the admissible family and agrees with the observations can
be continued by positive AdamW steps. A sufficiently small geometric tail
preserves all hidden answers at every update and at the full weight limit.
The certificate uses membership and observed labels alone. -/
theorem adamW_identified_checkpoint (r B : ℕ) (batch : ℕ → List (LabeledPosition A))
    (draw : ComplexParameter A H d L → ℂ) (inputs : Finset (Word A (T+1)))
    (family : Set (Word A (T+1) → A)) (f : Word A (T+1) → A) (hf : f ∈ family)
    (hid : InjOn (observe inputs) family)
    (hfit : fitsObserved r (observe inputs f) (adamWRun r batch (wsdRate B) draw B).weights)
    (hclass : (Model.fromWeights (adamWRun r batch (wsdRate B) draw B).weights).predictedToken r ∈ family) :
    ∃ cap > (0 : ℝ), ∀ a, 0 < a → a ≤ cap →
      let fit := adamWRun r batch (adamWContinuedRate B a) draw
      ∃ w, Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧ (Model.fromWeights w).realizes r f ∧
        ∀ k, (Model.fromWeights (fit (B+k)).weights).realizes r f := by
  let x := (adamWRun r batch (wsdRate B) draw B).weights
  let g := (Model.fromWeights x).predictedToken (T := T) r
  have he : g = f := hid hclass hf (observe_eq_iff_eqOn.mpr (fun y hy =>
    hfit.2 (y, f y) (mem_observe.mpr ⟨hy,rfl⟩)))
  have hx : (Model.fromWeights x).realizes r f := he ▸ hfit.1
  obtain ⟨δ, hδ, hball⟩ := Metric.isOpen_iff.mp (isOpen_realizingWeights (H := H) (d := d) (layers := L) r f) x hx
  let C := max ‖sgdInitialWeights true draw‖ (120 : ℝ) / 10 + 12
  have hC : 0 < C := by dsimp [C]; positivity
  refine ⟨min 10 (δ/(4*C)), lt_min (by norm_num) (div_pos hδ (by positivity)), fun a ha hcap => ?_⟩
  obtain ⟨w, hw, hlim, hsteps⟩ := adamW_tail_ball r B batch draw ha.le (hcap.trans (min_le_left _ _))
  have ht := (le_div_iff₀ (by positivity : 0 < 4*C)).mp (hcap.trans (min_le_right _ _))
  have small : C * (2*a) < δ := by nlinarith
  exact ⟨w, hw, hball (Metric.mem_ball.mpr ((dist_comm w x) ▸ (hlim.trans_lt small))),
    fun k => hball (Metric.mem_ball.mpr ((dist_comm _ x) ▸ ((hsteps k).trans_lt small)))⟩

example (batches : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :=
  adamW_tail_ball 1 20000 batches draw (a := 1/100000000) (by norm_num) (by norm_num)

example (B t : ℕ) : adamWContinuedRate B (1/100000000) t = summableWsdRate B t := rfl

/-- A minimum Flat Spectrum mask provides the same identifiability assumption
for every member. This mask is chosen before seeing the target or the draw. -/
theorem exists_minimum_adamW_identifying_mask {U V : Type*} [Fintype U] [Fintype V] [DecidableEq V]
    (cut : U × V ≃ Word A (T+1)) (code : A → ℂ) (hc : Function.Injective code)
    (family : Set (Word A (T+1) → A)) :
    ∃ inputs : Finset (Word A (T+1)),
      inputs.card = transversalNumber (recoveryHypergraph (targetKernel cut code '' family) univ
        FlatSpectrum.nuclearNorm) ∧ InjOn (observe inputs) family := by
  obtain ⟨inputs, hcard, hrecover⟩ := (isLeast_phase_observations cut code family).1
  exact ⟨inputs, hcard, (exists_learner_iff_injOn inputs family).1
    (exists_learner_of_nuclear_recovers cut code hc inputs family hrecover)⟩

/-- A two-word observation mask with genuinely hidden words, three possible
targets, trainable gates and subsequent positive AdamW steps. The witness
starts inside the certified set, so it makes no probabilistic assumption. -/
example (batches : ℕ → List (LabeledPosition Bool)) (mode : Option Bool) :
    ∃ S : Finset (Word Bool 8), S.card = 2 ∧
      ∃ cap > (0:ℝ), ∀ a, 0 < a → a ≤ cap →
        let fit := adamWRun 1 batches (adamWContinuedRate 0 a) (binaryWitnessDraw (layers := 2) 1 mode)
        ∃ w, Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧ (Model.fromWeights w).realizes 1 (lastTokenTask (T := 7) mode) ∧
          ∀ k, (Model.fromWeights (fit k).weights).realizes 1 (lastTokenTask (T := 7) mode) := by
  obtain ⟨S,hS,hlearn⟩ := (isLeast_lastToken_observations 7).1
  let draw := binaryWitnessDraw (layers := 2) 1 mode
  let f := lastTokenTask (T := 7) mode
  let x := (adamWRun 1 batches (wsdRate 0) draw 0).weights
  have hw : (Model.fromWeights x).realizes 1 f := by
    intro word a ha
    have he : a = !(f word) := (by decide : ∀ a b : Bool, a ≠ b → a = !b) a (f word) ha
    rw [he]
    exact sub_pos.mp ((by norm_num : (0:ℝ) < 1/4).trans_le
      (binaryWitness_gap (layers := 2) le_rfl mode 1 word (Fin.last 7)))
  have hp := (Model.fromWeights x).predictedToken_eq_of_realizes 1 hw
  have hf : f ∈ lastTokenFamily 7 := ⟨mode,rfl⟩
  have hfit : fitsObserved 1 (observe S f) x := by
    change (Model.fromWeights x).realizes 1 _ ∧ _
    rw [hp]
    exact ⟨hw,fun p h => (mem_observe.mp h).2⟩
  have h := adamW_identified_checkpoint 1 0 batches draw S (lastTokenFamily 7) f hf
    ((exists_learner_iff_injOn _ _).mp hlearn) hfit (hp ▸ hf)
  exact ⟨S,hS,by simpa only [Nat.zero_add] using h⟩

end Transformer.Grokking.FiniteRecovery.Argand
