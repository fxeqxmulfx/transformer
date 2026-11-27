module

public import Transformer.Grokking.FiniteRecovery.PeriodicForecast
public import Transformer.Grokking.FiniteRecovery.PhaseDecoding
public import Transformer.Grokking.FiniteRecovery.SeriesCodec

/-! Fitting the observed entries gives geometric convergence for additive
periodic forecasts and eventual exact token forecasts for covered encoded
contexts. Both procedures use the same half step on squared error. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Filter
open scoped Topology

section HalfFit

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- Move halfway from the current value toward the observed target.
This is a gradient step of size one quarter for squared distance on a real
inner product space. -/
noncomputable def halfFit (value initial : E) : ℕ → E
  | 0 => initial
  | step + 1 => value + (1 / 2 : ℝ) • (halfFit value initial step - value)

theorem halfFit_eq (value initial : E) (step : ℕ) :
    halfFit value initial step = value + (1 / 2 : ℝ) ^ step • (initial - value) := by
  induction step with
  | zero => simp [halfFit]
  | succ step ih => rw [halfFit, ih, add_sub_cancel_left, smul_smul, pow_succ']

theorem halfFit_tendsto [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul ℝ E] (value initial : E) :
    Tendsto (halfFit value initial) atTop (𝓝 value) := by
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
  change Tendsto (fun step => halfFit value initial step) atTop (𝓝 value)
  simpa only [halfFit_eq, zero_smul, add_zero] using (hpow.smul_const (initial - value)).const_add value

example := halfFit_eq (1 : ℂ) 0 2

example := halfFit_tendsto (1 : ℝ) 0

end HalfFit

section Periodic

variable {R : Type*} {period : ℕ} [NeZero period]

/-- At every horizon, the forecast error is halved at each training step. -/
theorem periodicForecast_halfFit (observedRow startRow : R → ℝ)
    (observedTime startTime : Fin period → ℝ) (r : R) (time step : ℕ) :
    periodicForecast (fun i => halfFit (observedRow i) (startRow i) step)
      (fun t => halfFit (observedTime t) (startTime t) step) r time =
      halfFit (periodicForecast observedRow observedTime r time)
        (periodicForecast startRow startTime r time) step := by
  simp only [periodicForecast, halfFit_eq, smul_eq_mul]
  ring

/-- Only time zero on each series and one full period on a reference
series are fitted. They give a convergent forecast at every future time. -/
theorem periodicFit_converges (offset startRow : R → ℝ) (shape startTime : Fin period → ℝ)
    (referenceRow r : R) (time : ℕ) :
    Tendsto (fun step => periodicForecast
      (fun i => halfFit (offset i + shape (periodIndex period 0)) (startRow i) step)
      (fun t => halfFit (offset referenceRow + shape t) (startTime t) step) r time)
      atTop (𝓝 (offset r + shape (periodIndex period time))) := by
  rw [← periodicForecast_eq offset shape referenceRow r time]
  simpa only [periodicForecast_halfFit] using
    halfFit_tendsto (periodicForecast (fun i => offset i + shape (periodIndex period 0))
      (fun t => offset referenceRow + shape t) r time) (periodicForecast startRow startTime r time)

example (a : Fin 1024 → ℝ) (b : Fin 4096 → ℝ) (r : Fin 1024) :=
  periodicFit_converges a (fun _ => 0) b (fun _ => 0) 0 r 1000000

example (a : Fin 32 → ℝ) (b : Fin 128 → ℝ) (r : Fin 32) :=
  periodicFit_converges a (fun _ => 0) b (fun _ => 0) 0 r 128

example (a : Fin 32 → ℝ) (b : Fin 128 → ℝ) (r : Fin 32) :=
  periodicForecast_halfFit a (fun _ => 0) b (fun _ => 0) r 128 10

end Periodic

section Encoded

variable {I X A Y : Type*}

/-- Fit a codeword independently for each observed encoded context.
A context with no label keeps its initial value. -/
noncomputable def fitEncodedContext (context : I → X) (sites : Finset I) (labels : ↥sites → A)
    (code : A → ℂ) (initial : X → ℂ) (step : ℕ) (x : X) : ℂ :=
  open Classical in
  if h : ∃ i : ↥sites, context i = x then
    halfFit (code (labels (Classical.choose h))) (initial x) step
  else initial x

/-- One observation per occurring context suffices. Compatibility ensures
that each selected representative supplies the right label for its whole fiber. -/
theorem SeriesEncoding.halfFit_eventually_decoded [Finite I] [Finite A] [Nonempty A]
    (encoding : SeriesEncoding I X A Y) (sites : Finset I) (target : I → A)
    (hcover : ∀ i, ∃ j : ↥sites, encoding.context j = encoding.context i)
    (hcompat : ∀ i j, encoding.context i = encoding.context j → target i = target j)
    (code : A → ℂ) (hinj : Function.Injective code) (initial : X → ℂ) :
    ∀ᶠ step in atTop,
      encoding.decoded (fun x => phaseDecode code
        (fitEncodedContext encoding.context sites (fun j => target j) code initial step x)) =
        fun i => encoding.decode i (target i) := by
  have h : ∀ᶠ step in atTop, ∀ i, phaseDecode code
      (fitEncodedContext encoding.context sites (fun j => target j) code initial step (encoding.context i)) =
        target i := by
    apply eventually_all.mpr
    intro i
    have hc := hcover i
    have he := hcompat (Classical.choose hc : ↥sites) i (Classical.choose_spec hc)
    apply phaseDecode_eventually_of_tendsto code hinj
    simpa only [fitEncodedContext, dite_eq_left hc, he] using
      halfFit_tendsto (code (target i)) (initial (encoding.context i))
  exact h.mono fun _ hs => funext fun i => congrArg (encoding.decode i) (hs i)

example (a : ZMod 3) :=
  SeriesEncoding.halfFit_eventually_decoded
    (⟨fun _ : Bool => (), fun i token => if i then token + 1 else token⟩ :
      SeriesEncoding Bool Unit (ZMod 3) (ZMod 3))
    {false} (fun _ => a) (fun _ => ⟨⟨false, Finset.mem_singleton_self _⟩, rfl⟩)
    (fun _ _ _ => rfl) ZMod.stdAddChar ZMod.injective_stdAddChar (fun _ => 0)

/-- Covered LGT, MASH and MASH-half windows eventually forecast the correct
next grid levels when the labels agree on contexts and no clipping occurs. -/
theorem FirstDifferenceGrid.halfFit_eventually_level [Finite I] {T vocab : ℕ} (hv : 0 < vocab)
    (grid : I → FirstDifferenceGrid) (past : I → Fin (T + 1) → ℝ) (future : I → ℝ)
    (sites : Finset I) (code : Fin vocab → ℂ) (hinj : Function.Injective code)
    (initial : Word (Fin vocab) T → ℂ) :
    letI : Nonempty (Fin vocab) := ⟨⟨0, hv⟩⟩
    let encoding := FirstDifferenceGrid.windowEncoding hv grid past
    let step i := (grid i).step T (past i (Fin.last T)) (future i)
    let target i := FirstDifferenceGrid.clipToken hv (step i)
    (∀ i, ∃ j : ↥sites, encoding.context j = encoding.context i) →
    (∀ i j, encoding.context i = encoding.context j → target i = target j) →
    (∀ i (j : Fin T), 0 ≤ (grid i).step j.val (past i j.castSucc) (past i j.succ) ∧
      (grid i).step j.val (past i j.castSucc) (past i j.succ) < vocab) →
    (∀ i, 0 ≤ step i ∧ step i < vocab) →
    ∀ᶠ k in atTop,
      encoding.decoded (fun x => phaseDecode code
        (fitEncodedContext encoding.context sites (fun j => target j) code initial k x)) =
        fun i => (grid i).level (T + 1) (future i) := by
  let : Nonempty (Fin vocab) := ⟨⟨0, hv⟩⟩
  dsimp only
  intro hcover hcompat hprefix hnext
  have h := (FirstDifferenceGrid.windowEncoding hv grid past).halfFit_eventually_decoded
    sites _ hcover hcompat code hinj initial
  apply h.mono
  intro k hk
  rw [hk]
  funext i
  exact FirstDifferenceGrid.windowEncoding_decode_step hv grid past i (future i)
    (hprefix i) (hnext i).1 (hnext i).2

private def unitGrid (tokenizer : SeriesTokenizer) : FirstDifferenceGrid := ⟨tokenizer, 1, 128, 0⟩

example (codec : SeriesTokenizer) (code : Fin 256 → ℂ) (hinj : Function.Injective code) :=
  FirstDifferenceGrid.halfFit_eventually_level (T := 0) (by decide) (fun _ : Unit => unitGrid codec)
    (fun _ _ => 0) (fun _ => 0) {()} code hinj (fun _ => 0)
    (fun _ => ⟨⟨(), Finset.mem_singleton_self _⟩, rfl⟩) (fun ⟨⟩ ⟨⟩ _ => rfl)
    (fun _ j => j.elim0) (by
      intro i
      cases codec <;> norm_num [unitGrid, FirstDifferenceGrid.step, FirstDifferenceGrid.offset,
        FirstDifferenceGrid.level, FirstDifferenceGrid.fraction, Int.fract, Int.floor_zero,
        Int.floor_div_natCast])

end Encoded

end Transformer.Grokking.FiniteRecovery
