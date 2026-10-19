module

public import Transformer.Landscape.Neural.SwiGLU
public import Transformer.Grokking.AdamW.ScheduledParameterBox

/-!
# The Rotate90 warmup and an exact-real AdamW parameter container

Sources: `domain.training.rate` and `experiments/rotate90_analytic` at
c59d4426; the browser Fast preset at grokking_abmod revision
f49969393a7482490d45483d8a4fd3046b1adede. Betas are (0.9, 0.98), optimizer
epsilon is 1e-8, decay is 1, and warmup counts the current update too.
The base rate is variable, covering 0.01, 0.003 and 0.001 in M9.5.

The existing initialized native-moment theorem supplies a coordinate
ceiling max(abs(initial), 1.7) for any coupled gradient sequence. This
ceiling is derived from the zero buffers, retained clock and actual update;
no future bound on gradients, parameters or adaptive directions is assumed.
The resulting product of closed intervals is compact. At the finite indices
of SiluWeights and GatedWeights, it is a compact Euclidean parameter set.

This is an exact-real recurrence bridge, not a certificate for FP32 kernels.
An AdamW path in a compact set need not converge or follow a gradient flow.
No attraction, stable fitting or held-out generalization follows here.
-/

@[expose] public section

namespace Transformer.Landscape.Neural

open Transformer.Grokking.AdamW

/-- Ten inclusive warmup updates, followed by a constant base rate.
Source: domain.training.rate at c59d4426, completed update count n. -/
noncomputable def rotate90Rate (base : ℝ) (n : ℕ) : ℝ :=
  base * min 1 (((n : ℝ) + 1) / 10)

/-- These rates keep native decay nonnegative without any loss-curvature
assumption. Source: the Fast schedule with weight decay 1. -/
theorem rotate90Rate_legal {base : ℝ} (h0 : 0 ≤ base) (h1 : base ≤ 1) (n : ℕ) :
    0 ≤ rotate90Rate base n ∧ rotate90Rate base n ≤ base ∧
      0 ≤ 1 - rotate90Rate base n := by
  have hn : 0 ≤ min (1 : ℝ) (((n : ℝ) + 1) / 10) := le_min (by norm_num) (by positivity)
  have hc := min_le_left (1 : ℝ) (((n : ℝ) + 1) / 10)
  have hr : rotate90Rate base n ≤ base := by
    unfold rotate90Rate
    nlinarith only [mul_nonneg h0 (sub_nonneg.mpr hc)]
  exact ⟨mul_nonneg h0 hn, hr, by linarith⟩

/-- The first update uses base/10; the tenth and all later updates use
base. Source: the source export's increment-before-warmup indexing. -/
theorem rotate90Rate_profile (base : ℝ) :
    rotate90Rate base 0 = base / 10 ∧ ∀ n, 9 ≤ n → rotate90Rate base n = base := by
  constructor
  · norm_num [rotate90Rate, div_eq_mul_inv]
  · intro n hn
    have hc : (9 : ℝ) ≤ n := by exact_mod_cast hn
    have ht : (1 : ℝ) ≤ ((n : ℝ) + 1) / 10 := by linarith
    rw [rotate90Rate, min_eq_left ht, mul_one]

example : (0 : ℝ) ≤ 1 / 100 ∧ (1 / 100 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 3 / 1000 ∧ (3 / 1000 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (1 / 1000 : ℝ) ≤ 1 := by norm_num

example : rotate90Rate (1 / 100) 0 = 1 / 1000 ∧
    rotate90Rate (1 / 100) 9 = 1 / 100 := by norm_num [rotate90Rate]

/-- Every always-updated coordinate stays in its computed initial-data box.
Source: ScheduledParameterBox.scalar_scheduled_original_parameter_box,
specialized to the actual inclusive warmup, epsilon and decay. Gradients
may depend on every parameter, example and previous optimizer state. -/
theorem rotate90_adamw_coordinate_box {I : Type*} (base : ℝ)
    (state : ℕ → I → ScalarState) (gradient : ℕ → I → ℝ)
    (h0 : 0 ≤ base) (h1 : base ≤ 1)
    (hstep : ∀ n i, state (n + 1) i = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      1 (rotate90Rate base n) (state n i) (gradient n i))
    (hm : ∀ i, (state 0 i).moment = 0) (hv : ∀ i, (state 0 i).variance = 0)
    (hc : ∀ i, (state 0 i).clock = 0) :
    ∀ n i, |(state n i).parameter| ≤ max |(state 0 i).parameter| (17 / 10) := by
  intro n i
  have hb := scalar_scheduled_original_parameter_box (1 / 100000000) 1
    (rotate90Rate base) (fun n => state n i) (fun n => gradient n i)
    (fun n => hstep n i) (hm i) (hv i) (hc i) (by norm_num) (by norm_num)
    (fun n => ⟨(rotate90Rate_legal h0 h1 n).1, by simpa using (rotate90Rate_legal h0 h1 n).2.2⟩) n
  simpa only [div_one] using hb

/-- A coordinate box in the full parameter space, computed only from the
initial vector. Source: the native-moment bound above, not a landscape guard. -/
def adamwParameterBox {I : Type*} (initial : EuclideanSpace ℝ I) :
    Set (EuclideanSpace ℝ I) :=
  {x | ∀ i, |x.ofLp i| ≤ max |initial.ofLp i| (17 / 10)}

/-- The derived box is compact in coordinate topology, hence in the usual
Euclidean topology at the models' finite indices. Source: Tychonoff's
theorem applied to the explicit closed intervals; no compactness premise. -/
theorem isCompact_adamwParameterBox {I : Type*} (initial : EuclideanSpace ℝ I) :
    IsCompact (adamwParameterBox initial) := by
  let intervals : Set (I → ℝ) := Set.univ.pi
    (fun i => Set.Icc (-(max |initial.ofLp i| (17 / 10))) (max |initial.ofLp i| (17 / 10)))
  have hi : IsCompact intervals := isCompact_univ_pi (fun i => isCompact_Icc)
  have he : WithLp.toLp 2 '' intervals = adamwParameterBox initial := by
    ext x
    constructor
    · rintro ⟨v, hv, rfl⟩
      exact fun i => abs_le.mpr (hv i (Set.mem_univ i))
    · intro hx
      exact ⟨x.ofLp, fun i _ => abs_le.mp (hx i), rfl⟩
  rw [← he]
  exact hi.image (PiLp.continuous_toLp 2 (fun _ : I => ℝ))

/-- An initialized source-matched AdamW path has a computed compact
parameter container. Source: M9.5's actual optimizer bridge. The recurrence
is an input, but future boundedness and convergence are not. -/
theorem exists_compact_rotate90_adamw_container {I : Type*} (base : ℝ)
    (state : ℕ → I → ScalarState) (gradient : ℕ → I → ℝ)
    (h0 : 0 ≤ base) (h1 : base ≤ 1)
    (hstep : ∀ n i, state (n + 1) i = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      1 (rotate90Rate base n) (state n i) (gradient n i))
    (hm : ∀ i, (state 0 i).moment = 0) (hv : ∀ i, (state 0 i).variance = 0)
    (hc : ∀ i, (state 0 i).clock = 0) :
    ∃ container : Set (EuclideanSpace ℝ I), IsCompact container ∧
      ∀ n, WithLp.toLp 2 (fun i => (state n i).parameter) ∈ container := by
  refine ⟨adamwParameterBox (WithLp.toLp 2 (fun i => (state 0 i).parameter)),
    isCompact_adamwParameterBox _, ?_⟩
  exact rotate90_adamw_coordinate_box base state gradient h0 h1 hstep hm hv hc

/-- A pure-decay parameter used only to witness the recurrence
hypotheses. It does not assert a neural loss has zero gradient there. -/
noncomputable def fastDecayParameter (base initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => (1 - rotate90Rate base n) * fastDecayParameter base initial n

/-- Zero retained buffers with the actual advancing clock, for that witness. -/
noncomputable def fastDecayState (base initial : ℝ) (n : ℕ) : ScalarState :=
  ⟨fastDecayParameter base initial n, 0, 0, n⟩

/-- The witness satisfies the native step with zero supplied gradients.
Source: ScalarRecurrence's decay and buffer equations, not a new optimizer. -/
theorem fastDecayState_step (base initial : ℝ) (n : ℕ) :
    fastDecayState base initial (n + 1) =
      scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) 1 (rotate90Rate base n)
        (fastDecayState base initial n) 0 := by
  simp [fastDecayState, fastDecayParameter, scalarNativeStep, nextBufferDirection]

example : fastDecayParameter (1 / 100) 1 0 = 1 ∧
    fastDecayParameter (1 / 100) 1 1 = 999 / 1000 := by
  norm_num [fastDecayParameter, rotate90Rate]

-- Both container theorems have the following legal, moving initialized path.
example : (0 : ℝ) ≤ 1 / 100 ∧ (1 / 100 : ℝ) ≤ 1 ∧
    (∀ n (i : Fin 2), fastDecayState (1 / 100) (i.val + 1) (n + 1) =
      scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) 1 (rotate90Rate (1 / 100) n)
        (fastDecayState (1 / 100) (i.val + 1) n) 0) ∧
    (∀ i : Fin 2, (fastDecayState (1 / 100) (i.val + 1) 0).moment = 0) ∧
    (∀ i : Fin 2, (fastDecayState (1 / 100) (i.val + 1) 0).variance = 0) ∧
    (∀ i : Fin 2, (fastDecayState (1 / 100) (i.val + 1) 0).clock = 0) := by
  exact ⟨by norm_num, by norm_num, fun n i => fastDecayState_step _ _ n,
    fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

example (initial : SiluWeights (Fin 97) (Fin 64)) :
    IsCompact (adamwParameterBox initial) := isCompact_adamwParameterBox initial

example (initial : GatedWeights (Fin 97) (Fin 64) (Fin 43)) :
    IsCompact (adamwParameterBox initial) := isCompact_adamwParameterBox initial

end Transformer.Landscape.Neural
