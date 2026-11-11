module

public import Transformer.Grokking.Protocol.Adam
public import Transformer.Grokking.Protocol.Decision
public import Transformer.Grokking.Protocol.Sampler
public import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
public import Mathlib.MeasureTheory.Integral.DominatedConvergence
public import Mathlib.Probability.Distributions.Uniform
public import Mathlib.Probability.ProductMeasure

/-!
# The convergence target: held-out accuracy tends to `1`

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 (left) and the appendix *Additional
experimental details*; openai/grok at 3d64b1d8; `experiments/grok_division/PROTOCOL.md`,
*Randomness*. The paper reports curves; it states no theorem. This file says what a theorem
would have to state, and proves none.

A run is fixed by a sample (`Sample`): a shuffle of the `N` equations, the initial parameters,
and a shuffle of the `k` training equations for each epoch. Its law (`law`) is the product of
the uniform law on shuffles, an initialization law `ν`, and the uniform law for every epoch,
all independent. The sample fixes the trajectory (`trajectory`): the author's Adam run whose
update `t` sees the gradient `lossGrad` of the loss of batch `t` of `batchSlots`. The held-out
accuracy after `t` updates (`heldoutAccuracy`) is that of `accuracy` on the held-out equations.

"Held-out accuracy tends to `1`" has three modes for a random sequence: almost surely
(`TendsToOneAlmostSurely`), in probability (`TendsToOneInProbability`) and in mean
(`TendsToOneInMean`). For measurable accuracies the first gives the other two
(`TendsToOneAlmostSurely.inProbability`, `TendsToOneAlmostSurely.inMean`). The target
(`Generalizes`) is the first, and since an accuracy below `1` is at most `1 - 1/|H|` it says:
for almost every sample, from some update on every held-out equation is answered
(`generalizes_iff`). The model `logits`, its labels and the gradient oracle are arguments, so
the predicate is a property of them and asserts nothing by itself; `Witness` shows a model for
which it holds and one for which it fails.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

section Modes

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (acc : ℕ → Ω → ℝ)

/-- The accuracies tend to `1` for almost every sample. -/
def TendsToOneAlmostSurely : Prop := ∀ᵐ ω ∂μ, Tendsto (fun t => acc t ω) atTop (𝓝 1)

/-- The accuracies tend to `1` in probability. -/
def TendsToOneInProbability : Prop := TendstoInMeasure μ acc atTop fun _ => 1

/-- The mean accuracy tends to `1`. -/
def TendsToOneInMean : Prop := Tendsto (fun t => ∫ ω, acc t ω ∂μ) atTop (𝓝 1)

variable {μ acc}

/-- **Almost sure convergence gives convergence in probability**, for measurable accuracies. -/
theorem TendsToOneAlmostSurely.inProbability [IsFiniteMeasure μ]
    (hm : ∀ t, AEStronglyMeasurable (acc t) μ) (h : TendsToOneAlmostSurely μ acc) :
    TendsToOneInProbability μ acc :=
  tendstoInMeasure_of_tendsto_ae hm h

/-- **Almost sure convergence gives convergence in mean**, for measurable accuracies in
`[0, 1]`. -/
theorem TendsToOneAlmostSurely.inMean [IsProbabilityMeasure μ]
    (hm : ∀ t, AEStronglyMeasurable (acc t) μ) (h01 : ∀ t ω, 0 ≤ acc t ω ∧ acc t ω ≤ 1)
    (h : TendsToOneAlmostSurely μ acc) : TendsToOneInMean μ acc := by
  have key := tendsto_integral_of_dominated_convergence (fun _ => (1 : ℝ)) hm (integrable_const 1)
    (fun t => ae_of_all μ fun ω => by
      rw [Real.norm_eq_abs, abs_le]
      constructor <;> linarith [h01 t ω]) h
  simpa [TendsToOneInMean] using key

end Modes

variable {ι X C : Type*} {N k : ℕ}

/-- Every set of shuffles is an event. -/
scoped instance permMeasurableSpace (n : ℕ) : MeasurableSpace (Equiv.Perm (Fin n)) := ⊤

/-- The uniform law on the shuffles of `n` items. -/
noncomputable def uniformPerm (n : ℕ) : Measure (Equiv.Perm (Fin n)) :=
  Measure.uniformOfFinset Finset.univ

instance (n : ℕ) : IsProbabilityMeasure (uniformPerm n) := ⟨by
  have hc : ((Fintype.card (Equiv.Perm (Fin n)) : ℕ) : ENNReal) ≠ 0 :=
    Nat.cast_ne_zero.2 Fintype.card_ne_zero
  simp only [uniformPerm, Measure.uniformOfFinset, Measure.coe_finsetSum, Measure.coe_smul,
    Finset.sum_apply, Pi.smul_apply, measure_univ, smul_eq_mul, mul_one, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul]
  exact ENNReal.mul_inv_cancel hc (ENNReal.natCast_ne_top _)⟩

/-- **A sample of the protocol**: the shuffle of the `N` equations, the initial parameters, and
the shuffles of the `k` training equations, one for each epoch. -/
abbrev Sample (N k : ℕ) (ι : Type*) := Equiv.Perm (Fin N) × (ι → ℝ) × (ℕ → Equiv.Perm (Fin k))

/-- **The law of a sample**: the three parts are independent, the shuffles uniform and the
initial parameters of law `ν`. -/
noncomputable def law (N k : ℕ) (ν : Measure (ι → ℝ)) : Measure (Sample N k ι) :=
  (uniformPerm N).prod (ν.prod (Measure.infinitePi fun _ : ℕ => uniformPerm k))

instance (ν : Measure (ι → ℝ)) [IsProbabilityMeasure ν] : IsProbabilityMeasure (law N k ν) := by
  unfold law
  infer_instance

variable [Fintype C] [LinearOrder C] [Nonempty C] [DecidableEq X]

/-- **The trajectory of a sample**: the parameters after `t` updates of the Adam run with
constants `h` whose update `s` sees the gradient `lossGrad` of the loss of batch `s`, a list
of training equations, at the batch size `b`. -/
noncomputable def trajectory (h : Hyper) (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X)
    (hk : k ≤ N) (b : ℕ) (ω : Sample N k ι) (t : ℕ) : ι → ℝ :=
  (adamRun h (fun s θ => lossGrad ((batchSlots k b ω.2.2 s).map (trainAt eqs hk ω.1)) θ)
    ω.2.1 t).params

/-- **The held-out accuracy** of a sample after `t` updates, for the model `logits` with the
labels `label`. -/
noncomputable def heldoutAccuracy (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (h : Hyper)
    (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X) (hk : k ≤ N) (b : ℕ) (t : ℕ)
    (ω : Sample N k ι) : ℝ :=
  accuracy (logits (trajectory h lossGrad eqs hk b ω t)) label (heldout eqs k ω.1)

/-- **The target**: for almost every sample the held-out accuracy tends to `1`. -/
def Generalizes (ν : Measure (ι → ℝ)) (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (h : Hyper)
    (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X) (hk : k ≤ N) (b : ℕ) : Prop :=
  TendsToOneAlmostSurely (law N k ν) (heldoutAccuracy logits label h lossGrad eqs hk b)

/-- **The target says that the held-out equations are eventually all answered**: for almost
every sample, from some update on the model predicts the label of every held-out equation. -/
theorem generalizes_iff {ν : Measure (ι → ℝ)} {logits : (ι → ℝ) → X → C → ℝ} {label : X → C}
    {h : Hyper} {lossGrad : List X → (ι → ℝ) → ι → ℝ} {eqs : Fin N → X} (hk : k < N) {b : ℕ} :
    Generalizes ν logits label h lossGrad eqs hk.le b ↔
      ∀ᵐ ω ∂law N k ν, ∀ᶠ t in atTop, ∀ x ∈ heldout eqs k ω.1,
        predict (logits (trajectory h lossGrad eqs hk.le b ω t) x) = label x :=
  eventually_congr (Eventually.of_forall fun ω =>
    tendsto_accuracy_one_iff (heldout_nonempty eqs hk ω.1))

/-- The held-out accuracy lies in `[0, 1]`. -/
lemma heldoutAccuracy_mem (logits : (ι → ℝ) → X → C → ℝ) (label : X → C) (h : Hyper)
    (lossGrad : List X → (ι → ℝ) → ι → ℝ) (eqs : Fin N → X) (hk : k ≤ N) (b t : ℕ)
    (ω : Sample N k ι) :
    0 ≤ heldoutAccuracy logits label h lossGrad eqs hk b t ω ∧
      heldoutAccuracy logits label h lossGrad eqs hk b t ω ≤ 1 :=
  ⟨accuracy_nonneg, accuracy_le_one⟩

end Transformer.Grokking.Protocol
