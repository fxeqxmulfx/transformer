module

public import Mathlib.MeasureTheory.Function.SpecialFunctions.Basic
public import Transformer.Grokking.Protocol.Target

/-!
# The trajectory of a sample is a measurable function of the sample

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`, and `grok/data.py`,
`ArithmeticIterator`; `experiments/grok_division/PROTOCOL.md`, *The optimizer* and *The
split*; `review/grok_division_convergence.md`, bridge 2.

The update of the optimizer is made of sums, products, quotients and square roots, so a run
whose start and whose gradients depend measurably on a point depends measurably on it, in the
parameters and in both moments (`measurable_adamRun`). A sample is a shuffle of the equations,
initial parameters, and a shuffle for each epoch. The batch of an update is a function of two
shuffles, which take finitely many values, so if the gradient oracle is measurable in the
parameters for every batch, then every parameter after every update is a measurable function
of the sample (`measurable_trajectory`). The oracle is an argument here; `GradientMeasurable`
proves the hypothesis for the gradient of the source's backward pass.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

variable {Ω ι X : Type*} [MeasurableSpace Ω] {N k : ℕ}

/-- **A run whose start and whose gradients depend measurably on a point depends measurably on
it**, in every coordinate of the parameters and of both moments. -/
theorem measurable_adamRun (h : Hyper) {grad : Ω → ℕ → (ι → ℝ) → ι → ℝ} {θ₀ : Ω → ι → ℝ}
    (hθ : ∀ i, Measurable fun ω => θ₀ ω i)
    (hg : ∀ (s : ℕ) (y : Ω → ι → ℝ), (∀ i, Measurable fun ω => y ω i) →
      ∀ i, Measurable fun ω => grad ω s (y ω) i) (t : ℕ) :
    (∀ i, Measurable fun ω => (adamRun h (grad ω) (θ₀ ω) t).params i) ∧
      (∀ i, Measurable fun ω => (adamRun h (grad ω) (θ₀ ω) t).first i) ∧
        ∀ i, Measurable fun ω => (adamRun h (grad ω) (θ₀ ω) t).second i := by
  induction t with
  | zero => exact ⟨hθ, fun _ => measurable_const, fun _ => measurable_const⟩
  | succ t ih =>
    obtain ⟨hp, hm, hv⟩ := ih
    have hgi := hg (t + 1) _ hp
    have hm' : ∀ i, Measurable fun ω => h.beta1 * (adamRun h (grad ω) (θ₀ ω) t).first i +
        (1 - h.beta1) * grad ω (t + 1) (adamRun h (grad ω) (θ₀ ω) t).params i := fun i =>
      ((hm i).const_mul _).add ((hgi i).const_mul _)
    have hv' : ∀ i, Measurable fun ω => h.beta2 * (adamRun h (grad ω) (θ₀ ω) t).second i +
        (1 - h.beta2) * grad ω (t + 1) (adamRun h (grad ω) (θ₀ ω) t).params i ^ 2 := fun i =>
      ((hv i).const_mul _).add (((hgi i).pow_const 2).const_mul _)
    exact ⟨fun i => ((hp i).const_mul _).sub
      (((hm' i).div (((hv' i).sqrt.div_const _).add_const _)).const_mul _), hm', hv'⟩

/-- **Every parameter of a sample's trajectory is a measurable function of the sample**, if
the gradient oracle is measurable in the parameters for every batch. -/
theorem measurable_trajectory (h : Hyper) {lossGrad : List X → (ι → ℝ) → ι → ℝ}
    (hL : ∀ B i, Measurable fun θ => lossGrad B θ i) (eqs : Fin N → X) (hk : k ≤ N) (b t : ℕ)
    (i : ι) : Measurable fun ω : Sample N k ι => trajectory h lossGrad eqs hk b ω t i := by
  refine (measurable_adamRun (Ω := Sample N k ι) h
    (grad := fun ω s θ => lossGrad ((batchSlots k b ω.2.2 s).map (trainAt eqs hk ω.1)) θ)
    (θ₀ := fun ω => ω.2.1)
    (fun j => (measurable_pi_apply j).comp (measurable_fst.comp measurable_snd))
    (fun s y hy j => ?_) t).1 i
  have hF : Measurable fun x : (Equiv.Perm (Fin N) × Equiv.Perm (Fin k)) × (ι → ℝ) =>
      lossGrad (((((List.finRange k).map x.1.2).drop
        ((s - 1) % batchesPerEpoch k b * b)).take b).map (trainAt eqs hk x.1.1)) x.2 j :=
    measurable_from_prod_countable_right fun d =>
      hL (((((List.finRange k).map d.2).drop
        ((s - 1) % batchesPerEpoch k b * b)).take b).map (trainAt eqs hk d.1)) j
  have hkey : Measurable fun ω : Sample N k ι =>
      ((ω.1, ω.2.2 ((s - 1) / batchesPerEpoch k b)), y ω) :=
    (measurable_fst.prodMk ((measurable_pi_apply _).comp
      (measurable_snd.comp measurable_snd))).prodMk (measurable_pi_iff.2 hy)
  have h2 := hF.comp hkey
  exact h2

/-! The witnesses: a run on one coordinate whose gradient is the parameter itself, started at
a real number; and the trajectory of a sample for an oracle that returns the parameters. -/

example (t : ℕ) : Measurable fun x : ℝ =>
    (adamRun (Hyper.source 1) (fun _ θ => θ) (fun _ : Unit => x) t).params () :=
  (measurable_adamRun (Ω := ℝ) (Hyper.source 1) (grad := fun _ _ θ => θ)
    (θ₀ := fun x _ => x) (fun _ => measurable_id) (fun _ _ hy => hy) t).1 ()

example (t : ℕ) : Measurable fun ω : Sample 4 2 Unit =>
    trajectory (Hyper.source 0) (fun (_ : List (Fin 4)) θ => θ) id (by norm_num) 2 ω t () :=
  measurable_trajectory (Hyper.source 0) (fun _ i => measurable_pi_apply i) id (by norm_num) 2 t ()

end Transformer.Grokking.Protocol
