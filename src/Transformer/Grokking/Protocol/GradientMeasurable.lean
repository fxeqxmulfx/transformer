module

public import Mathlib.Analysis.Calculus.FDeriv.Measurable
public import Transformer.Grokking.Protocol.Continuity
public import Transformer.Grokking.Protocol.Differentiability
public import Transformer.Grokking.Protocol.TargetMean

/-!
# The gradient of the backward pass is a measurable function of the parameters

Source: openai/grok at 3d64b1d8, `grok/training.py`, `TrainableTransformer._step`, and
PyTorch's backward pass; `experiments/grok_division/PROTOCOL.md`, *The loss and the accuracy*;
`review/grok_division_convergence.md`, bridge 2.

The gates of `sourceGrad` are indicators of the signs of the hidden inputs, so no continuity
of it in the parameters is used here. It is measurable (`measurable_sourceGrad`), which is
what the law of a run needs. The gates of a row of tokens take finitely many values, the
patterns (`gatePattern`, `gate_eq_patternGate`), and the pattern is a measurable function of
the parameters because the hidden inputs are continuous (`measurable_gatePattern`). For a
fixed pattern the loss of an example is continuous in the parameters and the coordinate
together (`continuous_exampleLoss_update`), so its derivative in the coordinate is measurable
in the parameters (`measurable_deriv_exampleLoss`). The gradient of a batch is the mean of
the derivatives of its examples (`sourceGrad_eq_sum`), each with the pattern of its own row.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy sum_exp_pos)

variable {S : Sizes}

/-- A pattern of gates: which hidden units pass their input. -/
abbrev Pattern (S : Sizes) := Fin S.blocks → Fin S.context → Fin S.hidden → Bool

/-- The activations of a pattern: a unit passes its input where the pattern says so. -/
def patternGate (β : Pattern S) : Activation S := fun l t u v => if β l t u then v else 0

/-- The pattern of the gates of the backward pass at `θ₀` for the tokens read. -/
noncomputable def gatePattern (θ₀ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) :
    Pattern S :=
  fun l t u => decide (0 < preactivation θ₀ (decodeTo θ₀ (relu S) tokens l) l t u)

/-- The gates of the backward pass are those of their pattern. -/
lemma gate_eq_patternGate (θ₀ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) :
    gate θ₀ tokens = patternGate (gatePattern θ₀ tokens) := by
  funext l t u v
  simp [gate, patternGate, gatePattern]

/-- The activations of a pattern are continuous. -/
lemma continuous_patternGate (β : Pattern S) (l : Fin S.blocks) (t : Fin S.context)
    (u : Fin S.hidden) : Continuous (patternGate β l t u) := by
  unfold patternGate
  split_ifs
  · exact continuous_id
  · exact continuous_const

/-- The activations of a pattern are differentiable. -/
lemma differentiable_patternGate (β : Pattern S) (l : Fin S.blocks) (t : Fin S.context)
    (u : Fin S.hidden) : Differentiable ℝ (patternGate β l t u) := by
  unfold patternGate
  split_ifs
  · exact differentiable_fun_id
  · exact differentiable_const 0

/-- **The pattern of the gates is a measurable function of the parameters.** -/
theorem measurable_gatePattern (tokens : Fin S.context → Fin S.vocab) :
    Measurable fun θ₀ : Param S → ℝ => gatePattern θ₀ tokens :=
  measurable_pi_iff.2 fun l => measurable_pi_iff.2 fun t => measurable_pi_iff.2 fun u =>
    measurable_to_bool (by
      have hopen : IsOpen {θ₀ : Param S → ℝ |
          0 < preactivation θ₀ (decodeTo θ₀ (relu S) tokens l) l t u} :=
        isOpen_lt continuous_const (continuous_preactivation continuous_apply
          (continuous_decodeTo continuous_relu continuous_apply tokens l) l t u)
      convert hopen.measurableSet using 1
      ext θ₀
      simp [gatePattern])

/-- **With continuous activations the loss of an example is continuous in the parameters and
a coordinate together.** -/
theorem continuous_exampleLoss_update {act : Activation S}
    (hact : ∀ l t u, Continuous (act l t u)) (p : Param S) (e : Example S) :
    Continuous fun a : (Param S → ℝ) × ℝ => exampleLoss (Function.update a.1 p a.2) act e := by
  have hline : ∀ q, Continuous fun a : (Param S → ℝ) × ℝ => Function.update a.1 p a.2 q :=
    fun q => by
      by_cases h : q = p
      · subst h
        simpa using continuous_snd
      · simp only [Function.update_of_ne h]
        exact (continuous_apply q).comp continuous_fst
  unfold exampleLoss
  refine continuous_list_sum _ fun pc _ => ?_
  exact ((continuous_finsetSum _ fun q _ => Real.continuous_exp.comp
    (continuous_logits hact hline e.tokens pc.1 q)).log fun a => (sum_exp_pos _ pc.2).ne').sub
      (continuous_logits hact hline e.tokens pc.1 pc.2)

/-- **For a fixed pattern the derivative of an example's loss in a coordinate is measurable in
the parameters.** -/
theorem measurable_deriv_exampleLoss (β : Pattern S) (p : Param S) (e : Example S) :
    Measurable fun θ : Param S → ℝ =>
      deriv (fun s => exampleLoss (Function.update θ p s) (patternGate β) e) (θ p) := by
  have hc : Continuous (Function.uncurry fun (θ : Param S → ℝ) (s : ℝ) =>
      exampleLoss (Function.update θ p s) (patternGate β) e) := by
    rw [Function.uncurry_def]
    exact continuous_exampleLoss_update (continuous_patternGate β) p e
  have hm := (measurable_deriv_with_param hc).comp (measurable_id.prodMk (measurable_pi_apply p))
  exact hm

/-- The derivative of an example's loss with the gates of the current parameters is measurable
in the parameters. -/
theorem measurable_deriv_exampleLoss_gate (p : Param S) (e : Example S) :
    Measurable fun θ : Param S → ℝ =>
      deriv (fun s => exampleLoss (Function.update θ p s) (gate θ e.tokens) e) (θ p) := by
  have h : Measurable fun x : Pattern S × (Param S → ℝ) =>
      deriv (fun s => exampleLoss (Function.update x.2 p s) (patternGate x.1) e) (x.2 p) :=
    measurable_from_prod_countable_right fun β => measurable_deriv_exampleLoss β p e
  have h2 := h.comp ((measurable_gatePattern e.tokens).prodMk measurable_id)
  simp only [gate_eq_patternGate]
  exact h2

/-- With differentiable activations the loss of an example is differentiable along a
coordinate. -/
lemma differentiable_exampleLoss {act : Activation S}
    (hact : ∀ l t u, Differentiable ℝ (act l t u)) (θ : Param S → ℝ) (p : Param S)
    (e : Example S) : Differentiable ℝ fun s => exampleLoss (Function.update θ p s) act e :=
  differentiable_list_sum_map e.targets fun pc _ =>
    ((Differentiable.fun_sum fun q _ => (differentiable_logits hact (differentiable_update θ p)
      e.tokens pc.1 q).exp).log fun _ => (sum_exp_pos _ pc.2).ne').fun_sub
      (differentiable_logits hact (differentiable_update θ p) e.tokens pc.1 pc.2)

/-- **The gradient of a batch is the mean over its targets of the derivatives of its
examples**, each with the gates of its own row of tokens. -/
theorem sourceGrad_eq_sum (B : List (Example S)) (θ : Param S → ℝ) (p : Param S) :
    sourceGrad B θ p = (B.map fun e =>
      deriv (fun s => exampleLoss (Function.update θ p s) (gate θ e.tokens) e) (θ p)).sum /
        targetCount B :=
  HasDerivAt.deriv ((hasDerivAt_list_sum_map B fun e _ =>
    (differentiable_exampleLoss (differentiable_gate θ e.tokens) θ p e (θ p)).hasDerivAt).div_const
      _)

/-- A sum over a list of measurable functions is measurable. -/
lemma measurable_list_sum_map {α Ω : Type*} [MeasurableSpace Ω] (l : List α) {f : α → Ω → ℝ}
    (h : ∀ a ∈ l, Measurable (f a)) : Measurable fun ω => (l.map fun a => f a ω).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact (h a List.mem_cons_self).add (ih fun b hb => h b (List.mem_cons_of_mem a hb))

/-- **The gradient of the backward pass is measurable in the parameters**, coordinate by
coordinate and for every batch. -/
theorem measurable_sourceGrad (B : List (Example S)) (p : Param S) :
    Measurable fun θ : Param S → ℝ => sourceGrad B θ p := by
  simp only [sourceGrad_eq_sum]
  exact (measurable_list_sum_map B fun e _ => measurable_deriv_exampleLoss_gate p e).div_const _

/-! The witnesses: the loss of an example of the paper's decoder with the source's activation,
in the parameters and a coordinate together; the same along a coordinate with every gate shut;
a list of two measurable functions; and the gradient of a batch of the paper's decoder. -/

example (p : Param ⟨239, 128, 4, 32, 512, 2, 6⟩) (e : Example ⟨239, 128, 4, 32, 512, 2, 6⟩) :
    Continuous fun a : (Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ) × ℝ =>
      exampleLoss (Function.update a.1 p a.2) (relu _) e :=
  continuous_exampleLoss_update continuous_relu p e

example (θ : Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ) (p : Param ⟨239, 128, 4, 32, 512, 2, 6⟩)
    (e : Example ⟨239, 128, 4, 32, 512, 2, 6⟩) :
    Differentiable ℝ fun s =>
      exampleLoss (Function.update θ p s) (patternGate fun _ _ _ => false) e :=
  differentiable_exampleLoss (differentiable_patternGate _) θ p e

example : Measurable fun x : ℝ => ([1, 2].map fun a : ℝ => a * x).sum :=
  measurable_list_sum_map _ fun a _ => measurable_id.const_mul a

example (B : List (Example ⟨239, 128, 4, 32, 512, 2, 6⟩))
    (p : Param ⟨239, 128, 4, 32, 512, 2, 6⟩) :
    Measurable fun θ : Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ => sourceGrad B θ p :=
  measurable_sourceGrad B p

end Transformer.Grokking.Protocol
