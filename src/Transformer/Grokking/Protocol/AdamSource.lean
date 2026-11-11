module

public import Transformer.Grokking.Protocol.Adam

/-!
# The statements of the author's optimizer step, in the order of the code

Source: openai/grok at 3d64b1d8, `grok/training.py`, `CustomAdamW.step`, lines 946 to 999 with
`weight_decay_form == "to_zero"`, `amsgrad` off and `noise_factor == 0`, the settings of every
run of arXiv:2201.02177v1 (Power et al.) used here; `experiments/grok_division/PROTOCOL.md`,
*The optimizer*.

The code updates each parameter tensor entry by entry with these statements, here for one
entry with value `p`, gradient `g` and moments `m`, `v` (`sourceStep`):

1. `if group["weight_decay"] > 0: p.mul_(1 - group["lr"] * group["weight_decay"])`;
2. `state["step"] += 1`, `bias_correction1 = 1 - beta1 ** state["step"]`, and the same for
   `beta2`;
3. `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)` and
   `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`;
4. `denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])`;
5. `step_size = group["lr"] / bias_correction1`, `upd = exp_avg / denom`,
   `p.add_(-step_size * upd)`.

Read in exact reals, with `group["lr"]` the learning rate `Hyper.rateAt` of the update, their
composition is `adamStep` when the weight decay is not negative (`adamStep_eq_sourceStep`).
The decay of statement 1 is skipped at weight decay `0`, where the formula of `adamStep`
multiplies by `1` (`adamStep_params_of_decay_eq_zero`). For a negative weight decay the code
skips the decay and `adamStep` does not; no run uses one.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {ι : Type*} {h : Hyper}

/-- **The statements of `CustomAdamW.step` for one entry**, in the order of the code: the new
value and the new moments. -/
noncomputable def sourceStep (h : Hyper) (t : ℕ) (g p m v : ℝ) : ℝ × ℝ × ℝ :=
  let lr := h.rateAt t
  let decayed := if 0 < h.decay then p * (1 - lr * h.decay) else p
  let biasCorrection1 := 1 - h.beta1 ^ t
  let biasCorrection2 := 1 - h.beta2 ^ t
  let expAvg := m * h.beta1 + (1 - h.beta1) * g
  let expAvgSq := v * h.beta2 + (1 - h.beta2) * (g * g)
  let denom := √expAvgSq / √biasCorrection2 + h.eps
  let stepSize := lr / biasCorrection1
  let upd := expAvg / denom
  (decayed + -stepSize * upd, expAvg, expAvgSq)

/-- **The update of the run is the code's**, entry by entry, when the weight decay is not
negative. -/
theorem adamStep_eq_sourceStep (hd : 0 ≤ h.decay) (t : ℕ) (g : ι → ℝ) (s : AdamState ι) (i : ι) :
    ((adamStep h t g s).params i, (adamStep h t g s).first i, (adamStep h t g s).second i) =
      sourceStep h t (g i) (s.params i) (s.first i) (s.second i) := by
  have hv : h.beta2 * s.second i + (1 - h.beta2) * g i ^ 2 =
      s.second i * h.beta2 + (1 - h.beta2) * (g i * g i) := by ring
  simp only [adamStep, sourceStep, hv]
  refine Prod.ext ?_ (Prod.ext (by ring) rfl)
  rcases hd.lt_or_eq with hpos | hzero
  · simp only [ite_eq_left hpos]
    ring
  · simp only [← hzero, lt_irrefl, ite_false]
    ring

/-- **At weight decay `0` the update is Adam's**: the parameter is not rescaled. -/
theorem adamStep_params_of_decay_eq_zero (hd : h.decay = 0) (t : ℕ) (g : ι → ℝ)
    (s : AdamState ι) (i : ι) :
    (adamStep h t g s).params i = s.params i - h.rateAt t / (1 - h.beta1 ^ t) *
      ((adamStep h t g s).first i /
        (√((adamStep h t g s).second i) / √(1 - h.beta2 ^ t) + h.eps)) := by
  simp only [adamStep, hd]
  ring

/-! The witnesses: the source's AdamW and Adam constants, on one entry. -/

example (t : ℕ) (g : Unit → ℝ) (s : AdamState Unit) :=
  adamStep_eq_sourceStep (h := Hyper.source 1) (by norm_num [Hyper.source]) t g s ()

example (t : ℕ) (g : Unit → ℝ) (s : AdamState Unit) :=
  adamStep_params_of_decay_eq_zero (h := Hyper.source 0) rfl t g s ()

end Transformer.Grokking.Protocol
