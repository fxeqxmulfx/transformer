module

public import Transformer.Phasor.Router

/-!
# A router that does not route

R0 of `review/residual_connections.md`, *The ordinary residual inside mHC*, its first half. A
router of the main model (`Router`) computes its read, write and skip logits as
`scale · projection + bias` of the normalized four-stream state. Give it zero scales, zero phase
coordinates, read biases `−ln 3` and write biases `0`, and leave its projections and skip biases
arbitrary (`Router.IsPlain`). Then the router ignores the state:

* every read gate is `σ(−ln 3) = 1/4`, so the four sum to exactly one (`gate_neg_log_three`,
  `Router.IsPlain.read_const`);
* every write gate is `2σ(0) = 1` (`writeGate_zero`);
* the skip `D P D*` has `D = 1`, and the last step of finite Sinkhorn normalizes rows, so it maps
  four equal streams to themselves (`mix_finiteSinkhorn_const`).

So a block with such a router maps four equal streams `h` to four copies of the ordinary residual
`h + α R_D(raw(R_D h))` (`step_const_of_isPlain`). `Transformer.Phasor.OneStream` runs this
through the whole forward pass. Exact real arithmetic, every `ε` and every depth scale `α`.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

variable {D : ℕ}

/-- `σ(−ln 3) = 1/(1 + 3) = 1/4`: four such read gates sum to one. -/
theorem gate_neg_log_three : gate (-Real.log 3) = 1 / 4 := by
  rw [gate, neg_neg, Real.exp_log (by norm_num)]
  norm_num

/-- `2σ(0) = 1`. -/
theorem writeGate_zero : writeGate 0 = 1 := by
  rw [writeGate, gate, neg_zero, Real.exp_zero]
  norm_num

/-- **A router that does not route**: zero scales, zero phase coordinates, read biases `−ln 3`
and write biases `0`. The projections and the skip biases are arbitrary. -/
structure Router.IsPlain (R : Router D) : Prop where
  scale : R.scale = 0
  readBias : R.readBias = fun _ => -Real.log 3
  writeBias : R.writeBias = 0
  phase : R.phase = 0

/-- One plain router: zero projections and zero skip biases. -/
noncomputable def Router.plain (D : ℕ) : Router D where
  readProj := 0
  writeProj := 0
  skipProj := 0
  scale := 0
  writeBias := 0
  skipBias := 0
  phase := 0
  readBias := fun _ => -Real.log 3

/-- `Router.plain` is plain. -/
theorem Router.plain_isPlain (D : ℕ) : (Router.plain D).IsPlain :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- A plain router's read logits are `−ln 3` at every state. -/
theorem Router.IsPlain.routing_read {R : Router D} (hR : R.IsPlain) (eps : ℝ) (x : State D)
    (i : Fin 4) : (R.routing eps x).read i = -Real.log 3 := by
  simp [Router.routing, hR.scale, hR.readBias]

/-- A plain router's write logits are `0` at every state. -/
theorem Router.IsPlain.routing_write {R : Router D} (hR : R.IsPlain) (eps : ℝ) (x : State D)
    (i : Fin 4) : (R.routing eps x).write i = 0 := by
  simp [Router.routing, hR.scale, hR.writeBias]

/-- **A plain router reads four equal streams as `R_D` of one**: the gates `1/4` sum to one. -/
theorem Router.IsPlain.read_const {R : Router D} (hR : R.IsPlain) (eps : ℝ) (h : EucSpace D) :
    R.read eps (fun _ => h) = GPTMini.rmsNormEps eps h := by
  rw [Router.read, weightedSum_const]
  simp only [hR.routing_read, gate_neg_log_three, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin]
  norm_num

/-- **The skip keeps four equal streams**: the twenty-cycle Sinkhorn map ends with a row
normalization, so each output stream is the sum of its row, one, times the common stream. -/
theorem mix_finiteSinkhorn_const (logits : Matrix (Fin 4) (Fin 4) ℝ) (h : EucSpace D) :
    mix (finiteSinkhorn logits 20) (fun _ => h) = fun _ => h := by
  funext i
  rw [mix, weightedSum_const, twenty_iterations_row_sum, one_smul]

/-- **A block with a plain router keeps four equal streams equal**: it maps `(h, h, h, h)` to four
copies of the ordinary residual `h + α R_D(raw(R_D h))`. -/
theorem step_const_of_isPlain {R : Router D} (hR : R.IsPlain) (eps scale : ℝ)
    (raw : State D → EucSpace D → EucSpace D) (h : EucSpace D) :
    step eps scale ⟨R.routing eps, raw⟩ (fun _ => h) =
      fun _ => h + scale • GPTMini.rmsNormEps eps (raw (fun _ => h) (GPTMini.rmsNormEps eps h)) := by
  funext i
  rw [Router.step_apply, Router.routing_phase_eq_zero R eps _ hR.phase, phaseMix_zero,
    mix_finiteSinkhorn_const, hR.routing_write, writeGate_zero, one_smul, hR.read_const]


/-- The model's width `D = 128` and `ε = 2⁻²³`: a plain router reads four equal streams as one. -/
example (h : EucSpace 128) :
    (Router.plain 128).read (1 / 8388608) (fun _ => h) = GPTMini.rmsNormEps (1 / 8388608) h :=
  (Router.plain_isPlain 128).read_const _ h

/-- A block of the model's width with a plain router and any raw branch keeps equal streams equal. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (h : EucSpace 128) :
    step (1 / 8388608) (1 / 4) ⟨(Router.plain 128).routing (1 / 8388608), raw⟩ (fun _ => h) =
      fun _ => h + (1 / 4 : ℝ) • GPTMini.rmsNormEps (1 / 8388608)
        (raw (fun _ => h) (GPTMini.rmsNormEps (1 / 8388608) h)) :=
  step_const_of_isPlain (Router.plain_isPlain 128) _ _ raw h

end Transformer.Phasor
