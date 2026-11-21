module

public import Transformer.Phasor.PlainRouter

/-!
# The sum of the streams

T3d of `review/residual_connections.md`, *What the extra streams can hold*. The finite Sinkhorn map
`P` of a block ends with a row normalization, so its rows sum to one and it keeps four equal streams
equal (R0). Its columns need not sum to one. When they do, the mix keeps the *sum* of the streams,
whatever the streams are (`sum_mix_of_col_sum`), and with zero phases the sum after a block is the
sum before plus the total of the write gates times the branch (`sum_phaseMHCStep_zero_phases`).
With equal read gates the branch reads the sum alone (`weightedSum_const_weights`). So such blocks
never read the part of the streams that sums to zero, and it keeps summing to zero. The paper's
block has the same sum for a doubly stochastic skip (`gatedHyperStep_sum`); here the hypothesis is
only on the columns of the finite map, whose rows already sum to one.

The columns sum to one when the exponentiated skip logits have all row and column sums equal to a
common `c ≠ 0`: then the first Sinkhorn cycle divides by `c` and every later one is the identity
(`sinkhornIterate_balanced`), so the twenty-cycle map is `exp(logits)/c`. Main's initial skip logits
`2 I` are such (`col_sum_finiteSinkhorn_two`), and so are zero logits.

* `Router.read_of_const`, `Router.sum_step`: for a router with zero scales and phases, equal read
  biases and balanced skip biases, at any streams.

Exact real arithmetic, every `ε` and every depth scale.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

section General

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The sum of the mixed streams: the stream `j` enters with the sum of its column. -/
theorem sum_mix (H : Matrix ι ι ℝ) (x : ι → E) :
    ∑ i, mix H x i = ∑ j, (∑ i, H i j) • x j := by
  simp only [mix, weightedSum, Finset.sum_smul]
  exact Finset.sum_comm

/-- **Columns summing to one keep the sum of the streams.** -/
theorem sum_mix_of_col_sum {H : Matrix ι ι ℝ} (hH : ∀ j, ∑ i, H i j = 1) (x : ι → E) :
    ∑ i, mix H x i = ∑ j, x j := by
  simp only [sum_mix, hH, one_smul]

/-- The uniform mixer `J/4` keeps the sum of four streams. -/
example (x : Fin 4 → EucSpace 128) :
    ∑ i, mix (Matrix.of fun _ _ => (1 / 4 : ℝ)) x i = ∑ j, x j :=
  sum_mix_of_col_sum (fun _ => by simp) x

/-- **Equal read gates read the sum of the streams.** -/
theorem weightedSum_const_weights (g : ℝ) (x : ι → E) :
    weightedSum (fun _ => g) x = g • ∑ j, x j := by
  rw [weightedSum, Finset.smul_sum]

/-- **The sum of the streams after a block with zero phases**, when the twenty-cycle map has
columns summing to one: the sum before plus the total of the write gates times the branch. -/
theorem sum_phaseMHCStep_zero_phases (P : PhaseAction E) (readLogits writeLogits : ι → ℝ)
    {skipLogits : Matrix ι ι ℝ} (hcol : ∀ j, ∑ i, finiteSinkhorn skipLogits 20 i j = 1)
    (F : E → E) (x : ι → E) :
    ∑ i, phaseMHCStep P readLogits writeLogits skipLogits (fun _ => 0) F x i =
      ∑ j, x j + (∑ i, writeGate (writeLogits i)) •
        F (weightedSum (fun j => gate (readLogits j)) x) := by
  simp only [phaseMHCStep, phaseMix_zero, Finset.sum_add_distrib, sum_mix_of_col_sum hcol,
    Finset.sum_smul]

/-- **One cycle of a balanced matrix**: when all rows and all columns of `M` sum to `c ≠ 0`, one
Sinkhorn cycle divides `M` by `c`. -/
theorem sinkhornCycle_balanced {M : Matrix ι ι ℝ} {c : ℝ} (hc : c ≠ 0)
    (hr : ∀ i, ∑ j, M i j = c) (hcol : ∀ j, ∑ i, M i j = c) :
    sinkhornCycle M = Matrix.of fun i j => M i j / c := by
  have hcn : columnNormalize M = Matrix.of fun i j => M i j / c := by
    ext a b
    simp only [columnNormalize, Matrix.of_apply, hcol]
  ext i j
  simp only [sinkhornCycle, rowNormalize, hcn, Matrix.of_apply, ← Finset.sum_div, hr,
    div_self hc, div_one]

/-- **Every cycle after the first is the identity**: `t + 1` cycles map a balanced `M` to `M/c`. -/
theorem sinkhornIterate_balanced {M : Matrix ι ι ℝ} {c : ℝ} (hc : c ≠ 0)
    (hr : ∀ i, ∑ j, M i j = c) (hcol : ∀ j, ∑ i, M i j = c) (t : ℕ) :
    sinkhornIterate M (t + 1) = Matrix.of fun i j => M i j / c := by
  induction t with
  | zero => exact sinkhornCycle_balanced hc hr hcol
  | succ t ih =>
    change sinkhornCycle (sinkhornIterate M (t + 1)) = _
    rw [ih, sinkhornCycle_balanced one_ne_zero
      (fun i => by simp only [Matrix.of_apply, ← Finset.sum_div, hr, div_self hc])
      (fun j => by simp only [Matrix.of_apply, ← Finset.sum_div, hcol, div_self hc])]
    ext i j
    simp only [Matrix.of_apply, div_one]

/-- **The twenty-cycle map of balanced logits** is `exp(logits)/c`. -/
theorem finiteSinkhorn_balanced {logits : Matrix ι ι ℝ} {c : ℝ} (hc : c ≠ 0)
    (hr : ∀ i, ∑ j, Real.exp (logits i j) = c) (hcol : ∀ j, ∑ i, Real.exp (logits i j) = c) :
    finiteSinkhorn logits 20 = Matrix.of fun i j => Real.exp (logits i j) / c :=
  sinkhornIterate_balanced (M := Matrix.of fun i j => Real.exp (logits i j)) hc hr hcol 19

/-- So its columns sum to one, as well as its rows. -/
theorem col_sum_finiteSinkhorn_of_balanced {logits : Matrix ι ι ℝ} {c : ℝ} (hc : c ≠ 0)
    (hr : ∀ i, ∑ j, Real.exp (logits i j) = c) (hcol : ∀ j, ∑ i, Real.exp (logits i j) = c)
    (j : ι) : ∑ i, finiteSinkhorn logits 20 i j = 1 := by
  rw [finiteSinkhorn_balanced hc hr hcol]
  simp only [Matrix.of_apply, ← Finset.sum_div, hcol, div_self hc]

end General

/-- Main's initial skip logits `2 I`: `exp` puts `e²` on the diagonal and `1` elsewhere, so every
row sums to `e² + 3`. -/
theorem sum_exp_two_smul_one (i : Fin 4) :
    ∑ j, Real.exp ((2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) i j) = Real.exp 2 + 3 := by
  fin_cases i <;> simp [Fin.sum_univ_four, Matrix.one_apply] <;> ring

/-- And every column. -/
theorem sum_exp_two_smul_one' (j : Fin 4) :
    ∑ i, Real.exp ((2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) i j) = Real.exp 2 + 3 := by
  fin_cases j <;> simp [Fin.sum_univ_four, Matrix.one_apply] <;> ring

/-- **At main's initial skip logits the columns of the twenty-cycle map sum to one.** -/
theorem col_sum_finiteSinkhorn_two (j : Fin 4) :
    ∑ i, finiteSinkhorn (2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) 20 i j = 1 :=
  col_sum_finiteSinkhorn_of_balanced (by positivity) sum_exp_two_smul_one sum_exp_two_smul_one' j

/-- One cycle at main's initial skip logits divides their exponentials by `e² + 3`. -/
example := sinkhornCycle_balanced (M := Matrix.of fun i j =>
  Real.exp ((2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) i j)) (by positivity) sum_exp_two_smul_one
  sum_exp_two_smul_one'

/-- And so does every positive number of cycles. -/
example (t : ℕ) := sinkhornIterate_balanced (M := Matrix.of fun i j =>
  Real.exp ((2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) i j)) (by positivity) sum_exp_two_smul_one
  sum_exp_two_smul_one' t

/-- The twenty-cycle map at main's initial skip logits. -/
example := finiteSinkhorn_balanced (logits := 2 • (1 : Matrix (Fin 4) (Fin 4) ℝ))
  (by positivity) sum_exp_two_smul_one sum_exp_two_smul_one'

/-- A block of main's width at the initial skip logits with zero phases, on any streams. -/
example (F : EucSpace 128 → EucSpace 128) (x : Fin 4 → EucSpace 128) :=
  sum_phaseMHCStep_zero_phases (headPhaseAction 128) 0 0 col_sum_finiteSinkhorn_two F x

variable {D : ℕ}

/-- **A router with zero scales and equal read biases reads the sum of the streams**, at any
streams: `R_D(σ(r) ∑_s x_s)`. -/
theorem Router.read_of_const {R : Router D} (hs : R.scale = 0) {r : ℝ}
    (hb : R.readBias = fun _ => r) (eps : ℝ) (x : State D) :
    R.read eps x = GPTMini.rmsNormEps eps (gate r • ∑ j, x j) := by
  simp only [Router.read, Router.routing, hs, hb, Pi.zero_apply, zero_mul, zero_add,
    weightedSum_const_weights]

/-- **The sum of the streams after a block whose router has zero scales and phases and balanced
skip biases**, at any streams: the sum before plus the write gates' total times the branch. -/
theorem Router.sum_step {R : Router D} (hs : R.scale = 0) (hp : R.phase = 0) {c : ℝ} (hc : c ≠ 0)
    (hr : ∀ i, ∑ j, Real.exp (R.skipBias i j) = c)
    (hcol : ∀ j, ∑ i, Real.exp (R.skipBias i j) = c) (eps scale : ℝ)
    (raw : State D → EucSpace D → EucSpace D) (x : State D) :
    ∑ i, step eps scale ⟨R.routing eps, raw⟩ x i =
      ∑ j, x j + (∑ i, writeGate (R.writeBias i)) •
        (scale • GPTMini.rmsNormEps eps (raw x (R.read eps x))) := by
  have hskip : (R.routing eps x).skip = R.skipBias := by
    ext i j
    simp [Router.routing, hs]
  have hw : ∀ i, (R.routing eps x).write i = R.writeBias i := fun i => by
    simp [Router.routing, hs]
  have hP : ∀ j, ∑ i, finiteSinkhorn (R.routing eps x).skip 20 i j = 1 := by
    rw [hskip]
    exact col_sum_finiteSinkhorn_of_balanced hc hr hcol
  change ∑ i, phaseMHCStep (headPhaseAction D) (R.routing eps x).read (R.routing eps x).write
    (R.routing eps x).skip (R.routing eps x).phase
      (fun z => scale • GPTMini.rmsNormEps eps (raw x (GPTMini.rmsNormEps eps z))) x i = _
  rw [Router.routing_phase_eq_zero R eps x hp, sum_phaseMHCStep_zero_phases _ _ _ hP]
  simp only [hw]
  rfl

/-- The plain router of R0 has zero skip biases, which are balanced: at main's width and `ε`, the
sum of the streams gains four times the branch, whatever the streams. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (x : State 128) :=
  Router.sum_step (R := Router.plain 128) rfl rfl (c := 4) (by norm_num)
    (fun _ => by simp [Router.plain]) (fun _ => by simp [Router.plain])
    (1 / 8388608) (1 / 4) raw x

/-- A plain router at main's width reads the sum of any four streams with the gate `1/4`. -/
example (x : State 128) :=
  Router.read_of_const (R := Router.plain 128) rfl rfl (1 / 8388608) x

end Transformer.Phasor
