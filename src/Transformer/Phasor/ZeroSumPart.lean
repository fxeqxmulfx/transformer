module

public import Transformer.Phasor.MeanStream

/-!
# The part of the streams that sums to zero

T3d of `review/residual_connections.md`, *What the extra streams can hold*. Split four streams into
their mean, which every read sees, and a part `y` with `∑_s y_s = 0`. A block with equal read
gates and zero phases never reads `y` and carries it by its skip map alone: adding `y` to its
input adds `P y` to its output (`phaseMHCStep_add_of_sum_zero`; `Router.IsBalanced.step_add` for a
router that reads the mean). When the columns of `P` sum to one, `P y` again sums to zero
(`sum_mix_of_col_sum`), so the part stays unread through any number of such blocks. At main's
initial skip logits `2 I` the skip multiplies it by `(e² − 1)/(e² + 3) ≈ 0.61`
(`mix_finiteSinkhorn_two_of_sum_zero`): it fades, and is never lost.

This is where `HideRouter` puts the stream it hides. A branch that reads through other positions,
as attention does, sees only their means too (`Attention.heads_collapse`); the induction over
positions and layers is by hand. Exact real arithmetic, every `ε` and every depth scale.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

/-- **Equal read gates and zero phases carry a zero-sum part by the skip alone**: adding `y` with
`∑ y = 0` to the streams adds `P y` after the block and changes nothing that is read. -/
theorem phaseMHCStep_add_of_sum_zero {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (P : PhaseAction E) (g : ℝ) (writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (F : E → E) (x : ι → E) {y : ι → E} (hy : ∑ s, y s = 0) :
    phaseMHCStep P (fun _ => g) writeLogits skipLogits (fun _ => 0) F (x + y) =
      phaseMHCStep P (fun _ => g) writeLogits skipLogits (fun _ => 0) F x +
        mix (finiteSinkhorn skipLogits 20) y := by
  have hread : weightedSum (fun _ => gate g) (x + y) = weightedSum (fun _ => gate g) x := by
    simp only [weightedSum_const_weights, Pi.add_apply, Finset.sum_add_distrib, hy, add_zero]
  funext i
  simp only [phaseMHCStep, phaseMix_zero, Pi.add_apply]
  rw [hread]
  simp only [Residual.mix, weightedSum, Pi.add_apply, smul_add, Finset.sum_add_distrib]
  abel

/-- At main's width and initial skip logits, on the zero-sum part `(v, −v, 0, 0)`. -/
example (F : EucSpace 128 → EucSpace 128) (x : Fin 4 → EucSpace 128) (v : EucSpace 128) :=
  phaseMHCStep_add_of_sum_zero (headPhaseAction 128) (-Real.log 3) 0 (2 • 1) F x
    (y := ![v, -v, 0, 0]) (by simp [Fin.sum_univ_four])

variable {D : ℕ}

/-- **A block with a router that reads the mean carries a zero-sum part by its skip alone**, for
a branch that sees the state only through what it reads. -/
theorem Router.IsBalanced.step_add {R : Router D} (hR : R.IsBalanced) (eps scale : ℝ)
    (f : EucSpace D → EucSpace D) (x : State D) {y : State D} (hy : ∑ s, y s = 0) :
    step eps scale ⟨R.routing eps, fun _ => f⟩ (x + y) =
      step eps scale ⟨R.routing eps, fun _ => f⟩ x + mix (finiteSinkhorn R.skipBias 20) y := by
  have hz : ∀ z, step eps scale ⟨R.routing eps, fun _ => f⟩ z =
      phaseMHCStep (headPhaseAction D) (fun _ => -Real.log 3) R.writeBias R.skipBias (fun _ => 0)
        (fun w => scale • GPTMini.rmsNormEps eps (f (GPTMini.rmsNormEps eps w))) z := fun z => by
    have h1 : (R.routing eps z).read = fun _ => -Real.log 3 := by
      funext i
      simp [Router.routing, hR.scale, hR.readBias]
    have h2 : (R.routing eps z).write = R.writeBias := by
      funext i
      simp [Router.routing, hR.scale]
    have h3 : (R.routing eps z).skip = R.skipBias := by
      ext i j
      simp [Router.routing, hR.scale]
    change phaseMHCStep (headPhaseAction D) (R.routing eps z).read (R.routing eps z).write
      (R.routing eps z).skip (R.routing eps z).phase
        (fun w => scale • GPTMini.rmsNormEps eps (f (GPTMini.rmsNormEps eps w))) z = _
    rw [h1, h2, h3, Router.routing_phase_eq_zero R eps z hR.phase]
  rw [hz, hz, phaseMHCStep_add_of_sum_zero _ _ _ _ _ _ hy]

/-- `Router.uneven` at main's width, `ε` and depth scale `1/√8`, with any branch, on the zero-sum
part `(v, −v, 0, 0)`. -/
example (f : EucSpace 128 → EucSpace 128) (x : State 128) (v : EucSpace 128) :=
  (Router.uneven_isBalanced 128).step_add (1 / 8388608) (1 / Real.sqrt 8) f x
    (y := ![v, -v, 0, 0]) (by simp [Fin.sum_univ_four])

/-- **At main's initial skip logits `2 I` the skip keeps a zero-sum part**, scaled by
`(e² − 1)/(e² + 3)`: the map is `(e² I + J − I)/(e² + 3)` and `J y = 0`. -/
theorem mix_finiteSinkhorn_two_of_sum_zero {y : State D} (hy : ∑ s, y s = 0) :
    mix (finiteSinkhorn (2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)) 20) y =
      ((Real.exp 2 - 1) / (Real.exp 2 + 3)) • y := by
  rw [finiteSinkhorn_balanced (by positivity) sum_exp_two_smul_one sum_exp_two_smul_one']
  rw [Fin.sum_univ_four] at hy
  have h3 : y 3 = -(y 0 + y 1 + y 2) := eq_neg_of_add_eq_zero_left (by rw [add_comm]; exact hy)
  funext i
  fin_cases i <;> simp [Residual.mix, weightedSum, Fin.sum_univ_four, Matrix.one_apply, h3] <;>
    module

/-- At main's width, on `(v, −v, 0, 0)`. -/
example (v : EucSpace 128) :=
  mix_finiteSinkhorn_two_of_sum_zero (D := 128) (y := ![v, -v, 0, 0]) (by simp [Fin.sum_univ_four])

end Transformer.Phasor
