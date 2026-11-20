module

public import Transformer.Phasor.Router
public import Transformer.ComplexFFN.Defs

/-!
# The feed-forward branch of the main model: a complex product

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
*Architecture*: the branch is `R_D(W_F (W_g z ⊙_ℂ W_u z) / √2)` on `z = Router.read x`, the plain
complex-product FFN, with `F` complex hidden units, no biases and no gate nonlinearity. The raw map
is `Transformer.ComplexFFN.plainFFN`: the real part of unit `k` leaves through the column `outRe k`
of `W_F`, its imaginary part through `outIm k`. The projections are real-linear here; the model's
are the complex-linear ones (stored `[[A, −B], [B, A]]`), and no bound needs it.

* `FFN.sublayer`: the router of the sublayer and this raw map as a `PeriRMS.Sublayer`;
* `FFN.step_apply`: the block is the phase-conjugated skip plus the twice-sigmoid write gate times
  the depth scale times `R_D(W_F (W_g z ⊙_ℂ W_u z) / √2)`, with `z` what the router reads;
* `FFN.sublayer_raw_zero`: with zero gate projections the raw branch vanishes.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.ComplexFFN

variable {D F : ℕ}

/-- The parameters of the feed-forward branch of one layer: `F` complex units with gate and value
projections `W_g, W_u : ℝ^D → ℂ` and the columns of `W_F` that take the real and the imaginary part
of a unit to the streams. -/
structure FFN (D F : ℕ) where
  gate : Fin F → EucSpace D →ₗ[ℝ] ℂ
  value : Fin F → EucSpace D →ₗ[ℝ] ℂ
  outRe : Fin F → EucSpace D
  outIm : Fin F → EucSpace D

/-- All projections zero: the raw branch is zero, so a feed-forward block only mixes the streams. -/
noncomputable def FFN.zero (D F : ℕ) : FFN D F where
  gate := fun _ => 0
  value := fun _ => 0
  outRe := fun _ => 0
  outIm := fun _ => 0

/-- The feed-forward sublayer: the router `R` and the raw map `W_F (W_g z ⊙_ℂ W_u z) / √2`, before
the output RMS and the depth scale that `PeriRMS.step` applies. -/
noncomputable def FFN.sublayer (f : FFN D F) (R : Router D) (eps : ℝ) : Sublayer D where
  routing := R.routing eps
  raw := fun _ z => plainFFN f.gate f.value f.outRe f.outIm z

/-- The block of the feed-forward sublayer: the README's FFN residual, on the read
`z = Router.read x`. -/
theorem FFN.step_apply (f : FFN D F) (R : Router D) (eps scale : ℝ) (x : State D) (i : Fin 4) :
    step eps scale (f.sublayer R eps) x i =
      phaseMix (headPhaseAction D) (finiteSinkhorn (R.routing eps x).skip 20)
          (R.routing eps x).phase x i +
        writeGate ((R.routing eps x).write i) •
          (scale • GPTMini.rmsNormEps eps
            (plainFFN f.gate f.value f.outRe f.outIm (R.read eps x))) :=
  rfl

/-- With zero gate projections the raw branch is zero, at every state and read. -/
theorem FFN.sublayer_raw_zero (f : FFN D F) (R : Router D) (eps : ℝ)
    (hg : ∀ k, f.gate k = 0) (x : State D) (z : EucSpace D) :
    (f.sublayer R eps).raw x z = 0 := by
  simp [FFN.sublayer, plainFFN, complexReadout, plainHidden, hg]

/-- Zero gates in the model's sizes, `D = 128` and `F` units: the hypothesis of the vanishing. -/
example (R : Router 128) (x : State 128) (z : EucSpace 128) :
    ((FFN.zero 128 4).sublayer R (1 / 8388608)).raw x z = 0 :=
  FFN.sublayer_raw_zero _ R _ (fun _ => rfl) x z

end Transformer.Phasor
