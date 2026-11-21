module

public import Transformer.Phasor.HideSkip

/-!
# A router of main that hides four equal streams

T3d of `review/residual_connections.md`, *What the extra streams can hold*. Give a router of main
zero scales, the skip biases `hideLogits` and the phase parameters `hidePhase` (`HideSkip`). Its
block, on four equal streams `h` and at every even width:

* `Router.step_of_hide`: stream `i` is the hidden part `a_i h` plus its write;
* `Router.sum_step_of_hide`: the streams sum to the write gates' total times the branch, so `h`
  has left the sum, and a next router with equal read biases, which reads the sum
  (`Router.read_of_const`), does not see it;
* `Router.weightedSum_step_of_hide`: a read with the gate `g₀` on stream `0` and `γ` on the others
  sees `(g₀ − γ) a_0 h` besides the branch, and `a_0 h ≠ 0` (`phaseMix_hide_zero_ne`).

`Router.hide` is such a router, with read biases `−ln 3` and write biases `0`. Exact real
arithmetic, every `ε` and every depth scale.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

variable {D : ℕ}

/-- Main's router that hides: zero scales and projections, read biases `−ln 3`, write biases `0`,
the skip biases `hideLogits` and the phase parameters `hidePhase`. -/
noncomputable def Router.hide (D : ℕ) : Router D where
  readProj := 0
  writeProj := 0
  skipProj := 0
  scale := 0
  readBias := fun _ => -Real.log 3
  writeBias := 0
  phase := hidePhase
  skipBias := hideLogits

/-- Main computes the angles `hideAngle` from the phase parameters `hidePhase`. -/
theorem Router.routing_phase_of_hide {R : Router D} (hp : R.phase = hidePhase) (eps : ℝ)
    (x : State D) : (R.routing eps x).phase = hideAngle := by
  funext s
  refine Fin.cases rfl (fun t => ?_) s
  rw [Router.routing_phase_succ, hp]
  fin_cases t <;> rfl

/-- At main's width and `ε`. -/
example (x : State 128) := Router.routing_phase_of_hide (R := Router.hide 128) rfl (1 / 8388608) x

/-- **A block whose router has zero scales, the skip biases `hideLogits` and the phases
`hidePhase`**, on four equal streams: stream `i` is the hidden part `a_i h` plus its write. -/
theorem Router.step_of_hide {R : Router D} (hs : R.scale = 0) (hk : R.skipBias = hideLogits)
    (hp : R.phase = hidePhase) (eps scale : ℝ) (raw : State D → EucSpace D → EucSpace D)
    (h : EucSpace D) (i : Fin 4) :
    step eps scale ⟨R.routing eps, raw⟩ (fun _ => h) i =
      phaseMix (headPhaseAction D) hideSkip hideAngle (fun _ => h) i + writeGate (R.writeBias i) •
        (scale • GPTMini.rmsNormEps eps (raw (fun _ => h) (R.read eps (fun _ => h)))) := by
  have hskip : (R.routing eps (fun _ => h)).skip = hideLogits := by
    ext a b
    simp [Router.routing, hs, hk]
  have hw : (R.routing eps (fun _ => h)).write i = R.writeBias i := by
    simp [Router.routing, hs]
  rw [Router.step_apply, hskip, finiteSinkhorn_hideLogits, Router.routing_phase_of_hide hp, hw]

/-- At main's width, `ε` and depth scale `1/√8`, with any branch. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (h : EucSpace 128) (i : Fin 4) :=
  Router.step_of_hide (R := Router.hide 128) rfl rfl rfl (1 / 8388608) (1 / Real.sqrt 8) raw h i

/-- **`h` leaves the sum of the streams**: after such a block on `(h, h, h, h)` the streams sum to
the write gates' total times the branch, at every even width. -/
theorem Router.sum_step_of_hide {R : Router D} (hD : Even D) (hs : R.scale = 0)
    (hk : R.skipBias = hideLogits) (hp : R.phase = hidePhase) (eps scale : ℝ)
    (raw : State D → EucSpace D → EucSpace D) (h : EucSpace D) :
    ∑ i, step eps scale ⟨R.routing eps, raw⟩ (fun _ => h) i = (∑ i, writeGate (R.writeBias i)) •
      (scale • GPTMini.rmsNormEps eps (raw (fun _ => h) (R.read eps (fun _ => h)))) := by
  simp only [Router.step_of_hide hs hk hp, Finset.sum_add_distrib, sum_phaseMix_hide hD, zero_add,
    Finset.sum_smul]

/-- At main's width: `h` leaves the sum, which gains four branches. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (h : EucSpace 128) :=
  Router.sum_step_of_hide (R := Router.hide 128) ⟨64, rfl⟩ rfl rfl rfl (1 / 8388608)
    (1 / Real.sqrt 8) raw h

/-- **The next read sees it again with unequal gates**: gates `γ` off stream `0` read the branch
and `(g₀ − γ) a_0 h`. -/
theorem Router.weightedSum_step_of_hide {R : Router D} (hD : Even D) (hs : R.scale = 0)
    (hk : R.skipBias = hideLogits) (hp : R.phase = hidePhase) (eps scale : ℝ)
    (raw : State D → EucSpace D → EucSpace D) (h : EucSpace D) (g : Fin 4 → ℝ) {γ : ℝ}
    (hg : ∀ s, s ≠ 0 → g s = γ) :
    weightedSum g (step eps scale ⟨R.routing eps, raw⟩ (fun _ => h)) =
      (γ * ∑ i, writeGate (R.writeBias i) + (g 0 - γ) * writeGate (R.writeBias 0)) •
          (scale • GPTMini.rmsNormEps eps (raw (fun _ => h) (R.read eps (fun _ => h)))) +
        (g 0 - γ) • phaseMix (headPhaseAction D) hideSkip hideAngle (fun _ => h) 0 := by
  rw [weightedSum_eq_sum_add g hg, Router.sum_step_of_hide hD hs hk hp,
    Router.step_of_hide hs hk hp]
  module

/-- At main's width: the gates `1/2, 1/4, 1/4, 1/4` see `a_0 h / 4`. -/
example (raw : State 128 → EucSpace 128 → EucSpace 128) (h : EucSpace 128) :=
  Router.weightedSum_step_of_hide (R := Router.hide 128) ⟨64, rfl⟩ rfl rfl rfl (1 / 8388608)
    (1 / Real.sqrt 8) raw h ![1 / 2, 1 / 4, 1 / 4, 1 / 4] (γ := 1 / 4)
    (fun s hs => by fin_cases s <;> simp_all)

end Transformer.Phasor
