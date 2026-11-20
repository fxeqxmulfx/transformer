module

public import Transformer.MHC.PhaseStep
public import Transformer.MHC.Section4_Parameterization
public import Transformer.GPTMini.RMSNorm
public import Transformer.SequenceModel.RotateHead

/-!
# Four-stream finite mHC with Peri-RMS sublayers

Local adaptation of arXiv:2512.24880v2, §§4.1–4.2, Eqs. (3), (8), (9),
and the diff-gpt inverse-width SSMax model frozen at commit c6c517b.
Migrated from its `research/mhc_scaled/Bounds.lean` and examples.
Input RMS, raw attention / feed-forward, output RMS, and the residual coefficient
are explicit. The coefficient is outside output RMS and before twice-sigmoid
write-back. Exactly twenty column-then-row Sinkhorn cycles are used.

The skip map of each block is conjugated by one phase per stream, `D P D*` (`phaseMHCStep`), as in
DiffGPT's main model (`fe1d528`, `research/main_model/`): `Routing.phase` holds the angles, each
stream is read as `d/2` complex coordinates by the half-split pairing of `headPhase`, and zero
angles, the default, are the paper's block (`step_of_phase_zero`). The bounds do not depend on
the angles, nor on how the streams are paired into complex coordinates.

Routing is input-dependent. Raw branch closures may contain the token's
causal context. This is an exact-real forward bound, not a translation of
PyTorch, a floating-point guarantee, or a bound on the full block Jacobian.
The paper's finite row normalization suffices; no exact double stochasticity
is assumed, and the released denominator-regularized algorithm is distinct.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Transformer Transformer.Residual Transformer.MHC
open scoped BigOperators

/-- Four residual streams; feature width is the current task's model width. -/
abbrev State (d : ℕ) := Fin 4 → EucSpace d

/-- The phases `e^{iφ}` of a stream of width `d`, read as `d/2` complex coordinates by the half-split
pairing: `headPhase`, isometries that compose by adding angles. -/
noncomputable def headPhaseAction (d : ℕ) : PhaseAction (EucSpace d) where
  rot := SequenceModel.headPhase d
  rot_zero := SequenceModel.headPhase_zero d
  rot_add := SequenceModel.headPhase_add d
  norm_rot := SequenceModel.headPhase_norm d

/-- Logits after the routing predictor, before sigmoid / finite Sinkhorn. -/
structure Routing where
  read : Fin 4 → ℝ
  write : Fin 4 → ℝ
  skip : Matrix (Fin 4) (Fin 4) ℝ
  /-- The angles `φ_s` of the stream phases `r_s = e^{iφ_s}` of the skip map `D P D*`. The main
  model has `φ_0 = 0`, `φ_s = arctan b_s` with one learned `b_s` per sublayer, shared by all tokens;
  the bounds hold for every angle and every dependence on the streams. Zero: the paper's skip. -/
  phase : Fin 4 → ℝ := fun _ => 0

/-- A raw attention or feed-forward branch and its input-dependent routing logits.
`raw` excludes both Peri-RMS operations and the depth coefficient. -/
structure Sublayer (d : ℕ) where
  routing : State d → Routing
  raw : State d → EucSpace d → EucSpace d

/-- mHC Eq. (3) with both current Peri-RMS sites and post-RMS scaling.
Read gates are sigmoid, write gates are twice sigmoid; the skip matrix is
the paper's finite twenty-cycle map, without denominator regularization, conjugated by the
stream phases (`phaseMHCStep`). -/
noncomputable def step {d : ℕ} (eps scale : ℝ) (branch : Sublayer d)
    (x : State d) : State d :=
  let r := branch.routing x
  phaseMHCStep (headPhaseAction d) r.read r.write r.skip r.phase
    (fun z => scale • GPTMini.rmsNormEps eps
      (branch.raw x (GPTMini.rmsNormEps eps z))) x

/-- Zero phases give the paper's finite block `finiteMHCStep` for every input. -/
theorem step_of_phase_zero {d : ℕ} (eps scale : ℝ) (branch : Sublayer d) (x : State d)
    (h : (branch.routing x).phase = fun _ => 0) :
    step eps scale branch x =
      finiteMHCStep (branch.routing x).read (branch.routing x).write (branch.routing x).skip
        (fun z => scale • GPTMini.rmsNormEps eps
          (branch.raw x (GPTMini.rmsNormEps eps z))) x := by
  unfold step
  simp only [h]
  exact phaseMHCStep_zero_phases _ _ _ _ _ _

/-- Finite row normalization suffices for the combined per-stream bound.
The factor two comes from mHC's actual write gate, not the stream count.
No exact double stochasticity assumption is made for finite Sinkhorn, and the stream phases
are arbitrary. Local Peri-RMS extension of arXiv:2512.24880v2, Eqs. (3), (8), (9). -/
theorem step_norm_le {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (branch : Sublayer d)
    (x : State d) (R : ℝ) (hx : ∀ i, ‖x i‖ ≤ R) (i : Fin 4) :
    ‖step eps scale branch x i‖ ≤ R + 2 * scale * Real.sqrt (d : ℝ) := by
  let r := branch.routing x
  let v := branch.raw x (GPTMini.rmsNormEps eps
    (weightedSum (fun j => gate (r.read j)) x))
  have hv : ‖scale • GPTMini.rmsNormEps eps v‖ ≤ scale * Real.sqrt (d : ℝ) := by
    rw [norm_smul, Real.norm_of_nonneg hscale]
    exact mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hd v) hscale
  have h := phaseMHCStep_norm_le (headPhaseAction d) r.read r.write r.skip r.phase
    (fun z => scale • GPTMini.rmsNormEps eps (branch.raw x (GPTMini.rmsNormEps eps z))) x R
    (scale * Real.sqrt (d : ℝ)) (by positivity) hx hv i
  calc ‖step eps scale branch x i‖ ≤ R + 2 * (scale * Real.sqrt (d : ℝ)) := h
    _ = R + 2 * scale * Real.sqrt (d : ℝ) := by ring

/-- The current normalized embedding is copied into all four streams. -/
noncomputable def embeddingState {d : ℕ} (eps : ℝ) (embedding : EucSpace d) : State d :=
  fun _ => GPTMini.rmsNormEps eps embedding

/-- An unweighted mean collapses the four streams before the existing final RMS.
This is the explicit readout convention for the proposed adaptation. -/
noncomputable def collapse {d : ℕ} (x : State d) : EucSpace d :=
  weightedSum (fun _ => 1 / 4) x

/-- Stream averaging preserves a common norm bound; no learned readout is assumed.
Local mean-head extension of the streams in arXiv:2512.24880v2, Eq. (3). -/
theorem collapse_norm_le {d : ℕ} (x : State d) (R : ℝ)
    (hx : ∀ i, ‖x i‖ ≤ R) : ‖collapse x‖ ≤ R := by
  apply weightedSum_convex_norm_le
  · intro i
    norm_num
  · norm_num
  · exact hx

/-- Nonzero stream feature with norm one. -/
noncomputable def witness : EucSpace 1 := EuclideanSpace.single (0 : Fin 1) 1

/-- A nontrivial branch whose routing and raw output depend on its inputs. -/
noncomputable def witnessBranch : Sublayer 1 where
  routing x := {
    read := fun i => ‖x i‖ - 1
    write := fun i => ‖x i‖ - 1
    skip := fun i j => if i = j then ‖x i‖ else 0
  }
  raw x z := z + x 0

example : ‖witness‖ = 1 := by simp [witness, PiLp.norm_single]

example : ‖step (1 / 8388608) (1 / 2) witnessBranch (fun _ => witness) 0‖ ≤
    (1 : ℝ) + 2 * (1 / 2) * Real.sqrt (1 : ℝ) := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using
    step_norm_le (d := 1) (1 / 8388608) (1 / 2) (by norm_num) (by norm_num)
      (by norm_num) witnessBranch (fun _ => witness) 1
      (by intro i; simp [witness, PiLp.norm_single]) 0

/-- A width-two branch, one complex coordinate per stream, with a different phase on every stream. -/
noncomputable def phasedBranch : Sublayer 2 where
  routing x := {
    read := fun i => ‖x i‖ - 1
    write := fun i => ‖x i‖ - 1
    skip := fun i j => if i = j then ‖x i‖ else 0
    phase := fun i => (i : ℝ)
  }
  raw x z := z + x 0

example (x : State 2) (hx : ∀ i, ‖x i‖ ≤ 1) :
    ‖step (1 / 8388608) (1 / 2) phasedBranch x 0‖ ≤ 1 + 2 * (1 / 2) * Real.sqrt (2 : ℝ) := by
  simpa only [Nat.cast_ofNat] using
    step_norm_le (d := 2) (1 / 8388608) (1 / 2) (by norm_num) (by norm_num) (by norm_num)
      phasedBranch x 1 hx 0

/-- The paper's block is the case of zero angles: `witnessBranch` has no phases. -/
example : step (1 / 8388608) (1 / 2) witnessBranch (fun _ => witness) =
    finiteMHCStep (witnessBranch.routing fun _ => witness).read
      (witnessBranch.routing fun _ => witness).write
      (witnessBranch.routing fun _ => witness).skip
      (fun z => (1 / 2 : ℝ) • GPTMini.rmsNormEps (1 / 8388608)
        (witnessBranch.raw (fun _ => witness) (GPTMini.rmsNormEps (1 / 8388608) z)))
      (fun _ => witness) :=
  step_of_phase_zero _ _ _ _ rfl

example : ‖collapse (fun _ => witness)‖ ≤ 1 := by
  exact collapse_norm_le (fun _ => witness) 1
    (by intro i; simp [witness, PiLp.norm_single])

end Transformer.MHC.PeriRMS
