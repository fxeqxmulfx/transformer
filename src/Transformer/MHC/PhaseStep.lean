module

public import Transformer.MHC.PhaseSkip
public import Transformer.MHC.Section4_Parameterization

/-!
# The finite mHC block with a phase-conjugated skip

The main model's residual block (`research/main_model/README.txt`, *Architecture*),

    h'_s = ∑_t P_st r_s conj(r_t) h_t + write_s F(∑_t read_t h_t),

is the finite block of arXiv:2512.24880v2, §4.2 (`finiteMHCStep`) with the skip `P`, the paper's
twenty column-then-row Sinkhorn cycles of the skip logits, replaced by `D P D*` (`phaseMix`). The
read gates (sigmoid) and the write gates (twice sigmoid) are the real gates of the paper, unchanged.

* `phaseMHCStep_zero_phases`: zero phases are `finiteMHCStep` exactly, for every input.
* `phaseMHCStep_norm_le`: a bound `R` on every stream and `B` on the branch output give
  `R + 2B` on every output stream, the bound of `gatedHyperStep_norm_le`, and no more is assumed:
  the finite Sinkhorn map is only row-normalized, not doubly stochastic.

Coefficients are those of one token; they may depend on the streams. Exact real arithmetic.
Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/mhc.json`.
-/

@[expose] public section

namespace Transformer.MHC

open Transformer.Residual

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- One block of the main model's mHC: the phase-conjugated skip of the finite twenty-cycle
Sinkhorn map of `skipLogits`, plus the branch `F` of the sigmoid-gated read of the streams written
back with twice-sigmoid gates. -/
noncomputable def phaseMHCStep (P : PhaseAction E) (readLogits writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (φ : ι → ℝ) (F : E → E) (x : ι → E) : ι → E :=
  fun i => phaseMix P (finiteSinkhorn skipLogits 20) φ x i +
    writeGate (writeLogits i) • F (weightedSum (fun j => gate (readLogits j)) x)

/-- Zero-initialized phases give the paper's block for every input. -/
theorem phaseMHCStep_zero_phases (P : PhaseAction E) (readLogits writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (F : E → E) (x : ι → E) :
    phaseMHCStep P readLogits writeLogits skipLogits (fun _ => 0) F x =
      finiteMHCStep readLogits writeLogits skipLogits F x := by
  funext i
  simp only [phaseMHCStep, finiteMHCStep, gatedHyperStep, hyperStep, phaseMix_zero]

/-- **The phase-conjugated block keeps the bound `R + 2B`** of the paper's block: stream norms at
most `R` before, branch output at most `B`, write gates below two. Only row normalization of the
skip map is used, never column normalization. Not a bound of the Jacobian of the router. -/
theorem phaseMHCStep_norm_le [Nonempty ι] (P : PhaseAction E) (readLogits writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (φ : ι → ℝ) (F : E → E) (x : ι → E) (R B : ℝ) (hB : 0 ≤ B)
    (hx : ∀ i, ‖x i‖ ≤ R) (hF : ‖F (weightedSum (fun j => gate (readLogits j)) x)‖ ≤ B)
    (i : ι) : ‖phaseMHCStep P readLogits writeLogits skipLogits φ F x i‖ ≤ R + 2 * B := by
  have hpos : ∀ j k, 0 < finiteSinkhorn skipLogits 20 j k :=
    sinkhornIterate_pos _ (fun j k => Real.exp_pos (skipLogits j k)) 20
  have hc := writeGate_bounds (writeLogits i)
  calc
    ‖phaseMHCStep P readLogits writeLogits skipLogits φ F x i‖ ≤
        ‖phaseMix P (finiteSinkhorn skipLogits 20) φ x i‖ +
          ‖writeGate (writeLogits i) • F (weightedSum (fun j => gate (readLogits j)) x)‖ :=
      norm_add_le _ _
    _ = ‖phaseMix P (finiteSinkhorn skipLogits 20) φ x i‖ +
          writeGate (writeLogits i) * ‖F (weightedSum (fun j => gate (readLogits j)) x)‖ := by
      rw [norm_smul, Real.norm_of_nonneg hc.1.le]
    _ ≤ R + writeGate (writeLogits i) * B :=
      add_le_add (phaseMix_norm_le P _ φ x R (fun j k => (hpos j k).le)
        (twenty_iterations_row_sum skipLogits) hx i)
        (mul_le_mul_of_nonneg_left hF hc.1.le)
    _ ≤ R + 2 * B := add_le_add le_rfl (mul_le_mul_of_nonneg_right hc.2.le hB)

/-- Four unit streams in `ℂ`, zero logits, four distinct phases and a constant branch of norm one:
every hypothesis holds and the output streams are at most `1 + 2`. -/
example : ∀ i : Fin 4,
    ‖phaseMHCStep complexPhase (fun _ => 0) (fun _ => 0) (0 : Matrix (Fin 4) (Fin 4) ℝ)
      (fun j => (j : ℝ)) (fun _ => (1 : ℂ)) (fun _ => (1 : ℂ)) i‖ ≤ 1 + 2 * 1 :=
  fun i => phaseMHCStep_norm_le complexPhase _ _ _ _ _ _ 1 1 zero_le_one (fun _ => by simp)
    (by simp) i

/-- Zero phases: the same four streams, the paper's block. -/
example :
    phaseMHCStep complexPhase (fun _ => 0) (fun _ => 0) (0 : Matrix (Fin 4) (Fin 4) ℝ)
      (fun _ => 0) (fun _ => (1 : ℂ)) (fun _ => (1 : ℂ)) =
    finiteMHCStep (fun _ => 0) (fun _ => 0) (0 : Matrix (Fin 4) (Fin 4) ℝ)
      (fun _ => (1 : ℂ)) (fun _ => (1 : ℂ)) :=
  phaseMHCStep_zero_phases complexPhase _ _ _ _ _

end Transformer.MHC
