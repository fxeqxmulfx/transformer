module

public import Transformer.MHC.Section4_Parameterization
public import Transformer.GPTMini.RMSNorm

/-!
# Four-stream finite mHC with Peri-RMS sublayers

Local adaptation of arXiv:2512.24880v2, §§4.1–4.2, Eqs. (3), (8), (9),
and the diff-gpt inverse-width SSMax model frozen at commit c6c517b.
Migrated from its `research/mhc_scaled/Bounds.lean` and examples.
Input RMS, raw attention / SwiGLU, output RMS, and the residual coefficient
are explicit. The coefficient is outside output RMS and before twice-sigmoid
write-back. Exactly twenty column-then-row Sinkhorn cycles are used.

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

/-- Logits after the routing predictor, before sigmoid / finite Sinkhorn. -/
structure Routing where
  read : Fin 4 → ℝ
  write : Fin 4 → ℝ
  skip : Matrix (Fin 4) (Fin 4) ℝ

/-- A raw attention or SwiGLU branch and its input-dependent routing logits.
`raw` excludes both Peri-RMS operations and the depth coefficient. -/
structure Sublayer (d : ℕ) where
  routing : State d → Routing
  raw : State d → EucSpace d → EucSpace d

/-- mHC Eq. (3) with both current Peri-RMS sites and post-RMS scaling.
Read gates are sigmoid, write gates are twice sigmoid; the skip matrix is
the paper's finite twenty-cycle map, without denominator regularization. -/
noncomputable def step {d : ℕ} (eps scale : ℝ) (branch : Sublayer d)
    (x : State d) : State d :=
  let r := branch.routing x
  finiteMHCStep r.read r.write r.skip
    (fun z => scale • GPTMini.rmsNormEps eps
      (branch.raw x (GPTMini.rmsNormEps eps z))) x

/-- Finite row normalization suffices for the combined per-stream bound.
The factor two comes from mHC's actual write gate, not the stream count.
No exact double stochasticity assumption is made for finite Sinkhorn.
Local Peri-RMS extension of arXiv:2512.24880v2, Eqs. (3), (8), (9). -/
theorem step_norm_le {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (branch : Sublayer d)
    (x : State d) (R : ℝ) (hx : ∀ i, ‖x i‖ ≤ R) (i : Fin 4) :
    ‖step eps scale branch x i‖ ≤ R + 2 * scale * Real.sqrt (d : ℝ) := by
  let r := branch.routing x
  let H := finiteSinkhorn r.skip 20
  let v := branch.raw x (GPTMini.rmsNormEps eps
    (weightedSum (fun j => gate (r.read j)) x))
  have hpos : ∀ j k, 0 < H j k :=
    sinkhornIterate_pos _ (fun j k => Real.exp_pos (r.skip j k)) 20
  have hmix : ‖mix H x i‖ ≤ R :=
    weightedSum_convex_norm_le (H i) x R (fun j => (hpos i j).le)
      (twenty_iterations_row_sum r.skip i) hx
  have hc := writeGate_bounds (r.write i)
  have hv : ‖scale • GPTMini.rmsNormEps eps v‖ ≤ scale * Real.sqrt (d : ℝ) := by
    rw [norm_smul, Real.norm_of_nonneg hscale]
    exact mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hd v) hscale
  have hw : ‖writeGate (r.write i) • (scale • GPTMini.rmsNormEps eps v)‖ ≤
      2 * scale * Real.sqrt (d : ℝ) := by
    rw [norm_smul, Real.norm_of_nonneg hc.1.le]
    calc
      writeGate (r.write i) * ‖scale • GPTMini.rmsNormEps eps v‖ ≤
          writeGate (r.write i) * (scale * Real.sqrt (d : ℝ)) :=
        mul_le_mul_of_nonneg_left hv hc.1.le
      _ ≤ 2 * (scale * Real.sqrt (d : ℝ)) :=
        mul_le_mul_of_nonneg_right hc.2.le (by positivity)
      _ = 2 * scale * Real.sqrt (d : ℝ) := by ring
  change ‖mix H x i + writeGate (r.write i) •
    (scale • GPTMini.rmsNormEps eps v)‖ ≤ R + 2 * scale * Real.sqrt (d : ℝ)
  exact (norm_add_le _ _).trans (add_le_add hmix hw)

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

example : ‖collapse (fun _ => witness)‖ ≤ 1 := by
  exact collapse_norm_le (fun _ => witness) 1
    (by intro i; simp [witness, PiLp.norm_single])

end Transformer.MHC.PeriRMS
