module

public import Transformer.Grokking.Protocol.ReadoutStationary
public import Transformer.Grokking.Protocol.SolverScale

/-!
# Witnesses for the statements on the readout's gradient

Source: openai/grok at 3d64b1d8, `grok/transformer.py` and `grok/training.py`;
`experiments/grok_division/PROTOCOL.md`, *The model* and *The optimizer*;
`review/grok_division_convergence.md`, bridge 4.

The hypotheses of `ReadoutStationary` are those of a run whose readout converges. One run is
known to converge: with every parameter zero, a decoder with a block has the logits `0` along
every coordinate line (`logits_update_zero`), so the backward pass returns `0`
(`sourceGrad_zero`) and the run stays at zero, with or without weight decay
(`batchRun_zero`, `divisionRun_eq_zero`, `divisionRun_zeroSample`). It is the witness of
`eventually_exists_margin_lt` and of `division_eventually_margin_lt`; its margins are `0`. No
other run is proved to converge here, and by those theorems one that keeps its batches fitted
with a margin does not.

The hypotheses of `sum_out_mul_sourceGrad_le_of_margin` are those of a fitted batch. The built
parameters `solver` fit the answer of every equation with a margin of at least `1/900` and
logits of at most `1/870` (`solverScale_le_margin`, `solverScale_mem`), so the readout of
`solver` paired with its gradient on the answer of `3 / 5 = 20` is negative: the last witness.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {S : Sizes}

/-- With every other parameter zero, a decoder with a block has the logits `0` along every
coordinate line. -/
lemma logits_update_zero (hS : 0 < S.blocks) (p : Param S) (s : ℝ) (act : Activation S)
    (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context) :
    logits (Function.update (0 : Param S → ℝ) p s) act tokens t = fun _ => 0 := by
  funext q
  by_cases hp : ∃ c i, p = .out c i
  · obtain ⟨c, i, rfl⟩ := hp
    have hφ : decodeTo (0 : Param S → ℝ) act tokens S.blocks t i = 0 :=
      abs_eq_zero.1 (le_antisymm (by
        simpa using abs_decodeTo_le hS (θ := (0 : Param S → ℝ)) (G := 0) (B := 0) (by simp)
          (by simp) act tokens t i) (abs_nonneg _))
    rw [logits_update_out, hφ]
    simp [logits]
  · simp only [logits]
    refine Finset.sum_eq_zero fun i _ => ?_
    rw [Function.update_of_ne fun h => hp ⟨q, i, h.symm⟩]
    simp

/-- **At the parameters zero the backward pass returns zero**, in a decoder with a block. -/
theorem sourceGrad_zero (hS : 0 < S.blocks) (B : List (Example S)) :
    sourceGrad B (0 : Param S → ℝ) = 0 := by
  funext p
  unfold sourceGrad frozenLoss exampleLoss
  simp only [logits_update_zero hS]
  exact deriv_const _ _

/-- A run from zero on fields that vanish at zero stays at zero, for every weight decay. -/
lemma adamRun_eq_zero {ι : Type*} (h : Hyper) {grad : ℕ → (ι → ℝ) → ι → ℝ}
    (hg : ∀ s, grad s 0 = 0) (t : ℕ) : adamRun h grad 0 t = ⟨0, 0, 0⟩ := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [adamRun, ih]
    simp only [adamStep, hg]
    congr 1 <;> funext i <;> simp

/-- **The run on batches from the parameters zero stays there**, in a decoder with a block. -/
theorem batchRun_zero (hS : 0 < S.blocks) (h : Hyper) (batch : ℕ → List (Example S)) (t : ℕ) :
    batchRun h batch 0 t = 0 := by
  rw [batchRun, adamRun_eq_zero h fun s => sourceGrad_zero hS (batch s)]

/-- The sample with the identity shuffles and every initial parameter zero. -/
def zeroSample : Sample (97 * 96) 4656 (Param paperSizes) := (1, 0, fun _ => 1)

/-- **The division run of a sample whose initial parameters are zero stays at zero**, for every
weight decay and whatever its shuffles. -/
theorem divisionRun_eq_zero (decay : ℝ) {ω : Sample (97 * 96) 4656 (Param paperSizes)}
    (h0 : ω.2.1 = 0) (t : ℕ) : divisionRun decay ω t = 0 := by
  change batchRun (Hyper.source decay) (fun s => (divisionBatch ω s).map supervised) ω.2.1 t = 0
  rw [h0]
  exact batchRun_zero (by decide) _ _ t

/-- The division run of the sample `zeroSample` stays at zero, for every weight decay. -/
theorem divisionRun_zeroSample (decay : ℝ) (t : ℕ) : divisionRun decay zeroSample t = 0 :=
  divisionRun_eq_zero decay rfl t

/-! The witnesses. -/

example (t : ℕ) : divisionRun 1 zeroSample t = 0 := divisionRun_eq_zero 1 rfl t

example : sourceGrad [twoTargets] 0 = 0 := sourceGrad_zero (by decide) _

example (t : ℕ) : batchRun (Hyper.source 1) (fun _ => [twoTargets]) 0 t = 0 :=
  batchRun_zero (by decide) _ _ t

/-- The run of Adam from zero on one batch of two targets: its readout converges and its
normalizations stay bounded, and no target leads by `1`. -/
example : ∀ᶠ t in atTop, ∃ e ∈ [twoTargets], ∃ pc ∈ e.targets,
    margin (logits (batchRun (Hyper.source 0) (fun _ => [twoTargets]) 0 t) (relu _) e.tokens pc.1)
      pc.2 < 1 :=
  eventually_exists_margin_lt (batch := fun _ => [twoTargets]) (θ₀ := 0) (M := 0) (by decide)
    (by decide) (Hyper.source_regular 0) rfl (fun _ => by decide)
    (fun c i => ⟨0, tendsto_const_nhds.congr fun t =>
      (congrFun (batchRun_zero (by decide) _ _ t) (.out c i)).symm⟩)
    (fun t l i => by rw [batchRun_zero (by decide)]; simp)
    (fun t l i => by rw [batchRun_zero (by decide)]; simp) one_pos

/-- The division run of Adam from zero converges, and in every batch an answer or a closing
token leads by less than `1`. -/
example : ∀ᶠ t in atTop, ∃ x ∈ divisionBatch zeroSample (t + 1),
    margin (answerLogits (divisionRun 0 zeroSample t) x) (residueToken x.1) < 1 ∨
      margin (closingLogits (divisionRun 0 zeroSample t) x) eosToken < 1 :=
  division_eventually_margin_lt zeroSample (fun p => ⟨0, tendsto_const_nhds.congr fun t =>
    (congrFun (divisionRun_zeroSample 0 t) p).symm⟩) one_pos

example : ∃ M, ∀ t p, |divisionRun 0 zeroSample t p| ≤ M :=
  exists_abs_le_of_tendsto fun p => ⟨0, tendsto_const_nhds.congr fun t =>
    (congrFun (divisionRun_zeroSample 0 t) p).symm⟩

/-- The readout of `solver` pairs negatively with its gradient on the answer of `3 / 5 = 20`,
which `solver` fits with a margin of at least `1/900` and logits of at most `1`. -/
example : ∑ c, ∑ i, solver (.out c i) * sourceGrad [supervisedAnswer (20, 4)] solver (.out c i) ≤
    -(1 / 900 * Real.exp (-(2 * 1)) / (239 : ℕ)) :=
  sum_out_mul_sourceGrad_le_of_margin (S := paperSizes) (by decide) (by decide) (by norm_num)
    (fun e he pc hpc c => by
      obtain rfl := List.mem_singleton.1 he
      obtain rfl := List.mem_singleton.1 hpc
      have hs := solverScale_mem (20, 4)
      have h := answerLogits_solver (20, 4) c
      rw [show logits solver (relu paperSizes) (supervisedAnswer (20, 4)).tokens 4 c =
        answerLogits solver (20, 4) c from rfl, h, markLogits]
      split_ifs <;> rw [abs_le] <;> constructor <;> linarith)
    (fun e he pc hpc => by
      obtain rfl := List.mem_singleton.1 he
      obtain rfl := List.mem_singleton.1 hpc
      exact (solverScale_mem (20, 4)).1.trans (solverScale_le_margin (20, 4)))

end Transformer.Grokking.Protocol
