module

public import Transformer.Grokking.Protocol.AdamStationary
public import Transformer.Grokking.Protocol.DivisionTarget
public import Transformer.Grokking.Protocol.ReadoutPairing

/-!
# Adam without weight decay: a readout that converges does not keep the batches fitted

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and the appendix *Additional experimental
details*: the experiment of the figure "used Adam optimizer with no weight decay"; openai/grok
at 3d64b1d8, `grok/training.py`; `experiments/grok_division/PROTOCOL.md`, *The optimizer* and
*The loss and the accuracy*; `review/grok_division_convergence.md`, bridge 4.

The paper states no convergence. This file excludes one way of proving one: through a limit
of the parameters. Let a run of the source's optimizer without weight decay see at every
update the gradient of the backward pass for a batch with a target (`batchRun`), let every
readout parameter converge, and let the gains and biases of the second normalizations stay
bounded. Then for every `μ > 0`, from some update on, every batch has a target that leads the
other tokens by less than `μ` at the parameters the batch is seen at
(`eventually_exists_margin_lt`).

The proof has no continuity and no descent in it. The readout's gradients are bounded by the
normalization (`abs_sourceGrad_out_le_of_norm`), so they tend to `0` where the readout
converges (`tendsto_adamRun_grad_zero`), and so does their pairing with the readout. On a
batch fitted with the margin `μ` that pairing is at most `− μ e^{−2Z} / vocab`
(`sum_out_mul_sourceGrad_le_of_margin`), with `Z` a bound of the logits that the convergent
readout and the bounded normalization give (`abs_logits_le`).

For division (`division_eventually_margin_lt`): on every sample whose Adam trajectory
converges, from some update on every batch has an equation whose answer or whose closing token
leads by less than `μ`. Read the other way: a trajectory that from some update on fits its
batches with a margin does not converge, so a proof that held-out answers become and stay
correct cannot pass through a limit of the parameters in that regime. Nothing here says which
of the two happens, and nothing is said of runs with weight decay, whose update has another
term (`AdamStationary`). The witnesses are in `ReadoutWitness`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {S : Sizes}

/-- **The parameters of a run on batches**: after `t` updates of the optimizer with constants
`h` whose update `s` sees the gradient of the backward pass for the batch `batch s`. -/
noncomputable abbrev batchRun (h : Hyper) (batch : ℕ → List (Example S)) (θ₀ : Param S → ℝ)
    (t : ℕ) : Param S → ℝ :=
  (adamRun h (fun s θ => sourceGrad (batch s) θ) θ₀ t).params

/-- **A run of Adam without weight decay whose readout converges does not keep its batches
fitted with a margin**: if every readout parameter converges and the gains and biases of the
second normalizations stay bounded, then for every `μ > 0`, from some update on, every batch
has a target that leads by less than `μ` at the parameters the batch is seen at. -/
theorem eventually_exists_margin_lt (hS : 0 < S.blocks) (hV : 2 ≤ S.vocab) {h : Hyper}
    (hh : h.Regular) (hd : h.decay = 0) {batch : ℕ → List (Example S)}
    (hN : ∀ t, 0 < targetCount (batch (t + 1))) {θ₀ : Param S → ℝ} {M : ℝ}
    (hout : ∀ c i, ∃ a, Tendsto (fun t => batchRun h batch θ₀ t (.out c i)) atTop (𝓝 a))
    (hgain : ∀ t l i, |batchRun h batch θ₀ t (.gain2 l i)| ≤ M)
    (hbias : ∀ t l i, |batchRun h batch θ₀ t (.bias2 l i)| ≤ M) {μ : ℝ} (hμ : 0 < μ) :
    ∀ᶠ t in atTop, ∃ e ∈ batch (t + 1), ∃ pc ∈ e.targets,
      margin (logits (batchRun h batch θ₀ t) (relu S) e.tokens pc.1) pc.2 < μ := by
  choose a ha using hout
  have hzero : ∀ c i, Tendsto (fun t =>
      sourceGrad (batch (t + 1)) (batchRun h batch θ₀ t) (.out c i)) atTop (𝓝 0) := fun c i =>
    tendsto_adamRun_grad_zero hh hd (ha c i) fun t =>
      abs_sourceGrad_out_le_of_norm hS _ (hgain t) (hbias t) c i
  have hpair : Tendsto (fun t => ∑ c, ∑ i, batchRun h batch θ₀ t (.out c i) *
      sourceGrad (batch (t + 1)) (batchRun h batch θ₀ t) (.out c i)) atTop (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ fun c _ =>
      tendsto_finsetSum Finset.univ fun i _ => (ha c i).mul (hzero c i)
  obtain ⟨W, hW⟩ := Finite.bddAbove_range fun ci : Fin S.vocab × Fin S.width => |a ci.1 ci.2| + 1
  have hWt : ∀ᶠ t in atTop, ∀ c i, |batchRun h batch θ₀ t (.out c i)| ≤ W := by
    simp only [eventually_all]
    intro c i
    filter_upwards [(ha c i).abs.eventually (gt_mem_nhds (lt_add_one _))] with t ht
    exact ht.le.trans (hW ⟨(c, i), rfl⟩)
  have hvocab : (0 : ℝ) < S.vocab := Nat.cast_pos.2 (by omega)
  have hκ : 0 < μ * Real.exp (-(2 * (S.width * (W * (√S.width * M + M))))) / S.vocab := by
    positivity
  filter_upwards [hWt, hpair.eventually (lt_mem_nhds (neg_neg_of_pos hκ))] with t hWt hlt
  by_contra hcon
  push Not at hcon
  exact absurd (sum_out_mul_sourceGrad_le_of_margin hV (hN t) hμ.le (fun e _ pc _ c =>
    abs_logits_le hS hWt (hgain t) (hbias t) (relu S) e.tokens pc.1 c) hcon) (not_le.2 hlt)

/-- Sequences that converge in each of finitely many coordinates are bounded together. -/
lemma exists_abs_le_of_tendsto {ι : Type*} [Finite ι] {u : ℕ → ι → ℝ}
    (h : ∀ p, ∃ a, Tendsto (fun t => u t p) atTop (𝓝 a)) : ∃ M, ∀ t p, |u t p| ≤ M := by
  have h1 : ∀ p, ∃ C, ∀ t, |u t p| ≤ C := fun p => by
    obtain ⟨a, ha⟩ := h p
    obtain ⟨C, hC⟩ := (Metric.isBounded_range_of_tendsto _ ha).exists_norm_le
    exact ⟨C, fun t => hC _ ⟨t, rfl⟩⟩
  choose C hC using h1
  obtain ⟨M, hM⟩ := Finite.bddAbove_range C
  exact ⟨M, fun t p => (hC p t).trans (hM ⟨p, rfl⟩)⟩

/-- The parameters of the division run with weight decay `decay` after `t` updates. -/
noncomputable abbrev divisionRun (decay : ℝ) (ω : Sample (97 * 96) 4656 (Param paperSizes))
    (t : ℕ) : Param paperSizes → ℝ :=
  trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512 ω t

/-- The equations of the batch of update `t` of a sample. -/
abbrev divisionBatch (ω : Sample (97 * 96) 4656 (Param paperSizes)) (t : ℕ) : List Equation :=
  (batchSlots 4656 512 ω.2.2 t).map (trainAt equations (by norm_num : 4656 ≤ 97 * 96) ω.1)

/-- **Every batch of the division protocol has an equation**: the last batch of an epoch has
48 of them. -/
lemma length_divisionBatch_pos (ω : Sample (97 * 96) 4656 (Param paperSizes)) (t : ℕ) :
    0 < (divisionBatch ω t).length := by
  have hb : batchesPerEpoch 4656 512 = 10 := by decide
  have hm := Nat.mod_lt (t - 1) (by norm_num : 0 < 10)
  rw [List.length_map, length_batchSlots, hb]
  omega

/-- **Division under Adam without weight decay: a trajectory that converges does not keep its
batches fitted with a margin.** On a sample whose parameters all converge, for every `μ > 0`,
from some update on every batch has an equation whose answer or whose closing token leads by
less than `μ` at the parameters the batch is seen at. -/
theorem division_eventually_margin_lt (ω : Sample (97 * 96) 4656 (Param paperSizes))
    (hlim : ∀ p, ∃ a, Tendsto (fun t => divisionRun 0 ω t p) atTop (𝓝 a)) {μ : ℝ} (hμ : 0 < μ) :
    ∀ᶠ t in atTop, ∃ x ∈ divisionBatch ω (t + 1),
      margin (answerLogits (divisionRun 0 ω t) x) (residueToken x.1) < μ ∨
        margin (closingLogits (divisionRun 0 ω t) x) eosToken < μ := by
  obtain ⟨M, hM⟩ := exists_abs_le_of_tendsto hlim
  have key := eventually_exists_margin_lt (S := paperSizes) (h := Hyper.source 0)
    (batch := fun s => (divisionBatch ω s).map supervised) (θ₀ := ω.2.1) (M := M) (by decide)
    (by decide) (Hyper.source_regular 0) rfl
    (fun t => by
      rw [targetCount_supervised]
      exact Nat.mul_pos (by norm_num) (length_divisionBatch_pos ω (t + 1)))
    (fun c i => hlim (.out c i)) (fun t l i => hM t (.gain2 l i))
    (fun t l i => hM t (.bias2 l i)) hμ
  filter_upwards [key] with t ht
  obtain ⟨e, he, pc, hpc, hlt⟩ := ht
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he
  refine ⟨x, hx, ?_⟩
  simp only [supervised, List.mem_cons, List.not_mem_nil, or_false] at hpc
  rcases hpc with rfl | rfl
  · exact Or.inl hlt
  · exact Or.inr hlt

end Transformer.Grokking.Protocol
