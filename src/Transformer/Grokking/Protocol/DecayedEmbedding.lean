module

public import Transformer.Grokking.Protocol.AbsentBatches
public import Transformer.Grokking.Protocol.AdamIdle
public import Transformer.Grokking.Protocol.ReadoutWitness
public import Transformer.Grokking.Protocol.UnreadToken

/-!
# Under weight decay the embedding of a residue converges to zero only

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "For
most experiments we used AdamW optimizer with learning rate `10⁻³`, weight decay `1`,
`β₁ = 0.9`, `β₂ = 0.98`, … minibatch size 512"; openai/grok at 3d64b1d8, `grok/training.py`
(`CustomAdamW.step`), `grok/transformer.py` (`Transformer.embed`) and `grok/data.py`
(`ArithmeticIterator`); `experiments/grok_division/PROTOCOL.md`, *The optimizer*;
`review/grok_division_convergence.md`, section 5. The paper states no limit of the
parameters.

Three facts of the protocol meet here. A batch in which no equation shows the residue `r`
reads the token of `r` nowhere (`read_ne_residueToken`), so the backward pass returns `0` in
its embedding (`divisionGrad_emb_eq_zero`, from `UnreadToken`): the update is idle there. A
coordinate that converges under a weight decay that is not `0` and is idle at infinitely many
updates converges to `0` (`AdamIdle`; the source's `β₁² = 0.81 < 0.98 = β₂`). And for almost
every sample infinitely many batches show no `r` (`AbsentBatches`). Hence, for every
positive weight decay and whatever the law of the initial parameters:

    for almost every sample, a coordinate of the embedding of a residue that converges
    converges to `0`

(`emb_eq_zero_of_tendsto_divisionRun` for one sample, `ae_emb_eq_zero_of_tendsto_divisionRun`).
The source applies the decay only when it is positive, so the statements here ask for a
positive one; `AdamIdle` is about the formula of `adamStep`, at any decay that is not `0`.

Parameters that give every residue the same embedding give every equation the same logits
(`logits_read_eq_of_emb_eq`, from `UnreadToken`): the three other tokens read are the same in
all equations. So for almost every sample, if the parameters converge, their limit does not
tell the equations apart (`ae_logits_limit_eq`). `DecayedLimit` draws the consequence for
the margins along the run.

Nothing here says that a run converges. Whether a run from the source's initialization
converges with positive probability is not known here. Under the law whose initial parameters
are zero the run of almost every sample stays at zero (`ae_divisionRun_dirac_zero`), which is
the witness below.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- The tokens of an equation that does not show `r` are not the token of `r`. -/
lemma read_ne_residueToken {r : Fin 97} {x : Equation} (h : ¬Shows r x) (t : Fin 6) :
    read x t ≠ residueToken r := by
  have hn : numerator x ≠ r := fun e => h (Or.inl e)
  have hd : denominator x ≠ r := fun e => h (Or.inr (Or.inl e))
  have hq : x.1 ≠ r := fun e => h (Or.inr (Or.inr e))
  fin_cases t
  · exact (residueToken_ne_eosToken r).symm
  · exact fun e => hn (residueToken_injective e)
  · exact (residueToken_ne_divToken r).symm
  · exact fun e => hd (residueToken_injective e)
  · exact (residueToken_ne_eqToken r).symm
  · exact fun e => hq (residueToken_injective e)

/-- **A batch in which no equation shows a residue leaves its embedding idle**: the backward
pass returns `0` there. -/
theorem divisionGrad_emb_eq_zero {B : List Equation} {r : Fin 97} (hB : ∀ x ∈ B, ¬Shows r x)
    (θ : Param paperSizes → ℝ) (i : Fin 128) :
    divisionGrad B θ (.emb (residueToken r) i) = 0 :=
  sourceGrad_emb_eq_zero (fun e he t => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 he
    exact read_ne_residueToken (hB x hx) t) θ i

/-- **Under weight decay the embedding of a residue that is missing from infinitely many
batches converges to zero only**: if the weight decay is positive, a coordinate of the
embedding of `r` converges to `a` along the run of a sample, and infinitely many of its batches
show no `r`, then `a = 0`. -/
theorem emb_eq_zero_of_tendsto_divisionRun {decay : ℝ} (hd : 0 < decay)
    (ω : Sample (97 * 96) 4656 (Param paperSizes)) {r : Fin 97} {i : Fin 128} {a : ℝ}
    (hlim : Tendsto (fun t => divisionRun decay ω t (.emb (residueToken r) i)) atTop (𝓝 a))
    (habs : ∃ᶠ t in atTop, ∀ x ∈ divisionBatch ω (t + 1), ¬Shows r x) : a = 0 :=
  eq_zero_of_tendsto_adamRun_of_frequently_idle (h := Hyper.source decay)
    (grad := fun s θ => divisionGrad (divisionBatch ω s) θ) (θ₀ := ω.2.1)
    (i := .emb (residueToken r) i) (Hyper.source_regular decay) (by norm_num [Hyper.source])
    (by norm_num [Hyper.source]) hd.ne' hlim
    (habs.mono fun _ ht => divisionGrad_emb_eq_zero ht _ i)

/-- **Division under weight decay: for almost every sample a coordinate of the embedding of a
residue that converges converges to zero**, for every positive weight decay and every law of
the initial parameters. -/
theorem ae_emb_eq_zero_of_tendsto_divisionRun {ν : Measure (Param paperSizes → ℝ)} [SFinite ν]
    {decay : ℝ} (hd : 0 < decay) :
    ∀ᵐ ω ∂law (97 * 96) 4656 ν, ∀ (r : Fin 97) (i : Fin 128) (a : ℝ),
      Tendsto (fun t => divisionRun decay ω t (.emb (residueToken r) i)) atTop (𝓝 a) →
        a = 0 := by
  filter_upwards [ae_all_iff.2 fun r => ae_frequently_batch_not_shows (ν := ν) r]
    with ω hω r i a hlim
  exact emb_eq_zero_of_tendsto_divisionRun hd ω hlim (hω r)

/-- **Parameters that give every residue the same embedding give every equation the same
logits**: the other three tokens read are the same in all equations. -/
theorem logits_read_eq_of_emb_eq {θ : Param paperSizes → ℝ}
    (h : ∀ (r s : Fin 97) (i : Fin 128),
      θ (.emb (residueToken r) i) = θ (.emb (residueToken s) i)) (x y : Equation) :
    logits θ (relu paperSizes) (read x) = logits θ (relu paperSizes) (read y) :=
  logits_eq_of_emb_eq fun t i => by
    fin_cases t
    · rfl
    · exact h _ _ i
    · rfl
    · exact h _ _ i
    · rfl
    · exact h _ _ i

/-- **Division under weight decay: for almost every sample the limit of the parameters, if
there is one, does not tell the equations apart**: it gives every equation the same logits
at every position. -/
theorem ae_logits_limit_eq {ν : Measure (Param paperSizes → ℝ)} [SFinite ν] {decay : ℝ}
    (hd : 0 < decay) :
    ∀ᵐ ω ∂law (97 * 96) 4656 ν, ∀ θ' : Param paperSizes → ℝ,
      (∀ p, Tendsto (fun t => divisionRun decay ω t p) atTop (𝓝 (θ' p))) →
        ∀ x y : Equation,
          logits θ' (relu paperSizes) (read x) = logits θ' (relu paperSizes) (read y) := by
  filter_upwards [ae_emb_eq_zero_of_tendsto_divisionRun (ν := ν) hd] with ω hω θ' hlim x y
  exact logits_read_eq_of_emb_eq
    (fun r s i => by rw [hω r i _ (hlim _), hω s i _ (hlim _)]) x y

/-- **Under the law whose initial parameters are zero the run of almost every sample stays at
zero**, for every weight decay. Such a run converges, so the statements for almost every
sample are about runs that exist. -/
theorem ae_divisionRun_dirac_zero (decay : ℝ) :
    ∀ᵐ ω ∂law (97 * 96) 4656 (Measure.dirac (0 : Param paperSizes → ℝ)),
      ∀ t, divisionRun decay ω t = 0 := by
  have h0 : ∀ᵐ θ ∂Measure.dirac (0 : Param paperSizes → ℝ), θ = 0 := by
    rw [ae_dirac_eq]
    exact rfl
  filter_upwards [ae_law_init (N := 97 * 96) (k := 4656) h0] with ω hω t
  exact divisionRun_eq_zero decay hω t

/-! The witnesses: `3 / 5 = 20`, which does not show `7`; a split that keeps the residue `1`
out of training, with the initial parameters zero, whose run stays at zero under the weight
decay `1`; and the law whose initial parameters are zero, under which the run of almost every
sample converges. -/

example : read (20, 4) 1 ≠ residueToken 7 := read_ne_residueToken (by decide) 1

example (θ : Param paperSizes → ℝ) : divisionGrad [(20, 4)] θ (.emb (residueToken 7) 0) = 0 :=
  divisionGrad_emb_eq_zero (fun x hx => by rw [List.mem_singleton.1 hx]; decide) θ 0

/-- The hypotheses of `emb_eq_zero_of_tendsto_divisionRun` hold together. -/
example : ∃ ω : Sample (97 * 96) 4656 (Param paperSizes),
    Tendsto (fun t => divisionRun 1 ω t (.emb (residueToken 1) 0)) atTop (𝓝 0) ∧
      ∃ᶠ t in atTop, ∀ x ∈ divisionBatch ω (t + 1), ¬Shows 1 x := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  refine ⟨(π, 0, fun _ => 1), tendsto_const_nhds.congr fun t =>
    (congrFun (divisionRun_eq_zero 1 rfl t) _).symm, Frequently.of_forall fun t x hx => ?_⟩
  obtain ⟨s, -, rfl⟩ := List.mem_map.1 hx
  have h1 : (1 : Fin 97) ∈ hidden := by decide
  rintro (h | h | h)
  · exact (hπ s).1 (h ▸ h1)
  · exact (hπ s).2.1 (h ▸ h1)
  · exact (hπ s).2.2 (h ▸ h1)

example (ω : Sample (97 * 96) 4656 (Param paperSizes)) (h0 : ω.2.1 = 0)
    (habs : ∃ᶠ t in atTop, ∀ x ∈ divisionBatch ω (t + 1), ¬Shows 1 x) : (0 : ℝ) = 0 :=
  emb_eq_zero_of_tendsto_divisionRun one_pos ω (r := 1) (i := 0)
    (tendsto_const_nhds.congr fun t => (congrFun (divisionRun_eq_zero 1 h0 t) _).symm) habs

example : ∀ᵐ ω ∂law (97 * 96) 4656 (Measure.dirac (0 : Param paperSizes → ℝ)),
    ∀ (r : Fin 97) (i : Fin 128) (a : ℝ),
      Tendsto (fun t => divisionRun 1 ω t (.emb (residueToken r) i)) atTop (𝓝 a) → a = 0 :=
  ae_emb_eq_zero_of_tendsto_divisionRun one_pos

example : logits (0 : Param paperSizes → ℝ) (relu paperSizes) (read (20, 4)) =
    logits 0 (relu paperSizes) (read (0, 4)) :=
  logits_read_eq_of_emb_eq (fun _ _ _ => rfl) _ _

example := ae_logits_limit_eq (ν := Measure.dirac (0 : Param paperSizes → ℝ)) (decay := 1)
  one_pos

end Transformer.Grokking.Protocol
