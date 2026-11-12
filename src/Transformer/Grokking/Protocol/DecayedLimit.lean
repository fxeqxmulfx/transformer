module

public import Transformer.Grokking.Protocol.DecayedEmbedding
public import Transformer.Grokking.Protocol.LimitNotFitted
public import Transformer.Grokking.Protocol.Unlabelled

/-!
# A run with weight decay that converges does not keep its answers ahead by a margin

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "For
most experiments we used AdamW optimizer with learning rate `10⁻³`, weight decay `1`";
openai/grok at 3d64b1d8, `grok/training.py`; `experiments/grok_division/PROTOCOL.md`, *The
optimizer* and *The loss and the accuracy*; `review/grok_division_convergence.md`, section 5.
The paper states no convergence.

`DecayedEmbedding` says: for almost every sample of a run with a positive weight decay,
the limit of the parameters, if there is one, gives every equation the same logits. The
logits are continuous in the parameters (`Continuity`). So along parameters that converge to
such a limit only the answers that are the best residue of the limit can keep a margin: there
is a residue `r` such that for every `μ > 0`, from some index on, the answer of every equation
with another answer than `r` leads by less than `μ` (`exists_eventually_margin_lt_of_tendsto`).
At most 96 equations have the answer `r` and 4656 are held out, so some held-out equation has
another answer (`exists_mem_heldout_fst_ne`).

Hence, for every positive weight decay and every law of the initial parameters, for almost
every sample whose parameters converge:

* there is a residue `r` such that for every `μ > 0`, from some update on, no equation with
  another answer than `r` has its answer ahead by `μ` (`ae_exists_eventually_margin_lt`), so
  at most 96 of the 9312 equations keep a margin;
* for every `μ > 0`, from some update on, some held-out equation has its answer ahead by less
  than `μ` (`ae_eventually_exists_heldout_margin_lt`).

Read the other way, for almost every sample: a run with weight decay in which, again and
again, every held-out answer is ahead by one positive margin does not converge. So a proof
that training with weight decay reaches and keeps correct held-out answers with a margin
cannot pass through a limit of the parameters. `ReadoutStationary` and `LimitNotFitted` say
the like of Adam without weight decay, by another argument.

What is not said. It is not said that a run converges, nor that it does not. A run that
converges and whose held-out answers are right from some update on, with margins that tend to
`0`, is not excluded: at its limit the answers would be tied. Whether a run from the source's
initialization converges with positive probability is not known here; under the law whose
initial parameters are zero the run of almost every sample does
(`ae_divisionRun_dirac_zero`), which is the witness below.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- **Along parameters that converge to parameters that do not tell the equations apart, only
the answers of one residue can keep a margin**: there is a residue `r` such that for every
`μ > 0`, from some index on, the answer of every equation with another answer leads by less
than `μ`. -/
theorem exists_eventually_margin_lt_of_tendsto {u : ℕ → Param paperSizes → ℝ}
    {θ' : Param paperSizes → ℝ} (hlim : ∀ p, Tendsto (fun t => u t p) atTop (𝓝 (θ' p)))
    (hconst : ∀ x y : Equation, answerLogits θ' x = answerLogits θ' y) :
    ∃ r : Fin 97, ∀ μ : ℝ, 0 < μ → ∀ᶠ t in atTop, ∀ x : Equation, x.1 ≠ r →
      margin (answerLogits (u t) x) (residueToken x.1) < μ := by
  obtain ⟨r, -, hr⟩ := Finset.exists_max_image Finset.univ
    (fun r : Fin 97 => answerLogits θ' (0, 0) (residueToken r)) Finset.univ_nonempty
  have hθ : Tendsto u atTop (𝓝 θ') := tendsto_pi_nhds.2 hlim
  refine ⟨r, fun μ hμ => eventually_all.2 fun x => ?_⟩
  by_cases hx : x.1 = r
  · exact Eventually.of_forall fun _ h => absurd hx h
  have hz : ∀ c : Fin 239, Tendsto (fun t => answerLogits (u t) x c) atTop
      (𝓝 (answerLogits θ' x c)) := fun c =>
    ((continuous_logits_relu (S := paperSizes) (read x) 4 c).tendsto θ').comp hθ
  have hlt : answerLogits θ' x (residueToken x.1) - answerLogits θ' x (residueToken r) < μ := by
    have := hr x.1 (Finset.mem_univ _)
    rw [hconst x (0, 0)]
    linarith
  filter_upwards [((hz _).sub (hz _)).eventually (gt_mem_nhds hlt)] with t ht _
  exact (margin_le _ fun h => hx (residueToken_injective h).symm).trans_lt ht

/-- **Some held-out equation has another answer than a given residue**: at most 96 equations
have a given answer, and 4656 are held out. -/
theorem exists_mem_heldout_fst_ne (π : Equiv.Perm (Fin (97 * 96))) (r : Fin 97) :
    ∃ x ∈ heldout equations 4656 π, x.1 ≠ r := by
  by_contra hcon
  push Not at hcon
  have hlt : ∀ i : Fin 4656, 4656 + (i : ℕ) < 97 * 96 := fun i => by
    have := i.isLt
    omega
  have h := Finset.card_le_card_of_injOn (s := (Finset.univ : Finset (Fin 4656)))
    (t := Finset.univ.filter fun j : Fin (97 * 96) => (equations j).1 = r)
    (fun i => π ⟨4656 + i, hlt i⟩)
    (fun i _ => Finset.mem_coe.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _,
      hcon _ (Finset.mem_image.2 ⟨⟨4656 + i, hlt i⟩,
        Finset.mem_filter.2 ⟨Finset.mem_univ _, Nat.le_add_right _ _⟩, rfl⟩)⟩))
    (fun a _ b _ hab => by
      have := congrArg Fin.val (π.injective hab)
      simp only at this
      exact Fin.ext (by omega))
  have h96 := card_quotient_eq_le r
  rw [Finset.card_univ, Fintype.card_fin] at h
  omega

/-- **Division under weight decay: for almost every sample whose parameters converge, only the
answers of one residue can keep a margin.** For every positive weight decay and every law of
the initial parameters: there is a residue `r` such that for every `μ > 0`, from some
update on, the answer of every equation with another answer than `r` leads by less than
`μ`. -/
theorem ae_exists_eventually_margin_lt {ν : Measure (Param paperSizes → ℝ)} [SFinite ν]
    {decay : ℝ} (hd : 0 < decay) :
    ∀ᵐ ω ∂law (97 * 96) 4656 ν,
      (∀ p, ∃ a, Tendsto (fun t => divisionRun decay ω t p) atTop (𝓝 a)) →
        ∃ r : Fin 97, ∀ μ : ℝ, 0 < μ → ∀ᶠ t in atTop, ∀ x : Equation, x.1 ≠ r →
          margin (answerLogits (divisionRun decay ω t) x) (residueToken x.1) < μ := by
  filter_upwards [ae_logits_limit_eq (ν := ν) hd] with ω hω hlim
  choose θ' hθ' using hlim
  exact exists_eventually_margin_lt_of_tendsto hθ' fun x y => congrFun (hω θ' hθ' x y) 4

/-- **Division under weight decay: for almost every sample whose parameters converge, from some
update on a held-out answer leads by less than any given margin.** -/
theorem ae_eventually_exists_heldout_margin_lt {ν : Measure (Param paperSizes → ℝ)}
    [SFinite ν] {decay : ℝ} (hd : 0 < decay) :
    ∀ᵐ ω ∂law (97 * 96) 4656 ν,
      (∀ p, ∃ a, Tendsto (fun t => divisionRun decay ω t p) atTop (𝓝 a)) →
        ∀ μ : ℝ, 0 < μ → ∀ᶠ t in atTop, ∃ x ∈ heldout equations 4656 ω.1,
          margin (answerLogits (divisionRun decay ω t) x) (residueToken x.1) < μ := by
  filter_upwards [ae_exists_eventually_margin_lt (ν := ν) hd] with ω hω hlim μ hμ
  obtain ⟨r, hr⟩ := hω hlim
  obtain ⟨x, hx, hne⟩ := exists_mem_heldout_fst_ne ω.1 r
  exact (hr μ hμ).mono fun t ht => ⟨x, hx, ht x hne⟩

/-! The witnesses: the parameters zero, which do not tell the equations apart; the identity
split; and the law whose initial parameters are zero, under which the run of almost every
sample converges, with the weight decay `1`. -/

example : ∃ r : Fin 97, ∀ μ : ℝ, 0 < μ → ∀ᶠ _ : ℕ in atTop, ∀ x : Equation, x.1 ≠ r →
    margin (answerLogits (0 : Param paperSizes → ℝ) x) (residueToken x.1) < μ :=
  exists_eventually_margin_lt_of_tendsto (u := fun _ => 0) (θ' := 0)
    (fun _ => tendsto_const_nhds)
    fun x y => congrFun (logits_read_eq_of_emb_eq (fun _ _ _ => rfl) x y) 4

example : ∃ x ∈ heldout equations 4656 (1 : Equiv.Perm (Fin (97 * 96))), x.1 ≠ 96 :=
  exists_mem_heldout_fst_ne 1 96

/-- Under the law whose initial parameters are zero the hypothesis holds for almost every
sample, and so does the conclusion: only the answers of one residue keep a margin. -/
example : ∀ᵐ ω ∂law (97 * 96) 4656 (Measure.dirac (0 : Param paperSizes → ℝ)),
    ∃ r : Fin 97, ∀ μ : ℝ, 0 < μ → ∀ᶠ t in atTop, ∀ x : Equation, x.1 ≠ r →
      margin (answerLogits (divisionRun 1 ω t) x) (residueToken x.1) < μ := by
  filter_upwards [ae_exists_eventually_margin_lt
    (ν := Measure.dirac (0 : Param paperSizes → ℝ)) (decay := 1) one_pos,
    ae_divisionRun_dirac_zero 1] with ω hω h0
  exact hω fun p => ⟨0, tendsto_const_nhds.congr fun t => (congrFun (h0 t) p).symm⟩

example : ∀ᵐ ω ∂law (97 * 96) 4656 (Measure.dirac (0 : Param paperSizes → ℝ)),
    ∀ μ : ℝ, 0 < μ → ∀ᶠ t in atTop, ∃ x ∈ heldout equations 4656 ω.1,
      margin (answerLogits (divisionRun 1 ω t) x) (residueToken x.1) < μ := by
  filter_upwards [ae_eventually_exists_heldout_margin_lt
    (ν := Measure.dirac (0 : Param paperSizes → ℝ)) (decay := 1) one_pos,
    ae_divisionRun_dirac_zero 1] with ω hω h0
  exact hω fun p => ⟨0, tendsto_const_nhds.congr fun t => (congrFun (h0 t) p).symm⟩

end Transformer.Grokking.Protocol
