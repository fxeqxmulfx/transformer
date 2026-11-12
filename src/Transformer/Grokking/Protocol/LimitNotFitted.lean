module

public import Transformer.Grokking.Protocol.Continuity
public import Transformer.Grokking.Protocol.OrbitAnchors
public import Transformer.Grokking.Protocol.ReadoutWitness

/-!
# The limit of a trajectory of Adam without weight decay does not fit the training set

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and the appendix *Additional experimental
details*: the experiment of the figure "used Adam optimizer with no weight decay"; openai/grok
at 3d64b1d8, `grok/training.py`; `experiments/grok_division/PROTOCOL.md`, *The optimizer*;
`review/grok_division_convergence.md`, section 5.

`ReadoutStationary` says, along a trajectory whose parameters converge: for every `μ > 0`,
from some update on, every batch has an equation whose answer or closing token leads by less
than `μ`. The logits are continuous in the parameters (`Continuity`), so a margin that is
above `μ` at the limit of the logits is at least `μ` at all late updates
(`eventually_le_margin`). Hence at the limit the margins of the training set are not all
positive (`exists_margin_le_zero_of_tendsto_division`): some training equation has an answer
or a closing token that does not lead every other token there.

So if the parameters of a run without weight decay converge, they converge to parameters that
do not fit the training set with a margin: a limit of the parameters and a training set that
is fitted with a margin at the limit exclude each other. Nothing here says whether a
trajectory converges, and nothing is said of runs with weight decay.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

/-- **A margin above `μ` at the limit of the logits is at least `μ` at all late indices.** -/
theorem eventually_le_margin {V : Type*} [Fintype V] {z : ℕ → V → ℝ} {z' : V → ℝ} {y : V}
    {μ : ℝ} (hz : ∀ c, Tendsto (fun t => z t c) atTop (𝓝 (z' c))) (hV : ∃ c, c ≠ y)
    (h : μ < margin z' y) : ∀ᶠ t in atTop, μ ≤ margin (z t) y := by
  have h1 : ∀ c : {c // c ≠ y}, ∀ᶠ t in atTop, μ < z t y - z t c := fun c =>
    ((hz y).sub (hz c)).eventually (lt_mem_nhds (h.trans_le (margin_le z' c.2)))
  have hne : Nonempty {c // c ≠ y} := hV.elim fun c hc => ⟨⟨c, hc⟩⟩
  filter_upwards [eventually_all.2 h1] with t ht
  exact le_ciInf fun c => (ht c).le

/-- **The limit of a trajectory of Adam without weight decay does not fit the training set
with positive margins**: if the parameters of a sample's trajectory converge, then at the
limit some training equation has an answer or a closing token with a margin of at most `0`. -/
theorem exists_margin_le_zero_of_tendsto_division
    (ω : Sample (97 * 96) 4656 (Param paperSizes)) {θ' : Param paperSizes → ℝ}
    (hlim : ∀ p, Tendsto (fun t => divisionRun 0 ω t p) atTop (𝓝 (θ' p))) :
    ∃ i : Fin 4656,
      margin (answerLogits θ' (trainEq ω.1 i)) (residueToken (trainEq ω.1 i).1) ≤ 0 ∨
        margin (closingLogits θ' (trainEq ω.1 i)) eosToken ≤ 0 := by
  by_contra hcon
  push Not at hcon
  obtain ⟨i₀, hi₀⟩ := Finite.exists_min fun i : Fin 4656 =>
    min (margin (answerLogits θ' (trainEq ω.1 i)) (residueToken (trainEq ω.1 i).1))
      (margin (closingLogits θ' (trainEq ω.1 i)) eosToken)
  set m := min (margin (answerLogits θ' (trainEq ω.1 i₀)) (residueToken (trainEq ω.1 i₀).1))
    (margin (closingLogits θ' (trainEq ω.1 i₀)) eosToken) with hm
  have hmpos : 0 < m := lt_min (hcon i₀).1 (hcon i₀).2
  have hθ : Tendsto (fun t => divisionRun 0 ω t) atTop (𝓝 θ') := tendsto_pi_nhds.2 hlim
  have hz : ∀ (x : Equation) (pos : Fin 6) (c : Fin 239), Tendsto (fun t =>
      logits (divisionRun 0 ω t) (relu paperSizes) (read x) pos c) atTop
      (𝓝 (logits θ' (relu paperSizes) (read x) pos c)) := fun x pos c =>
    ((continuous_logits_relu (S := paperSizes) (read x) pos c).tendsto θ').comp hθ
  have hev : ∀ᶠ t in atTop, ∀ i : Fin 4656,
      m / 2 ≤ margin (answerLogits (divisionRun 0 ω t) (trainEq ω.1 i))
        (residueToken (trainEq ω.1 i).1) ∧
      m / 2 ≤ margin (closingLogits (divisionRun 0 ω t) (trainEq ω.1 i)) eosToken :=
    eventually_all.2 fun i =>
      (eventually_le_margin (hz _ 4) ⟨eosToken, (residueToken_ne_eosToken _).symm⟩
        ((half_lt_self hmpos).trans_le ((hi₀ i).trans (min_le_left _ _)))).and
      (eventually_le_margin (hz _ 5) ⟨eqToken, by decide⟩
        ((half_lt_self hmpos).trans_le ((hi₀ i).trans (min_le_right _ _))))
  obtain ⟨t, ht, x, hx, hlt⟩ := (hev.and
    (division_eventually_margin_lt ω (fun p => ⟨_, hlim p⟩) (half_pos hmpos))).exists
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
  rcases hlt with h | h
  · exact absurd (ht i).1 (not_le.2 h)
  · exact absurd (ht i).2 (not_le.2 h)

/-! The witnesses: constant rows of two logits, and the trajectory from zero, which converges
to zero, where every margin is `0`. -/

example : ∀ᶠ _ : ℕ in atTop, (1 : ℝ) ≤ margin (![(2 : ℝ), 0]) 0 := by
  have : Nonempty {c : Fin 2 // c ≠ 0} := ⟨⟨1, by decide⟩⟩
  refine eventually_le_margin (z := fun _ => ![(2 : ℝ), 0]) (fun _ => tendsto_const_nhds)
    ⟨1, by decide⟩ (lt_of_lt_of_le (by norm_num : (1 : ℝ) < 2) (le_ciInf fun c => ?_))
  obtain ⟨c, hc⟩ := c
  fin_cases c
  · exact absurd rfl hc
  · simp

example : ∃ i : Fin 4656,
    margin (answerLogits 0 (trainEq zeroSample.1 i))
      (residueToken (trainEq zeroSample.1 i).1) ≤ 0 ∨
      margin (closingLogits 0 (trainEq zeroSample.1 i)) eosToken ≤ 0 :=
  exists_margin_le_zero_of_tendsto_division zeroSample fun p =>
    tendsto_const_nhds.congr fun t => (congrFun (divisionRun_zeroSample 0 t) p).symm

end Transformer.Grokking.Protocol
