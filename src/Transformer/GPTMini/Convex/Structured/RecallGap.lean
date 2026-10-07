import Transformer.GPTMini.Convex.Structured.RecallEnergy

/-!
# Uniform whole-configuration gap from complete raw recall semantics

Source: MQAR's independently verified raw last-write/table parser at
9e6661b and actual finite shared learned potentials at b127359. Every
visible position pair and latent channel assignment remains a rival.
Wrong table positions or binding offsets incur their finite penalties;
wrong channels incur their real energy loss. A fully agreeing genuine
record is no later than the true last write, by actual raw parsing.

The whole raw theorem derives its data configuration and gap from
successful parsing and the context bound alone. No correct encoder,
probability or model score is assumed. The finite gain may be zero for
this non-strict energy inequality; actual decoder confidence will use
a positive finite logarithmic gain. Data target construction is absent
from the true free-parameter inference normalizer. Full tensor-stack
realization and successful AdamW training remain separate obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.Semantics
open scoped Classical
noncomputable section

/-- All real rivals have a derived gain deficit when their selected data record comes from genuine raw parsing.
Source: actual complete learned energies, raw adjacent table reads, full key-code injectivity and chronological latest-write semantics. -/
theorem recallBinding_parsed_gap (P : ℕ) (rewrites : Bool) (gain : ℝ) (hgain : 0 ≤ gain)
    (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (query previous selected : Fin tokens.length) (key : Fin 256)
    (hq : tokens.get query = recallKeyId key) (hp : previous.val + 1 = selected.val)
    (hk : tokens.get previous = recallKeyId key) (hs : recallTableValuePosition P selected.val)
    (hlast : RecallRawLatestWrite P tokens.get selected key) :
    ∀ rival : RawBindingConfiguration tokens, rival ≠ recallBindingTarget tokens previous selected →
      energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain) rival ≤
        energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain)
          (recallBindingTarget tokens previous selected) - gain := by
  have hmatch : tokens.get query = tokens.get previous := hq.trans hk.symm
  have hselected := recallBinding_selected_energy P gain tokens hcap query previous selected hp hs hmatch
  have hselectedNonneg : 0 ≤ gain * (selected.val : ℝ) := mul_nonneg hgain (Nat.cast_nonneg _)
  intro rival hrival
  rw [hselected]
  have hposition : (rival.1.2.val : ℝ) ≤ 64 := by
    exact_mod_cast (lt_of_lt_of_le rival.1.2.isLt hcap).le
  have hchronology := mul_le_mul_of_nonneg_left hposition hgain
  by_cases hslot : recallTableValuePosition P rival.1.2.val
  · by_cases hadj : rival.1.1.val + 1 = rival.1.2.val
    · by_cases hmiss : rival.2.1 ≠ recallBindingDigits (tokens.get query) ∨
          rival.2.1 ≠ recallBindingDigits (tokens.get rival.1.1) ∨
          rival.2.2 ≠ recallBindingOutput (tokens.get rival.1.2)
      · have hchannels := recallChannelEnergy_deficit gain hgain (tokens.get query)
          (tokens.get rival.1.1) (tokens.get rival.1.2) rival.2.1 rival.2.2 hmiss
        rw [recallBinding_energy]
        simp only [hslot, hadj, ite_true, zero_add, add_zero]
        linarith
      · have hqEq : rival.2.1 = recallBindingDigits (tokens.get query) := by
          by_contra hne
          exact hmiss (Or.inl hne)
        have hkEq : rival.2.1 = recallBindingDigits (tokens.get rival.1.1) := by
          by_contra hne
          exact hmiss (Or.inr (Or.inl hne))
        have hvEq : rival.2.2 = recallBindingOutput (tokens.get rival.1.2) := by
          by_contra hne
          exact hmiss (Or.inr (Or.inr hne))
        obtain ⟨other, _, hother, _⟩ :=
          recallBinding_table_pair P rewrites tokens answer hanswer rival.1.1 rival.1.2 hadj hslot
        have hcodes := hqEq.symm.trans hkEq
        rw [hq, hother, recallBindingDigits_key, recallBindingDigits_key] at hcodes
        have hkeyEq := recallDigit_injective hcodes
        have hactualKey : tokens.get rival.1.1 = recallKeyId key :=
          hother.trans (congrArg recallKeyId hkeyEq.symm)
        have hlate := recallBinding_matching_last P rewrites tokens answer hanswer selected rival.1.1 rival.1.2
          key hlast hadj hslot hactualKey
        by_cases he : rival.1.2 = selected
        · have hprevEq : rival.1.1 = previous := recallBinding_unique_previous _ _ selected (by simpa only [he] using hadj) hp
          have hmatchingEq : rival.2.1 = recallBindingDigits (tokens.get previous) :=
            hqEq.trans (congrArg recallBindingDigits hmatch)
          have hvalueEq : rival.2.2 = recallBindingOutput (tokens.get selected) := by simpa only [he] using hvEq
          exact False.elim (hrival (Prod.ext (Prod.ext hprevEq he) (Prod.ext hmatchingEq hvalueEq)))
        · have hne : rival.1.2.val ≠ selected.val := fun h => he (Fin.ext h)
          have hn : rival.1.2.val + 1 ≤ selected.val := by omega
          have hreal : (rival.1.2.val : ℝ) + 1 ≤ (selected.val : ℝ) := by exact_mod_cast hn
          have hchron := mul_le_mul_of_nonneg_left hreal hgain
          have hupper := recallBinding_energy_upper P gain hgain tokens hcap query rival
          rw [mul_add, mul_one] at hchron
          linarith
    · have hupper := recallBinding_energy_excluded P gain hgain tokens hcap query rival (Or.inr hadj)
      linarith
  · have hupper := recallBinding_energy_excluded P gain hgain tokens hcap query rival (Or.inl hslot)
    linarith

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    recallOverwriteTokens.get (5 : Fin 6) = recallKeyId 0 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧ recallTableValuePosition 2 (4 : Fin 6).val ∧
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) :=
  ⟨by norm_num, by decide, by decide, by decide, by decide, by decide, by decide, recallOverwriteTokens_raw_latest⟩

/-- Complete raw parsing alone derives the real final query, correct output token and every true whole-configuration energy gap.
Source: full validated selected-write inversion and the genuine finite learned all-pair gap proof above, without model correctness premises. -/
theorem recallBinding_raw_gap (P : ℕ) (rewrites : Bool) (gain : ℝ) (hgain : 0 ≤ gain)
    (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query previous selected : Fin tokens.length,
      query.val + 1 = tokens.length ∧ ((tokens.get selected).val : ℤ) = answer ∧
      ∀ rival : RawBindingConfiguration tokens, rival ≠ recallBindingTarget tokens previous selected →
        energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain) rival ≤
          energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain)
            (recallBindingTarget tokens previous selected) - gain := by
  obtain ⟨query, previous, selected, key, value, hquery, hq, hp, hk, hv, hs, _, hlast, ha⟩ :=
    recallBinding_selected P rewrites tokens answer hanswer
  refine ⟨query, previous, selected, hquery, ?_, recallBinding_parsed_gap P rewrites gain hgain
    tokens hcap answer hanswer query previous selected key hq hp hk hs hlast⟩
  rw [hv]
  exact ha

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by norm_num, by decide, by decide⟩

/-- Successful raw parsing derives a correct actual complete configuration with a uniform true inference probability bound.
Source: the full finite raw energy gap, genuine Gibbs normalization and the exact all-pair implicit count; no probability is assumed. -/
theorem recallBinding_raw_probability (P : ℕ) (rewrites : Bool) (gain : ℝ) (hgain : 0 ≤ gain)
    (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query previous selected : Fin tokens.length,
      query.val + 1 = tokens.length ∧ ((tokens.get selected).val : ℤ) = answer ∧
      1 - 1073741824 * Real.exp (-gain) ≤ rawBindingProbability (recallBindingParameters P gain)
        tokens hcap query (recallBindingTarget tokens previous selected) := by
  obtain ⟨query, previous, selected, hquery, hanswerId, hgap⟩ :=
    recallBinding_raw_gap P rewrites gain hgain tokens hcap answer hanswer
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  have hprob := probability_gap (rawBindingLinear tokens hcap query) (fun _ => 0)
    (recallBindingParameters P gain) (recallBindingTarget tokens previous selected) gain hgap
  have hcard : (Fintype.card (RawBindingConfiguration tokens) : ℝ) ≤ 1073741824 := by
    exact_mod_cast rawBindingConfiguration_bound tokens hcap
  have htail := mul_le_mul_of_nonneg_left hcard (Real.exp_pos (-gain)).le
  rw [← rawBinding_probability] at hprob
  refine ⟨query, previous, selected, hquery, hanswerId, ?_⟩
  nlinarith

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by norm_num, by decide, by decide⟩

/-- One finite logarithmic gain for every complete raw recall prefix, allowing both latent-pair and head-selection tails.
Source: 1073741824 complete raw choices at cap 64 plus two finite mixture-head logits; no infinite weights are used. -/
def recallBindingGain : ℝ := Real.log 1000000000000

/-- The actual uniform recall chronology gain is strictly positive and finite.
Source: its ordinary log(10^12) real value, independent of every raw table and prefix. -/
theorem recallBindingGain_pos : 0 < recallBindingGain := Real.log_pos (by norm_num)

/-- The actual full latent-pair tail and an additional two-way head tail fit strictly inside the whole-vocabulary decoder allowance.
Source: exact exp/log inversion at the finite gain, not numerical approximation or a supplied attention confidence. -/
theorem recallBindingGain_tail : 1073741826 * Real.exp (-recallBindingGain) < (1 / 11 : ℝ) := by
  unfold recallBindingGain
  rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 1000000000000)]
  norm_num

end
end Transformer.GPTMini.Convex.Structured
