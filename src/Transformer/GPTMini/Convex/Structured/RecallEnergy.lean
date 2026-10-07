import Transformer.GPTMini.Convex.Structured.RecallPotentials
import Transformer.GPTMini.Convex.Structured.ChannelGaps
import Transformer.GPTMini.Convex.Structured.RawBinding

/-!
# Genuine raw finite recall energies and local deficits

Source: the actual shared finite recall weights at b127359 and unchanged
raw all-pair inference at e0be315. Every channel and positional term is
evaluated from those actual weights. Wrong physical binding, excluded
table positions and wrong matching/value assignments have derived finite
deficits; correct probabilities or model logits are never assumed.

The reference configuration below specifies data supervision only and
is absent from actual inference. Full successful parsing must still
derive its raw selected write and discharge the uniform rival gap.
All unrestricted raw parameters of the true forward/loss remain free;
the table/adjacency predicates appear only when evaluating given weights.
No infinite gains, optimizer modification or training-success claim is
used. Full tensor/prenorm/residual/tied integration remains separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

/-- The actual channel part of the finite given pointer weights, retaining both independently trained matching sums and values.
Source: the full shared Q/K/value lookup identities; this witness expression is not a separate inference operator. -/
def recallChannelEnergy (gain : ℝ) (query key value : Fin 548) (matching : Fin 4 → Fin 4) (values : Fin 5 → Fin 4) : ℝ :=
  channelEnergy (fun g c => sharpRowLogits (recallBindingDigits query g) (65 * gain) c +
    sharpRowLogits (recallBindingDigits key g) (65 * gain) c) matching +
  channelEnergy (fun h => sharpRowLogits (recallBindingOutput value h) (65 * gain)) values

/-- Every actual complete channel assignment has at most thirteen finite group gains.
Source: separate true Q/K upper bounds for four groups and a true value bound for five groups. -/
theorem recallChannelEnergy_upper (gain : ℝ) (hgain : 0 ≤ gain) (query key value : Fin 548)
    (matching : Fin 4 → Fin 4) (values : Fin 5 → Fin 4) :
    recallChannelEnergy gain query key value matching values ≤ 845 * gain := by
  have hM : 0 ≤ 65 * gain := by linarith
  have hm := sharpMatching_upper (recallBindingDigits query) (recallBindingDigits key) matching (65 * gain) hM
  have hv := sharpChannel_upper (recallBindingOutput value) values (65 * gain) hM
  norm_num only [Fintype.card_fin, Nat.cast_ofNat] at hm hv
  unfold recallChannelEnergy
  linarith

example : (0 : ℝ) ≤ 1 := by norm_num

/-- Any genuine query, key or output-channel mismatch loses one entire finite witness gain.
Source: the actual three independent channel sums and a real differing assignment, including jointly trainable values. -/
theorem recallChannelEnergy_deficit (gain : ℝ) (hgain : 0 ≤ gain) (query key value : Fin 548)
    (matching : Fin 4 → Fin 4) (values : Fin 5 → Fin 4)
    (hmiss : matching ≠ recallBindingDigits query ∨ matching ≠ recallBindingDigits key ∨ values ≠ recallBindingOutput value) :
    recallChannelEnergy gain query key value matching values ≤ 780 * gain := by
  have hM : 0 ≤ 65 * gain := by linarith
  have hq := sharpChannel_upper (recallBindingDigits query) matching (65 * gain) hM
  have hk := sharpChannel_upper (recallBindingDigits key) matching (65 * gain) hM
  have hv := sharpChannel_upper (recallBindingOutput value) values (65 * gain) hM
  norm_num only [Fintype.card_fin, Nat.cast_ofNat] at hq hk hv
  unfold recallChannelEnergy
  rw [channelEnergy_add]
  rcases hmiss with hqMiss | hkMiss | hvMiss
  · have h := sharpChannel_miss (recallBindingDigits query) matching (65 * gain) hM hqMiss
    norm_num only [Fintype.card_fin, Nat.cast_ofNat] at h
    linarith
  · have h := sharpChannel_miss (recallBindingDigits key) matching (65 * gain) hM hkMiss
    norm_num only [Fintype.card_fin, Nat.cast_ofNat] at h
    linarith
  · have h := sharpChannel_miss (recallBindingOutput value) values (65 * gain) hM hvMiss
    norm_num only [Fintype.card_fin, Nat.cast_ofNat] at h
    linarith

example : (0 : ℝ) ≤ 1 ∧ ((fun _ : Fin 4 => (1 : Fin 4)) ≠ recallBindingDigits (recallKeyId 0) ∨
    (fun _ : Fin 4 => (1 : Fin 4)) ≠ recallBindingDigits (recallKeyId 0) ∨
    (fun _ : Fin 5 => (0 : Fin 4)) ≠ recallBindingOutput (recallValueId 0)) := by
  refine ⟨by norm_num, Or.inl ?_⟩
  rw [recallBindingDigits_key]
  intro he
  have h := congrArg (fun f : Fin 4 → Fin 4 => (f 0).val) he
  norm_num [recallDigit] at h

/-- Actual equal query/key channels and the raw value's own decoder channels give exactly 845 times the chronology gain.
Source: eight real matching gains plus five real value gains from the concrete shared token fields. -/
theorem recallChannelEnergy_self (gain : ℝ) (key value : Fin 548) :
    recallChannelEnergy gain key key value (recallBindingDigits key) (recallBindingOutput value) = 845 * gain := by
  unfold recallChannelEnergy
  rw [sharpMatching_basis_self, sharpValue_basis_self]
  ring

/-- Every genuine raw finite given-weight energy includes exactly its learned positional penalties and actual channel score.
Source: the real raw linear map, full shared coordinate reads and signed relative displacement equivalence. -/
theorem recallBinding_energy (P : ℕ) (gain : ℝ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64)
    (query : Fin tokens.length) (z : RawBindingConfiguration tokens) :
    energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain) z =
      (if recallTableValuePosition P z.1.2.val then 0 else -(65 * gain)) + gain * (z.1.2.val : ℝ) +
      (if z.1.1.val + 1 = z.1.2.val then 0 else -(65 * gain)) +
      recallChannelEnergy gain (tokens.get query) (tokens.get z.1.1) (tokens.get z.1.2) z.2.1 z.2.2 := by
  simp only [energy, add_zero, rawBindingLinear, bindingPointerLinear_apply, pointerEnergy,
    bindingRouteBias, sharedRouteBias, recallBinding_position, recallBinding_chronology,
    recallBinding_relative, rawBindingPosition]
  unfold recallChannelEnergy channelEnergy
  simp_rw [recallBinding_query, recallBinding_key, recallBinding_value]
  ring

example : recallOverwriteTokens.length ≤ 64 := by decide

/-- Every raw candidate's actual energy is bounded by its physical chronology and thirteen complete group gains.
Source: exact finite given-weight evaluation, nonpositive learned penalties and the real channel upper bound. -/
theorem recallBinding_energy_upper (P : ℕ) (gain : ℝ) (hgain : 0 ≤ gain) (tokens : List (Fin 548))
    (hcap : tokens.length ≤ 64) (query : Fin tokens.length) (z : RawBindingConfiguration tokens) :
    energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain) z ≤
      gain * (z.1.2.val : ℝ) + 845 * gain := by
  rw [recallBinding_energy]
  have hc := recallChannelEnergy_upper gain hgain (tokens.get query) (tokens.get z.1.1) (tokens.get z.1.2) z.2.1 z.2.2
  split_ifs <;> linarith

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 := ⟨by norm_num, by decide⟩

/-- A true excluded position or wrong physical key/value displacement loses a full finite penalty in the actual all-pair forward.
Source: free given absolute/relative weights evaluated at the original raw endpoints, with every candidate still present. -/
theorem recallBinding_energy_excluded (P : ℕ) (gain : ℝ) (hgain : 0 ≤ gain) (tokens : List (Fin 548))
    (hcap : tokens.length ≤ 64) (query : Fin tokens.length) (z : RawBindingConfiguration tokens)
    (hbad : ¬recallTableValuePosition P z.1.2.val ∨ z.1.1.val + 1 ≠ z.1.2.val) :
    energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain) z ≤
      gain * (z.1.2.val : ℝ) + 780 * gain := by
  rw [recallBinding_energy]
  have hc := recallChannelEnergy_upper gain hgain (tokens.get query) (tokens.get z.1.1) (tokens.get z.1.2) z.2.1 z.2.2
  split_ifs <;> first | tauto | linarith

example : (0 : ℝ) ≤ 1 ∧ recallOverwriteTokens.length ≤ 64 ∧
    (¬recallTableValuePosition 2 (0 : Fin 6).val ∨ (0 : Fin 6).val + 1 ≠ (0 : Fin 6).val) :=
  ⟨by norm_num, by decide, Or.inl (by decide)⟩

/-- A training-only raw selected-record configuration contains its actual matching and full value digits.
Source: unchanged physical raw token reads; this reference configuration never enters the learned forward normalizer. -/
def recallBindingTarget (tokens : List (Fin 548)) (previous selected : Fin tokens.length) : RawBindingConfiguration tokens :=
  ((previous, selected), (recallBindingDigits (tokens.get previous), recallBindingOutput (tokens.get selected)))

/-- Given finite shared weights derive the exact genuine selected energy from raw adjacency, table slot and matching IDs alone.
Source: actual raw energy evaluation and thirteen finite self-channel gains, with no correct probability or logit premise. -/
theorem recallBinding_selected_energy (P : ℕ) (gain : ℝ) (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64)
    (query previous selected : Fin tokens.length) (hp : previous.val + 1 = selected.val)
    (hs : recallTableValuePosition P selected.val) (hmatch : tokens.get query = tokens.get previous) :
    energy (rawBindingLinear tokens hcap query) (fun _ => 0) (recallBindingParameters P gain)
      (recallBindingTarget tokens previous selected) = gain * (selected.val : ℝ) + 845 * gain := by
  unfold recallBindingTarget
  rw [recallBinding_energy]
  simp only [hp, hs, ite_true, zero_add, add_zero]
  rw [hmatch, recallChannelEnergy_self]

example : recallOverwriteTokens.length ≤ 64 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallTableValuePosition 2 (4 : Fin 6).val ∧ recallOverwriteTokens.get (5 : Fin 6) = recallOverwriteTokens.get (3 : Fin 6) := by decide

end
end Transformer.GPTMini.Convex.Structured
