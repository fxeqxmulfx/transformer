module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale.Batch
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget

/-!
# Bounded complex realization of a declared categorical decoder

Source: `review/flat_spectrum.md`, CM44. A fixed input-state map may repeat
states and its declared raw decoder may be many-to-one. The actual conjugate
head realizes a chosen margin, with every complex coefficient charged. This
does not identify unknown targets or make the input-query matrix unitary.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {I X C : Type*} [Fintype I] [DecidableEq I]

/-- Pull back the actual realified complex code along the fixed declared
input-state map, CM44. The map is frozen during head optimization. -/
def features (U : Matrix I I ℂ) (state : X → I) : I × Bool → X → ℝ :=
  fun k x => complexFeatures U k (state x)

/-- The source unitary bound survives repeated query states, CM44. This
is a coordinate bound and imposes no injectivity on the input-state map. -/
theorem features_bound (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (state : X → I) (x : X) (k : I × Bool) : |features U state k x| ≤ 1 := by
  exact ConvexDepth.complexFeatures_bound U hU k (state x)

/-- The complete real/imaginary coefficient budget bounds every actual
query logit, CM44. Repeated states do not create uncharged coefficients. -/
theorem head_bound (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (state : X → I) {W : I × Bool → C → ℝ} {R : ℝ}
    (hW : W ∈ headBudget R) (x : X) (c : C) :
    |headLogits (features U state) W x c| ≤ R := by
  have h := headLogits_bound (features U state) hW (by norm_num : (0 : ℝ) ≤ 1)
    x (features_bound U hU state x) c
  simpa only [mul_one] using h

/-- CFt5's unitary inversion gives the declared margin on every mapped
input, CM44. The raw decoder need not be injective or correct at that input. -/
theorem logits [DecidableEq C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (state : X → I) (decoder : I → C)
    (a : ℝ) (x : X) :
    headLogits (features U state) (complexCoefficients (complexInitializer U decoder a)) x =
      fun c => if c = decoder (state x) then a else 0 := by
  change headLogits (complexFeatures U)
    (complexCoefficients (complexInitializer U decoder a)) (state x) = _
  exact complexInitializer_logits U hU decoder a (state x)

/-- Margin24 fits the same explicit native budget64 times the number of
source states, CM44. Every query uses the actual complete conjugate tensor. -/
theorem native_bound [Fintype C] [DecidableEq C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (state : X → I) (decoder : I → C)
    (x : X) (c : C) :
    |headLogits (features U state)
      (complexCoefficients (complexInitializer U decoder 24)) x c| ≤
      64 * (Fintype.card I : ℝ) := by
  apply head_bound U hU state
  exact complexInitializer_native_budget U hU decoder

/-- Every nonnegative margin through24 fits one common complete-head
budget, CM44. A native confidence bank can compare these feasible tensors. -/
theorem initializer_mem [Fintype C] [DecidableEq C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (decoder : I → C) {a : ℝ}
    (hzero : 0 ≤ a) (htop : a ≤ 24) :
    complexCoefficients (complexInitializer U decoder a) ∈
      headBudget (64 * (Fintype.card I : ℝ)) := by
  intro c
  have h := complexInitializer_budget U hU decoder a c
  rw [abs_of_nonneg hzero] at h
  have hn : (0 : ℝ) ≤ Fintype.card I := Nat.cast_nonneg _
  calc
    _ ≤ 2 * a * (Fintype.card I : ℝ) := h
    _ ≤ 64 * (Fintype.card I : ℝ) := by nlinarith

/-- Any positive declared margin preserves the decoder's strict raw
prediction, CM44. This says nothing about whether the decoder is correct. -/
theorem declared_winner [DecidableEq C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (state : X → I) (decoder : I → C)
    {a : ℝ} (ha : 0 < a) (x : X) (c : C) (hc : c ≠ decoder (state x)) :
    headLogits (features U state) (complexCoefficients (complexInitializer U decoder a)) x c <
      headLogits (features U state) (complexCoefficients (complexInitializer U decoder a))
        x (decoder (state x)) := by
  rw [logits U hU]
  simpa [hc] using ha

omit [Fintype I] [DecidableEq I] in
/-- Ordinary full-vocabulary CE of a correct margin, CM44. The class count
and margin are explicit arguments; no empirical correctness is assumed. -/
noncomputable def marginCE (classes : ℕ) (a : ℝ) : ℝ :=
  Real.log (1 + ((classes : ℝ) - 1) * Real.exp (-a))

omit [Fintype I] [DecidableEq I] in
/-- Increasing the margin cannot increase the correct-label CE for a
nonempty vocabulary, CM44. Wrong targets have the separate linear penalty. -/
theorem marginCE_antitone {classes : ℕ} (hclasses : 1 ≤ classes) {a b : ℝ}
    (hab : a ≤ b) : marginCE classes b ≤ marginCE classes a := by
  have hn : (1 : ℝ) ≤ classes := by exact_mod_cast hclasses
  have hnon : 0 ≤ (classes : ℝ) - 1 := sub_nonneg.mpr hn
  have he := Real.exp_le_exp.mpr (neg_le_neg hab)
  unfold marginCE
  apply Real.log_le_log (by positivity)
  simpa only [add_comm] using add_le_add_left (mul_le_mul_of_nonneg_left he hnon) 1

omit [Fintype I] [DecidableEq I] in
/-- Exact raw CE for an arbitrary target, CM44. A wrong declared label
adds the complete margin, even though its own predicted class is unchanged. -/
theorem row_ce [Fintype C] [DecidableEq C] (declared target : C) (a : ℝ) :
    crossEntropy (fun c => if c = declared then a else 0) target =
      marginCE (Fintype.card C) a + if declared = target then 0 else a := by
  by_cases h : declared = target
  · subst target
    simpa [marginCE] using fixed_margin_ce declared a
  · have heq : crossEntropy (fun c => if c = declared then a else 0) target =
        crossEntropy (fun c => if c = declared then a else 0) declared + a := by
      simp [crossEntropy, Ne.symm h]
    rw [heq, fixed_margin_ce, ite_eq_right h]
    exact rfl

/-- Actual complex-head CE has the same declared-label error identity,
CM44. Flatness supplies the feature bound, not the target on the right. -/
theorem initializer_ce [Fintype C] [DecidableEq C] (U : Matrix I I ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (state : X → I) (decoder : I → C)
    (a : ℝ) (x : X) (target : C) :
    crossEntropy (headLogits (features U state)
      (complexCoefficients (complexInitializer U decoder a)) x) target =
      marginCE (Fintype.card C) a + if decoder (state x) = target then 0 else a := by
  rw [logits U hU]
  exact row_ce _ target a

/-- Six inputs use two of four source states; repeated queries are allowed. -/
def witnessState (x : Fin 6) : Fin 4 := if x.val < 3 then 0 else 2

/-- Each raw Boolean label has two possible source states. -/
def witnessDecoder (i : Fin 4) : Bool := if i.val < 2 then false else true

example := features_bound fourier4 fourier4_unitary witnessState 5 (0, true)
example := head_bound fourier4 fourier4_unitary witnessState
  (W := fun _ (_ : Bool) => 1) (R := 8) (by intro c; norm_num [Fintype.card_prod]) 5 true
example := logits fourier4 fourier4_unitary witnessState witnessDecoder 1 5
example := native_bound fourier4 fourier4_unitary witnessState witnessDecoder 5 true
example := initializer_mem fourier4 fourier4_unitary witnessDecoder
  (by norm_num : (0 : ℝ) ≤ 1) (by norm_num : (1 : ℝ) ≤ 24)
example := declared_winner fourier4 fourier4_unitary witnessState witnessDecoder
  (by norm_num : (0 : ℝ) < 24) 5 false (by decide)
example := initializer_ce fourier4 fourier4_unitary witnessState witnessDecoder 24 5 false
example := marginCE_antitone (by norm_num : 1 ≤ (193 : ℕ))
  (by norm_num : (1 : ℝ) ≤ 24)

/-- A genuine wrong target incurs margin24 on top of the correct CE, CM44. -/
theorem witness_wrong_ce :
    crossEntropy (headLogits (features fourier4 witnessState)
      (complexCoefficients (complexInitializer fourier4 witnessDecoder 24)) 5) false =
      Real.log (1 + Real.exp (-24)) + 24 := by
  rw [initializer_ce fourier4 fourier4_unitary]
  norm_num [marginCE, witnessState, witnessDecoder]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence
