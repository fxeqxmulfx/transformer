import Transformer.GPTMini.Semantics.DepthTransition

/-!
# Quantitative true flags and strictly earlier depth occurrences

Source: actual complete original detector transitions at f11b6e2,
DepthNormalization's shared r=1/4224 and DepthRecurrence's strict
opposite-ending predecessor. Incoming norm at most 4064 puts the
true attention residual within the proved norm-4096 RMS domain.
Every real written flag is zero or lies in [r^2,128].

Actual flag positivity is exactly raw-type match and a strictly
earlier genuine incoming feature at least r^2. The current position
is excluded using its derived zero opposite self, not by replacing
the original causal mask or attention operator with a strict mask.

These local transition statements must still be instantiated by an
induction on actual raw-token hidden states. They do not assume any
desired label, supply a prefix oracle or prove training success.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The same genuine pre-FFN RMS square lies in [r^2,128] on bounded incoming detector states.
Source: actual M+32 attention residual, protected norm lower bound and original RMS scale bounds. -/
theorem depthDetector_RMSSquare_bounds (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4064) (i : Fin T) :
    depthScaleLower ^ 2 ≤ (depthScale mode eps (depthDetectorAttentionState mode stage eps positions x i)) ^ 2 ∧
      (depthScale mode eps (depthDetectorAttentionState mode stage eps positions x i)) ^ 2 ≤ 128 := by
  have hl := depthDetectorAttentionState_norm_lower mode stage eps positions x i (hinput.1 i)
  have ha := depthDetectorAttentionState_norm mode stage eps heps positions x i 4064 (hnorm i)
    (depthDetectorInput_nonneg mode stage x hinput)
  have hu : ‖depthDetectorAttentionState mode stage eps positions x i‖ ≤ 4096 := by linarith
  have hs := depthScale_lower mode eps heps.le hclip _ hl hu
  have hp := depthScale_pos mode eps heps.le _ hl
  refine ⟨?_, (depthScale_upper mode eps heps.le _ hl).2⟩
  simpa only [pow_two] using mul_le_mul hs hs depthScaleLower_bounds.1.le hp.le

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4064) := by
  refine ⟨by norm_num, by norm_num, depthDetectorInput_active .easy, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)

/-- Every actual newly written depth flag is zero or at least r^2, and is no larger than 128.
Source: complete real block formula and its own true pre-FFN RMS square bounds, including arbitrary wrong-type signals. -/
theorem depthDetectorBlock_feature_bounds (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4064) (i : Fin T) (b : Fin 2) :
    (blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) = 0 ∨ depthScaleLower ^ 2 ≤
      blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i (depthFeatureCoordinate mode stage b)) ∧
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i (depthFeatureCoordinate mode stage b) ≤ 128 := by
  have hs := depthDetector_RMSSquare_bounds mode stage eps heps hclip positions x hinput hnorm i
  have hn (j : Fin T) : ‖x j‖ ≤ 4096 := (hnorm j).trans (by norm_num)
  rw [depthDetectorBlock_feature mode stage eps heps hclip hT positions x hinput hn i b]
  split_ifs
  · exact ⟨Or.inr hs.1, hs.2⟩
  · exact ⟨Or.inl rfl, by norm_num⟩

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .hard 0 (fun _ : Fin 2 => depthAxis .hard 0 + depthAxis .hard 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .hard 0 + depthAxis .hard 1) j‖ ≤ 4064) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .hard, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .hard).2.trans (by norm_num)

/-- Threshold detection and strict positivity coincide for every actual new flag.
Source: the genuine zero-or-r^2 gap and the strictly positive shared lower RMS scalar. -/
theorem depthDetectorBlock_feature_threshold_iff (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4064) (i : Fin T) (b : Fin 2) :
    depthScaleLower ^ 2 ≤ blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) ↔
    0 < blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i (depthFeatureCoordinate mode stage b) := by
  have hr := sq_pos_of_pos depthScaleLower_bounds.1
  have hf := depthDetectorBlock_feature_bounds mode stage eps heps hclip hT positions x hinput hnorm i b
  constructor
  · intro hp
    linarith
  · intro hp
    rcases hf.1 with hz | hg
    · rw [hz] at hp
      exact False.elim ((lt_irrefl 0) hp)
    · exact hg

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4064) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)

/-- Actual new flag positivity is exactly a matching raw type and a strictly earlier real incoming occurrence feature.
Source: genuine causal head positivity, zero opposite self and full real FFN transition.
The original attention remains self-inclusive; strict precedence is proved from the raw-type representation. -/
theorem depthDetectorBlock_feature_ordered (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4096) (i : Fin T) (b : Fin 2) :
    0 < blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) ↔ x i (depthTypeCoordinate mode b) = 1 ∧
      ∃ j : Fin T, j.val < i.val ∧ depthScaleLower ^ 2 ≤ x j (depthSourceCoordinate mode stage b) := by
  have hbound (j : Fin T) : 1 ≤ ‖x j‖ ∧ ‖x j‖ ≤ 4096 :=
    ⟨depthState_norm_lower mode (x j) (hinput.1 j), hnorm j⟩
  rw [depthDetectorBlock_feature_positive_iff mode stage eps heps hclip hT positions x hinput hnorm i b]
  constructor
  · rintro ⟨hk, hp⟩
    obtain ⟨j, hji, hf⟩ := (depthAttentionSignal_positive_iff mode (depthSourceCoordinate mode stage)
      eps heps.le hclip hT x b i hbound (hinput.2.2.2.2.1 b) (hinput.2.2.2.2.2 b i hk)).mp hp
    refine ⟨hk, j, ?_, hf⟩
    by_contra hnot
    have he : j = i := Fin.ext (by omega)
    subst j
    rw [hinput.2.2.2.2.2 b i hk] at hf
    have hr := sq_pos_of_pos depthScaleLower_bounds.1
    linarith
  · rintro ⟨hk, j, hji, hf⟩
    refine ⟨hk, ?_⟩
    exact (depthAttentionSignal_positive_iff mode (depthSourceCoordinate mode stage) eps heps.le hclip hT
      x b i hbound (hinput.2.2.2.2.1 b) (hinput.2.2.2.2.2 b i hk)).mpr ⟨j, hji.le, hf⟩

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4096) := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy, ?_⟩
  intro j
  exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)

/-- A current wrong raw type has exactly zero actual newly written flag, regardless of the other branch's signal.
Source: genuine complete-block type/presence formula with the raw type retained through real prenorm and residuals. -/
theorem depthDetectorBlock_feature_wrong_type (mode : Mode) (stage : Fin 3) (eps : ℝ)
    (heps : 0 < eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128) (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (hinput : DepthDetectorInput mode stage x)
    (hnorm : ∀ j, ‖x j‖ ≤ 4096) (i : Fin T) (b : Fin 2) (hk : x i (depthTypeCoordinate mode b) = 0) :
    blockForward (depthConfig mode) (depthDetectorBlock mode stage) eps positions x i
      (depthFeatureCoordinate mode stage b) = 0 := by
  rw [depthDetectorBlock_feature mode stage eps heps hclip hT positions x hinput hnorm i b, hk]
  apply ite_eq_right
  intro h
  norm_num at h

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    DepthDetectorInput .easy 0 (fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) ∧
    (∀ j : Fin 2, ‖(fun _ : Fin 2 => depthAxis .easy 0 + depthAxis .easy 1) j‖ ≤ 4096) ∧
    (depthAxis .easy 0 + depthAxis .easy 1) (depthTypeCoordinate .easy 1) = 0 := by
  refine ⟨by norm_num, by norm_num, by decide, depthDetectorInput_active .easy, ?_, ?_⟩
  · intro j
    exact (depthDetectorInput_active_norm .easy).2.trans (by norm_num)
  · change (depthAxis .easy 0 + depthAxis .easy 1) (depthCoordinate .easy 2) = 0
    rw [depthActiveDetector_coordinate]
    norm_num

end Transformer.GPTMini.Semantics
