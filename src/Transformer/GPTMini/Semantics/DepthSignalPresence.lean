import Transformer.GPTMini.Semantics.DepthAttentionBounds

/-!
# Exact presence gap in actual depth attention signals

Source: original depth attnSubLayer at f11b6e2, context cap 128 in
Basis, and the proved true RMS amplitude floor r^3 with r=1/4224.
A genuine visible feature at least r^2 supplies a real signal at
least r^3/128=2*depthThreshold when its current value is zero.

If every raw residual feature is zero or at least r^2, the actual
signal is positive exactly when a visible such feature exists. The
local residual norm domain and zero-current condition remain explicit;
the ordered layer induction must derive them from actual raw inputs.
No Boolean indicator defines the head or replaces its finite softmax.

An ordinary two-position real-vector control satisfies all local
hypotheses. It is an operator witness, not a whole-prefix encoder or
the definition of the full model's hidden state.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- A real-coordinate operator control has a nonzero earlier feature and a zero current feature.
Source: two ordinary residual unit vectors, solely for local hypothesis witnesses. -/
noncomputable def depthPresenceControl (j : Fin 2) : EucSpace (depthConfig .easy).d_model :=
  if j.val = 0 then depthAxis .easy 1 else depthAxis .easy 0

/-- The control's genuine residual norms and feature coordinates are evaluated from their actual real vectors.
Source: unit-axis norms and the disjoint original depth working coordinates. -/
theorem depthPresenceControl_coordinate (j : Fin 2) :
    ‖depthPresenceControl j‖ = 1 ∧
      depthPresenceControl j (depthCoordinate .easy 1) = if j.val = 0 then 1 else 0 := by
  unfold depthPresenceControl
  split_ifs <;> rw [depthAxis_norm, depthAxis_coordinate] <;> norm_num

/-- All required norm, nonnegativity, separated-feature and zero-self conditions hold simultaneously in the same real control.
Source: evaluated actual control coordinates; r is the proved fixed positive RMS lower scalar. -/
theorem depthPresenceControl_conditions :
    (∀ j, 1 ≤ ‖depthPresenceControl j‖ ∧ ‖depthPresenceControl j‖ ≤ 4096) ∧
    (∀ j, 0 ≤ depthPresenceControl j (depthCoordinate .easy 1)) ∧
    (∀ j, depthPresenceControl j (depthCoordinate .easy 1) = 0 ∨
      depthScaleLower ^ 2 ≤ depthPresenceControl j (depthCoordinate .easy 1)) ∧
    depthPresenceControl 1 (depthCoordinate .easy 1) = 0 ∧
    depthScaleLower ^ 2 ≤ depthPresenceControl 0 (depthCoordinate .easy 1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro j
    rw [(depthPresenceControl_coordinate j).1]
    norm_num
  · intro j
    rw [(depthPresenceControl_coordinate j).2]
    split_ifs <;> norm_num
  · intro j
    rw [(depthPresenceControl_coordinate j).2]
    by_cases hj : j.val = 0
    · exact Or.inr (by norm_num [hj, depthScaleLower])
    · exact Or.inl (ite_eq_right hj)
  · rw [(depthPresenceControl_coordinate 1).2]
    norm_num
  · rw [(depthPresenceControl_coordinate 0).2]
    norm_num [depthScaleLower]

/-- A genuine visible residual occurrence yields twice the shared finite FFN threshold in the actual original head signal.
Source: derived real prenorm floor, context-bounded true causal mean and original zero-current XSA. -/
theorem depthAttentionSignal_occurrence (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i j : Fin T)
    (hl : 1 ≤ ‖x j‖) (hu : ‖x j‖ ≤ 4096) (hn : ∀ r, 0 ≤ x r (source branch))
    (hself : x i (source branch) = 0) (hvisible : j.val ≤ i.val)
    (hf : depthScaleLower ^ 2 ≤ x j (source branch)) :
    2 * depthThreshold ≤ depthAttentionSignal mode source eps x branch i := by
  have ha := depthFeatureAmplitude_floor mode source eps heps hclip x branch j hl hu hf
  have hp : 0 ≤ depthScaleLower ^ 3 := by have hr := depthScaleLower_bounds.1; positivity
  rw [depthAttentionSignal_zero_self mode source eps x branch i hself, depthThreshold_bounds.2]
  exact depthPrefixMass_context hT (depthFeatureAmplitude mode source eps x branch) i j
    (depthScaleLower ^ 3) hp
    (fun r => depthFeatureAmplitude_nonneg mode source eps x branch r (hn r)) hvisible ha

example : (2 : ℕ) ≤ 128 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    1 ≤ ‖depthPresenceControl 0‖ ∧ ‖depthPresenceControl 0‖ ≤ 4096 ∧
    (∀ j, 0 ≤ depthPresenceControl j (depthCoordinate .easy 1)) ∧
    depthPresenceControl 1 (depthCoordinate .easy 1) = 0 ∧ (0 : Fin 2).val ≤ (1 : Fin 2).val ∧
    depthScaleLower ^ 2 ≤ depthPresenceControl 0 (depthCoordinate .easy 1) := by
  have h := depthPresenceControl_conditions
  exact ⟨by decide, by norm_num, by norm_num, (h.1 0).1, (h.1 0).2, h.2.1, h.2.2.2.1, by decide, h.2.2.2.2⟩

/-- The actual head signal is positive exactly at a visible genuine feature in the separated local representation domain.
Source: actual occurrence gap and causal absence; no presence condition defines the attention computation. -/
theorem depthAttentionSignal_positive_iff (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i : Fin T)
    (hbound : ∀ j, 1 ≤ ‖x j‖ ∧ ‖x j‖ ≤ 4096)
    (hfeature : ∀ j, x j (source branch) = 0 ∨ depthScaleLower ^ 2 ≤ x j (source branch))
    (hself : x i (source branch) = 0) :
    0 < depthAttentionSignal mode source eps x branch i ↔
      ∃ j : Fin T, j.val ≤ i.val ∧ depthScaleLower ^ 2 ≤ x j (source branch) := by
  have hn (j : Fin T) : 0 ≤ x j (source branch) := by
    rcases hfeature j with hz | hp
    · rw [hz]
    · exact (sq_nonneg depthScaleLower).trans hp
  constructor
  · intro hs
    by_contra habsent
    have hz : ∀ j, j.val ≤ i.val → x j (source branch) = 0 := by
      intro j hj
      rcases hfeature j with hz | hp
      · exact hz
      · exact False.elim (habsent ⟨j, hj, hp⟩)
    have hzero := depthAttentionSignal_absent mode source eps x branch i hz
    linarith
  · rintro ⟨j, hj, hp⟩
    have hgap := depthAttentionSignal_occurrence mode source eps heps hclip hT x branch i j
      (hbound j).1 (hbound j).2 hn hself hj hp
    have ha := depthThreshold_bounds.1
    linarith

example : (2 : ℕ) ≤ 128 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    (∀ j, 1 ≤ ‖depthPresenceControl j‖ ∧ ‖depthPresenceControl j‖ ≤ 4096) ∧
    (∀ j, depthPresenceControl j (depthCoordinate .easy 1) = 0 ∨
      depthScaleLower ^ 2 ≤ depthPresenceControl j (depthCoordinate .easy 1)) ∧
    depthPresenceControl 1 (depthCoordinate .easy 1) = 0 := by
  have h := depthPresenceControl_conditions
  exact ⟨by decide, by norm_num, by norm_num, h.1, h.2.2.1, h.2.2.2.1⟩

/-- The real original head provides exactly the zero-or-positive-gap input required by the unchanged three-hinge FFN.
Source: derived separated raw features, genuine visible occurrence and causal absence. -/
theorem depthAttentionSignal_separated (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 128)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i : Fin T)
    (hbound : ∀ j, 1 ≤ ‖x j‖ ∧ ‖x j‖ ≤ 4096)
    (hfeature : ∀ j, x j (source branch) = 0 ∨ depthScaleLower ^ 2 ≤ x j (source branch))
    (hself : x i (source branch) = 0) :
    depthAttentionSignal mode source eps x branch i = 0 ∨ 2 * depthThreshold ≤ depthAttentionSignal mode source eps x branch i := by
  classical
  by_cases h : ∃ j : Fin T, j.val ≤ i.val ∧ depthScaleLower ^ 2 ≤ x j (source branch)
  · obtain ⟨j, hj, hp⟩ := h
    have hn (r : Fin T) : 0 ≤ x r (source branch) := by
      rcases hfeature r with hz | hf
      · rw [hz]
      · exact (sq_nonneg depthScaleLower).trans hf
    exact Or.inr (depthAttentionSignal_occurrence mode source eps heps hclip hT x branch i j
      (hbound j).1 (hbound j).2 hn hself hj hp)
  · apply Or.inl
    apply depthAttentionSignal_absent mode source eps x branch i
    intro j hj
    rcases hfeature j with hz | hp
    · exact hz
    · exact False.elim (h ⟨j, hj, hp⟩)

example : (2 : ℕ) ≤ 128 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ j, 1 ≤ ‖depthPresenceControl j‖ ∧ ‖depthPresenceControl j‖ ≤ 4096) ∧
    (∀ j, depthPresenceControl j (depthCoordinate .easy 1) = 0 ∨
      depthScaleLower ^ 2 ≤ depthPresenceControl j (depthCoordinate .easy 1)) ∧
    depthPresenceControl 1 (depthCoordinate .easy 1) = 0 := by
  have h := depthPresenceControl_conditions
  exact ⟨by decide, by norm_num, by norm_num, h.1, h.2.2.1, h.2.2.2.1⟩

end Transformer.GPTMini.Semantics
