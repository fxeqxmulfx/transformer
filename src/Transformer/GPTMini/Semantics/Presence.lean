import Transformer.GPTMini.Semantics.Basic

/-!
# Softmax detects the presence of an encoded prefix feature

Source: CausalMHA.forward at f11b6e2. Every visible finite-softmax weight
is positive. Consequently a faithful zero/unit value feature has a
positive probe exactly when some visible position carries the feature,
for arbitrary learned queries, keys and temperature. This is the
prefix-existence operation used by ordered subsequence detectors.

The semantic value encoding is a local hypothesis about the hidden
representation, not an assumption that final task answers are correct.
XSA preserves the operation when the query self-value is zero. The
quantitative bound gives a positive signal floor at a finite context
and temperature. Future occurrences are
masked before the test; they cannot create a past-prefix signal. The
sign criterion applies at every finite learned temperature.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators Classical

/-- A zero/unit value feature turns the actual head probe into its weighted indicator sum.
Source: attnOutput at f11b6e2, before the self-value projection. -/
theorem output_indicator_signal (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (P : Fin T → Prop) [DecidablePred P] (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hvalues : ∀ j, v j = if P j then direction else 0) :
    inner (𝕜 := ℝ) (attnOutput cfg alpha eps q k v i) direction =
      ∑ j : Fin T, if P j then causalAttnWeights cfg alpha eps q k i j else 0 := by
  rw [attnOutput, sum_inner]
  apply Finset.sum_congr rfl
  intro j hj
  rw [real_inner_smul_left, hvalues j]
  split_ifs
  · rw [real_inner_self_eq_norm_sq, hunit]
    ring
  · simp

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    ∀ j : Fin 2, (if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0 := by
  exact ⟨by simp [PiLp.norm_single], fun _ => rfl⟩

/-- A positive actual attention signal is equivalent to a visible semantic feature occurrence.
Source: positivity of every unmasked weight in the original finite softmax at f11b6e2. -/
theorem output_presence (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (P : Fin T → Prop) [DecidablePred P] (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hvalues : ∀ j, v j = if P j then direction else 0) :
    0 < inner (𝕜 := ℝ) (attnOutput cfg alpha eps q k v i) direction ↔
      ∃ j, j.val ≤ i.val ∧ P j := by
  rw [output_indicator_signal cfg alpha eps q k v i P direction hunit hvalues]
  rw [Finset.sum_pos_iff_of_nonneg (fun j _ => by
    split_ifs
    · exact causalAttnWeights_nonneg cfg alpha eps q k i j
    · exact le_rfl)]
  constructor
  · rintro ⟨j, hj, hpositive⟩
    by_cases hp : P j
    · rw [ite_eq_left hp] at hpositive
      refine ⟨j, ?_, hp⟩
      by_contra hji
      rw [causalAttnWeights_zero_above cfg alpha eps q k i j (not_le.mp hji)] at hpositive
      exact (lt_irrefl 0) hpositive
    · rw [ite_eq_right hp] at hpositive
      exact False.elim ((lt_irrefl 0) hpositive)
  · rintro ⟨j, hji, hp⟩
    exact ⟨j, Finset.mem_univ _, by rw [ite_eq_left hp]; exact weight_pos cfg alpha eps q k i j hji⟩

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    ∀ j : Fin 2, (if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0 := by
  exact ⟨by simp [PiLp.norm_single], fun _ => rfl⟩

/-- The full RoPE/QKNorm/softmax/XSA head retains the prefix-existence feature at a zero self-value.
Source: the actual attentionHead at f11b6e2; Q/K need not be uniform or preselected. -/
theorem head_presence (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T)
    (P : Fin T → Prop) [DecidablePred P] (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hvalues : ∀ j, v j = if P j then direction else 0) (hself : v i = 0) :
    0 < inner (𝕜 := ℝ) (attentionHead cfg alpha eps q k v positions i) direction ↔
      ∃ j, j.val ≤ i.val ∧ P j := by
  have he : attentionHead cfg alpha eps q k v positions i = attnOutput cfg alpha eps
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
      (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) v i := by
    simp [attentionHead, xsaProjection, hself, normL2]
  rw [he]
  exact output_presence cfg alpha eps _ _ v i P direction hunit hvalues

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 2, (if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) ∧
    (if (1 : Fin 2).val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) = 0 := by
  exact ⟨by simp [PiLp.norm_single], fun _ => rfl, by simp⟩

/-- A real feature occurrence has an explicit positive signal floor under QKNorm.
Source: causalAttnWeights_bounds at f11b6e2, using the original learned head temperature. -/
theorem presence_lower_bound (cfg : Config) {T : ℕ} (alpha eps : ℝ) (heps : 0 ≤ eps)
    (q k v : Fin T → EucSpace cfg.head_dim) (i j : Fin T)
    (P : Fin T → Prop) [DecidablePred P] (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hvalues : ∀ r, v r = if P r then direction else 0)
    (hji : j.val ≤ i.val) (hp : P j) :
    (((i.val + 1 : ℕ) : ℝ))⁻¹ * Real.exp (-(2 * Real.exp alpha)) ≤
      inner (𝕜 := ℝ) (attnOutput cfg alpha eps q k v i) direction := by
  rw [output_indicator_signal cfg alpha eps q k v i P direction hunit hvalues]
  have hsingle : causalAttnWeights cfg alpha eps q k i j ≤
      ∑ r : Fin T, if P r then causalAttnWeights cfg alpha eps q k i r else 0 := by
    have h := Finset.single_le_sum (s := Finset.univ)
      (f := fun r : Fin T => if P r then causalAttnWeights cfg alpha eps q k i r else 0)
      (fun r _ => by split_ifs; exact causalAttnWeights_nonneg cfg alpha eps q k i r; exact le_rfl)
      (Finset.mem_univ j)
    simpa only [ite_eq_left hp] using h
  simpa only [Nat.cast_add, Nat.cast_one] using
    (causalAttnWeights_bounds cfg alpha eps heps q k i j hji).1.trans hsingle

example : (0 : ℝ) ≤ 1 ∧ ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 2, (if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if j.val = 0 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) ∧
    (0 : Fin 2).val ≤ (1 : Fin 2).val ∧ (0 : Fin 2).val = 0 := by
  exact ⟨by norm_num, by simp [PiLp.norm_single], fun _ => rfl, by decide, rfl⟩

/-- If the encoded feature occurs only in the future, its current attention signal is exactly zero.
Source: the original causal mask at f11b6e2, applied to the faithful prefix feature. -/
theorem absence_signal_zero (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (P : Fin T → Prop) [DecidablePred P] (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1)
    (hvalues : ∀ j, v j = if P j then direction else 0)
    (habsent : ∀ j, j.val ≤ i.val → ¬P j) :
    inner (𝕜 := ℝ) (attnOutput cfg alpha eps q k v i) direction = 0 := by
  rw [output_indicator_signal cfg alpha eps q k v i P direction hunit hvalues]
  apply Finset.sum_eq_zero
  intro j hj
  by_cases hji : j.val ≤ i.val
  · exact ite_eq_right (habsent j hji)
  · rw [causalAttnWeights_zero_above cfg alpha eps q k i j (not_le.mp hji)]
    split_ifs <;> rfl

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 2, (if j.val = 1 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) =
      if j.val = 1 then EuclideanSpace.single (0 : Fin 16) (1 : ℝ) else 0) ∧
    ∀ j : Fin 2, j.val ≤ (0 : Fin 2).val → ¬j.val = 1 := by
  refine ⟨by simp [PiLp.norm_single], fun _ => rfl, ?_⟩
  intro j hj
  change j.val ≤ 0 at hj
  omega

end Transformer.GPTMini.Semantics
