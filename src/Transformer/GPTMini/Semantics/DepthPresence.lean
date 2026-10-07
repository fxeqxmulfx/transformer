import Transformer.GPTMini.Semantics.DepthStep
import Transformer.GPTMini.Semantics.RecallMarkerWeights

/-!
# Actual uniform softmax/XSA presence with nonconstant amplitudes

Source: CausalMHA.forward at f11b6e2 and the bounded context of Basis
depth at cbafbe9. Zero Q/K gives the genuine diagonal-inclusive causal
mean, even when the original finite array contains future positions.
Values may have different amplitudes at every position, as required
by the actual position-dependent RMS and homogeneous FFN outputs.

When the current value is zero, original XSA retains this exact mean.
Nonnegative values give a nonnegative probe. A genuine prior occurrence
with amplitude at least L contributes at least L/128 throughout the
depth context; complete absence gives exactly zero. Thus a propagated
amplitude floor yields the separated input required by the real FFN
step without assuming unit-valued hidden features.

The amplitude array is a local operator input, not a prefix oracle.
The raw embedding, simultaneous matrices and layer recurrence must
still derive these representation conditions for the complete model.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- Explicit arithmetic causal mean of real coordinate amplitudes, retaining the actual i+1 diagonal-inclusive denominator.
Source: evaluated zero-score original softmax; no Boolean presence decision occurs in this definition. -/
noncomputable def depthPrefixMass {T : ℕ} (amplitude : Fin T → ℝ) (i : Fin T) : ℝ :=
  (1 / ((i.val + 1 : ℕ) : ℝ)) * ∑ j, if j.val ≤ i.val then amplitude j else 0

/-- The true RoPE/QKNorm/softmax/XSA head equals the explicit causal mean of arbitrary real amplitudes at zero self-value.
Source: actual zero Q/K causal weights and the genuine original XSA formula, with every future position still masked. -/
theorem depthUniformHead_mass (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim)
    (i : Fin T) (hself : amplitude i = 0) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0) (fun j => amplitude j • direction) positions i =
      depthPrefixMass amplitude i • direction := by
  unfold attentionHead xsaProjection
  simp only [rope_zero, ite_true, hself, zero_smul, normL2, smul_zero, inner_zero_right, sub_zero]
  unfold attnOutput
  have hw (j : Fin T) : causalAttnWeights cfg alpha eps (fun _ => 0) (fun _ => 0) i j =
      if j.val ≤ i.val then 1 / ((i.val + 1 : ℕ) : ℝ) else 0 := by
    by_cases hji : j.val ≤ i.val
    · rw [ite_eq_left hji, recall_uniform_visible_weight cfg alpha eps i j hji]
    · rw [ite_eq_right hji, recall_uniform_future_weight cfg alpha eps i j (by omega)]
  simp_rw [hw, smul_smul]
  have hc (j : Fin T) : (if j.val ≤ i.val then 1 / ((i.val + 1 : ℕ) : ℝ) else 0) * amplitude j =
      (1 / ((i.val + 1 : ℕ) : ℝ)) * (if j.val ≤ i.val then amplitude j else 0) := by
    by_cases hji : j.val ≤ i.val <;> simp only [hji, ↓reduceIte, zero_mul, mul_zero]
  simp_rw [hc]
  rw [← Finset.sum_smul, ← Finset.mul_sum]
  rfl

example : (fun j : Fin 2 => if j.val = 0 then (3 : ℝ) else 0) 1 = 0 := by norm_num

/-- A genuine unit-coordinate probe recovers the same actual amplitude mean through the complete head.
Source: the derived full head formula and the true real self-inner-product of the chosen coordinate. -/
theorem depthUniformHead_probe (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim)
    (hunit : ‖direction‖ = 1) (i : Fin T) (hself : amplitude i = 0) :
    inner (𝕜 := ℝ) (attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
      (fun j => amplitude j • direction) positions i) direction = depthPrefixMass amplitude i := by
  rw [depthUniformHead_mass cfg alpha eps positions amplitude direction i hself,
    real_inner_smul_left, real_inner_self_eq_norm_sq, hunit]
  ring

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (fun j : Fin 2 => if j.val = 0 then (3 : ℝ) else 0) 1 = 0 := by
  exact ⟨by simp [PiLp.norm_single], by norm_num⟩

/-- Nonnegative actual amplitudes give a nonnegative causal signal at every prefix position.
Source: the original positive uniform softmax mass, including exactly the visible finite coordinates. -/
theorem depthPrefixMass_nonneg {T : ℕ} (amplitude : Fin T → ℝ) (i : Fin T)
    (hnonneg : ∀ j, 0 ≤ amplitude j) : 0 ≤ depthPrefixMass amplitude i := by
  unfold depthPrefixMass
  apply mul_nonneg (by positivity)
  apply Finset.sum_nonneg
  intro j hj
  split_ifs
  · exact hnonneg j
  · exact le_refl 0

example : ∀ j : Fin 2, 0 ≤ (if j.val = 0 then (3 : ℝ) else 0) := by intro j; split_ifs <;> norm_num

/-- A uniform actual amplitude cap bounds the whole genuine causal mean without multiplying it by the sequence length.
Source: the derived i+1 visible count and the exact original softmax denominator. -/
theorem depthPrefixMass_upper {T : ℕ} (amplitude : Fin T → ℝ) (i : Fin T) (cap : ℝ)
    (hcap : ∀ j, amplitude j ≤ cap) : depthPrefixMass amplitude i ≤ cap := by
  rw [depthPrefixMass, one_div_mul_eq_div]
  apply (div_le_iff₀ (by positivity : 0 < ((i.val + 1 : ℕ) : ℝ))).mpr
  calc
    (∑ j : Fin T, if j.val ≤ i.val then amplitude j else 0) ≤ ∑ j : Fin T, if j.val ≤ i.val then cap else 0 := by
      apply Finset.sum_le_sum
      intro j hj
      split_ifs
      · exact hcap j
      · exact le_refl 0
    _ = cap * ((i.val + 1 : ℕ) : ℝ) := by
      have hs : (∑ j : Fin T, if j.val ≤ i.val then cap else 0) =
          (∑ j : Fin T, if j.val ≤ i.val then (1 : ℝ) else 0) * cap := by
        rw [Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro j hj
        split_ifs <;> ring
      rw [hs, recall_causal_visible_count]
      ring

example : ∀ j : Fin 2, (if j.val = 0 then (3 : ℝ) else 0) ≤ 3 := by intro j; split_ifs <;> norm_num

/-- Any visible genuine occurrence contributes its actual amplitude floor divided by the internal prefix length.
Source: a single nonnegative softmax summand, retaining arbitrary amplitudes at every other position. -/
theorem depthPrefixMass_occurrence {T : ℕ} (amplitude : Fin T → ℝ) (i j : Fin T) (floor : ℝ)
    (hnonneg : ∀ r, 0 ≤ amplitude r) (hvisible : j.val ≤ i.val) (hfloor : floor ≤ amplitude j) :
    floor / ((i.val + 1 : ℕ) : ℝ) ≤ depthPrefixMass amplitude i := by
  have hs : amplitude j ≤ ∑ r : Fin T, if r.val ≤ i.val then amplitude r else 0 := by
    have h := Finset.single_le_sum (s := Finset.univ)
      (f := fun r : Fin T => if r.val ≤ i.val then amplitude r else 0)
      (fun r _ => by split_ifs; exact hnonneg r; exact le_refl 0) (Finset.mem_univ j)
    simpa only [ite_eq_left hvisible] using h
  rw [depthPrefixMass, one_div_mul_eq_div]
  exact div_le_div_of_nonneg_right (hfloor.trans hs) (by positivity)

example : (∀ r : Fin 2, 0 ≤ (if r.val = 0 then (3 : ℝ) else 0)) ∧
    (0 : Fin 2).val ≤ (1 : Fin 2).val ∧ (1 : ℝ) ≤ 3 := by
  exact ⟨by intro r; split_ifs <;> norm_num, by decide, by norm_num⟩

/-- Complete absence in the actual visible prefix gives exactly zero, regardless of future amplitudes.
Source: original causal masking and the actual finite amplitude sum. -/
theorem depthPrefixMass_absent {T : ℕ} (amplitude : Fin T → ℝ) (i : Fin T)
    (habsent : ∀ j, j.val ≤ i.val → amplitude j = 0) : depthPrefixMass amplitude i = 0 := by
  have hs : (∑ j : Fin T, if j.val ≤ i.val then amplitude j else 0) = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    by_cases hji : j.val ≤ i.val
    · rw [ite_eq_left hji, habsent j hji]
    · exact ite_eq_right hji
  rw [depthPrefixMass, hs, mul_zero]

example : ∀ j : Fin 2, j.val ≤ (0 : Fin 2).val → (if j.val = 1 then (3 : ℝ) else 0) = 0 := by
  intro j hj
  have hne : j.val ≠ 1 := by omega
  simp only [ite_eq_right hne]

/-- A genuine positive amplitude floor gives a uniform presence gap over the whole original depth context, including OOD length 128.
Source: the real i+1 causal denominator and the actual bounded context, not a supplied constant model weight. -/
theorem depthPrefixMass_context {T : ℕ} (hT : T ≤ 128) (amplitude : Fin T → ℝ) (i j : Fin T) (floor : ℝ)
    (hfloor : 0 ≤ floor) (hnonneg : ∀ r, 0 ≤ amplitude r) (hvisible : j.val ≤ i.val) (hoccurs : floor ≤ amplitude j) :
    floor / 128 ≤ depthPrefixMass amplitude i := by
  have hi : i.val + 1 ≤ 128 := by have h := i.isLt; omega
  have hc : (((i.val + 1 : ℕ) : ℝ)) ≤ 128 := by exact_mod_cast hi
  exact (div_le_div_of_nonneg_left hfloor (by positivity) hc).trans
    (depthPrefixMass_occurrence amplitude i j floor hnonneg hvisible hoccurs)

example : (2 : ℕ) ≤ 128 ∧ (0 : ℝ) ≤ 1 ∧ (∀ r : Fin 2, 0 ≤ (if r.val = 0 then (3 : ℝ) else 0)) ∧
    (0 : Fin 2).val ≤ (1 : Fin 2).val ∧ (1 : ℝ) ≤ 3 := by
  exact ⟨by decide, by norm_num, by intro r; split_ifs <;> norm_num, by decide, by norm_num⟩

/-- Zero-or-bounded real amplitudes derive the separated actual presence signal needed by the three-hinge FFN.
Source: the genuine causal mean's absence and occurrence bounds, without assuming identical amplitudes or a ready Boolean output. -/
theorem depthPrefixMass_separated {T : ℕ} (hT : T ≤ 128) (amplitude : Fin T → ℝ) (i : Fin T) (floor : ℝ)
    (hfloor : 0 ≤ floor) (hseparated : ∀ j, amplitude j = 0 ∨ floor ≤ amplitude j) :
    depthPrefixMass amplitude i = 0 ∨ floor / 128 ≤ depthPrefixMass amplitude i := by
  classical
  have hnonneg : ∀ j, 0 ≤ amplitude j := by
    intro j
    rcases hseparated j with hz | hp
    · rw [hz]
    · exact hfloor.trans hp
  by_cases hex : ∃ j : Fin T, j.val ≤ i.val ∧ amplitude j ≠ 0
  · obtain ⟨j, hv, hn⟩ := hex
    have hp : floor ≤ amplitude j := by
      rcases hseparated j with hz | hp
      · exact False.elim (hn hz)
      · exact hp
    exact Or.inr (depthPrefixMass_context hT amplitude i j floor hfloor hnonneg hv hp)
  · apply Or.inl
    apply depthPrefixMass_absent amplitude i
    intro j hv
    by_contra hn
    exact hex ⟨j, hv, hn⟩

example : (2 : ℕ) ≤ 128 ∧ (0 : ℝ) ≤ 1 ∧
    ∀ j : Fin 2, (if j.val = 0 then (3 : ℝ) else 0) = 0 ∨ 1 ≤ (if j.val = 0 then (3 : ℝ) else 0) := by
  refine ⟨by decide, by norm_num, ?_⟩
  intro j
  split_ifs <;> norm_num

end Transformer.GPTMini.Semantics
