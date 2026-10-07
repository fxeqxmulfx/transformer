import Transformer.GPTMini.Semantics.DepthPresence

/-!
# Genuine uniform depth heads when the current value is nonzero

Source: original CausalMHA/XSA and clipped normL2 at f11b6e2. A head
whose values all lie along one unit coordinate has the actual causal
mean times 1-(a_i/max(|a_i|,epsilon))^2 after XSA. This factor is in
[0,1] for nonnegative epsilon, including clipping and the zero value.

Thus nonnegative scalar features remain nonnegative and their head
output norm stays below their genuine amplitude cap. No zero-current
premise is needed for these bounds. The earlier exact presence floor
still applies when the current feature is zero, as derived from the
opposite-letter depth recurrence.

At final readout an occurrence at the current position is retained in
its own residual coordinate; earlier occurrences supply an actual
uniform head signal when that coordinate is zero. Raw state induction
must derive those alternatives and their amplitude bounds separately.
This is an original-operator capacity result, not a training assertion.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The actual scalar component of a clipped one-coordinate self-value.
Source: normL2's genuine max(norm,epsilon) denominator. -/
noncomputable def depthSelfFraction (eps amplitude : ℝ) : ℝ :=
  amplitude / max |amplitude| eps

/-- Original normL2 retains the same unit direction and this exact scalar at every amplitude, including zero.
Source: the real L2 norm of a scalar multiple of the actual unit direction. -/
theorem depthNormL2_collinear {D : ℕ} (eps amplitude : ℝ) (direction : EucSpace D)
    (hunit : ‖direction‖ = 1) :
    normL2 eps (amplitude • direction) = depthSelfFraction eps amplitude • direction := by
  rw [normL2, norm_smul, Real.norm_eq_abs, hunit, mul_one, smul_smul]
  unfold depthSelfFraction
  congr 1
  ring

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 := by simp [PiLp.norm_single]

/-- The genuine clipped self component has squared magnitude at most one.
Source: the proved actual normL2 bound, evaluated on an ordinary one-coordinate value. -/
theorem depthSelfFraction_sq_le (eps amplitude : ℝ) (heps : 0 ≤ eps) :
    (depthSelfFraction eps amplitude) ^ 2 ≤ 1 := by
  let direction : EucSpace 1 := EuclideanSpace.single 0 1
  have hunit : ‖direction‖ = 1 := by simp [direction, PiLp.norm_single]
  have hn := normL2_norm_le eps heps (amplitude • direction)
  rw [depthNormL2_collinear eps amplitude direction hunit, norm_smul, Real.norm_eq_abs, hunit, mul_one] at hn
  have hb := abs_le.mp hn
  nlinarith

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- Original XSA's attenuation along this value coordinate is always between zero and one.
Source: the actual clipped self-value bound, with the full subtraction retained. -/
theorem depthSelfFraction_attenuation (eps amplitude : ℝ) (heps : 0 ≤ eps) :
    0 ≤ 1 - (depthSelfFraction eps amplitude) ^ 2 ∧
      1 - (depthSelfFraction eps amplitude) ^ 2 ≤ 1 := by
  have h := depthSelfFraction_sq_le eps amplitude heps
  constructor <;> nlinarith [sq_nonneg (depthSelfFraction eps amplitude)]

example : (0 : ℝ) ≤ 0 := by norm_num

/-- The genuine pre-XSA causal attention output is exactly the real prefix mean along the same value direction.
Source: the actual diagonal-inclusive zero-score softmax, with all future coordinates still present and masked. -/
theorem depthUniformAttnOutput_mass (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim) (i : Fin T) :
    attnOutput cfg alpha eps (fun _ => 0) (fun _ => 0) (fun j => amplitude j • direction) i =
      depthPrefixMass amplitude i • direction := by
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

/-- The entire original RoPE/QKNorm/softmax/XSA head has this exact collinear output, without a zero-current premise.
Source: genuine prefix averaging followed by the actual clipped self-value projection. -/
theorem depthUniformHead_collinear (cfg : Config) (alpha eps : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim)
    (hunit : ‖direction‖ = 1) (i : Fin T) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0) (fun j => amplitude j • direction) positions i =
      (depthPrefixMass amplitude i * (1 - (depthSelfFraction eps (amplitude i)) ^ 2)) • direction := by
  unfold attentionHead xsaProjection
  simp only [rope_zero, ite_true]
  rw [depthUniformAttnOutput_mass cfg alpha eps amplitude direction i,
    depthNormL2_collinear eps (amplitude i) direction hunit,
    real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq, hunit, smul_smul, ← sub_smul]
  congr 1
  ring

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 := by simp [PiLp.norm_single]

/-- Nonnegative feature values give an actual attenuated coefficient between zero and the same uniform feature cap.
Source: genuine causal mean bounds and original XSA attenuation in [0,1]. -/
theorem depthUniformCoefficient_bounds {T : ℕ} (eps cap : ℝ) (heps : 0 ≤ eps)
    (amplitude : Fin T → ℝ) (hnonneg : ∀ j, 0 ≤ amplitude j) (hcap : ∀ j, amplitude j ≤ cap) (i : Fin T) :
    0 ≤ depthPrefixMass amplitude i * (1 - (depthSelfFraction eps (amplitude i)) ^ 2) ∧
      depthPrefixMass amplitude i * (1 - (depthSelfFraction eps (amplitude i)) ^ 2) ≤ cap := by
  have hm := depthPrefixMass_nonneg amplitude i hnonneg
  have hu := depthPrefixMass_upper amplitude i cap hcap
  have hf := depthSelfFraction_attenuation eps (amplitude i) heps
  constructor
  · exact mul_nonneg hm hf.1
  · calc depthPrefixMass amplitude i * (1 - (depthSelfFraction eps (amplitude i)) ^ 2)
        ≤ depthPrefixMass amplitude i * 1 := mul_le_mul_of_nonneg_left hf.2 hm
      _ ≤ cap := by simpa only [mul_one] using hu

example : (0 : ℝ) ≤ 1 / 100000 ∧ (∀ j : Fin 2, 0 ≤ (if j.val = 0 then (3 : ℝ) else 0)) ∧
    (∀ j : Fin 2, (if j.val = 0 then (3 : ℝ) else 0) ≤ 3) := by
  exact ⟨by norm_num, by intro j; split_ifs <;> norm_num, by intro j; split_ifs <;> norm_num⟩

/-- A genuine unit-coordinate head has norm at most the feature cap even when its actual current value is nonzero.
Source: the exact original head formula, rather than the looser generic twice-value bound. -/
theorem depthUniformHead_norm (cfg : Config) (alpha eps cap : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim)
    (hunit : ‖direction‖ = 1) (hnonneg : ∀ j, 0 ≤ amplitude j) (hcap : ∀ j, amplitude j ≤ cap) (i : Fin T) :
    ‖attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0) (fun j => amplitude j • direction) positions i‖ ≤ cap := by
  have hb := depthUniformCoefficient_bounds eps cap heps amplitude hnonneg hcap i
  rw [depthUniformHead_collinear cfg alpha eps positions amplitude direction hunit i,
    norm_smul, Real.norm_eq_abs, abs_of_nonneg hb.1, hunit, mul_one]
  exact hb.2

example : (0 : ℝ) ≤ 1 / 100000 ∧ ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 2, 0 ≤ (if j.val = 0 then (3 : ℝ) else 0)) ∧
    (∀ j : Fin 2, (if j.val = 0 then (3 : ℝ) else 0) ≤ 3) := by
  exact ⟨by norm_num, by simp [PiLp.norm_single], by intro j; split_ifs <;> norm_num, by intro j; split_ifs <;> norm_num⟩

/-- Reading the actual unit feature after original XSA remains nonnegative, including a positive current feature.
Source: genuine full-head coefficient and its original clipped attenuation; this protects the final residual's own occurrence. -/
theorem depthUniformHead_probe_nonneg (cfg : Config) (alpha eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (amplitude : Fin T → ℝ) (direction : EucSpace cfg.head_dim)
    (hunit : ‖direction‖ = 1) (hnonneg : ∀ j, 0 ≤ amplitude j) (i : Fin T) :
    0 ≤ inner (𝕜 := ℝ) (attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
      (fun j => amplitude j • direction) positions i) direction := by
  rw [depthUniformHead_collinear cfg alpha eps positions amplitude direction hunit i,
    real_inner_smul_left, real_inner_self_eq_norm_sq, hunit]
  norm_num
  exact mul_nonneg (depthPrefixMass_nonneg amplitude i hnonneg) (depthSelfFraction_attenuation eps (amplitude i) heps).1

example : (0 : ℝ) ≤ 1 / 100000 ∧ ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 ∧
    (∀ j : Fin 2, 0 ≤ (if j.val = 0 then (3 : ℝ) else 0)) := by
  exact ⟨by norm_num, by simp [PiLp.norm_single], by intro j; split_ifs <;> norm_num⟩

end Transformer.GPTMini.Semantics
