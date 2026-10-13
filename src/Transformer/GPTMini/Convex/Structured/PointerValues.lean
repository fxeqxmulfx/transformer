import Transformer.GPTMini.Convex.Structured.ChannelMarginals
import Transformer.GPTMini.Convex.Structured.PointerTraining

/-!
# Exact joint Gibbs values through the compact pointer

New proposed value head following Factorial/PointerTraining at 87ffa1b.
The actual complete route/matching/value distribution factors into its
contracted memory weight and two local channel distributions. Its true
expected value-code coordinate equals a sum over memory positions of
the small local value-channel mean. Inference therefore reads jointly
trained values without enumerating every complete latent configuration.

The computed expectation is from the same affine-energy Gibbs model
whose fully supervised loss is globally convex, not a separately chosen
surrogate readout. Matching, values and chronology are still freely
trained; desired routes/channels are not arguments of these forward means.
The fixed code only converts output channels to residual coordinates.
Source: exact finite normalization/distributivity, motivated by §7's
log-sum-exp construction in arXiv:2305.05465v6. Raw task capability and
the genuine residual/tied-embedding integer adapter remain unproved here.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {G C H D J : Type*} [Fintype G] [Fintype C] [Fintype H] [Fintype D] [Fintype J]

/-- The actual full joint pointer probability before compact channel contraction.
Source: the complete affine route/matching/value energy and its exact small-sum partition. -/
def pointerProbability (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ)
    (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) : ℝ :=
  Real.exp (pointerEnergy query key value bias z) / pointerPartition query key value bias

/-- Every actual complete pointer configuration retains positive mass at finite matching/value parameters.
Source: the real exponential energy and proved compact positive route/value normalizer. -/
theorem pointerProbability_pos [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ)
    (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    0 < pointerProbability query key value bias z :=
  div_pos (Real.exp_pos _) (pointerPartition_pos query key value bias)

/-- The real joint distribution factors exactly into its contracted memory weight and conditional channel probabilities.
Source: actual exponential additivity and cancellation of the proved positive matching/value partitions. -/
theorem pointerProbability_factor [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ)
    (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    pointerProbability query key value bias z = pointerWeight query key value bias z.1 *
      channelProbability (fun g c => query g c + key z.1 g c) z.2.1 * channelProbability (value z.1) z.2.2 := by
  have hm := (channelPartition_pos (fun g c => query g c + key z.1 g c)).ne'
  have hv := (channelPartition_pos (value z.1)).ne'
  have hp := (pointerPartition_pos query key value bias).ne'
  simp only [pointerProbability, pointerEnergy, Real.exp_add, pointerWeight, channelProbability]
  field_simp

/-- The actual jointly trained pointer probabilities sum to one across every implicit complete configuration.
Source: the real factorization, channel normalization and contracted physical memory weights. -/
theorem pointerProbability_sum [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ) :
    ∑ z, pointerProbability query key value bias z = 1 := by
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type, pointerProbability_factor]
  simp only [← Finset.mul_sum, channelProbability_sum, mul_one]
  exact pointerWeight_sum query key value bias

/-- The actual compact expected value-code coordinate, computed from trained log potentials at each memory token.
Source: the proposed exact memory/group readout, with no observed route or value channel supplied at inference. -/
def pointerMean (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) (h : H) (code : D → ℝ) : ℝ :=
  ∑ j, pointerWeight query key value bias j * channelMean (value j) h code

/-- Addition of actual output code coordinates commutes with the full compact learned-value readout.
Source: the true memory-weighted local expectations, before choosing any task target. -/
theorem pointerMean_add (query : G → C → ℝ) (key : J → G → C → ℝ)
    (value : J → H → D → ℝ) (bias : J → ℝ) (h : H) (left right : D → ℝ) :
    pointerMean query key value bias h (fun d => left d + right d) =
      pointerMean query key value bias h left + pointerMean query key value bias h right := by
  unfold pointerMean
  simp_rw [channelMean_add, mul_add, Finset.sum_add_distrib]

/-- The compact learned-value head equals the actual joint Gibbs expectation, not just its normalizer.
Source: the derived full probability factorization and exact local value marginal. -/
theorem pointerProbability_mean [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ)
    (h : H) (code : D → ℝ) :
    (∑ z, pointerProbability query key value bias z * code (z.2.2 h)) =
      pointerMean query key value bias h code := by
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type, pointerProbability_factor]
  unfold pointerMean
  apply Finset.sum_congr rfl
  intro j _
  simp_rw [mul_assoc, ← Finset.mul_sum, channelProbability_mean]
  rw [← Finset.sum_mul, channelProbability_sum, one_mul]

/-- The computed pointer expectation preserves a constant value code exactly.
Source: actual local channel means and derived global memory-weight normalization. -/
theorem pointerMean_const [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ)
    (h : H) (a : ℝ) : pointerMean query key value bias h (fun _ => a) = a := by
  unfold pointerMean
  simp_rw [channelMean_const]
  rw [← Finset.sum_mul, pointerWeight_sum, one_mul]

/-- Every compact expected value-code coordinate obeys the real code bounds at arbitrary trained parameters.
Source: positive actual memory weights and exact bounded local expectations, before any task correctness premise. -/
theorem pointerMean_bounds [Nonempty J] [Nonempty C] [Nonempty D]
    (query : G → C → ℝ) (key : J → G → C → ℝ) (value : J → H → D → ℝ) (bias : J → ℝ)
    (h : H) (code : D → ℝ) (lower upper : ℝ) (hcode : ∀ d, lower ≤ code d ∧ code d ≤ upper) :
    lower ≤ pointerMean query key value bias h code ∧ pointerMean query key value bias h code ≤ upper := by
  have hlo := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (channelMean_bounds (value j) h code lower upper hcode).1
      (pointerWeight_pos query key value bias j).le)
  have hup := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (channelMean_bounds (value j) h code lower upper hcode).2
      (pointerWeight_pos query key value bias j).le)
  rw [← Finset.sum_mul, pointerWeight_sum, one_mul] at hlo hup
  exact ⟨hlo, hup⟩

example : ∀ d : Fin 2, (-1 : ℝ) ≤ (if d = 0 then (1 : ℝ) else -1) ∧
    (if d = 0 then (1 : ℝ) else -1) ≤ 1 := by
  intro d
  fin_cases d <;> norm_num

/-- The compact inference distribution is the identical joint Gibbs model used by the convex shared-table training loss.
Source: actual token-local Q/K/value lookups, pointerLinear and the exact joint partition contraction. -/
theorem pointerProbability_shared {V : ℕ} (query : Fin V) (keys values : J → Fin V) (positions : J → ℝ)
    (θ : PointerParameters V G C H D)
    (z : PointerConfiguration (G := G) (C := C) (H := H) (D := D) (J := J)) :
    probability (pointerLinear query keys values positions) (fun _ => 0) θ z =
      pointerProbability (pointerQuery θ query) (fun j => pointerKey θ (keys j))
        (fun j => pointerValue θ (values j)) (fun j => θ.2 * positions j) z := by
  unfold probability partition
  simp only [energy, pointerLinear, LinearMap.coe_mk, AddHom.coe_mk, add_zero]
  rw [← pointerPartition_eq]
  rfl

/-- A finite untrained pointer reads the actual mixture of local value channels, without a route oracle.
Source: exact compact inference at all-zero free log potentials. -/
example : pointerMean (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ))
    (fun _ : Fin 2 => fun _ : Fin 1 => fun _ : Fin 2 => 0)
    (fun _ : Fin 2 => fun _ : Fin 1 => fun _ : Fin 2 => 0) (fun _ : Fin 2 => 0) 0
    (fun d => if d = 0 then 1 else 0) = 1 / 2 := by
  norm_num [pointerMean, pointerWeight, pointerPartition, channelPartition, channelMean,
    channelWeight, Fin.sum_univ_two, Fin.prod_univ_one]

end
end Transformer.GPTMini.Convex.Structured
