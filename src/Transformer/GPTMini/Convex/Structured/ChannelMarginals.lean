import Transformer.GPTMini.Convex.Structured.Factorial

/-!
# Exact compact readout of implicit value channels

New continuation of the structured pointer contraction at 87ffa1b.
Inference computes one local categorical mean per value group, rather
than enumerating the Cartesian product of value assignments. The actual
complete Gibbs distribution's expected coordinate equals that small
local mean. This is proved for arbitrary real channel readout functions,
including the fixed signed-axis output label decoder under investigation.

Matching and value potentials remain freely trained. The output code
is only a channel-to-output readout, not a fixed input interaction bank.
Observed channels occur only in training targets, not in these actual
inference means. Compact channel readout is therefore exact even at
arbitrary untrained parameters; task correctness still requires a
separate semantic capacity and real residual/tied-output construction.
Source algebra: finite distributivity and normalized Gibbs expectations,
following arXiv:2305.05465v6, §7's finite exponential normalizer.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {G C : Type*} [Fintype C]

/-- The actually computed local categorical probability in one value group.
Source: free value log potentials and their small positive exponential sum. -/
def channelWeight (potential : G → C → ℝ) (g : G) (c : C) : ℝ :=
  Real.exp (potential g c) / ∑ d, Real.exp (potential g d)

/-- The compact local readout of an arbitrary real-valued channel code.
Source: the proposed value output coordinate, computed with only the local categorical sum. -/
def channelMean (potential : G → C → ℝ) (g : G) (code : C → ℝ) : ℝ :=
  ∑ c, channelWeight potential g c * code c

/-- Each actually evaluated channel probability is positive at finite raw value parameters.
Source: positive local exponential weights and a nonempty local choice domain. -/
theorem channelWeight_pos [Nonempty C] (potential : G → C → ℝ) (g : G) (c : C) :
    0 < channelWeight potential g c :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)

/-- The small local channel weights sum to one independently of every other group's parameters.
Source: the actual local normalizer in the proposed value readout. -/
theorem channelWeight_sum [Nonempty C] (potential : G → C → ℝ) (g : G) :
    ∑ c, channelWeight potential g c = 1 := by
  unfold channelWeight
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

/-- The actual local code mean is its weighted exponential sum divided by its small partition.
Source: the explicitly computed local categorical probabilities. -/
theorem channelMean_eq_ratio (potential : G → C → ℝ) (g : G) (code : C → ℝ) :
    channelMean potential g code = (∑ c, Real.exp (potential g c) * code c) /
      ∑ c, Real.exp (potential g c) := by
  unfold channelMean channelWeight
  simp_rw [div_mul_eq_mul_div]
  rw [← Finset.sum_div]

/-- The full implicit weighted configuration sum contracts to a product with only one modified local factor.
Source: exact finite distributivity and a single-coordinate code readout; no approximate marginal is introduced. -/
theorem channelWeightedPartition [Fintype G] (potential : G → C → ℝ) (g : G) (code : C → ℝ) :
    (∑ assignment : G → C, Real.exp (channelEnergy potential assignment) * code (assignment g)) =
      ∏ h, ∑ c, Real.exp (potential h c) * (if h = g then code c else 1) := by
  rw [Fintype.prod_sum]
  apply Finset.sum_congr rfl
  intro assignment _
  rw [Finset.prod_mul_distrib, Fintype.prod_ite_eq', ← channelEnergy_exp]

/-- Compact local readout is exactly the full implicit Gibbs expectation at arbitrary joint value parameters.
Source: the true channel distribution, exact weighted contraction and cancellation of every other positive group partition. -/
theorem channelProbability_mean [Fintype G] [Nonempty C] (potential : G → C → ℝ) (g : G) (code : C → ℝ) :
    (∑ assignment : G → C, channelProbability potential assignment * code (assignment g)) =
      channelMean potential g code := by
  rw [channelMean_eq_ratio]
  unfold channelProbability
  simp_rw [div_mul_eq_mul_div]
  rw [← Finset.sum_div, channelWeightedPartition, channelPartition, ← Finset.prod_div_distrib]
  have hl : ∀ h : G,
      (∑ c, Real.exp (potential h c) * (if h = g then code c else 1)) / (∑ c, Real.exp (potential h c)) =
        if h = g then (∑ c, Real.exp (potential h c) * code c) / (∑ c, Real.exp (potential h c)) else 1 := by
    intro h
    by_cases he : h = g
    · subst h
      simp only [ite_true]
    · simp only [ite_eq_right he, mul_one]
      exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'
  simp_rw [hl]
  exact Fintype.prod_ite_eq' g _

/-- The genuine compact value readout respects addition of output code coordinates.
Source: linearity of an actual categorical expectation, while its trained log potentials remain arbitrary. -/
theorem channelMean_add (potential : G → C → ℝ) (g : G) (left right : C → ℝ) :
    channelMean potential g (fun c => left c + right c) =
      channelMean potential g left + channelMean potential g right := by
  unfold channelMean
  simp_rw [mul_add, Finset.sum_add_distrib]

/-- A constant output channel is preserved exactly by the actually computed normalized mean.
Source: the local categorical normalization, used later for the true residual constant/readout coupling. -/
theorem channelMean_const [Nonempty C] (potential : G → C → ℝ) (g : G) (a : ℝ) :
    channelMean potential g (fun _ => a) = a := by
  unfold channelMean
  rw [← Finset.sum_mul, channelWeight_sum, one_mul]

/-- Actual code scaling commutes with the compact learned-value readout.
Source: linearity in fixed output codes, not convexity in nonlinear probabilities. -/
theorem channelMean_scale (potential : G → C → ℝ) (g : G) (a : ℝ) (code : C → ℝ) :
    channelMean potential g (fun c => a * code c) = a * channelMean potential g code := by
  unfold channelMean
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro c _
  ring

/-- The actual compact code mean lies between the smallest and largest allowed code coordinates.
Source: positive computed channel weights with derived unit total mass, needed for the real output error bound. -/
theorem channelMean_bounds [Nonempty C] (potential : G → C → ℝ) (g : G)
    (code : C → ℝ) (lower upper : ℝ) (hcode : ∀ c, lower ≤ code c ∧ code c ≤ upper) :
    lower ≤ channelMean potential g code ∧ channelMean potential g code ≤ upper := by
  have hlo := Finset.sum_le_sum (fun c (_ : c ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hcode c).1 (channelWeight_pos potential g c).le)
  have hup := Finset.sum_le_sum (fun c (_ : c ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hcode c).2 (channelWeight_pos potential g c).le)
  rw [← Finset.sum_mul, channelWeight_sum, one_mul] at hlo hup
  exact ⟨hlo, hup⟩

example : ∀ c : Fin 2, (0 : ℝ) ≤ (if c = 0 then (1 : ℝ) else 0) ∧
    (if c = 0 then (1 : ℝ) else 0) ≤ 1 := by
  intro c
  fin_cases c <;> norm_num

/-- A two-channel control shows the compact local readout uses the genuine untrained distribution.
Source: zero free potentials give the exact categorical mean one half, without an observed channel input. -/
example : channelMean (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) 0
    (fun c => if c = 0 then 1 else 0) = 1 / 2 := by
  norm_num [channelMean, channelWeight, Fin.sum_univ_two]

/-- The same finite control verifies equality to the full implicit configuration expectation.
Source: the proved exact marginal identity, independent of successful task training. -/
example : (∑ a : Fin 1 → Fin 2, channelProbability (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) a *
    (if a 0 = 0 then (1 : ℝ) else 0)) = 1 / 2 := by
  calc
    _ = channelMean (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) 0 (fun c => if c = 0 then 1 else 0) := by
      convert channelProbability_mean (fun _ : Fin 1 => fun _ : Fin 2 => (0 : ℝ)) (0 : Fin 1)
        (fun c : Fin 2 => if c = 0 then 1 else 0)
    _ = 1 / 2 := by norm_num [channelMean, channelWeight, Fin.sum_univ_two]

end
end Transformer.GPTMini.Convex.Structured
