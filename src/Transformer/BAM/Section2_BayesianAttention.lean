module

public import Mathlib.Analysis.SpecialFunctions.Log.ERealExp
public import Mathlib.Probability.ProbabilityMassFunction.Constructions
public import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# §2.1 Bayesian attention and the joint probability `p_ij`

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, "Bayesian Attention Mechanism: A
Probabilistic Framework for Positional Encoding and Context Length Extrapolation", §2.1.

A query attends to the keys `ι` with scores `s : ι → EReal`. Scores are extended reals because the
causal mask of Appendix B.1 "changes the scores ... to `−∞`"; a score `−∞` gets the weight
`e^{−∞} = 0` (`weight`, through Mathlib's `EReal.exp`). No score of the paper is `+∞`.
`softmax s` is the attention row `σ(s)` of Appendix A, `partition s = Σ_z e^{s_z}` its denominator.

* Definition 1: `σ(s) V = Σ_j p_ij v_j = 𝐄_{j|i}[V]`. When one score is finite, the weights are a
  probability distribution on the keys and attention is the expectation of the values under it
  (`attention_eq_expectation`).
* Theorem 1: for an additive score `f_cont + g_pos`, `p_ij = p(f_cont) · p(g_pos) / Z` with
  `Z = Σ_z p(f_cont(q_i, k_z)) p(g_pos(i, z))` (`softmax_add`); the factor its proof multiplies
  in, `Σ_z e^{f_z} · Σ_z e^{g_z} / Σ_z e^{f_z + g_z}`, is `1 / Z`
  (`partition_mul_div_partition_add`). No hypothesis is needed: masks (`−∞`) are allowed.
* `softmax_add_eq_of_proportional`: the same with the position marginal replaced by any prior
  proportional to `e^{g}`; Lemmas 1 and 2 and the GGD prior of §2.3 are its instances.
-/

@[expose] public section

namespace Transformer.BAM

open Finset MeasureTheory

/-- The weight `e^s` of an extended score, with `e^{−∞} = 0` (Appendix B.1: the causal mask sets
scores to `−∞`, and the softmax gives them `0`). -/
noncomputable def weight (s : EReal) : ℝ := (EReal.exp s).toReal

@[simp] theorem weight_coe (x : ℝ) : weight x = Real.exp x := by
  simp [weight, ENNReal.toReal_ofReal (Real.exp_pos x).le]

@[simp] theorem weight_bot : weight ⊥ = 0 := by simp [weight]

@[simp] theorem weight_zero : weight 0 = 1 := by simp [weight]

/-- `e^{a + b} = e^a e^b` for extended scores, masks included. -/
theorem weight_add (a b : EReal) : weight (a + b) = weight a * weight b := by
  simp [weight, EReal.exp_add]

theorem weight_nonneg (s : EReal) : 0 ≤ weight s := ENNReal.toReal_nonneg

/-- A finite score has a positive weight. -/
theorem weight_pos {s : EReal} (h₁ : s ≠ ⊥) (h₂ : s ≠ ⊤) : 0 < weight s := by
  rw [← EReal.coe_toReal h₂ h₁, weight_coe]
  exact Real.exp_pos _

/-- A score at most `0`, a mask or a penalty, has a weight at most `1`. -/
theorem weight_le_one {s : EReal} (h : s ≤ 0) : weight s ≤ 1 := by
  have h1 : EReal.exp s ≤ 1 := by simpa using EReal.exp_monotone h
  unfold weight
  simpa using ENNReal.toReal_mono ENNReal.one_ne_top h1

variable {ι : Type*} [Fintype ι]

/-- The partition function `Σ_z e^{s_z}`: `Z_cont`, `Z_pos` and `Z_joint` of Appendix K.1. -/
noncomputable def partition (s : ι → EReal) : ℝ := ∑ z, weight (s z)

/-- The softmax `σ(s)_j = e^{s_j} / Σ_z e^{s_z}` (Appendix A). -/
noncomputable def softmax (s : ι → EReal) (j : ι) : ℝ := weight (s j) / partition s

theorem partition_nonneg (s : ι → EReal) : 0 ≤ partition s :=
  sum_nonneg fun z _ => weight_nonneg (s z)

theorem weight_le_partition (s : ι → EReal) (z : ι) : weight (s z) ≤ partition s :=
  single_le_sum (fun y _ => weight_nonneg (s y)) (mem_univ z)

theorem softmax_nonneg (s : ι → EReal) (j : ι) : 0 ≤ softmax s j :=
  div_nonneg (weight_nonneg _) (partition_nonneg s)

theorem softmax_le_one (s : ι → EReal) (j : ι) : softmax s j ≤ 1 :=
  div_le_one_of_le₀ (weight_le_partition s j) (partition_nonneg s)

/-- Every weight vanishes when the partition function does: all scores are `−∞`. -/
theorem weight_eq_zero_of_partition_eq_zero {s : ι → EReal} (h : partition s = 0) (j : ι) :
    weight (s j) = 0 :=
  (sum_eq_zero_iff_of_nonneg fun z _ => weight_nonneg (s z)).1 h j (mem_univ j)

/-- One finite score makes the partition function positive. -/
theorem partition_pos {s : ι → EReal} {z : ι} (h₁ : s z ≠ ⊥) (h₂ : s z ≠ ⊤) :
    0 < partition s :=
  (weight_pos h₁ h₂).trans_le (weight_le_partition s z)

theorem sum_softmax {s : ι → EReal} (h : partition s ≠ 0) : ∑ j, softmax s j = 1 := by
  rw [show ∑ j, softmax s j = partition s / partition s by simp [softmax, partition, sum_div]]
  exact div_self h

theorem sum_softmax_le_one (s : ι → EReal) : ∑ j, softmax s j ≤ 1 := by
  by_cases h : partition s = 0
  · simp [softmax, h]
  · exact (sum_softmax h).le

/-- Self-attention `σ(s) V = Σ_j p_ij v_j` of a query with scores `s` against the keys and values
`v` (§2.1, Definition 1). -/
noncomputable def attention {E : Type*} [AddCommMonoid E] [Module ℝ E] (s : ι → EReal)
    (v : ι → E) : E :=
  ∑ j, softmax s j • v j

/-- **Definition 1: attention is an expectation over the values.** If one score is finite, the
weights `p_ij = σ(s)_j` are a probability distribution `p` on the keys, and
`σ(s) V = Σ_j p_ij v_j = 𝐄_{j|i}[V]` (§2.1, Definition 1). -/
theorem attention_eq_expectation [MeasurableSpace ι] [MeasurableSingletonClass ι] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] {s : ι → EReal}
    (h : ∃ z, s z ≠ ⊥ ∧ s z ≠ ⊤) (v : ι → E) :
    ∃ p : PMF ι, (∀ j, (p j).toReal = softmax s j) ∧
      attention s v = ∫ j, v j ∂p.toMeasure := by
  obtain ⟨z, h₁, h₂⟩ := h
  have hsum : ∑ j, ENNReal.ofReal (softmax s j) = 1 := by
    rw [← ENNReal.ofReal_sum_of_nonneg fun j _ => softmax_nonneg s j,
      sum_softmax (partition_pos h₁ h₂).ne', ENNReal.ofReal_one]
  refine ⟨PMF.ofFintype _ hsum, fun j => ?_, ?_⟩
  · simp [ENNReal.toReal_ofReal (softmax_nonneg s j)]
  · rw [PMF.integral_eq_sum]
    exact sum_congr rfl fun j _ => by simp [ENNReal.toReal_ofReal (softmax_nonneg s j)]

/-- The hypothesis of `attention_eq_expectation` holds for real scores: here two keys, scored
`0` and `1`, with values in `ℝ`. -/
example : ∃ p : PMF (Fin 2),
    (∀ j, (p j).toReal = softmax (fun j : Fin 2 => (((j : ℕ) : ℝ) : EReal)) j) ∧
      attention (fun j : Fin 2 => (((j : ℕ) : ℝ) : EReal)) (fun j : Fin 2 => ((j : ℕ) : ℝ)) =
        ∫ j : Fin 2, ((j : ℕ) : ℝ) ∂p.toMeasure :=
  attention_eq_expectation (s := fun j : Fin 2 => (((j : ℕ) : ℝ) : EReal))
    ⟨0, EReal.coe_ne_bot _, EReal.coe_ne_top _⟩ _

/-- The normalizing scalar of Theorem 1, `Z = Σ_z p(f_cont(q_i, k_z)) · p(g_pos(i, z))`: the
overlap of the content and position marginals (§2.1). -/
noncomputable def overlap (f g : ι → EReal) : ℝ := ∑ z, softmax f z * softmax g z

/-- `Z = Σ_z e^{f_z + g_z} / (Σ_z e^{f_z} · Σ_z e^{g_z})`. -/
theorem overlap_eq (f g : ι → EReal) :
    overlap f g = partition (f + g) / (partition f * partition g) := by
  simp only [overlap, softmax, partition, Pi.add_apply, weight_add, sum_div, div_mul_div_comm]

/-- **Theorem 1.** If the score is additive, `f_cont(q_i, k_j) + g_pos(i, j)`, the attention
weight is the product of the content and position marginals over `Z`:
`p_ij = p(f_cont) · p(g_pos) / Z` (§2.1). Masks (`−∞`) are allowed; no hypothesis is needed. -/
theorem softmax_add (f g : ι → EReal) (j : ι) :
    softmax (f + g) j = softmax f j * softmax g j / overlap f g := by
  by_cases hf : partition f = 0
  · simp [softmax, hf, weight_eq_zero_of_partition_eq_zero hf, weight_add]
  by_cases hg : partition g = 0
  · simp [softmax, hg, weight_eq_zero_of_partition_eq_zero hg, weight_add]
  rw [overlap_eq, softmax, softmax, softmax, div_mul_div_comm,
    div_div_div_cancel_right₀ (mul_ne_zero hf hg), Pi.add_apply, weight_add]

/-- **Theorem 1, the inserted factor.** The proof of Theorem 1 multiplies in
`Σ_z e^{f_z} · Σ_z e^{g_z} / Σ_z e^{f_z + g_z}`, which is `1 / Z` (§2.1). -/
theorem partition_mul_div_partition_add (f g : ι → EReal) :
    partition f * partition g / partition (f + g) = 1 / overlap f g := by
  rw [overlap_eq, one_div_div]

/-- Theorem 1 with the position marginal replaced by a prior `π` proportional to `e^{g}`:
`p_ij = p(f_j) π_j / Σ_z p(f_z) π_z`. Lemmas 1 and 2 and the GGD prior of §2.3 are instances. -/
theorem softmax_add_eq_of_proportional {f g : ι → EReal} {π : ι → ℝ} {c : ℝ} (hc : c ≠ 0)
    (hπ : ∀ z, π z = c * weight (g z)) (j : ι) :
    softmax (f + g) j = softmax f j * π j / ∑ z, softmax f z * π z := by
  by_cases hf : partition f = 0
  · simp [softmax, hf, weight_eq_zero_of_partition_eq_zero hf, weight_add]
  have hP : partition (f + g) = ∑ z, weight (f z) * weight (g z) := by
    simp [partition, weight_add]
  have hsum : ∑ z, softmax f z * π z = c / partition f * partition (f + g) := by
    rw [hP, mul_sum]
    exact sum_congr rfl fun z _ => by
      rw [softmax, hπ]
      ring
  rw [hsum, softmax, softmax, hπ, Pi.add_apply, weight_add,
    show weight (f j) / partition f * (c * weight (g j)) =
      c / partition f * (weight (f j) * weight (g j)) by ring,
    mul_div_mul_left _ _ (div_ne_zero hc hf)]

/-- The hypotheses of `softmax_add_eq_of_proportional` hold for the prior `2 e^{g}`. -/
example (f g : Fin 3 → EReal) (j : Fin 3) :
    softmax (f + g) j = softmax f j * (2 * weight (g j)) / ∑ z, softmax f z * (2 * weight (g z)) :=
  softmax_add_eq_of_proportional (π := fun z => 2 * weight (g z)) two_ne_zero (fun _ => rfl) j

end Transformer.BAM
