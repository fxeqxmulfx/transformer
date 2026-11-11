module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Defs
public import Mathlib.Analysis.Convex.Function
public import Mathlib.Tactic

/-!
# Convex parameter domain and linear dictionary readout

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR1-CR2.
The complete matching matrix is learned directly. Its row-simplex domain
is convex, its output logits are jointly linear in every parameter, and
its identity setting copies every dictionary exactly. This retains a
memory coordinate per key, consistent with the cut-width bound AR4.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators

variable {κ ν : Type*} [Fintype κ] [DecidableEq ν]

/-- The learned matching matrices form a convex set (CR2). -/
theorem convex_stochastic : Convex ℝ {A : Matrix κ κ ℝ | IsStochastic A} := by
  intro A hA B hB a b ha hb hab
  constructor
  · intro q k
    exact add_nonneg (mul_nonneg ha (hA.1 q k)) (mul_nonneg hb (hB.1 q k))
  · intro q
    change (∑ k, (a * A q k + b * B q k)) = 1
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hA.2 q, hB.2 q]
    simpa using hab

/-- The identity is feasible; existence of a feasible model is unconditional
for every finite key set (CR1). -/
theorem stochastic_one [DecidableEq κ] : IsStochastic (1 : Matrix κ κ ℝ) := by
  constructor
  · intro q k
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · intro q
    simp [Matrix.one_apply]

/-- Addition of the full parameter matrices adds the actual logits (CR2). -/
lemma logits_add (A B : Matrix κ κ ℝ) (q : κ) (D : κ → ν) :
    logits (A + B) q D = logits A q D + logits B q D := by
  funext c
  simp only [logits, Matrix.add_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  split_ifs <;> simp

/-- Scaling every matching weight scales the actual logits (CR2). -/
lemma logits_smul (a : ℝ) (A : Matrix κ κ ℝ) (q : κ) (D : κ → ν) :
    logits (a • A) q D = a • logits A q D := by
  funext c
  simp only [logits, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  split_ifs <;> simp

/-- The identity matching matrix returns one-hot answer logits on every
dictionary, including dictionaries never used in training (CR1). -/
theorem logits_one [DecidableEq κ] (q : κ) (D : κ → ν) (c : ν) :
    logits (1 : Matrix κ κ ℝ) q D c = if D q = c then 1 else 0 := by
  classical
  simp only [logits, Matrix.one_apply]
  rw [Finset.sum_eq_single q]
  · simp
  · intro k _ hk
    simp [Ne.symm hk]
  · simp

/-- Feasible matching weights give nonnegative logits (CR1). -/
lemma logits_nonneg {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (q : κ) (D : κ → ν) (c : ν) : 0 ≤ logits A q D c := by
  apply Finset.sum_nonneg
  intro k _
  split_ifs
  · exact hA.1 q k
  · exact le_rfl

/-- Each class logit is at most the row's unit mass (CR1). -/
lemma logits_le_one {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (q : κ) (D : κ → ν) (c : ν) : logits A q D c ≤ 1 := by
  rw [← hA.2 q]
  apply Finset.sum_le_sum
  intro k _
  split_ifs
  · exact le_rfl
  · exact hA.1 q k

/-- A target with all the row's mass leaves zero weight on every key whose
value differs from the target (`review/flat_spectrum.md`, CR3). -/
lemma weight_eq_zero_of_target_mass {A : Matrix κ κ ℝ} (hA : IsStochastic A)
    (q : κ) (D : κ → ν) (hmass : logits A q D (D q) = 1)
    (k : κ) (hk : D k ≠ D q) : A q k = 0 := by
  have hle : ∀ j ∈ (Finset.univ : Finset κ),
      (if D j = D q then A q j else 0) ≤ A q j := by
    intro j _
    split_ifs
    · exact le_rfl
    · exact hA.1 q j
  have he : (∑ j, if D j = D q then A q j else 0) = ∑ j, A q j :=
    hmass.trans (hA.2 q).symm
  have hj := (Finset.sum_eq_sum_iff_of_le hle).mp he k (Finset.mem_univ k)
  simpa [hk] using hj.symm

/-- Two keys with a genuinely nonconstant input dictionary witness the
model, feasibility, and its output bounds together. -/
example : IsStochastic (1 : Matrix Bool Bool ℝ) ∧
    logits (1 : Matrix Bool Bool ℝ) true id true = 1 ∧
    logits (1 : Matrix Bool Bool ℝ) true id false = 0 := by
  exact ⟨stochastic_one, by simp [logits_one], by simp [logits_one]⟩

example := logits_nonneg (ν := Bool) (stochastic_one (κ := Bool)) true id false

example := logits_le_one (ν := Bool) (stochastic_one (κ := Bool)) true id true

example := weight_eq_zero_of_target_mass (stochastic_one (κ := Bool)) true id
  (by simp [logits_one]) false (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexRecall
