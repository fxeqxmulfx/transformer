module

public import Transformer.Perceptron.Section5_MapDegenerateBranches

/-!
# Primitive and rank-one reductions for two original ReLU neurons

ArXiv:2601.21366v2, §2, `eq:primitive.field`, §5.2, `thm: circle`,
and §6, `eq:ratio-const-correct`. A primitive of twice ReLU differs
from the squared positive part by its actual value at zero. Positive
and negative rescalings of the original neuron input are tracked
explicitly, so no primitive or original neuron is silently replaced.

The signed Gram action of two neurons is the sum of two signed rank-one
terms. Cancellation makes the two nontrivial terms collinear. At an
actual pattern point with nonzero neuron input, its sign determines
the original membership in the active set. These elementary reductions
are used to analyze the zero scalar alternative of the continued
matrix branches, without imposing non-degeneracy on the source claim.
-/

@[expose] public section

open Real Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- The original arbitrary primitive retains exactly its additive constant.
Source: arXiv:2601.21366v2, §2, the convention `φ'=2σ` and ReLU potential. -/
theorem reluPrimitive_eq (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (s : ℝ) : φ s = max s 0 ^ 2 + φ 0 := by
  have hd (r : ℝ) : HasDerivAt (fun t => φ t - max t 0 ^ 2) 0 r :=
    ((hφ r).sub (hasDerivAt_reluSq r)).congr_deriv (sub_self _)
  have h := is_const_of_deriv_eq_zero (fun r => (hd r).differentiableAt)
    (fun r => (hd r).deriv) s 0
  simp only [max_self, zero_pow (by decide : 2 ≠ 0), sub_zero] at h
  linarith

/-- The original weighted primitive has an explicit constant contribution.
Source: arXiv:2601.21366v2, §2 and §5.2, `eq:primitive.field`. -/
theorem potential_reluPrimitive_eq {d : ℕ} (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (x : EucSpace d) :
    potential φ ω a x = potential (fun s => max s 0 ^ 2) ω a x + φ 0 * ∑ j, ω j := by
  unfold potential
  calc
    _ = ∑ j, (ω j * max (inner (𝕜 := ℝ) (a j) x) 0 ^ 2 + φ 0 * ω j) := by
      apply Finset.sum_congr rfl
      intro j _
      rw [reluPrimitive_eq φ hφ (inner (𝕜 := ℝ) (a j) x)]
      ring
    _ = _ := by rw [Finset.sum_add_distrib, Finset.mul_sum]

/-- Nonnegative input rescaling preserves the squared positive part.
Source: arXiv:2601.21366v2, §5.2, the quadratic ReLU branches. -/
theorem reluSq_mul_nonneg (c : ℝ) (hc : 0 ≤ c) (s : ℝ) :
    max (c * s) 0 ^ 2 = c ^ 2 * max s 0 ^ 2 := by
  by_cases hs : 0 ≤ s
  · rw [max_eq_left hs, max_eq_left (mul_nonneg hc hs)]
    ring
  · rw [max_eq_right (le_of_not_ge hs),
      max_eq_right (mul_nonpos_of_nonneg_of_nonpos hc (le_of_not_ge hs))]
    simp

/-- Nonpositive input rescaling uses the opposite original half-line.
Source: arXiv:2601.21366v2, §5.2, complementary ReLU branches. -/
theorem reluSq_mul_nonpos (c : ℝ) (hc : c ≤ 0) (s : ℝ) :
    max (c * s) 0 ^ 2 = c ^ 2 * max (-s) 0 ^ 2 := by
  have h := reluSq_mul_nonneg (-c) (neg_nonneg.mpr hc) (-s)
  simpa only [neg_mul_neg, neg_sq] using h

/-- The two original complementary squared parts form one quadratic.
Source: arXiv:2601.21366v2, §5.2, `eq:quad-combo-zero`. -/
theorem reluSq_add_neg (s : ℝ) : max s 0 ^ 2 + max (-s) 0 ^ 2 = s ^ 2 := by
  by_cases hs : 0 ≤ s
  · rw [max_eq_left hs, max_eq_right (neg_nonpos.mpr hs)]
    simp
  · rw [max_eq_right (le_of_not_ge hs), max_eq_left (by linarith)]
    simp

/-- Zero effective neurons leave an actual constant analytic potential.
Source: arXiv:2601.21366v2, §3.1, the excluded constant cases of `thm: circle`. -/
theorem isAnalyticOnSphere_of_zero_effective_neurons {d : ℕ} (φ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hz : ∀ j, ω j = 0 ∨ a j = 0) : IsAnalyticOnSphere φ ω a := by
  refine ⟨fun _ => φ 0 * ∑ j, ω j, fun _ _ => analyticAt_const, fun x _ => ?_⟩
  rw [potential, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rcases hz j with hω | ha
  · simp [hω]
  · simp [ha, mul_comm]

/-- The original sign of each rank-one term of a signed Gram action.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`. -/
noncomputable def signedBranchWeight {d : ℕ} (ω : Idx d → ℝ)
    (J : Finset (Idx d)) (j : Idx d) : ℝ := if j ∈ J then ω j else -ω j

/-- The signed coefficient vanishes exactly when its original weight does.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`. -/
@[simp] theorem signedBranchWeight_eq_zero_iff {d : ℕ} (ω : Idx d → ℝ)
    (J : Finset (Idx d)) (j : Idx d) : signedBranchWeight ω J j = 0 ↔ ω j = 0 := by
  by_cases hj : j ∈ J <;> simp [signedBranchWeight, hj]

/-- The two original signed rank-one terms retain both original neurons.
Source: arXiv:2601.21366v2, §6, `eq: nondeg.normalized`, in dimension two. -/
theorem signedGram_two_eq (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (J : Finset (Idx 2)) (x : EucSpace 2) :
    signedGram ω a J x = (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) (a 0) x) • a 0 +
      (signedBranchWeight ω J 1 * inner (𝕜 := ℝ) (a 1) x) • a 1 := by
  classical
  rw [signedGram, ← Fintype.sum_ite_mem J, ← Fintype.sum_ite_mem Jᶜ]
  simp only [Fin.sum_univ_two, Finset.mem_compl]
  by_cases h₀ : (0 : Idx 2) ∈ J <;> by_cases h₁ : (1 : Idx 2) ∈ J <;>
    simp [signedBranchWeight, h₀, h₁, neg_smul] <;> abel

/-- Cancellation of two rank-one actions makes the nontrivial second
neuron a multiple of the first. Source: arXiv:2601.21366v2, §6,
the zero-scalar case of `eq:ratio-const-correct`. -/
theorem exists_smul_of_two_rankone_cancel {d : ℕ} (u v : EucSpace d) (A B : ℝ)
    (hB : B ≠ 0) (hv : v ≠ 0)
    (hz : ∀ x, (A * inner (𝕜 := ℝ) u x) • u + (B * inner (𝕜 := ℝ) v x) • v = 0) :
    ∃ c : ℝ, v = c • u := by
  have hb : B * inner (𝕜 := ℝ) v v ≠ 0 := mul_ne_zero hB (inner_self_ne_zero.mpr hv)
  have h := eq_neg_of_add_eq_zero_right (hz v)
  refine ⟨-(B * inner (𝕜 := ℝ) v v)⁻¹ * (A * inner (𝕜 := ℝ) u v), ?_⟩
  calc
    v = (B * inner (𝕜 := ℝ) v v)⁻¹ • ((B * inner (𝕜 := ℝ) v v) • v) := by
      rw [smul_smul, inv_mul_cancel₀ hb, one_smul]
    _ = _ := by rw [h, smul_neg, smul_smul, ← neg_smul, neg_mul]

/-- Nonzero original neuron input determines its actual active-set sign.
Source: arXiv:2601.21366v2, §5.2, the activation regions `J_I`. -/
theorem mem_pattern_iff_inner_pos_of_ne {d : ℕ} (a : Idx d → EucSpace d)
    (J : Finset (Idx d)) (x : EucSpace d) (hpat : IsActivationPattern a J x)
    (j : Idx d) (hne : inner (𝕜 := ℝ) (a j) x ≠ 0) :
    j ∈ J ↔ 0 < inner (𝕜 := ℝ) (a j) x := by
  constructor
  · exact fun hj => lt_of_le_of_ne (hpat.1 j hj) hne.symm
  · intro hp
    by_contra hj
    exact (not_lt_of_ge (hpat.2 j hj)) hp

/-- The primitive and both rescaling hypotheses hold at the original
squared-ReLU primitive, with genuine positive and negative scales. -/
example : (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    (0 : ℝ) ≤ 1 ∧ (-1 : ℝ) ≤ 0 ∧
    (∀ j : Idx 2, (0 : Idx 2 → ℝ) j = 0 ∨ (0 : Idx 2 → EucSpace 2) j = 0) :=
  ⟨hasDerivAt_reluSq, zero_le_one, by norm_num, fun _ => Or.inl rfl⟩

/-- Rank-one cancellation has nonzero original vector and coefficient
witnesses. The opposite coefficients cancel the same actual rank-one map. -/
example : (-1 : ℝ) ≠ 0 ∧ (basePoint 1 : EucSpace 2) ≠ 0 ∧
    ∀ x : EucSpace 2, ((1 : ℝ) * inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) •
      (basePoint 1 : EucSpace 2) + ((-1 : ℝ) * inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) •
        (basePoint 1 : EucSpace 2) = 0 := by
  refine ⟨by norm_num, ?_, fun x => by simp⟩
  intro h
  have hn := norm_basePoint_one
  rw [h, norm_zero] at hn
  exact zero_ne_one hn

/-- The actual sign test has a nonzero original neuron input at an
inactive pattern point. The second zero neuron remains in the pattern. -/
example : IsActivationPattern
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∅ (basePoint 1 : EucSpace 2) ∧
    inner (𝕜 := ℝ) ((Pi.single 0 (-(basePoint 1 : EucSpace 2)) : Idx 2 → EucSpace 2) 0)
      (basePoint 1 : EucSpace 2) ≠ 0 := by
  refine ⟨⟨fun j hj => by simp at hj, ?_⟩, ?_⟩
  · intro j hj
    fin_cases j <;> simp [basePoint]
  · simp [basePoint]

end Transformer.Perceptron
