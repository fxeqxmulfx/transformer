/-
# §6.2 — The ODE for the Gram matrix of an attention trajectory

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
`thm: orthogonal`. The derivative closes on the pairwise inner products,
without a choice of ambient coordinates or an assumed symmetry of the flow.
-/

import Transformer.Perspective.ConeFields
import Transformer.Perspective.Section1_IPS
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

/-- Softmax weights written in terms of a Gram matrix.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. -/
noncomputable def gramWeight (n : ℕ) (β : ℝ) (G : Idx n → Idx n → ℝ) (i k : Idx n) : ℝ :=
  exp (β * G i k) / (∑ l : Idx n, exp (β * G i l))

/-- The contribution from moving the first token to a pairwise inner
product. For a symmetric Gram matrix the other contribution swaps the indices.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
noncomputable def gramHalfDrift (n : ℕ) (β : ℝ) (G : Idx n → Idx n → ℝ) (i j : Idx n) : ℝ :=
  ∑ k : Idx n, gramWeight n β G i k * (G k j - G i k * G i j)

/-- The closed autonomous ODE for the pairwise inner products. The
ambient field is defined on all real matrices; on Gram matrices it is the
exact derivative of `SA`, as `hasDerivAt_tokenGram` proves.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
noncomputable def gramDrift (n : ℕ) (β : ℝ) (G : Idx n → Idx n → ℝ) : Idx n → Idx n → ℝ :=
  fun i j => gramHalfDrift n β G i j + gramHalfDrift n β G j i

/-- The matrix of all pairwise inner products of the tokens.

Source: arXiv:2312.10794v5, §6.2, `thm: orthogonal`. -/
noncomputable def tokenGram {d n : ℕ} (X : SphereTuple d n) : Idx n → Idx n → ℝ :=
  fun i j => inner ℝ (X i : EucSpace d) (X j : EucSpace d)

/-- The matrix softmax weights are smooth, because each row partition
function is positive.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. -/
theorem contDiff_gramWeight (n : ℕ) (β : ℝ) (i k : Idx n) :
    ContDiff ℝ 1 (fun G => gramWeight n β G i k) := by
  have hentry (i j : Idx n) : ContDiff ℝ 1 (fun G : Idx n → Idx n → ℝ => G i j) :=
    (contDiff_apply ℝ ℝ j).comp (contDiff_apply ℝ (Idx n → ℝ) i)
  have he (j : Idx n) : ContDiff ℝ 1 (fun G : Idx n → Idx n → ℝ => exp (β * G i j)) :=
    (contDiff_const.mul (hentry i j)).exp
  exact (he k).div (ContDiff.sum (fun j _ => he j)) (fun G =>
    (Finset.sum_pos (fun j _ => exp_pos _) ⟨i, Finset.mem_univ _⟩).ne')

/-- The Gram-matrix field is smooth and therefore locally Lipschitz.

Source: arXiv:2312.10794v5, §6.2, `thm: orthogonal`; this permits
comparison of the actual Gram matrix with a scalar-angle matrix by uniqueness. -/
theorem contDiff_gramDrift (n : ℕ) (β : ℝ) : ContDiff ℝ 1 (gramDrift n β) := by
  have hentry (i j : Idx n) : ContDiff ℝ 1 (fun G : Idx n → Idx n → ℝ => G i j) :=
    (contDiff_apply ℝ ℝ j).comp (contDiff_apply ℝ (Idx n → ℝ) i)
  have hhalf (i j : Idx n) : ContDiff ℝ 1 (fun G => gramHalfDrift n β G i j) :=
    ContDiff.sum fun k _ => (contDiff_gramWeight n β i k).mul
      ((hentry k j).sub ((hentry i k).mul (hentry i j)))
  exact contDiff_pi.2 fun i => contDiff_pi.2 fun j => (hhalf i j).add (hhalf j i)

/-- Taking the inner product of one token velocity with another yields
the first half of the Gram-matrix ODE, including its self-attention term.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
theorem inner_SA_drift (d n : ℕ) (β : ℝ) (X : ℝ → SphereTuple d n)
    (t : ℝ) (i j : Idx n) :
    inner ℝ (proj d (X t i : EucSpace d)
      ((partitionSA d n β X t i)⁻¹ • ∑ k : Idx n,
        exp (β * tokenGram (X t) i k) • (X t k : EucSpace d))) (X t j : EucSpace d)
      = gramHalfDrift n β (tokenGram (X t)) i j := by
  simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_smul_right,
    sum_inner, inner_sum, gramHalfDrift, gramWeight, tokenGram,
    partitionSA]
  rw [Finset.mul_sum, Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- Every `SA` trajectory induces an integral curve of the Gram field.
This follows by differentiating the actual pairwise inner products.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 2. -/
theorem hasDerivAt_tokenGram (d n : ℕ) (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : SA d n β X) (t : ℝ) :
    HasDerivAt (fun s => tokenGram (X s)) (gramDrift n β (tokenGram (X t))) t := by
  refine hasDerivAt_pi.2 fun i => hasDerivAt_pi.2 fun j => ?_
  have hi := inner_SA_drift d n β X t i j
  have hj := inner_SA_drift d n β X t j i
  dsimp only [tokenGram] at hi hj
  have hd := (hX t i).inner ℝ (hX t j)
  rw [real_inner_comm _ (X t i : EucSpace d), hj, hi] at hd
  simpa only [tokenGram, gramDrift, add_comm] using hd

/-- Unit sphere tokens give ones on the diagonal of their Gram matrix.

Source: arXiv:2312.10794v5, §6.2, `thm: orthogonal`. -/
theorem tokenGram_self {d n : ℕ} (X : SphereTuple d n) (i : Idx n) :
    tokenGram X i i = 1 := by
  rw [tokenGram, real_inner_self_eq_norm_sq, norm_coe_tuple, one_pow]

/-- The actual token Gram matrix is symmetric.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`. -/
theorem tokenGram_symm {d n : ℕ} (X : SphereTuple d n) (i j : Idx n) :
    tokenGram X i j = tokenGram X j i :=
  real_inner_comm (X j : EucSpace d) (X i : EucSpace d)

/-- Each token Gram entry lies in `[-1, 1]`.

Source: arXiv:2312.10794v5, §6.2, the cosine in `thm: orthogonal`. -/
theorem tokenGram_mem_Icc {d n : ℕ} (X : SphereTuple d n) (i j : Idx n) :
    tokenGram X i j ∈ Set.Icc (-1 : ℝ) 1 :=
  real_inner_mem_Icc_of_norm_eq_one (norm_coe_tuple X i) (norm_coe_tuple X j)

/-- Consensus has every Gram entry equal to one.

Source: arXiv:2312.10794v5, §6.2, the limiting common angle. -/
theorem tokenGram_consensus {d n : ℕ} (x : SSphere d) (i j : Idx n) :
    tokenGram (fun _ => x : SphereTuple d n) i j = 1 := by
  simp only [tokenGram, real_inner_self_eq_norm_sq,
    mem_sphere_zero_iff_norm.mp x.2, one_pow]

/-- The standard basis supplies a concrete orthogonal sphere configuration.

Source: arXiv:2312.10794v5, §6.2, proof of `thm: orthogonal`, part 1. -/
noncomputable def sphereBasisConfig (n : ℕ) : SphereTuple n n :=
  fun i => ⟨EuclideanSpace.single i 1,
    mem_sphere_zero_iff_norm.2 (by simp)⟩

/-- Distinct basis tokens are orthogonal.

Source: arXiv:2312.10794v5, §6.2, the hypothesis of `thm: orthogonal`. -/
theorem sphereBasisConfig_orthogonal (n : ℕ) (i j : Idx n) (hij : i ≠ j) :
    inner ℝ (sphereBasisConfig n i : EucSpace n)
      (sphereBasisConfig n j : EucSpace n) = 0 := by
  exact (EuclideanSpace.orthonormal_single (𝕜 := ℝ)).inner_eq_zero hij

/-- There are distinct indices witnessing the orthogonality hypothesis. -/
example : (0 : Idx 2) ≠ 1 := by norm_num

/-- A two-token consensus is an actual `SA` trajectory, witnessing the
hypothesis of the Gram derivative theorem at positive temperature. -/
example : SA 2 2 1 (fun _ _ => basePoint 1) :=
  SA_const_consensus 2 2 (by norm_num) 1 (basePoint 1)

end Perspective
end Transformer
