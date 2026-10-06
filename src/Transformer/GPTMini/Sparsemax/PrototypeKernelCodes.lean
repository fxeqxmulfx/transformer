import Transformer.GPTMini.Sparsemax.PrefixFeatureCodes

/-!
# Compact probability codes from a finite prototype kernel

Derived memory input architecture for sparsemax arXiv:1602.02068v2,
Eq. (1). Keep finitely many observed prototypes rather than enumerate every
possible prefix. A query is encoded by its normalized similarities to these
prototypes. The similarity is an exact-observation term plus a constant and
a squared feature inner product. All features and identities come from data.

The squared inner product computes pairwise feature interactions without
forming their tensor dictionary. The constant guarantees strictly positive
normalization even for unseen observations. The exact-observation component
will certify nonsingularity on distinct training prototypes in the next module.

There is one learned memory slot per registered prototype. The learned Gram
and values still use the earlier exact convex chart. Codes are dense, and
the output-only Gram freedom persists; neither sparsity of query routes nor
sample-independent parameter count is claimed by this construction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {Key : Type*} [DecidableEq Key]

/-- Target-independent similarity to every registered observation prototype.
Source: the derived compact input architecture for arXiv:1602.02068v2, Eq. (1).
The squared dot product encodes second-order feature interactions. -/
def prototypeKernelScores {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  Matrix.of (fun r j => (if observations r = prototypes j then 1 else 0) + 1 +
    (∑ d, queries r d * features j d) ^ 2)

/-- Every kernel entry is at least one, including unseen observations.
Source: the constant and nonnegative square in the derived arXiv:1602.02068v2, Eq. (1) input. -/
theorem prototypeKernelScores_ge_one {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (r : Fin R) (j : Fin (N + 1)) :
    1 ≤ prototypeKernelScores observations prototypes queries features r j := by
  change 1 ≤ (if observations r = prototypes j then (1 : ℝ) else 0) + 1 +
    (∑ d, queries r d * features j d) ^ 2
  split_ifs <;> nlinarith [sq_nonneg (∑ d, queries r d * features j d)]

/-- Normalizing mass over the registered prototypes, with no complete-prefix dictionary.
Source: the derived kernel probability input for arXiv:1602.02068v2, Eq. (1). -/
def prototypeKernelMass {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (r : Fin R) : ℝ :=
  ∑ j, prototypeKernelScores observations prototypes queries features r j

/-- A real registered slot ensures a positive denominator for every new query.
Source: the nonempty prototype memory and kernel lower bound before arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeKernelMass_pos {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (r : Fin R) :
    0 < prototypeKernelMass observations prototypes queries features r := by
  have he (j) := prototypeKernelScores_ge_one observations prototypes queries features r j
  have hs := Finset.single_le_sum (s := Finset.univ)
    (f := fun j => prototypeKernelScores observations prototypes queries features r j)
    (fun j _ => le_trans (by norm_num : (0 : ℝ) ≤ 1) (he j))
    (Finset.mem_univ (0 : Fin (N + 1)))
  change 0 < ∑ j, prototypeKernelScores observations prototypes queries features r j
  linarith [he 0]

/-- Normalize similarity profiles into data codes for the learned memory.
Source: the derived compact input architecture for arXiv:1602.02068v2, Eq. (1). -/
def prototypeKernelCodes {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  Matrix.of (fun r j => prototypeKernelScores observations prototypes queries features r j /
    prototypeKernelMass observations prototypes queries features r)

/-- Every input profile is a genuine probability code, for arbitrary real features.
Source: the derived normalization into the memory domain of arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeKernelCodes_mem {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) :
    prototypeKernelCodes observations prototypes queries features ∈ contextCodeDomain R N := by
  intro r
  have hm := prototypeKernelMass_pos observations prototypes queries features r
  refine ⟨?_, ?_, fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · intro j
    apply div_nonneg
    · exact le_trans (by norm_num) (prototypeKernelScores_ge_one _ _ _ _ r j)
    · exact le_of_lt hm
  · change (∑ j, prototypeKernelScores observations prototypes queries features r j /
      prototypeKernelMass observations prototypes queries features r) = 1
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hm)

/-- All similarity-profile entries are positive; this variant uses dense query codes.
Source: the derived positive kernel input before sparsemax arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeKernelCodes_pos {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (r : Fin R) (j : Fin (N + 1)) :
    0 < prototypeKernelCodes observations prototypes queries features r j := by
  apply div_pos
  · exact lt_of_lt_of_le (by norm_num) (prototypeKernelScores_ge_one _ _ _ _ r j)
  · exact prototypeKernelMass_pos _ _ _ _ r

/-- The kernel code can be evaluated by dot products, with no pair-feature enumeration.
Source: the derived arXiv:1602.02068v2, Eq. (1) input, exposing its finite computation. -/
theorem prototypeKernelCodes_apply {R N F : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (r : Fin R) (j : Fin (N + 1)) :
    prototypeKernelCodes observations prototypes queries features r j =
      ((if observations r = prototypes j then 1 else 0) + 1 +
        (∑ d, queries r d * features j d) ^ 2) /
      (∑ k, ((if observations r = prototypes k then 1 else 0) + 1 +
        (∑ d, queries r d * features k d) ^ 2)) := rfl

/-- Global output coordinates are evaluated by one normalized weighted kernel sum.
Source: the derived compact input chart for arXiv:1602.02068v2, Eq. (1).
The implementation needs neither a full prefix dictionary nor pair-feature vectors. -/
theorem prototypeKernelCodes_mul {R N F D : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) :
    prototypeKernelCodes observations prototypes queries features * Z = Matrix.of (fun r d =>
      (∑ j, prototypeKernelScores observations prototypes queries features r j * Z j d) /
        prototypeKernelMass observations prototypes queries features r) := by
  ext r d
  rw [Matrix.mul_apply]
  simp only [prototypeKernelCodes, Matrix.of_apply, div_mul_eq_mul_div, Finset.sum_div]

/-- A common constant output table stays constant on every new observation.
Source: probability normalization in the derived arXiv:1602.02068v2, Eq. (1) input. -/
theorem prototypeKernelCodes_const {R N F D : ℕ} (observations : Fin R → Key)
    (prototypes : Fin (N + 1) → Key) (queries : Matrix (Fin R) (Fin F) ℝ)
    (features : Matrix (Fin (N + 1)) (Fin F) ℝ) (value : Fin D → ℝ) :
    prototypeKernelCodes observations prototypes queries features *
      (Matrix.of (fun _ => value) : Matrix (Fin (N + 1)) (Fin D) ℝ) =
      (Matrix.of (fun _ => value) : Matrix (Fin R) (Fin D) ℝ) := by
  ext r d
  rw [Matrix.mul_apply]
  simp only [Matrix.of_apply]
  rw [← Finset.sum_mul, (prototypeKernelCodes_mem observations prototypes queries features r).2.1,
    one_mul]

/-- Even a query not present among the prototypes has a well-defined probability code. -/
example : prototypeKernelCodes (fun _ : Fin 1 => (1 : Fin 2))
    (fun _ : Fin 1 => (0 : Fin 2)) (0 : Matrix (Fin 1) (Fin 0) ℝ)
    (0 : Matrix (Fin 1) (Fin 0) ℝ) 0 0 = 1 := by
  norm_num [prototypeKernelCodes_apply, Fin.sum_univ_one]

end Transformer.GPTMini.Sparsemax
