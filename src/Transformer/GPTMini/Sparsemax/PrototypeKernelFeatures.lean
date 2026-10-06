import Transformer.GPTMini.Sparsemax.PrototypeKernelCodes

/-!
# Pair-feature kernel and nonsingular observed-prototype geometry

Derived compact memory input for sparsemax arXiv:1602.02068v2, Eq. (1).
The squared feature inner product equals the inner product of all ordered
feature pairs. Its training matrix is a genuine positive semidefinite Gram.
Thus computing one dot product and squaring it retains these interactions
without storing the expanded pair-feature dictionary.

On distinct prototype identities the exact-observation kernel is identity.
The complete training kernel is identity plus the constant Gram and the
pair-feature Gram. It is positive definite for any real input features,
including rank-deficient or zero features. Invertibility is derived from
the observations and construction, rather than assumed for the targets.

The features remain fixed data inputs. Learned Q/K embeddings reside in the
separate memory Gram, and their inverse chart is unchanged. The observed
prototype count controls memory size, rather than the universe of prefixes.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators Matrix

/-- All ordered pair products of a fixed observation feature row.
Source: the derived second-order input kernel for arXiv:1602.02068v2, Eq. (1). -/
def prototypeQuadraticFeatures {R F : ℕ} (features : Matrix (Fin R) (Fin F) ℝ) :
    Matrix (Fin R) (Fin F × Fin F) ℝ :=
  Matrix.of (fun r p => features r p.1 * features r p.2)

/-- One squared dot product is exactly the full pair-feature inner product.
Source: the derived arXiv:1602.02068v2, Eq. (1) input kernel, proved by finite sum factorization. -/
theorem prototypeQuadratic_inner {F : ℕ} (u v : Fin F → ℝ) :
    (∑ d, u d * v d) ^ 2 =
      ∑ p : Fin F × Fin F, (u p.1 * u p.2) * (v p.1 * v p.2) := by
  rw [Fintype.sum_prod_type, pow_two, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro d hd
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e he
  ring

/-- The squared-dot-product matrix on observed feature rows.
Source: the derived kernel preceding sparsemax arXiv:1602.02068v2, Eq. (1). -/
def prototypeFeatureKernel {R F : ℕ} (features : Matrix (Fin R) (Fin F) ℝ) :
    Matrix (Fin R) (Fin R) ℝ :=
  Matrix.of (fun r s => (∑ d, features r d * features s d) ^ 2)

/-- The efficiently evaluated kernel is a genuine Gram of pair features.
Source: the derived interaction identity for the memory input of arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeFeatureKernel_eq_product {R F : ℕ}
    (features : Matrix (Fin R) (Fin F) ℝ) :
    prototypeFeatureKernel features =
      prototypeQuadraticFeatures features * (prototypeQuadraticFeatures features)ᵀ := by
  ext r s
  rw [Matrix.mul_apply]
  change (∑ d, features r d * features s d) ^ 2 =
    ∑ p : Fin F × Fin F, (features r p.1 * features r p.2) *
      (features s p.1 * features s p.2)
  exact prototypeQuadratic_inner _ _

/-- The second-order input kernel is positive semidefinite for arbitrary real features.
Source: its proved genuine Gram representation before arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeFeatureKernel_posSemidef {R F : ℕ}
    (features : Matrix (Fin R) (Fin F) ℝ) : (prototypeFeatureKernel features).PosSemidef := by
  rw [prototypeFeatureKernel_eq_product]
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_self_mul_conjTranspose (prototypeQuadraticFeatures features)

/-- The constant part of the prototype kernel is another genuine feature Gram.
Source: the derived positive normalizer in the arXiv:1602.02068v2, Eq. (1) input construction. -/
theorem prototypeConstantKernel_posSemidef (R : ℕ) :
    (Matrix.of (fun _ : Fin R => fun _ : Fin R => (1 : ℝ))).PosSemidef := by
  let C : Matrix (Fin R) (Fin 1) ℝ := Matrix.of (fun _ _ => 1)
  have he : Matrix.of (fun _ : Fin R => fun _ : Fin R => (1 : ℝ)) = C * Cᵀ := by
    ext r s
    norm_num [Matrix.mul_apply, Fin.sum_univ_one, C]
  rw [he]
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_self_mul_conjTranspose C

/-- The training kernel on a finite set of registered observations.
Source: the derived compact input architecture for arXiv:1602.02068v2, Eq. (1). -/
def prototypeTrainingKernel {Key : Type*} [DecidableEq Key] {N F : ℕ}
    (prototypes : Fin (N + 1) → Key) (features : Matrix (Fin (N + 1)) (Fin F) ℝ) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  prototypeKernelScores prototypes prototypes features features

/-- Distinct observations give identity plus two genuine positive semidefinite Grams.
Source: the actual compact kernel for arXiv:1602.02068v2, Eq. (1); distinctness is a data premise. -/
theorem prototypeTrainingKernel_eq {Key : Type*} [DecidableEq Key] {N F : ℕ}
    (prototypes : Fin (N + 1) → Key) (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (hs : Function.Injective prototypes) : prototypeTrainingKernel prototypes features =
      1 + Matrix.of (fun _ : Fin (N + 1) => fun _ : Fin (N + 1) => (1 : ℝ)) +
        prototypeFeatureKernel features := by
  ext r s
  have he : prototypes r = prototypes s ↔ r = s := ⟨fun h => hs h, congrArg prototypes⟩
  simp only [prototypeTrainingKernel, prototypeKernelScores, prototypeFeatureKernel,
    Matrix.add_apply, Matrix.one_apply, Matrix.of_apply, he]

/-- Two real nonconstant feature rows inhabit the identity-separation premise. -/
example : prototypeTrainingKernel (fun r : Fin 2 => r)
    (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ))) =
    1 + Matrix.of (fun _ : Fin 2 => fun _ : Fin 2 => (1 : ℝ)) +
      prototypeFeatureKernel (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ))) :=
  prototypeTrainingKernel_eq _ _ (by intro r s h; exact h)

/-- Every distinct finite prototype set has positive definite training kernel.
Source: identity plus the two proved input Grams for arXiv:1602.02068v2, Eq. (1).
No full-rank feature premise, teacher route or target compatibility is assumed. -/
theorem prototypeTrainingKernel_posDef {Key : Type*} [DecidableEq Key] {N F : ℕ}
    (prototypes : Fin (N + 1) → Key) (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (hs : Function.Injective prototypes) : (prototypeTrainingKernel prototypes features).PosDef := by
  rw [prototypeTrainingKernel_eq prototypes features hs]
  exact (Matrix.PosDef.one.add_posSemidef (prototypeConstantKernel_posSemidef _)).add_posSemidef
    (prototypeFeatureKernel_posSemidef features)

/-- Nonconstant real features and distinct observations satisfy the positive-definite premise. -/
example : (prototypeTrainingKernel (fun r : Fin 2 => r)
    (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ)))).PosDef :=
  prototypeTrainingKernel_posDef _ _ (by intro r s h; exact h)

/-- Nonsingularity of the training kernel is proved from the prototype data.
Source: the positive definite compact input construction for arXiv:1602.02068v2, Eq. (1). -/
theorem prototypeTrainingKernel_det_unit {Key : Type*} [DecidableEq Key] {N F : ℕ}
    (prototypes : Fin (N + 1) → Key) (features : Matrix (Fin (N + 1)) (Fin F) ℝ)
    (hs : Function.Injective prototypes) : IsUnit (prototypeTrainingKernel prototypes features).det := by
  apply isUnit_iff_ne_zero.mpr
  exact ne_of_gt (prototypeTrainingKernel_posDef prototypes features hs).det_pos

/-- A genuine finite data set satisfies the nonsingularity premise. -/
example : IsUnit (prototypeTrainingKernel (fun r : Fin 2 => r)
    (Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 1 : ℝ)))).det :=
  prototypeTrainingKernel_det_unit _ _ (by intro r s h; exact h)

/-- One-dimensional nonconstant data for a concrete compact two-prototype model.
Source: the derived kernel input for arXiv:1602.02068v2, Eq. (1). -/
def prototypeExampleFeatures : Matrix (Fin 2) (Fin 1) ℝ :=
  Matrix.of (fun r _ => (r.val + 1 : ℝ))

/-- The actual kernel is `[[3,5],[5,18]]`, despite feature width only one.
Source: exact identities plus pair-feature interactions in the arXiv:1602.02068v2, Eq. (1) input. -/
theorem prototypeKernel_two_example : prototypeTrainingKernel (fun r : Fin 2 => r)
    prototypeExampleFeatures = Matrix.of (fun r s => if r = s then
      (if r = 0 then 3 else 18) else 5) := by
  ext r s
  fin_cases r <;> fin_cases s <;>
    norm_num [prototypeTrainingKernel, prototypeKernelScores, prototypeExampleFeatures,
      Fin.sum_univ_one]

/-- The example's exact-observation component supplies nonsingularity at small feature width.
Source: the concrete derived arXiv:1602.02068v2, Eq. (1) prototype kernel. -/
theorem prototypeKernel_two_det :
    (prototypeTrainingKernel (fun r : Fin 2 => r) prototypeExampleFeatures).det = 29 := by
  rw [prototypeKernel_two_example]
  rw [Matrix.det_fin_two]
  norm_num

end Transformer.GPTMini.Sparsemax
