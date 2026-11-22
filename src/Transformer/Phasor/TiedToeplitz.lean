module

public import Transformer.Phasor.PrefixToeplitz
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# An adjoint-tied channel projection for the prefix-conditioned Toeplitz read

Source: “Tied Toeplitz read” in `experiments/phasor_value_transport/README.md`.
One real-linear weight matrix maps an embedding into complex channels. Its
adjoint takes the transformed channels back to the external feature space.
The same weights participate on both sides of every conditional operator.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F m : ℕ}

/-- The Euclidean channel bank pairs real and imaginary components in the
real inner product. This is the native matrix with twice as many rows. -/
abbrev ToeplitzChannels (m : ℕ) := EuclideanSpace ℂ (Fin m)

/-- Tying removes the independent output matrix. All three conditional
controls and the current-value matrix remain freely learned. -/
structure TiedToeplitzRead (D m : ℕ) where
  value : EucSpace D →ₗ[ℝ] ToeplitzChannels m
  kernel : EucSpace D →ₗ[ℝ] (ℤ → ℝ)
  gain : EucSpace D →ₗ[ℝ] (Fin m → ℂ)
  bias : EucSpace D →ₗ[ℝ] (Fin m → ℂ)

/-- Exact inclusion into the untied architecture by selecting the transpose
of the two stacked real projections as the output weights. -/
noncomputable def TiedToeplitzRead.asRead (P : TiedToeplitzRead D m) : PrefixToeplitzRead D m :=
  let coord := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin m => ℂ)
  { value := coord.toLinearMap.comp P.value
    kernel := P.kernel
    gain := P.gain
    bias := P.bias
    output := (LinearMap.adjoint P.value).comp coord.symm.toLinearMap }

/-- The external dot product for any class vector can be evaluated in the
same learned channel basis as the current input. No classifier or class
vector is involved in constructing the inner feature. -/
theorem TiedToeplitzRead.score (P : TiedToeplitzRead D m) (q h x : EucSpace D) :
    inner ℝ q (P.asRead.read h x) =
      inner ℝ (P.value q) (WithLp.toLp 2
        (toeplitzAct (fun δ => (if δ = 0 then 1 else 0) + P.kernel h δ)
          (fun j => 1 + P.gain h j) (P.bias h) (P.value x).ofLp)) := by
  apply LinearMap.adjoint_inner_right

/-- At an empty prefix, every score is a shared Gram entry. This holds
for arbitrary vectors, including those outside a vocabulary table. -/
theorem TiedToeplitzRead.empty_score (P : TiedToeplitzRead D m) (q x : EucSpace D) :
    inner ℝ q (P.asRead.read 0 x) = inner ℝ (P.value q) (P.value x) := by
  rw [P.asRead.read_zero]
  exact LinearMap.adjoint_inner_right _ _ _

/-- Exchanging the external query and current normalized vector cannot
change an empty-prefix score. -/
theorem TiedToeplitzRead.empty_symmetric (P : TiedToeplitzRead D m) (q x : EucSpace D) :
    inner ℝ q (P.asRead.read 0 x) = inner ℝ x (P.asRead.read 0 q) := by
  rw [P.empty_score, P.empty_score, real_inner_comm]

/-- Every finite matrix of empty-prefix scores is positive semidefinite.
The whole quadratic form equals the squared norm of one weighted channel sum. -/
theorem TiedToeplitzRead.empty_gram (P : TiedToeplitzRead D m) {s : ℕ}
    (q : Fin s → EucSpace D) (a : Fin s → ℝ) :
    (∑ i, ∑ j, a i * a j * inner ℝ (q i) (P.asRead.read 0 (q j))) =
      ‖∑ i, a i • P.value (q i)‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [P.empty_score, sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [real_inner_comm (P.value (q i)) (P.value (q j))]
  ring

/-- A perturbation of the shared matrix reaches both occurrences. For any
fixed affine channel action the exact change has two first-order terms and
one quadratic remainder. This includes the action selected by a strict prefix. -/
theorem tied_channel_perturbation
    (W U : EucSpace D →ₗ[ℝ] ToeplitzChannels m)
    (M : ToeplitzChannels m →ₗ[ℝ] ToeplitzChannels m)
    (b : ToeplitzChannels m) (q x : EucSpace D) (t : ℝ) :
    inner ℝ q ((LinearMap.adjoint (W+t•U)) (M ((W+t•U) x) + b)) -
      inner ℝ q ((LinearMap.adjoint W) (M (W x) + b)) =
      t * (inner ℝ (U q) (M (W x)+b) + inner ℝ (W q) (M (U x))) +
        t^2 * inner ℝ (U q) (M (U x)) := by
  simp only [LinearMap.adjoint_inner_right, LinearMap.add_apply, LinearMap.smul_apply,
    map_add, map_smul, inner_add_right, real_inner_smul_right]
  ring

/-- Tying does not introduce any new dependency on the future. -/
theorem EmbeddingBackbone.tiedToeplitz_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : TiedToeplitzRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.toeplitzFeature P.asRead θ eps scale x l i = A.toeplitzFeature P.asRead θ eps scale y l i :=
  A.toeplitzFeature_congr_prefix _ _ _ _ _ _ _ _ h

example (A : EmbeddingBackbone 128 32 4 171) (P : TiedToeplitzRead 128 128)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.toeplitzFeature P.asRead θ 1 1 x l 41 =
      A.toeplitzFeature P.asRead θ 1 1 (fun j => if j ≤ 41 then x j else fill j) l 41 :=
  A.tiedToeplitz_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
