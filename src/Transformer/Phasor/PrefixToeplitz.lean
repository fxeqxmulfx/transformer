module

public import Transformer.Phasor.EmbeddingInput
public import Mathlib.Basic.Complex.Basic

/-!
# A prefix-conditioned Toeplitz operator on the next embedding

Source: “Prefix-conditioned Toeplitz read” in `experiments/phasor_value_transport/README.md`.
The strict prefix chooses a real convolution stencil, a complex diagonal and a
complex bias. They act on the learned channels of the current embedding. The
channel index is internal to the model and independent of the input vocabulary.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F m : ℕ}

/-- Finite Toeplitz multiplication, with zero extension outside the channel bank.
Only the difference between internal channel positions selects a coefficient. -/
noncomputable def toeplitzMix (c : ℤ → ℝ) (v : Fin m → ℂ) (j : Fin m) : ℂ :=
  ∑ k : Fin m, (c ((k.val : ℤ) - j.val) : ℂ) * v k

/-- A prefix chooses the stencil, complex gains and offsets for the next read. -/
noncomputable def toeplitzAct (c : ℤ → ℝ) (gain bias v : Fin m → ℂ) (j : Fin m) : ℂ :=
  bias j + gain j * toeplitzMix c v j

/-- For a fixed prefix the action on current channel values is affine.
In particular all current-input interactions beyond first order vanish. -/
theorem toeplitzAct_parallelogram (c : ℤ → ℝ) (g b u v w : Fin m → ℂ) (j : Fin m) :
    toeplitzAct c g b (u+v+w) j - toeplitzAct c g b (u+v) j =
      toeplitzAct c g b (u+w) j - toeplitzAct c g b u j := by
  simp only [toeplitzAct, toeplitzMix, Pi.add_apply, mul_add, Finset.sum_add_distrib]
  ring

/-- Transposing the value map reverses its stencil offsets and multiplies by
the channel gain. Complex covectors here pair bilinearly with complex values. -/
theorem toeplitzMix_transpose (c : ℤ → ℝ) (g q v : Fin m → ℂ) :
    (∑ j, q j * (g j * toeplitzMix c v j)) =
      ∑ k, (∑ j, q j * g j * (c ((k.val : ℤ)-j.val) : ℂ)) * v k := by
  simp only [toeplitzMix, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- A stencil supported on two offsets realizes a cyclic shift of an active
channel interval. The interval may use fewer channels than the whole bank. -/
def intervalStencil (n t : ℕ) (δ : ℤ) : ℝ :=
  if δ = t ∨ δ = (t : ℤ)-n then 1 else 0

/-- Every cyclic shift of every nonempty supported interval fits in a single
Toeplitz layer. The rest of the input channel bank may be padded with zeros. -/
theorem toeplitzMix_interval (v : Fin m → ℂ) (n t : ℕ) (j k : Fin m)
    (hj : j.val < n) (ht : t < n)
    (hk : k.val = if j.val+t < n then j.val+t else j.val+t-n)
    (hpad : ∀ i : Fin m, n ≤ i.val → v i = 0) :
    toeplitzMix (intervalStencil n t) v j = v k := by
  unfold toeplitzMix
  have hit : (k.val : ℤ)-j.val = t ∨ (k.val : ℤ)-j.val = (t : ℤ)-n := by
    split at hk <;> omega
  rw [Fintype.sum_eq_single k]
  · simp [intervalStencil, hit]
  · intro i hi
    by_cases h : n ≤ i.val
    · simp [hpad i h]
    · have miss : ¬((i.val : ℤ)-j.val = t ∨ (i.val : ℤ)-j.val = (t : ℤ)-n) := by
        intro he
        apply hi
        apply Fin.ext
        split at hk <;> omega
      simp [intervalStencil, miss]

example (v : Fin 16 → ℂ) (h : ∀ i : Fin 16, 11 ≤ i.val → v i = 0) :
    toeplitzMix (intervalStencil 11 4) v 9 = v 2 :=
  toeplitzMix_interval _ _ _ _ _ (by decide) (by decide) (by decide) h

example : toeplitzMix (intervalStencil 3 1) (![2,3,5] : Fin 3 → ℂ) 2 = 2 := by
  apply toeplitzMix_interval (k := (0 : Fin 3)) <;> decide

/-- Learned maps on embedding and context vectors. The native stencil stores
the finite set of offsets that occur in this definition. -/
structure PrefixToeplitzRead (D m : ℕ) where
  value : EucSpace D →ₗ[ℝ] (Fin m → ℂ)
  kernel : EucSpace D →ₗ[ℝ] (ℤ → ℝ)
  gain : EucSpace D →ₗ[ℝ] (Fin m → ℂ)
  bias : EucSpace D →ₗ[ℝ] (Fin m → ℂ)
  output : (Fin m → ℂ) →ₗ[ℝ] EucSpace D

/-- The zero context initializes the convolution and complex diagonal at the
identity. The only input-specific value is the current normalized embedding. -/
noncomputable def PrefixToeplitzRead.read (P : PrefixToeplitzRead D m)
    (h x : EucSpace D) : EucSpace D :=
  P.output (toeplitzAct (fun δ => (if δ = 0 then 1 else 0) + P.kernel h δ)
    (fun j => 1 + P.gain h j) (P.bias h) (P.value x))

/-- Without a previous embedding the affine map reduces exactly to the
learned value and output projections, including off-table inputs. -/
theorem PrefixToeplitzRead.read_zero (P : PrefixToeplitzRead D m) (x : EucSpace D) :
    P.read 0 x = P.output (P.value x) := by
  unfold PrefixToeplitzRead.read
  congr 1
  funext j
  simp only [toeplitzAct, toeplitzMix, map_zero, Pi.zero_apply, add_zero, zero_add, one_mul]
  rw [Fintype.sum_eq_single j]
  · simp
  · intro k hk
    have h : (k.val : ℤ)-j.val ≠ 0 := fun he => hk (Fin.ext (by omega))
    simp [h]

/-- The original Phasor builds the strict prefix control. The separate
current-vector path enters through its own learned complex channels. -/
noncomputable def EmbeddingBackbone.toeplitzFeature (A : EmbeddingBackbone D d H F)
    (P : PrefixToeplitzRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let previous := match i with
    | 0 => 0
    | j+1 => A.feature θ eps scale x l j
  P.read previous (GPTMini.rmsNormEps eps (x i))

/-- All causal prefixes of arbitrary lengths obey the same vector-input
operator. Later embeddings cannot change the selected feature. -/
theorem EmbeddingBackbone.toeplitzFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : PrefixToeplitzRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.toeplitzFeature P θ eps scale x l i = A.toeplitzFeature P θ eps scale y l i := by
  cases i with
  | zero => simp [EmbeddingBackbone.toeplitzFeature, h 0 (by omega)]
  | succ i =>
    simp only [EmbeddingBackbone.toeplitzFeature, EmbeddingBackbone.feature, h (i+1) (by omega),
      A.states_congr_prefix eps scale x y l i fun j hj => h j (by omega)]

example (A : EmbeddingBackbone 128 32 4 171) (P : PrefixToeplitzRead 128 128)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.toeplitzFeature P θ 1 1 x l 19 =
      A.toeplitzFeature P θ 1 1 (fun j => if j ≤ 19 then x j else fill j) l 19 :=
  A.toeplitzFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
