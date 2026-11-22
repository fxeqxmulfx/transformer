module

public import Transformer.Phasor.EmbeddingInput
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Learned phase memory on contextual embedding vectors

Source: the learned phase memory in `experiments/phasor_value_transport/README.md`.
A learned linear projection supplies the phase increment in each channel. The
read after a prefix is its sine and centered cosine, mixed back into the
contextual vector by two learned projections. Each sequence starts at phase
zero. All positions share the same three projections.

Segment concatenation rotates the current pair. Any earlier change of phase
has the same chord length after an arbitrary common suffix. The reverse pass
adds the sine/cosine cotangent over all subsequent reads.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- The angle accumulated after the first `n` projected contextual vectors. -/
noncomputable def prefixPhase (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ j ∈ Finset.range n, a j

/-- Processing a segment after a prefix rotates its two phase coordinates by
the total increment of the segment, at every prefix and segment length. -/
theorem prefixPhase_segment (a : ℕ → ℝ) (n m : ℕ) :
    let u := prefixPhase a n
    let v := prefixPhase (fun i => a (n+i)) m
    (Real.sin (prefixPhase a (n+m)), Real.cos (prefixPhase a (n+m))) =
      (Real.sin u * Real.cos v + Real.cos u * Real.sin v,
       Real.cos u * Real.cos v - Real.sin u * Real.sin v) := by
  dsimp only
  unfold prefixPhase
  rw [Finset.sum_range_add, Real.sin_add, Real.cos_add]

/-- An earlier phase difference has the same Euclidean chord after any
common subsequent angle. Both periodic coordinates are observed together. -/
theorem phase_pair_distance (a delta : ℝ) :
    (Real.sin (a+delta) - Real.sin a)^2 + (Real.cos (a+delta) - Real.cos a)^2 =
      2 - 2 * Real.cos delta := by
  have dot : Real.sin (a+delta) * Real.sin a + Real.cos (a+delta) * Real.cos a =
      Real.cos delta := by
    rw [Real.sin_add, Real.cos_add]
    linear_combination Real.cos delta * Real.sin_sq_add_cos_sq a
  linear_combination Real.sin_sq_add_cos_sq (a+delta) + Real.sin_sq_add_cos_sq a - 2 * dot

/-- Differentiating the two observed coordinates gives one local phase
cotangent for an arbitrary incoming pair of derivatives. -/
theorem phase_pair_adjoint (a da gs gc : ℝ) :
    HasDerivAt (fun t : ℝ => gs * Real.sin (a+t*da) + gc * (Real.cos (a+t*da)-1))
      ((gs * Real.cos a - gc * Real.sin a) * da) 0 := by
  have line : HasDerivAt (fun t : ℝ => a+t*da) da 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const da).const_add a
  have hs := (Real.hasDerivAt_sin (a+0*da)).comp 0 line
  have hc := (Real.hasDerivAt_cos (a+0*da)).comp 0 line
  convert! (hs.const_mul gs).add ((hc.sub_const 1).const_mul gc) using 1
  simp
  ring

/-- Transposing the inclusive sum sends every local phase cotangent to its
own increment and every earlier increment of the same sequence. -/
theorem phase_prefix_adjoint {n : ℕ} (d g : Fin n → ℝ) :
    ∑ i, g i * (∑ j, if j ≤ i then d j else 0) =
      ∑ j, (∑ i, if j ≤ i then g i else 0) * d j := by
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  split_ifs <;> simp

/-- One learned set of phase increments and two learned output projections.
The memory width is independent of the external vocabulary. -/
structure PhasePrefixRead (D m : ℕ) where
  increments : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  sineOut : (Fin m → ℝ) →ₗ[ℝ] EucSpace D
  cosineOut : (Fin m → ℝ) →ₗ[ℝ] EucSpace D

/-- At position `i` read the phase accumulated through that position.
Centering the cosine makes both residual coordinates zero at zero phase. -/
noncomputable def PhasePrefixRead.feature {D m : ℕ} (P : PhasePrefixRead D m)
    (h : ℕ → EucSpace D) (i : ℕ) : EucSpace D :=
  let a := fun k => prefixPhase (fun j => P.increments (h j) k) (i+1)
  h i + (P.sineOut (fun k => Real.sin (a k)) + P.cosineOut (fun k => Real.cos (a k)-1))

/-- With both output projections zero the hidden vector is exactly the
original contextual stream read at every position for every input. -/
theorem PhasePrefixRead.feature_zero_output {D m : ℕ} (P : PhasePrefixRead D m)
    (h : ℕ → EucSpace D) (i : ℕ) :
    ({P with sineOut := 0, cosineOut := 0} : PhasePrefixRead D m).feature h i = h i := by
  simp [PhasePrefixRead.feature]

/-- Suffix changes leave the phase read unchanged on arbitrary contextual
vectors, even if the inputs differ at every later position. -/
theorem PhasePrefixRead.feature_congr_prefix {D m : ℕ} (P : PhasePrefixRead D m)
    (x y : ℕ → EucSpace D) (i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    P.feature x i = P.feature y i := by
  have angle (k : Fin m) : prefixPhase (fun j => P.increments (x j) k) (i+1) =
      prefixPhase (fun j => P.increments (y j) k) (i+1) := by
    apply Finset.sum_congr rfl
    intro j hj
    change P.increments (x j) k = P.increments (y j) k
    exact congrArg (fun v => P.increments v k) (h j (Nat.le_of_lt_succ (Finset.mem_range.mp hj)))
  simp only [PhasePrefixRead.feature, angle, h i (by omega)]

example (P : PhasePrefixRead 128 32) (x fill : ℕ → EucSpace 128) (i : ℕ) :
    P.feature x i = P.feature (fun j => if j ≤ i then x j else fill j) i :=
  P.feature_congr_prefix _ _ _ (fun j hj => by simp [hj])

/-- The complete composition with the learned stream collapse at arbitrary
backbone depth is causal as a map on sequences of dense input vectors. -/
theorem EmbeddingBackbone.phaseFeature_congr_prefix {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    P.feature (A.feature θ eps scale x l) i = P.feature (A.feature θ eps scale y l) i := by
  apply P.feature_congr_prefix
  intro j hj
  unfold EmbeddingBackbone.feature
  rw [A.states_congr_prefix eps scale x y l j (fun k hk => h k (Nat.le_trans hk hj))]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    P.feature (A.feature θ 1 1 x l) 7 =
      P.feature (A.feature θ 1 1 (fun j => if j ≤ 7 then x j else fill j) l) 7 :=
  A.phaseFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
