module

public import Transformer.Phasor.PhasePrefixRead

/-!
# A learned phase read directly from normalized input embeddings

Source: the input phase memory in `experiments/phasor_value_transport/README.md`.
The phase branch projects the normalized embedding at each position. Its read
joins the contextual stream feature after the last transformer layer. The same
learned maps apply to every vector in every sequence.

The increment of a fixed segment now has the same value in every preceding
context. Exchanging two adjacent segments preserves the phase contribution.
Repeating a vector gives its learned increment times the repetition count.
The contextual feature still has its full causal dependence on embedding
order. Backpropagation has a direct phase path to the input and a separate
path through all transformer layers.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D m d H F : ℕ}

/-- The first `n` vectors of the left sequence followed by the right sequence.
Only the requested prefixes enter the finite scans below. -/
def joinEmbeddingSegment {α : Type*} (n : ℕ) (x y : ℕ → α) (j : ℕ) : α :=
  if j < n then x j else y (j-n)

/-- Learned phase coordinates through the first `n` vectors. The same input
RMS is used at the entrance to Phasor's four streams. -/
noncomputable def PhasePrefixRead.inputAngles (P : PhasePrefixRead D m)
    (eps : ℝ) (x : ℕ → EucSpace D) (n : ℕ) : Fin m → ℝ :=
  fun k => prefixPhase (fun j => P.increments (GPTMini.rmsNormEps eps (x j)) k) n

/-- Applying a particular segment after any prefix adds that segment's own
phase increment in every channel, independently of everything before it. -/
theorem PhasePrefixRead.inputAngles_join (P : PhasePrefixRead D m) (eps : ℝ)
    (x y : ℕ → EucSpace D) (n s : ℕ) :
    P.inputAngles eps (joinEmbeddingSegment n x y) (n+s) =
      P.inputAngles eps x n + P.inputAngles eps y s := by
  funext k
  simp only [PhasePrefixRead.inputAngles, prefixPhase, Pi.add_apply, Finset.sum_range_add]
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    simp [joinEmbeddingSegment, Finset.mem_range.mp hj]
  · apply Finset.sum_congr rfl
    intro j _
    simp [joinEmbeddingSegment]

/-- Swapping two whole embedding segments preserves the phase coordinates
at the end of both, including when the segment lengths differ. -/
theorem PhasePrefixRead.inputAngles_swap_segments (P : PhasePrefixRead D m) (eps : ℝ)
    (x y : ℕ → EucSpace D) (n s : ℕ) :
    P.inputAngles eps (joinEmbeddingSegment n x y) (n+s) =
      P.inputAngles eps (joinEmbeddingSegment s y x) (s+n) := by
  rw [P.inputAngles_join, P.inputAngles_join, add_comm]

/-- Repeating any dense vector after an arbitrary prefix gives exactly the
prefix phase plus `s` copies of that vector's learned increment. -/
theorem PhasePrefixRead.inputAngles_repeat (P : PhasePrefixRead D m) (eps : ℝ)
    (x : ℕ → EucSpace D) (v : EucSpace D) (n s : ℕ) (k : Fin m) :
    P.inputAngles eps (joinEmbeddingSegment n x (fun _ => v)) (n+s) k =
      P.inputAngles eps x n k + s * P.increments (GPTMini.rmsNormEps eps v) k := by
  rw [P.inputAngles_join]
  simp [PhasePrefixRead.inputAngles, prefixPhase]

/-- The two learned output projections of an arbitrary phase vector. -/
noncomputable def PhasePrefixRead.angleResidual (P : PhasePrefixRead D m)
    (angle : Fin m → ℝ) : EucSpace D :=
  P.sineOut (fun k => Real.sin (angle k)) + P.cosineOut (fun k => Real.cos (angle k)-1)

/-- The skip read joins the contextual feature outside the transformer stack.
Both branches receive the same sequence of dense input vectors. -/
noncomputable def EmbeddingBackbone.inputPhaseFeature (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  A.feature θ eps scale x l i + P.angleResidual (P.inputAngles eps x (i+1))

/-- Zero-initialized output matrices recover the learned stream model for
all inputs, depths and settings of the learned phase increment matrix. -/
theorem EmbeddingBackbone.inputPhaseFeature_zero (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) :
    A.inputPhaseFeature {P with sineOut := 0, cosineOut := 0} θ eps scale x l i =
      A.feature θ eps scale x l i := by
  simp [EmbeddingBackbone.inputPhaseFeature, PhasePrefixRead.angleResidual]

/-- Both the input phase and the ordered contextual path depend only on the
embedding prefix up through the selected output position. -/
theorem EmbeddingBackbone.inputPhaseFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.inputPhaseFeature P θ eps scale x l i = A.inputPhaseFeature P θ eps scale y l i := by
  have equal : P.inputAngles eps x (i+1) = P.inputAngles eps y (i+1) := by
    funext k
    unfold PhasePrefixRead.inputAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change P.increments (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt_succ (Finset.mem_range.mp hj))]
  simp only [EmbeddingBackbone.inputPhaseFeature, EmbeddingBackbone.feature, equal,
    A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.inputPhaseFeature P θ 1 1 x l 8 =
      A.inputPhaseFeature P θ 1 1 (fun j => if j ≤ 8 then x j else fill j) l 8 :=
  A.inputPhaseFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- The contextual feature variation and the input-phase angle variation
enter the derivative on separate paths with their own upstream gradients. -/
theorem input_phase_paths_adjoint (h dh a da g gs gc : ℝ) :
    HasDerivAt (fun t : ℝ => g*(h+t*dh) +
      (gs * Real.sin (a+t*da) + gc * (Real.cos (a+t*da)-1)))
      (g*dh + (gs * Real.cos a - gc * Real.sin a)*da) 0 := by
  have line : HasDerivAt (fun t : ℝ => h+t*dh) dh 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dh).const_add h
  exact (line.const_mul g).add (phase_pair_adjoint a da gs gc)

end Transformer.Phasor
