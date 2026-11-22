module

public import Transformer.Phasor.InputPhaseRead

/-!
# Ordered bilinear contributions to a learned phase

Source: the bilinear phase memory in `experiments/phasor_value_transport/README.md`.
Learned left and right projections of dense inputs contribute their
channelwise product for every ordered pair of positions. A first-order phase
and a local learned read are added before applying the sine and cosine maps.
Two running sums suffice for each channel, at every sequence length.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- Ordered products through a prefix. Only a strictly earlier left
projection can contribute to the right projection at position `j`. -/
noncomputable def orderedPhaseIntegral (left right : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ j ∈ Finset.range n, prefixPhase left j * right j

/-- Processing one more position requires only the earlier left sum and
the accumulated integral. The current left value enters at the next step. -/
theorem orderedPhaseIntegral_succ (left right : ℕ → ℝ) (n : ℕ) :
    orderedPhaseIntegral left right (n+1) =
      orderedPhaseIntegral left right n + prefixPhase left n * right n :=
  Finset.sum_range_succ ..

/-- Concatenating segments adds the two within-segment integrals and the
product of the first segment's left sum and the second segment's right sum. -/
theorem orderedPhaseIntegral_split (left right : ℕ → ℝ) (n m : ℕ) :
    orderedPhaseIntegral left right (n+m) = orderedPhaseIntegral left right n +
      orderedPhaseIntegral (fun j => left (n+j)) (fun j => right (n+j)) m +
      prefixPhase left n * prefixPhase (fun j => right (n+j)) m := by
  induction m with
  | zero => simp [orderedPhaseIntegral, prefixPhase]
  | succ m ih =>
    rw [Nat.add_succ, orderedPhaseIntegral_succ, ih, orderedPhaseIntegral_succ]
    rw [show prefixPhase left (n+m) = prefixPhase left n +
      prefixPhase (fun j => left (n+j)) m from Finset.sum_range_add ..]
    rw [show prefixPhase (fun j => right (n+j)) (m+1) = _ from Finset.sum_range_succ ..]
    simp only [prefixPhase]
    ring

/-- The integral is linear in each projection separately. Simultaneous
perturbations also have an exact second-order cross term at any length. -/
theorem orderedPhaseIntegral_perturb (a b da db : ℕ → ℝ) (t : ℝ) (n : ℕ) :
    orderedPhaseIntegral (fun j => a j+t*da j) (fun j => b j+t*db j) n =
      orderedPhaseIntegral a b n + t*(orderedPhaseIntegral da b n + orderedPhaseIntegral a db n) +
      t^2 * orderedPhaseIntegral da db n := by
  induction n with
  | zero => simp [orderedPhaseIntegral]
  | succ n ih =>
    simp only [orderedPhaseIntegral_succ, ih]
    have angleSum : prefixPhase (fun j => a j+t*da j) n =
        prefixPhase a n + t*prefixPhase da n := by
      simp [prefixPhase, Finset.sum_add_distrib, Finset.mul_sum]
    rw [angleSum]
    ring

/-- Agreement through the first `n` positions suffices for equality of the
ordered integral even when the two sequences differ everywhere afterward. -/
theorem orderedPhaseIntegral_congr_prefix (a b c d : ℕ → ℝ) (n : ℕ)
    (h : ∀ j < n, a j = c j ∧ b j = d j) :
    orderedPhaseIntegral a b n = orderedPhaseIntegral c d n := by
  apply Finset.sum_congr rfl
  intro j hj
  have hjn := Finset.mem_range.mp hj
  have earlier : prefixPhase a j = prefixPhase c j := by
    apply Finset.sum_congr rfl
    intro k hk
    exact (h k (Nat.lt_trans (Finset.mem_range.mp hk) hjn)).1
  rw [earlier, (h j hjn).2]

example (a b fill : ℕ → ℝ) (n : ℕ) :
    orderedPhaseIntegral a b n =
      orderedPhaseIntegral (fun j => if j < n then a j else fill j)
        (fun j => if j < n then b j else fill j) n :=
  orderedPhaseIntegral_congr_prefix _ _ _ _ _ (fun j hj => by simp [hj])

/-- Three learned maps on the same normalized representation. The transient
read only affects its current output; the two other maps write the integral. -/
structure BilinearPhaseRead (D m : ℕ) where
  left : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  right : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  transient : EucSpace D →ₗ[ℝ] (Fin m → ℝ)

/-- The full phase adds three terms: the ordinary first-order phase, the
ordered integral and the transient projection of the current embedding. -/
noncomputable def EmbeddingBackbone.bilinearPhaseFeature {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (B : BilinearPhaseRead D m)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let v := fun j => GPTMini.rmsNormEps eps (x j)
  let angle := fun k => P.inputAngles eps x (i+1) k +
    orderedPhaseIntegral (fun j => B.left (v j) k) (fun j => B.right (v j) k) (i+1) +
    B.transient (v i) k
  A.feature θ eps scale x l i + P.angleResidual angle

/-- A zero right and transient map recover the previously trained input-phase
model for every left map, including with nonzero output projections. -/
theorem EmbeddingBackbone.bilinearPhaseFeature_zero {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (B : BilinearPhaseRead D m)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.bilinearPhaseFeature P {B with right := 0, transient := 0} θ eps scale x l i =
      A.inputPhaseFeature P θ eps scale x l i := by
  simp [EmbeddingBackbone.bilinearPhaseFeature, EmbeddingBackbone.inputPhaseFeature,
    orderedPhaseIntegral]

/-- At any depth the complete bilinear-phase read depends only on the dense
embeddings observed by that position. -/
theorem EmbeddingBackbone.bilinearPhaseFeature_congr_prefix {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (B : BilinearPhaseRead D m)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ)
    (h : ∀ j ≤ i, x j = y j) :
    A.bilinearPhaseFeature P B θ eps scale x l i = A.bilinearPhaseFeature P B θ eps scale y l i := by
  have angles : P.inputAngles eps x (i+1) = P.inputAngles eps y (i+1) := by
    funext k
    unfold PhasePrefixRead.inputAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change P.increments (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt_succ (Finset.mem_range.mp hj))]
  have pairs (k : Fin m) := orderedPhaseIntegral_congr_prefix
    (fun j => B.left (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => B.right (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => B.left (GPTMini.rmsNormEps eps (y j)) k)
    (fun j => B.right (GPTMini.rmsNormEps eps (y j)) k) (i+1)
    (fun j hj => by rw [h j (by omega)]; exact ⟨rfl, rfl⟩)
  simp only [EmbeddingBackbone.bilinearPhaseFeature, EmbeddingBackbone.feature,
    A.states_congr_prefix eps scale x y l i h, angles, pairs, h i (by omega)]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (B : BilinearPhaseRead 128 32) (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128)
    (l : ℕ) :
    A.bilinearPhaseFeature P B θ 1 1 x l 9 =
      A.bilinearPhaseFeature P B θ 1 1 (fun j => if j ≤ 9 then x j else fill j) l 9 :=
  A.bilinearPhaseFeature_congr_prefix _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- Reverse differentiation of one scan step. The transient read and the
stored phase receive different cotangents. The earlier left sum receives
both its carried gradient and the new phase gradient times the right input. -/
theorem bilinearPhase_step_adjoint (p s a l r h dp ds da dl dr dh gp gs gt : ℝ) :
    HasDerivAt (fun t : ℝ =>
      let phase := (s+t*ds)+(a+t*da)+(p+t*dp)*(r+t*dr)
      gp*((p+t*dp)+(l+t*dl)) + gs*phase + gt*(phase+(h+t*dh)))
      ((gp+(gs+gt)*r)*dp + gp*dl + (gs+gt)*(ds+da) + (gs+gt)*p*dr + gt*dh) 0 := by
  have line (x dx : ℝ) : HasDerivAt (fun t : ℝ => x+t*dx) dx 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dx).const_add x
  have phase := ((line s ds).add (line a da)).add ((line p dp).mul (line r dr))
  have left := ((line p dp).add (line l dl)).const_mul gp
  convert! (left.add (phase.const_mul gs)).add ((phase.add (line h dh)).const_mul gt) using 1
  simp
  ring

end Transformer.Phasor
