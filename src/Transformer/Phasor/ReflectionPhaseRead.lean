module

public import Transformer.Phasor.InputPhaseRead

/-!
# Learned mixing of rotation and reflection memories

Source: the rotation and reflection memory in `experiments/phasor_value_transport/README.md`.
Each normalized input vector supplies a learned angle. It rotates one state
and reflects a second state. One learned scalar per channel mixes their two
observed coordinates. A pair of identical successive reflections cancels
for every projected direction and every surrounding sequence.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- The angle of a reflected state. Applying a reflection with learned
direction parameter `a j` sends the incoming angle `z` to `a j - z`. -/
def reflectedPhase (a : ℕ → ℝ) (z : ℝ) : ℕ → ℝ
  | 0 => z
  | n+1 => a n - reflectedPhase a z n

/-- Appending a segment adds its own reflected angle and preserves the
incoming magnitude with the orientation determined by its length. -/
theorem reflectedPhase_split (a : ℕ → ℝ) (z : ℝ) (n m : ℕ) :
    reflectedPhase a z (n+m) = (-1 : ℝ)^m * reflectedPhase a z n +
      reflectedPhase (fun j => a (n+j)) 0 m := by
  induction m with
  | zero => simp [reflectedPhase]
  | succ m ih =>
    rw [Nat.add_succ, reflectedPhase, ih, reflectedPhase, pow_succ]
    ring

/-- A common segment transmits every initial perturbation with derivative
exactly `1` or `-1`, regardless of the learned reflection directions. -/
theorem reflectedPhase_suffix_deriv (a : ℕ → ℝ) (z : ℝ) (n : ℕ) :
    HasDerivAt (fun v => reflectedPhase a v n) ((-1 : ℝ)^n) z := by
  induction n with
  | zero => simpa [reflectedPhase] using hasDerivAt_id' z
  | succ n ih => simpa [reflectedPhase, pow_succ] using ih.const_sub (a n)

/-- Reflected angles depend only on the requested embedding prefix. -/
theorem reflectedPhase_congr_prefix (a b : ℕ → ℝ) (z : ℝ) (n : ℕ)
    (h : ∀ j < n, a j = b j) : reflectedPhase a z n = reflectedPhase b z n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [reflectedPhase, reflectedPhase, h n (by omega), ih (fun j hj => h j (by omega))]

example (a fill : ℕ → ℝ) (z : ℝ) (n : ℕ) :
    reflectedPhase a z n = reflectedPhase (fun j => if j < n then a j else fill j) z n :=
  reflectedPhase_congr_prefix _ _ _ _ (fun j hj => by simp [hj])

/-- Insert two occurrences of an arbitrary vector before position `n`. -/
def insertIdenticalPair {α : Type*} (x : ℕ → α) (v : α) (n j : ℕ) : α :=
  if j < n then x j else if j < n+2 then v else x (j-2)

/-- Inserting two identical embeddings anywhere leaves the reflection
angle unchanged after any number of further steps, for every learned
projection `project` and every initial state. -/
theorem reflectedPhase_insert_pair {α : Type*} (project : α → ℝ) (x : ℕ → α) (v : α)
    (z : ℝ) (n m : ℕ) :
    reflectedPhase (fun j => project (insertIdenticalPair x v n j)) z (n+2+m) =
      reflectedPhase (fun j => project (x j)) z (n+m) := by
  induction m with
  | zero =>
    have same := reflectedPhase_congr_prefix
      (fun j => project (insertIdenticalPair x v n j)) (fun j => project (x j)) z n
      (fun j hj => by simp [insertIdenticalPair, hj])
    simp only [Nat.add_zero, reflectedPhase]
    simpa [insertIdenticalPair] using same
  | succ m ih =>
    change project (insertIdenticalPair x v n (n+2+m)) -
      reflectedPhase (fun j => project (insertIdenticalPair x v n j)) z (n+2+m) =
      project (x (n+m)) - reflectedPhase (fun j => project (x j)) z (n+m)
    rw [ih]
    simp [insertIdenticalPair, show ¬n+2+m < n by omega, show ¬n+2+m < n+2 by omega,
      show n+2+m-2 = n+m by omega]

/-- The linear interpolation uses the same learned gate for both coordinates
of a channel. Its real value may lie outside the unit interval. -/
def phaseMix (g rotation reflection : ℝ) : ℝ := rotation + g*(reflection-rotation)

/-- Both memories use the same learned projection on every normalized
input embedding. The classifier remains outside this contextual feature. -/
noncomputable def EmbeddingBackbone.reflectionPhaseFeature {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (gate : Fin m → ℝ)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let rotation := P.inputAngles eps x (i+1)
  let reflection := fun k =>
    reflectedPhase (fun j => P.increments (GPTMini.rmsNormEps eps (x j)) k) 0 (i+1)
  A.feature θ eps scale x l i +
    (P.sineOut (fun k => phaseMix (gate k) (Real.sin (rotation k)) (Real.sin (reflection k))) +
     P.cosineOut (fun k => phaseMix (gate k) (Real.cos (rotation k)-1) (Real.cos (reflection k)-1)))

/-- Zero gates give exactly the input-phase feature with the same learned
weights, even for nonzero sine and cosine output maps. -/
theorem EmbeddingBackbone.reflectionPhaseFeature_zero_gate {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.reflectionPhaseFeature P 0 θ eps scale x l i = A.inputPhaseFeature P θ eps scale x l i := by
  simp [EmbeddingBackbone.reflectionPhaseFeature, EmbeddingBackbone.inputPhaseFeature,
    PhasePrefixRead.angleResidual, phaseMix]

/-- Causality of the complete mixed read on arbitrary embedding sequences. -/
theorem EmbeddingBackbone.reflectionPhaseFeature_congr_prefix {D d H F m : ℕ}
    (A : EmbeddingBackbone D d H F) (P : PhasePrefixRead D m) (gate : Fin m → ℝ)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ)
    (h : ∀ j ≤ i, x j = y j) :
    A.reflectionPhaseFeature P gate θ eps scale x l i = A.reflectionPhaseFeature P gate θ eps scale y l i := by
  have angles : P.inputAngles eps x (i+1) = P.inputAngles eps y (i+1) := by
    funext k
    unfold PhasePrefixRead.inputAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change P.increments (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt_succ (Finset.mem_range.mp hj))]
  have reflections (k : Fin m) := reflectedPhase_congr_prefix
    (fun j => P.increments (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => P.increments (GPTMini.rmsNormEps eps (y j)) k) 0 (i+1)
    (fun j hj => by rw [h j (by omega)])
  simp only [EmbeddingBackbone.reflectionPhaseFeature, EmbeddingBackbone.feature,
    A.states_congr_prefix eps scale x y l i h, angles, reflections]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (g : Fin 32 → ℝ) (θ : StreamReadWeights 128) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.reflectionPhaseFeature P g θ 1 1 x l 5 =
      A.reflectionPhaseFeature P g θ 1 1 (fun j => if j ≤ 5 then x j else fill j) l 5 :=
  A.reflectionPhaseFeature_congr_prefix _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- Directional derivative of the complete mixed scalar read. A shared
gate receives the difference of the two read values, and sends the two
angle adjoints back in proportions `1-g` and `g`. -/
theorem phaseMix_pullback (a b g da db dg gs gc : ℝ) :
    let read := fun z => gs*Real.sin z + gc*(Real.cos z-1)
    HasDerivAt (fun t : ℝ => phaseMix (g+t*dg) (read (a+t*da)) (read (b+t*db)))
      ((1-g)*(gs*Real.cos a-gc*Real.sin a)*da + g*(gs*Real.cos b-gc*Real.sin b)*db +
        dg*(read b-read a)) 0 := by
  dsimp only [phaseMix]
  have gate : HasDerivAt (fun t : ℝ => g+t*dg) dg 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dg).const_add g
  have left := phase_pair_adjoint a da gs gc
  have right := phase_pair_adjoint b db gs gc
  convert! left.add (gate.mul (right.sub left)) using 1
  simp
  ring

end Transformer.Phasor
