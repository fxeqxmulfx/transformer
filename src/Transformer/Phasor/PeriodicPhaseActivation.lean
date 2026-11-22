module

public import Transformer.Phasor.ReflectionPhaseRead

/-!
# A learned periodic activation on phase increments

Source: the periodic phase activation in `experiments/phasor_value_transport/README.md`.
Each channel has its own trainable strength and center. Both phase scans
consume exactly the same transformed projection of each input embedding.
At unit strength, errors near any lift of the center modulo a half-turn
enter both scans at cubic order. The ordinary mixed model is the zero-
strength specialization.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- An activation on a learned scalar projection. Its sine correction
depends on the displacement from a learned center. -/
noncomputable def periodicPhaseIncrement (g b x : ℝ) : ℝ :=
  x - (g/2)*Real.sin (2*(x-b))

/-- Translating the input by any integer number of half-turns translates
the output by the same amount, for every learned strength and center. -/
theorem periodicPhaseIncrement_lift (g b e : ℝ) (n : ℤ) :
    periodicPhaseIncrement g b (b+n*Real.pi+e) =
      b+n*Real.pi+(e-(g/2)*Real.sin (2*e)) := by
  unfold periodicPhaseIncrement
  rw [show 2*(b+n*Real.pi+e-b) = 2*e+(n:ℝ)*(2*Real.pi) by ring,
    Real.sin_add_int_mul_two_pi]
  ring

/-- Uniform over every lift of the learned center, including negative
ones. The fifth-order term comes from the explicit real sine remainder. -/
theorem periodicPhaseIncrement_cubic (b e : ℝ) (n : ℤ) (small : |e| ≤ 1/2) :
    |periodicPhaseIncrement 1 b (b+n*Real.pi+e) - (b+n*Real.pi)| ≤
      (2/3)*|e|^3 + (4/25)*|e|^5 := by
  have two : |2*e| ≤ 1 := by rw [abs_mul, abs_of_pos (by norm_num : (0:ℝ)<2)]; linarith
  have sine := abs_le.mp (Real.sin_bound two)
  rw [abs_mul, abs_of_pos (by norm_num : (0:ℝ)<2)] at sine
  have cube := abs_le.mp (le_of_eq (abs_pow e 3))
  rw [periodicPhaseIncrement_lift, add_sub_cancel_left, abs_le]
  constructor <;> nlinarith [sine.1, sine.2, cube.1, cube.2]

example (b : ℝ) (n : ℤ) :
    |periodicPhaseIncrement 1 b (b+n*Real.pi+1/4) - (b+n*Real.pi)| ≤
      (2/3)*|(1/4:ℝ)|^3+(4/25)*|(1/4:ℝ)|^5 :=
  periodicPhaseIncrement_cubic _ _ _ (by norm_num)

/-- The activation differentiates the projection, center and strength
simultaneously. All three are trained by the ordinary backward pass. -/
theorem periodicPhaseIncrement_deriv (g b x dg db dx : ℝ) :
    HasDerivAt (fun t : ℝ => periodicPhaseIncrement (g+t*dg) (b+t*db) (x+t*dx))
      ((1-g*Real.cos (2*(x-b)))*dx + g*Real.cos (2*(x-b))*db -
        Real.sin (2*(x-b))*dg/2) 0 := by
  have line (v dv : ℝ) : HasDerivAt (fun t : ℝ => v+t*dv) dv 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dv).const_add v
  have angle := ((line x dx).sub (line b db)).const_mul 2
  have result := (line x dx).sub (((line g dg).div_const 2).mul angle.sin)
  convert result using 1 <;> first | rfl | (simp; ring)

/-- Rotation and reflection are both nonexpansive in the sum of absolute
increment perturbations, for the entire finite prefix. -/
theorem mixedPhase_prefix_error (a b : ℕ → ℝ) (n : ℕ) :
    |prefixPhase a n - prefixPhase b n| ≤ ∑ i ∈ Finset.range n, |a i-b i| ∧
    |reflectedPhase a 0 n - reflectedPhase b 0 n| ≤ ∑ i ∈ Finset.range n, |a i-b i| := by
  induction n with
  | zero => simp [prefixPhase, reflectedPhase]
  | succ n ih =>
    have addStep : prefixPhase a (n+1)-prefixPhase b (n+1) =
        (prefixPhase a n-prefixPhase b n)+(a n-b n) := by
      simp only [prefixPhase, Finset.sum_range_succ]; ring
    have refStep : reflectedPhase a 0 (n+1)-reflectedPhase b 0 (n+1) =
        -(reflectedPhase a 0 n-reflectedPhase b 0 n)+(a n-b n) := by
      simp only [reflectedPhase]; ring
    rw [addStep, refStep, Finset.sum_range_succ]
    exact ⟨(abs_add_le ..).trans (add_le_add_left ih.1 _),
      (abs_add_le ..).trans (by simpa only [abs_neg] using add_le_add_left ih.2 |a n-b n|)⟩

/-- Entire prefixes of embeddings near arbitrary lifts inherit the local
cubic error bound, even when the lifts change at every position. -/
theorem periodicPhase_prefix_cubic (b : ℝ) (k : ℕ → ℤ) (e : ℕ → ℝ) (n : ℕ)
    (small : ∀ j < n, |e j| ≤ 1/2) :
    let target := fun j => b + (k j:ℝ)*Real.pi
    let actual := fun j => periodicPhaseIncrement 1 b (target j+e j)
    let budget := ∑ j ∈ Finset.range n, ((2/3)*|e j|^3+(4/25)*|e j|^5)
    |prefixPhase actual n-prefixPhase target n| ≤ budget ∧
      |reflectedPhase actual 0 n-reflectedPhase target 0 n| ≤ budget := by
  dsimp only
  have total := Finset.sum_le_sum (s := Finset.range n)
    (fun j hj => periodicPhaseIncrement_cubic b (e j) (k j) (small j (Finset.mem_range.mp hj)))
  have both := mixedPhase_prefix_error
    (fun j => periodicPhaseIncrement 1 b (b+(k j:ℝ)*Real.pi+e j))
    (fun j => b+(k j:ℝ)*Real.pi) n
  exact ⟨both.1.trans total, both.2.trans total⟩

example (b : ℝ) (k : ℕ → ℤ) (n : ℕ) := periodicPhase_prefix_cubic b k (fun _ => 1/8) n
  (fun _ _ => by norm_num)

variable {D d H F m : ℕ}

/-- The shared activation feeds the rotation and reflection of every
channel. All projections operate on normalized dense embeddings. -/
noncomputable def EmbeddingBackbone.periodicReflectionFeature (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (mix strength center : Fin m → ℝ) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let step := fun k j => periodicPhaseIncrement (strength k) (center k)
    (P.increments (GPTMini.rmsNormEps eps (x j)) k)
  let rot := fun k => prefixPhase (step k) (i+1)
  let refl := fun k => reflectedPhase (step k) 0 (i+1)
  A.feature θ eps scale x l i +
    (P.sineOut (fun k => phaseMix (mix k) (Real.sin (rot k)) (Real.sin (refl k))) +
     P.cosineOut (fun k => phaseMix (mix k) (Real.cos (rot k)-1) (Real.cos (refl k)-1)))

/-- At zero strength this agrees with the mixed reflection model for
arbitrary centers and already active read projections. -/
theorem EmbeddingBackbone.periodicReflectionFeature_zero (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (mix center : Fin m → ℝ) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.periodicReflectionFeature P mix 0 center θ eps scale x l i =
      A.reflectionPhaseFeature P mix θ eps scale x l i := by
  simp [EmbeddingBackbone.periodicReflectionFeature, EmbeddingBackbone.reflectionPhaseFeature,
    periodicPhaseIncrement, PhasePrefixRead.inputAngles]

/-- Changing any vector beyond the scored position leaves the entire
periodic reflection feature unchanged. -/
theorem EmbeddingBackbone.periodicReflectionFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (mix g b : Fin m → ℝ) (θ : StreamReadWeights D)
    (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.periodicReflectionFeature P mix g b θ eps scale x l i =
      A.periodicReflectionFeature P mix g b θ eps scale y l i := by
  let f := fun k v => periodicPhaseIncrement (g k) (b k) (P.increments (GPTMini.rmsNormEps eps v) k)
  have rot (k : Fin m) : prefixPhase (f k ∘ x) (i+1) = prefixPhase (f k ∘ y) (i+1) :=
    Finset.sum_congr rfl (fun j hj => by simp only [Function.comp_apply, h j (by simpa using hj)])
  have refl (k : Fin m) : reflectedPhase (f k ∘ x) 0 (i+1) = reflectedPhase (f k ∘ y) 0 (i+1) :=
    reflectedPhase_congr_prefix _ _ _ _ (fun j hj => by simp only [Function.comp_apply, h j (by omega)])
  unfold EmbeddingBackbone.periodicReflectionFeature
  change A.feature θ eps scale x l i + (P.sineOut _ + P.cosineOut _) = _
  have eqFeature : A.feature θ eps scale x l i = A.feature θ eps scale y l i := by
    unfold EmbeddingBackbone.feature
    rw [A.states_congr_prefix eps scale x y l i h]
  rw [eqFeature]
  congr 2 <;> congr 1 <;> funext k
  · change phaseMix (mix k) (Real.sin (prefixPhase (f k ∘ x) (i+1)))
      (Real.sin (reflectedPhase (f k ∘ x) 0 (i+1))) =
      phaseMix (mix k) (Real.sin (prefixPhase (f k ∘ y) (i+1)))
        (Real.sin (reflectedPhase (f k ∘ y) 0 (i+1)))
    rw [rot k, refl k]
  · change phaseMix (mix k) (Real.cos (prefixPhase (f k ∘ x) (i+1))-1)
      (Real.cos (reflectedPhase (f k ∘ x) 0 (i+1))-1) =
      phaseMix (mix k) (Real.cos (prefixPhase (f k ∘ y) (i+1))-1)
        (Real.cos (reflectedPhase (f k ∘ y) 0 (i+1))-1)
    rw [rot k, refl k]

example (A : EmbeddingBackbone 64 16 4 86) (P : PhasePrefixRead 64 32)
    (mix g b : Fin 32 → ℝ) (θ : StreamReadWeights 64) (x fill : ℕ → EucSpace 64) :
    A.periodicReflectionFeature P mix g b θ 1 1 x 2 7 =
      A.periodicReflectionFeature P mix g b θ 1 1 (fun j => if j≤7 then x j else fill j) 2 7 :=
  A.periodicReflectionFeature_congr_prefix _ _ _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
