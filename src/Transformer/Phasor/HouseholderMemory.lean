module

public import Transformer.Phasor.SharedBilinearPhase
public import Transformer.Phasor.ReflectionPhaseRead

/-!
# Noncommuting Householder reflections on learned embedding projections

Source: “Householder memory” in `experiments/phasor_value_transport/README.md`.
Each channel carries one vector. A dense embedding supplies the free
coordinates of a hyperplane normal whose first coordinate is one. The next
state reflects the old state across that hyperplane. All input and output
projections and the initial states are learned. No inverse square root is needed.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

variable {D d H F r m : ℕ}

/-- Appending learned slopes to a leading one gives a nonsingular plane
normal for every point in the dense input space, including zero. -/
def householderNormal (x : Fin r → ℝ) : Fin (r+1) → ℝ := Fin.cons 1 x

/-- The squared norm of the chart normal is always positive. -/
theorem householderNormal_pos (x : Fin r → ℝ) :
    0 < householderNormal x ⬝ᵥ householderNormal x := by
  have h := Finset.sum_nonneg (s := Finset.univ) (f := fun i => x i * x i)
    (fun i _ => mul_self_nonneg (x i))
  simpa [householderNormal, dotProduct, Fin.sum_univ_succ] using (by linarith :
    0 < 1 + ∑ i, x i * x i)

/-- Coordinate form of a Householder reflection with an unnormalized normal. -/
noncomputable def householderStep (n s : Fin r → ℝ) : Fin r → ℝ :=
  s - (2 * (n ⬝ᵥ s) / (n ⬝ᵥ n)) • n

/-- The transpose action on a state cotangent is the same reflection. -/
theorem householderStep_adjoint (n s g : Fin r → ℝ) :
    householderStep n s ⬝ᵥ g = s ⬝ᵥ householderStep n g := by
  simp only [householderStep, sub_dotProduct, dotProduct_sub, dotProduct_smul,
    smul_dotProduct, smul_eq_mul, dotProduct_comm s n]
  ring

/-- Repeating any normal restores the complete incoming state. -/
theorem householderStep_involution (n s : Fin r → ℝ) :
    householderStep n (householderStep n s) = s := by
  by_cases h : n ⬝ᵥ n = 0
  · simp [householderStep, h]
  have flip : n ⬝ᵥ householderStep n s = -(n ⬝ᵥ s) := by
    simp only [householderStep, dotProduct_sub, dotProduct_smul, smul_eq_mul]
    field_simp
    ring
  ext k
  change householderStep n s k - (2 * (n ⬝ᵥ householderStep n s) / (n ⬝ᵥ n)) * n k = s k
  rw [flip]
  simp only [householderStep, Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- Exact squared state norm at any step for every normal. -/
theorem householderStep_norm (n s : Fin r → ℝ) :
    householderStep n s ⬝ᵥ householderStep n s = s ⬝ᵥ s := by
  rw [householderStep_adjoint, householderStep_involution]

/-- Differentiate the actual scalarized reflection with respect to all
normal coordinates. The last term accounts for the denominator's derivative. -/
theorem householderStep_normal_deriv (n s g dn : Fin r → ℝ) (h : n ⬝ᵥ n ≠ 0) :
    let u := (n ⬝ᵥ s) / (n ⬝ᵥ n)
    let v := (n ⬝ᵥ g) / (n ⬝ᵥ n)
    HasDerivAt (fun t : ℝ => householderStep (fun k => n k+t*dn k) s ⬝ᵥ g)
      (-2 * (u * (dn ⬝ᵥ g) + v * (dn ⬝ᵥ s) - 2*u*v*(dn ⬝ᵥ n))) 0 := by
  dsimp only
  let atT := fun (t : ℝ) k => n k+t*dn k
  have lin (x : Fin r → ℝ) : HasDerivAt (fun t => atT t ⬝ᵥ x) (dn ⬝ᵥ x) 0 := by
    simpa [atT] using sharedBilinear_frequency_deriv n x dn 0
  have den := sharedBilinear_frequency_deriv n n dn dn
  have quotient := (((lin s).fun_mul (lin g)).const_mul 2).fun_div den (by simpa [atT] using h)
  convert! quotient.const_sub (s ⬝ᵥ g) using 1
  · funext t
    simp only [householderStep, sub_dotProduct, smul_dotProduct, smul_eq_mul, atT]
    ring
  · simp [atT, dotProduct_comm n dn]
    field_simp
    ring

example : (householderNormal (![(1 : ℝ)]) ⬝ᵥ householderNormal ![1]) ≠ 0 :=
  (householderNormal_pos _).ne'

/-- Chronological scan over arbitrary normal slopes starting at any vector. -/
noncomputable def householderScan (a : ℕ → Fin r → ℝ) (s : Fin (r+1) → ℝ) :
    ℕ → Fin (r+1) → ℝ
  | 0 => s
  | i+1 => householderStep (householderNormal (a i)) (householderScan a s i)

/-- The complete prefix preserves its learned initial squared norm. -/
theorem householderScan_norm (a : ℕ → Fin r → ℝ) (s : Fin (r+1) → ℝ) (i : ℕ) :
    householderScan a s i ⬝ᵥ householderScan a s i = s ⬝ᵥ s := by
  induction i with
  | zero => rfl
  | succ i ih => rw [householderScan, householderStep_norm, ih]

/-- Changing every unobserved slope leaves the current vector unchanged. -/
theorem householderScan_congr_prefix (a b : ℕ → Fin r → ℝ) (s : Fin (r+1) → ℝ) (i : ℕ)
    (h : ∀ j < i, a j = b j) : householderScan a s i = householderScan b s i := by
  induction i with
  | zero => rfl
  | succ i ih => rw [householderScan, householderScan, h i (by omega), ih (fun j hj => h j (by omega))]

example (a filler : ℕ → Fin 3 → ℝ) (s : Fin 4 → ℝ) :
    householderScan a s 15 = householderScan (fun j => if j < 15 then a j else filler j) s 15 :=
  householderScan_congr_prefix _ _ _ _ (fun j hj => by simp [hj])

/-- Two identical embeddings cancel in every channel, for any learned
projection, insertion position and subsequent suffix, including after rewrites. -/
theorem householderScan_insert_pair {α : Type*} (project : α → Fin r → ℝ) (x : ℕ → α)
    (v : α) (s : Fin (r+1) → ℝ) (n k : ℕ) :
    householderScan (fun j => project (insertIdenticalPair x v n j)) s (n+2+k) =
      householderScan (fun j => project (x j)) s (n+k) := by
  induction k with
  | zero =>
    have eq := householderScan_congr_prefix
      (fun j => project (insertIdenticalPair x v n j)) (fun j => project (x j)) s n
      (fun j hj => by simp [insertIdenticalPair, hj])
    simp only [Nat.add_zero, householderScan]
    simpa [insertIdenticalPair, householderStep_involution] using eq
  | succ k ih =>
    change householderStep (householderNormal (project (insertIdenticalPair x v n (n+2+k))))
      (householderScan (fun j => project (insertIdenticalPair x v n j)) s (n+2+k)) =
      householderStep (householderNormal (project (x (n+k)))) (householderScan (fun j => project (x j)) s (n+k))
    rw [ih]
    simp [insertIdenticalPair, show ¬n+2+k < n by omega, show ¬n+2+k < n+2 by omega,
      show n+2+k-2 = n+k by omega]

/-- A finite witness that two attainable learned planes do not commute.
Reversing their order sends the first coordinate axis to opposite vectors. -/
theorem householderStep_order_witness :
    let a := householderNormal (![(0 : ℝ)])
    let b := householderNormal (![(1 : ℝ)])
    let e := ![(1 : ℝ),0]
    householderStep b (householderStep a e) 1 = 1 ∧
      householderStep a (householderStep b e) 1 = -1 := by
  norm_num [householderStep, householderNormal, dotProduct, Fin.sum_univ_succ]

/-- Shared input projection, trainable initial states and one output map. -/
structure HouseholderRead (D r m : ℕ) where
  slopes : EucSpace D →ₗ[ℝ] (Fin m → Fin r → ℝ)
  initial : Fin m → Fin (r+1) → ℝ
  output : (Fin m → Fin (r+1) → ℝ) →ₗ[ℝ] EucSpace D

/-- Reflect the learned state in each channel and return the centered
states to the contextual stream at every position. -/
noncomputable def EmbeddingBackbone.householderFeature (A : EmbeddingBackbone D d H F)
    (P : HouseholderRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let slope := fun k j => P.slopes (GPTMini.rmsNormEps eps (x j)) k
  A.feature θ eps scale x l i + P.output (fun k => householderScan (slope k) (P.initial k) (i+1) - P.initial k)

/-- A zero output projection recovers the complete stream backbone. -/
theorem EmbeddingBackbone.householderFeature_zero (A : EmbeddingBackbone D d H F)
    (P : HouseholderRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x : ℕ → EucSpace D) (l i : ℕ) : A.householderFeature {P with output := 0} θ eps scale x l i =
      A.feature θ eps scale x l i := by simp [EmbeddingBackbone.householderFeature]

/-- The reflected and contextual paths both respect the dense prefix. -/
theorem EmbeddingBackbone.householderFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : HouseholderRead D r m) (θ : StreamReadWeights D) (eps scale : ℝ)
    (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.householderFeature P θ eps scale x l i = A.householderFeature P θ eps scale y l i := by
  have same (k : Fin m) := householderScan_congr_prefix
    (fun j => P.slopes (GPTMini.rmsNormEps eps (x j)) k)
    (fun j => P.slopes (GPTMini.rmsNormEps eps (y j)) k) (P.initial k) (i+1)
    (fun j hj => by rw [h j (by omega)])
  simp only [EmbeddingBackbone.householderFeature, EmbeddingBackbone.feature, same,
    A.states_congr_prefix eps scale x y l i h]

example (A : EmbeddingBackbone 128 32 4 171) (P : HouseholderRead 128 3 16) (θ : StreamReadWeights 128)
    (x filler : ℕ → EucSpace 128) (l : ℕ) : A.householderFeature P θ 1 1 x l 12 =
      A.householderFeature P θ 1 1 (fun j => if j ≤ 12 then x j else filler j) l 12 :=
  A.householderFeature_congr_prefix _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
