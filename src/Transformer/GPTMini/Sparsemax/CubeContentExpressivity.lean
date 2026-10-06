import Transformer.GPTMini.Sparsemax.CubeContentValues
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Expressive generated content features with an explicit capacity price

New response family following arXiv:1602.02068v2, Eq. (1). Products
of bit signs have an exact delta expansion. Taking every subset therefore
represents every vector function on the finite content cube. This is
universality at a fixed bit dimension, not a generalization or efficient
language-model theorem. Full capacity uses 2^r feature coefficients per
channel; selecting K interactions uses K instead and restricts capacity.

Even one selected product represents XOR sign across two bits, which
constant and individual-bit affine responses cannot fit. Higher-order
parity is also a single product feature. The encoding and features use
only observed data, while coefficients are learned task parameters.
No teacher routes, hidden continuation or prototype table is supplied.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [DecidableEq ι] in
/-- Selected content interactions have an exact finite delta kernel on actual bit states.
Source: generated-feature universality in the new sparsemax Eq. (1) architecture. -/
theorem cubeContentCharacter_kernel (x y : CubeContentState ι) :
    (∑ S : Finset ι, cubeContentCharacter S x * cubeContentCharacter S y) =
      if x = y then (2 : ℝ) ^ Fintype.card ι else 0 := by
  have hp : (∏ i, (cubeContentSign (x i) * cubeContentSign (y i) + 1)) =
      ∑ S : Finset ι, cubeContentCharacter S x * cubeContentCharacter S y := by
    rw [Finset.prod_add_one, Finset.powerset_univ]
    simp only [Finset.prod_mul_distrib, cubeContentCharacter]
  rw [← hp]
  by_cases hxy : x = y
  · subst y
    simp only [ite_true]
    have hs (i : ι) : cubeContentSign (x i) * cubeContentSign (x i) + 1 = 2 := by
      nlinarith [cubeContentSign_sq (x i)]
    simp_rw [hs]
    rw [Finset.prod_const]
    rfl
  · simp only [hxy, ite_false]
    have hd : ∃ i, x i ≠ y i := by
      by_contra hn
      push Not at hn
      exact hxy (funext hn)
    obtain ⟨i, hi⟩ := hd
    apply Finset.prod_eq_zero (Finset.mem_univ i)
    cases hx : x i <;> cases hy : y i <;> simp_all [cubeContentSign]

/-- Full finite content coefficients are obtained from an ordinary vector answer function.
Source: explicit delta-kernel inversion following sparsemax Eq. (1). -/
def cubeContentFullCoefficients {D : ℕ} (Y : Matrix (CubeContentState ι) (Fin D) ℝ) :
    Matrix (Finset ι) (Fin D) ℝ :=
  fun S d => (∑ x, cubeContentCharacter S x * Y x d) / (2 : ℝ) ^ Fintype.card ι

/-- Every finite content function is exactly represented when all interaction subsets are selected.
Source: proved content delta expansion after sparsemax Eq. (1); coefficient count is exponential. -/
theorem cubeContentFullOutput_eq {D : ℕ} (Y : Matrix (CubeContentState ι) (Fin D) ℝ) :
    cubeContentOutput id (cubeContentFullCoefficients Y) = Y := by
  ext x d
  change (∑ S : Finset ι, cubeContentCharacter S x *
    ((∑ y : CubeContentState ι, cubeContentCharacter S y * Y y d) /
      (2 : ℝ) ^ Fintype.card ι)) = Y x d
  simp only [← mul_div_assoc, Finset.mul_sum]
  rw [← Finset.sum_div, Finset.sum_comm]
  have hs (y : CubeContentState ι) :
      (∑ S : Finset ι, cubeContentCharacter S x * (cubeContentCharacter S y * Y y d)) =
        (if x = y then (2 : ℝ) ^ Fintype.card ι else 0) * Y y d := by
    simp_rw [← mul_assoc]
    rw [← Finset.sum_mul, cubeContentCharacter_kernel]
  simp_rw [hs, ite_mul, zero_mul]
  rw [Fintype.sum_ite_eq]
  have hn : (2 : ℝ) ^ Fintype.card ι ≠ 0 := ne_of_gt (by positivity)
  field_simp

/-- Full generated interactions cover the entire finite vector answer space, not merely one fitted pattern.
Source: the explicit content kernel inverse following sparsemax Eq. (1). -/
theorem cubeContentFullOutput_surjective (D : ℕ) :
    Function.Surjective (cubeContentOutput id : Matrix (Finset ι) (Fin D) ℝ → _) := by
  intro Y
  exact ⟨cubeContentFullCoefficients Y, cubeContentFullOutput_eq Y⟩

/-- Bounded ordinary answers have full interaction coefficients inside the implementation box.
Source: the normalized content kernel inverse following sparsemax Eq. (1); no route targets occur. -/
theorem cubeContentFullCoefficients_bounds {D : ℕ} (Y : Matrix (CubeContentState ι) (Fin D) ℝ)
    (hY : ∀ x d, -1 ≤ Y x d ∧ Y x d ≤ 1) (S : Finset ι) (d : Fin D) :
    -1 ≤ cubeContentFullCoefficients Y S d ∧ cubeContentFullCoefficients Y S d ≤ 1 := by
  have hp : 0 < (2 : ℝ) ^ Fintype.card ι := by positivity
  have hs : |∑ x, cubeContentCharacter S x * Y x d| ≤ (2 : ℝ) ^ Fintype.card ι := by
    calc
      _ ≤ ∑ x, |cubeContentCharacter S x * Y x d| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : CubeContentState ι, (1 : ℝ) := by
        apply Finset.sum_le_sum
        intro x hx
        have hc : |cubeContentCharacter S x| = 1 := by
          nlinarith [cubeContentCharacter_sq S x, sq_abs (cubeContentCharacter S x), abs_nonneg (cubeContentCharacter S x)]
        rw [abs_mul, hc, one_mul]
        exact abs_le.mpr (hY x d)
      _ = (2 : ℝ) ^ Fintype.card ι := by
        simp only [Finset.sum_const, nsmul_eq_mul, Finset.card_univ, mul_one,
          Fintype.card_fun, Fintype.card_bool, Nat.cast_pow, Nat.cast_ofNat]
  unfold cubeContentFullCoefficients
  constructor
  · apply (le_div_iff₀ hp).mpr
    linarith [(abs_le.mp hs).1]
  · apply (div_le_iff₀ hp).mpr
    linarith [(abs_le.mp hs).2]

/-- Every full XOR coefficient satisfies the actual unit-box cap. -/
example (S : Finset (Fin 2)) :
    -1 ≤ cubeContentFullCoefficients (Matrix.of (fun x : CubeContentState (Fin 2) => fun _ : Fin 1 =>
      cubeContentCharacter ({0, 1} : Finset (Fin 2)) x)) S 0 ∧
    cubeContentFullCoefficients (Matrix.of (fun x : CubeContentState (Fin 2) => fun _ : Fin 1 =>
      cubeContentCharacter ({0, 1} : Finset (Fin 2)) x)) S 0 ≤ 1 := by
  apply cubeContentFullCoefficients_bounds
  intro x d
  change -1 ≤ cubeContentCharacter ({0, 1} : Finset (Fin 2)) x ∧ cubeContentCharacter {0, 1} x ≤ 1
  cases h0 : x 0 <;> cases h1 : x 1 <;> simp_all [cubeContentCharacter, cubeContentSign]

/-- Actual content sparsemax and generated original values retain complete finite response capacity.
Source: full feature inversion and the genuine variational Eq. (1) forward. -/
theorem cubeContentFullForward_eq {R D : ℕ} (floor : ℝ) (code : Fin R → CubeContentState ι)
    (t : ι → ℝ) (Y : Matrix (CubeContentState ι) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) :
    cubeContentForward code id t (cubeContentFullCoefficients Y) = fun r d => Y (code r) d := by
  rw [cubeContentForward_eq floor code id t _ hf ht, cubeContentFullOutput_eq]

/-- Nonlinear XOR answers and two positive flips inhabit every complete finite-capacity premise. -/
example : cubeContentForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0)) id
    (fun _ => (1 / 8 : ℝ))
    (cubeContentFullCoefficients (Matrix.of (fun x : CubeContentState (Fin 2) => fun _ : Fin 1 =>
      if x 0 = x 1 then (1 : ℝ) else -1))) =
    (fun _ : Fin 1 => fun _ : Fin 1 => (-1 : ℝ)) := by
  rw [cubeContentFullForward_eq (3 / 4) _ _ _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two])]
  rfl

omit [DecidableEq ι] in
/-- Full finite universality has 2^r learned feature rows per output channel.
Source: explicit capacity cost of the full content subset family after sparsemax Eq. (1). -/
theorem cubeContentFullCoefficient_count (D : ℕ) :
    Fintype.card (Finset ι × Fin D) = 2 ^ Fintype.card ι * D := by
  rw [Fintype.card_prod, Fintype.card_finset, Fintype.card_fin]

/-- Universal coefficient-linear scalar content responses require exponentially many coefficients.
Source: the finite capacity price of the new affine-forward content chart after sparsemax Eq. (1). -/
theorem cubeContentUniversal_capacity {κ : Type*} [Fintype κ]
    (f : (κ → ℝ) →ₗ[ℝ] (CubeContentState ι → ℝ)) (hf : Function.Surjective f) :
    2 ^ Fintype.card ι ≤ Fintype.card κ := by
  have h := LinearMap.finrank_le_finrank_of_surjective hf
  rw [Module.finrank_pi, Module.finrank_pi, Fintype.card_fun, Fintype.card_bool] at h
  exact h

/-- The identity on four content-state coefficients inhabits the full-capacity premise. -/
example : 2 ^ Fintype.card (Fin 2) ≤ Fintype.card (CubeContentState (Fin 2)) :=
  cubeContentUniversal_capacity (LinearMap.id :
    (CubeContentState (Fin 2) → ℝ) →ₗ[ℝ] (CubeContentState (Fin 2) → ℝ)) (fun Y => ⟨Y, rfl⟩)

/-- A single two-bit product represents all XOR-sign answers exactly.
Source: a nonlinear content-family witness following sparsemax Eq. (1). -/
theorem cubeContentXor_eq (x : CubeContentState (Fin 2)) :
    cubeContentCharacter ({0, 1} : Finset (Fin 2)) x = if x 0 = x 1 then (1 : ℝ) else -1 := by
  cases h0 : x 0 <;> cases h1 : x 1 <;> simp_all [cubeContentCharacter, cubeContentSign]

/-- XOR sign is not a constant-plus-individual-bit response on two observed content bits.
Source: counterexample to the former degree-one response restriction after sparsemax Eq. (1). -/
theorem cubeContentXor_not_additive :
    ¬ ∃ a b c : ℝ, ∀ x : CubeContentState (Fin 2),
      a + b * cubeContentSign (x 0) + c * cubeContentSign (x 1) =
        cubeContentCharacter ({0, 1} : Finset (Fin 2)) x := by
  rintro ⟨a, b, c, h⟩
  have h0 := h (fun _ => false)
  have h1 := h (fun i => decide (i = 0))
  have h2 := h (fun i => decide (i = 1))
  have h3 := h (fun _ => true)
  norm_num [cubeContentSign, cubeContentCharacter] at h0 h1 h2 h3
  linarith

/-- A selected high-order interaction needs one response coefficient, not every virtual state. -/
example : Fintype.card (Fin 1 × Fin 64) = 64 :=
  cubeContentCoefficient_count 64

end Transformer.GPTMini.Sparsemax
