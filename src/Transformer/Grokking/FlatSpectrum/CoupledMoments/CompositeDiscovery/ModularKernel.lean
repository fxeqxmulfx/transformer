module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.RowReduction

/-!
# Uniqueness of actual modular constraint systems

Source: `review/flat_spectrum.md`, CM28, component and digit uniqueness.
CM27 supplies exact residual lifting, not uniqueness. Here a trivial kernel
modulo a base modulus implies a trivial kernel at all its powers, by dividing
the actual integer representatives and using the exact residual identity.
Checked coprime component agreement also gives product agreement. Applied to
the actual anchored matrix, these statements justify uniqueness of normalized
solutions; they do not assert that any digit system has a solution.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

open ModularCoding

variable {I J : Type*} [Fintype J]

/-- The actual homogeneous integer system has only the zero residue vector,
CM28. This predicate is about a matrix and modulus, not a proposed answer. -/
def ZeroKernel (m : ℤ) (A : I → J → ℤ) : Prop :=
  ∀ x, Fits m A 0 x → ∀ j, m ∣ x j

/-- Actual integer row sums distribute over a difference, CM28. -/
theorem action_sub (A : I → J → ℤ) (x y : J → ℤ) (i : I) :
    action A (fun j => x j - y j) i = action A x i - action A y i := by
  unfold action
  simp only [mul_sub, Finset.sum_sub_distrib]

/-- Base kernel uniqueness propagates through an actual digit extension,
CM28. The moduli need not be coprime or prime. -/
theorem zeroKernel_mul {m p : ℤ} {A : I → J → ℤ} (hm : m ≠ 0)
    (hK : ZeroKernel m A) (hP : ZeroKernel p A) : ZeroKernel (m * p) A := by
  intro x hx j
  have hd : ∀ k, m ∣ x k := hK x fun i =>
    Int.dvd_trans (Int.dvd_mul_right m p) (hx i)
  let y : J → ℤ := fun k => x k / m
  have he : x = lift m 0 y := by
    funext k
    change x k = 0 + m * (x k / m)
    rw [zero_add, Int.mul_ediv_cancel_of_dvd (hd k)]
  have h0 : Fits m A 0 0 := by intro i; simp [action]
  have hr : ModularCoding.residual m A 0 0 = 0 := by
    ext i
    simp [ModularCoding.residual, action]
  have hy := (lift_iff hm h0 p y).mp (he ▸ hx)
  rw [hr] at hy
  obtain ⟨k, hk⟩ := hP y hy j
  refine ⟨k, ?_⟩
  have hj := congrFun he j
  change x j = 0 + m * y j at hj
  rw [hj, hk]
  ring

/-- A checked trivial kernel modulo a nonzero base gives uniqueness modulo
every power, CM28. This statement does not claim successful lifting. -/
theorem zeroKernel_pow {p : ℤ} {A : I → J → ℤ} (hp : p ≠ 0)
    (hK : ZeroKernel p A) (k : ℕ) : ZeroKernel (p ^ k) A := by
  induction k with
  | zero =>
    intro x _ j
    simpa only [pow_zero] using one_dvd (x j)
  | succ k ih =>
    rw [pow_succ]
    exact zeroKernel_mul (pow_ne_zero k hp) ih hK

/-- Actual zero-kernel certificates for coprime components combine under
the checked Bezout identity, CM28. No product-modulus shortcut is assumed. -/
theorem zeroKernel_crt {m n u v : ℤ} {A : I → J → ℤ}
    (h : u * m + v * n = 1) (hM : ZeroKernel m A) (hN : ZeroKernel n A) :
    ZeroKernel (m * n) A := by
  intro x hx j
  have hm : m ∣ x j := hM x (fun i =>
    Int.dvd_trans (Int.dvd_mul_right m n) (hx i)) j
  have hn : n ∣ x j := hN x (fun i =>
    Int.dvd_trans (Int.dvd_mul_left m n) (hx i)) j
  exact product_dvd h hm hn

/-- A trivial actual kernel makes any two solutions of the same anchored
system congruent at every coordinate, CM28, including normalized class codes. -/
theorem solutions_congr {m : ℤ} {A : I → J → ℤ} (hK : ZeroKernel m A)
    {b : I → ℤ} {x y : J → ℤ} (hx : Fits m A b x) (hy : Fits m A b y) :
    ∀ j, m ∣ x j - y j := by
  apply hK (fun j => x j - y j)
  intro i
  obtain ⟨k, hk⟩ := hx i
  obtain ⟨l, hl⟩ := hy i
  refine ⟨k - l, ?_⟩
  change 0 - action A (fun j => x j - y j) i = m * (k - l)
  rw [action_sub]
  linear_combination hk - hl

/-- A concrete nonzero integer matrix for the component/lift witnesses. -/
def triangular : Fin 2 → Fin 2 → ℤ := ![![1, 2], ![0, 1]]

/-- Eliminate the actual triangular equations for any modulus, CM28. -/
theorem triangular_zeroKernel (m : ℤ) : ZeroKernel m triangular := by
  intro x hx j
  have h0 : m ∣ -(x 0 + 2 * x 1) := by
    simpa [action, triangular, Fin.sum_univ_two] using hx 0
  have h1 : m ∣ -x 1 := by
    simpa [action, triangular, Fin.sum_univ_two] using hx 1
  obtain ⟨a, ha⟩ := h0
  obtain ⟨b, hb⟩ := h1
  fin_cases j
  · refine ⟨2 * b - a, ?_⟩
    change x 0 = m * (2 * b - a)
    linear_combination -ha + 2 * hb
  · refine ⟨-b, ?_⟩
    change x 1 = m * -b
    linear_combination -hb

example := action_sub triangular (![1, 2] : Fin 2 → ℤ) ![13, 14] 0

example : ZeroKernel 8 triangular :=
  zeroKernel_mul (m := 4) (p := 2) (by norm_num)
    (triangular_zeroKernel 4) (triangular_zeroKernel 2)

example : ZeroKernel 32 triangular :=
  zeroKernel_pow (p := 2) (by norm_num) (triangular_zeroKernel 2) 5

example : ZeroKernel 12 triangular :=
  zeroKernel_crt (m := 4) (n := 3) (u := 1) (v := -1) (by norm_num)
    (triangular_zeroKernel 4) (triangular_zeroKernel 3)

example : ∀ j : Fin 2, (12 : ℤ) ∣ (![1, 2] : Fin 2 → ℤ) j - ![13, 14] j :=
  solutions_congr (triangular_zeroKernel 12) (b := ![5, 2])
    (by norm_num [Fits, action, triangular, Fin.sum_univ_two, Fin.forall_fin_two])
    (by norm_num [Fits, action, triangular, Fin.sum_univ_two, Fin.forall_fin_two])

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
