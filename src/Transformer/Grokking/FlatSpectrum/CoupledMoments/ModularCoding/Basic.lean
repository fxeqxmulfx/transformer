module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers

/-!
# Exact residuals for lifting categorical phase equations

Source: `review/flat_spectrum.md`, CM27. Integer representatives of observed
categorical equations are lifted by solving the actual residual system.
Primality is unnecessary for the equivalence, and solvability is not assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding

variable {I J : Type*} [Fintype J]

/-- The actual finite row action of an integer constraint matrix, CM27. -/
def action (A : I → J → ℤ) (x : J → ℤ) (i : I) : ℤ := ∑ j, A i j * x j

/-- Every original equation holds modulo the declared integer modulus, CM27. -/
def Fits (m : ℤ) (A : I → J → ℤ) (b : I → ℤ) (x : J → ℤ) : Prop :=
  ∀ i, m ∣ b i - action A x i

/-- One digit extension of the actual representatives, CM27. -/
def lift (m : ℤ) (x y : J → ℤ) (j : J) : ℤ := x j + m * y j

/-- The divided original residual; its exactness requires `Fits`, CM27. -/
def residual (m : ℤ) (A : I → J → ℤ) (b : I → ℤ) (x : J → ℤ) (i : I) : ℤ :=
  (b i - action A x i) / m

/-- Actual integer row sums distribute over a digit extension, CM27. -/
theorem action_lift (A : I → J → ℤ) (m : ℤ) (x y : J → ℤ) (i : I) :
    action A (lift m x y) i = action A x i + m * action A y i := by
  simp only [action, lift, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- A known modular solution makes division of its residual exact; the next
residual is the old modulus times the digit residual, CM27. -/
theorem residual_identity {A : I → J → ℤ} {m : ℤ} {b : I → ℤ} {x : J → ℤ}
    (h : Fits m A b x) (y : J → ℤ) (i : I) :
    b i - action A (lift m x y) i = m * (residual m A b x i - action A y i) := by
  have he := Int.mul_ediv_cancel_of_dvd (h i)
  rw [action_lift]
  unfold residual
  linear_combination -he

/-- **Checked digit lifting**, CM27: a higher-modulus solution is equivalent
to solving the actual lower-modulus residual system. No field or existence
claim is hidden in the statement. -/
theorem lift_iff {A : I → J → ℤ} {m : ℤ} {b : I → ℤ} {x : J → ℤ}
    (hm : m ≠ 0) (h : Fits m A b x) (p : ℤ) (y : J → ℤ) :
    Fits (m * p) A b (lift m x y) ↔ Fits p A (residual m A b x) y := by
  constructor
  · intro hn i
    have hi := hn i
    rw [residual_identity h] at hi
    exact (Int.mul_dvd_mul_iff_left hm).mp hi
  · intro hn i
    rw [residual_identity h]
    exact (Int.mul_dvd_mul_iff_left hm).mpr (hn i)

example := action_lift (![![1, 2], ![3, 4]] : Fin 2 → Fin 2 → ℤ)
  4 ![1, 2] ![1, 1] 1

example := residual_identity
  (A := (![![1, 2], ![3, 4]] : Fin 2 → Fin 2 → ℤ))
  (m := 4) (b := ![9, 23]) (x := ![1, 2])
  (by norm_num [Fits, action, Fin.sum_univ_two, Fin.forall_fin_two]) ![1, 1] 1

example := lift_iff
  (A := (![![1, 2], ![3, 4]] : Fin 2 → Fin 2 → ℤ))
  (m := 4) (b := ![9, 23]) (x := ![1, 2]) (by norm_num)
  (by norm_num [Fits, action, Fin.sum_univ_two, Fin.forall_fin_two]) 2 ![1, 1]

/-- CM27's failed lift: a valid solution of `2*x=0` modulo two need not extend
to a solution modulo four. The observed residual obstruction must be retained. -/
theorem base_solution_need_not_lift :
    Fits 2 (fun _ _ : Fin 1 => (2 : ℤ)) (fun _ => 0) (fun _ => 1) ∧
    ¬ ∃ y : Fin 1 → ℤ,
      Fits 4 (fun _ _ : Fin 1 => (2 : ℤ)) (fun _ => 0) (lift 2 (fun _ => 1) y) := by
  constructor
  · intro i
    norm_num [action, Fin.sum_univ_one]
  · rintro ⟨y, hy⟩
    obtain ⟨k, hk⟩ := hy 0
    simp only [action, lift, Fin.sum_univ_one] at hk
    omega

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding
