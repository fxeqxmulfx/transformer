/-
# Counting lattice points by the theta function

`Transformer.ALM.ScalarSharp` removes the token count `n` from the softmax
bound for *scalar* integer keys.  Its one counting step is that at most two
integers sit at any given distance from the query.  In dimension `m` that step
has no analogue: the number of lattice points on the shell `‖x‖² = t` is the
sum-of-squares function `r_m(t)`, which is erratic and unbounded in `t`.

The way past it is not to count the shells but to sum over the whole lattice
at once.  The exponent is a *sum* of squared coordinates, so the exponential
factorizes and the lattice sum is a power of a one-dimensional series:

  `∑_{x ∈ ℤ^m} r^{‖x‖²} = (∑_{a ∈ ℤ} r^{a²})^m = θ(r)^m`.

The one-dimensional series is the one `ScalarSharp` already handles, and
`geom_sq_le` bounds its tail: `θ(r) ≤ 1 + 2r/(1-r³)`.

`theta1_le` is that bound on a finite window `[-M, M]`, and `box_sum_le` is
the factorization over the box `[-M, M]^m`.  Windows rather than the whole
lattice because the sums here are `Finset` sums; the box is chosen large
enough at the point of use, in `Transformer.ALM.VectorInt`.

* E. Grosswald, *Representations of Integers as Sums of Squares*, Springer
  1985, §1 — `θ(r)^m` as the generating function of `r_m(t)`, which is what
  the factorization below is.
-/

import Transformer.ALM.Lattice

open scoped BigOperators

namespace Transformer
namespace ALM

/-! ### The squared length of a lattice vector -/

/-- The squared Euclidean length of an integer vector, as a natural number.
Splitting it off keeps the counting below on the lattice, where the exponents
are natural numbers and `r ^ (a + b) = r ^ a * r ^ b` is literal. -/
def sqNorm {m : ℕ} (x : Fin m → ℤ) : ℕ := ∑ i, (x i).natAbs ^ 2

@[simp] lemma sqNorm_zero {m : ℕ} : sqNorm (0 : Fin m → ℤ) = 0 := by simp [sqNorm]

/-- Over the reals it is the sum of the squared coordinates. -/
lemma sqNorm_cast {m : ℕ} (x : Fin m → ℤ) :
    ((sqNorm x : ℕ) : ℝ) = ∑ i, ((x i : ℝ)) ^ 2 := by
  rw [sqNorm, Nat.cast_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  push_cast [Nat.cast_natAbs]
  rw [sq_abs]

/-! ### One dimension -/

/-- **The theta sum over a window.**  For `0 ≤ r < 1`,

  `∑_{a = -M}^{M} r^{a²} ≤ 1 + 2 r/(1 - r³)`,

uniformly in `M`: the term `a = 0` contributes `1`, and every other value of
`|a|` is taken by at most two integers, so the rest is twice the tail bounded
by `geom_sq_le` of `Transformer.ALM.Lattice`. -/
theorem theta1_le {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) (M : ℕ) :
    ∑ a ∈ Finset.Icc (-(M : ℤ)) (M : ℤ), r ^ (a.natAbs ^ 2)
      ≤ 1 + 2 * (r / (1 - r ^ 3)) := by
  set I := Finset.Icc (-(M : ℤ)) (M : ℤ) with hI
  have hmaps : ∀ a ∈ I, a.natAbs ∈ Finset.range (M + 1) := by
    intro a ha
    rw [hI, Finset.mem_Icc] at ha
    exact Finset.mem_range.mpr (by omega)
  have hfib : ∑ a ∈ I, r ^ (a.natAbs ^ 2)
      = ∑ t ∈ Finset.range (M + 1),
          ((I.filter (fun a => a.natAbs = t)).card : ℝ) * r ^ (t ^ 2) := by
    rw [← Finset.sum_fiberwise_of_maps_to' hmaps (fun t => r ^ (t ^ 2))]
    exact Finset.sum_congr rfl fun t _ => by rw [Finset.sum_const, nsmul_eq_mul]
  -- the two integers of absolute value `t`
  have hsub : ∀ t : ℕ, I.filter (fun a => a.natAbs = t)
      ⊆ ({-(t : ℤ), (t : ℤ)} : Finset ℤ) := by
    intro t a ha
    have ht : a.natAbs = t := (Finset.mem_filter.mp ha).2
    have hcases : a = -(t : ℤ) ∨ a = (t : ℤ) := by omega
    rcases hcases with h | h
    · rw [h]; exact Finset.mem_insert_self _ _
    · rw [h]; exact Finset.mem_insert_of_mem (Finset.mem_singleton.mpr rfl)
  have hcard : ∀ t : ℕ, ((I.filter (fun a => a.natAbs = t)).card : ℝ) ≤ 2 := by
    intro t
    have h1 := Finset.card_le_card (hsub t)
    have h2 : ({-(t : ℤ), (t : ℤ)} : Finset ℤ).card ≤ 2 :=
      le_trans (Finset.card_insert_le _ _) (by simp)
    exact_mod_cast le_trans h1 h2
  have hcard0 : ((I.filter (fun a => a.natAbs = 0)).card : ℝ) ≤ 1 := by
    have h1 := Finset.card_le_card (hsub 0)
    have hset : ({-((0 : ℕ) : ℤ), ((0 : ℕ) : ℤ)} : Finset ℤ) = {0} := by norm_num
    rw [hset, Finset.card_singleton] at h1
    exact_mod_cast h1
  have hmem0 : (0 : ℕ) ∈ Finset.range (M + 1) := Finset.mem_range.mpr (Nat.succ_pos M)
  rw [hfib, ← Finset.add_sum_erase _ _ hmem0]
  have hhead : ((I.filter (fun a => a.natAbs = 0)).card : ℝ) * r ^ ((0 : ℕ) ^ 2) ≤ 1 := by
    simpa using hcard0
  have htail : ∑ t ∈ (Finset.range (M + 1)).erase 0,
        ((I.filter (fun a => a.natAbs = t)).card : ℝ) * r ^ (t ^ 2)
      ≤ 2 * (r / (1 - r ^ 3)) := by
    calc ∑ t ∈ (Finset.range (M + 1)).erase 0,
          ((I.filter (fun a => a.natAbs = t)).card : ℝ) * r ^ (t ^ 2)
        ≤ ∑ t ∈ (Finset.range (M + 1)).erase 0, 2 * r ^ (t ^ 2) :=
          Finset.sum_le_sum fun t _ =>
            mul_le_mul_of_nonneg_right (hcard t) (pow_nonneg hr0 _)
      _ = 2 * ∑ t ∈ (Finset.range (M + 1)).erase 0, r ^ (t ^ 2) := by rw [Finset.mul_sum]
      _ ≤ 2 * (r / (1 - r ^ 3)) :=
          mul_le_mul_of_nonneg_left (geom_sq_le hr0 hr1 M) (by norm_num)
  linarith

/-! ### The box -/

/-- **The lattice sum factorizes.**  Over the box `[-M, M]^m`,

  `∑_x r^{‖x‖²} = (∑_{a = -M}^{M} r^{a²})^m ≤ (1 + 2r/(1 - r³))^m`,

because `‖x‖²` is the sum of the squared coordinates and `r^{a+b} = r^a r^b`.
This is the `m`-dimensional replacement for the "at most two keys per
distance" count, and it is the only place the dimension enters. -/
theorem box_sum_le {m : ℕ} {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) (M : ℕ) :
    ∑ x ∈ Fintype.piFinset (fun _ : Fin m => Finset.Icc (-(M : ℤ)) (M : ℤ)),
        r ^ (sqNorm x)
      ≤ (1 + 2 * (r / (1 - r ^ 3))) ^ m := by
  have hfact : ∀ x : Fin m → ℤ,
      r ^ (sqNorm x) = ∏ i, r ^ ((x i).natAbs ^ 2) := fun x =>
    (Finset.prod_pow_eq_pow_sum Finset.univ (fun i => (x i).natAbs ^ 2) r).symm
  rw [Finset.sum_congr rfl fun x _ => hfact x,
    ← Finset.prod_univ_sum (fun _ : Fin m => Finset.Icc (-(M : ℤ)) (M : ℤ))
      (fun _ a => r ^ (a.natAbs ^ 2))]
  calc ∏ _i : Fin m, ∑ a ∈ Finset.Icc (-(M : ℤ)) (M : ℤ), r ^ (a.natAbs ^ 2)
      ≤ ∏ _i : Fin m, (1 + 2 * (r / (1 - r ^ 3))) :=
        Finset.prod_le_prod (fun _ _ => Finset.sum_nonneg fun _ _ => pow_nonneg hr0 _)
          (fun _ _ => theta1_le hr0 hr1 M)
    _ = (1 + 2 * (r / (1 - r ^ 3))) ^ m := by
        rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-- The hypotheses are satisfiable, and at `m = 0` the box is a point. -/
example : ∑ x ∈ Fintype.piFinset (fun _ : Fin 0 => Finset.Icc (-(3 : ℤ)) (3 : ℤ)),
    (1 / 2 : ℝ) ^ (sqNorm x) = 1 := by
  simp [sqNorm]

end ALM
end Transformer
