/-
# Restricting a coordinate of a boolean function

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?",
Appendix E ("Technical Lemma").

Both lemmas of that appendix are about restricting a coordinate of a boolean
function to a fixed value, so this file works out what the restriction does
to the Fourier coefficients: it deletes every coefficient at a set containing
the coordinate, and folds it into the coefficient at the set without it.

The proof sketch in the paper is the multilinear one — "consider the
multilinear representation of `f`, and factor out terms containing `x₁`" —
and `restrict_eq` is exactly that factorization, summed over the subsets not
containing the coordinate.
-/

import Transformer.RASPL.Fourier

namespace Transformer
namespace RASPL

variable {n : ℕ}

/-- `f` with coordinate `i` held at the sign bit `b`; still a function on the
whole cube, but one that ignores coordinate `i`. -/
def restrict (i : Fin n) (b : Bool) (f : Cube n → ℝ) : Cube n → ℝ :=
  fun x => f (Function.update x i b)

/-- A character not involving `i` cannot see coordinate `i`. -/
lemma chi_update_of_notMem {i : Fin n} {S : Finset (Fin n)} (hi : i ∉ S) (x : Cube n)
    (b : Bool) : chi S (Function.update x i b) = chi S x :=
  Finset.prod_congr rfl fun j hj => by
    rw [Function.update_of_ne (fun h : j = i => hi (h ▸ hj))]

/-- Adjoining `i` to a character multiplies it by coordinate `i`. -/
lemma chi_insert {i : Fin n} {S : Finset (Fin n)} (hi : i ∉ S) (x : Cube n) :
    chi (insert i S) x = bitSign (x i) * chi S x := by
  rw [chi, Finset.prod_insert hi, chi]

/-- The subsets of `Fin n` in pairs: one not containing `i`, one containing
it.  This is the splitting the multilinear factorization runs over. -/
lemma sum_subsets_split (i : Fin n) (F : Finset (Fin n) → ℝ) :
    ∑ T : Finset (Fin n), F T
      = ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin n) => i ∉ S), (F S + F (insert i S)) := by
  classical
  rw [Finset.sum_add_distrib,
    ← Finset.sum_filter_add_sum_filter_not Finset.univ (fun S : Finset (Fin n) => i ∉ S) F]
  congr 1
  refine Finset.sum_nbij' (fun T => T.erase i) (fun S => insert i S) ?_ ?_ ?_ ?_ ?_
  · intro T _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact Finset.notMem_erase i T
  · intro S hS
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hS ⊢
    exact Finset.mem_insert_self i S
  · intro T hT
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hT
    exact Finset.insert_erase hT
  · intro S hS
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
    exact Finset.erase_insert hS
  · intro T hT
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hT
    rw [Finset.insert_erase hT]

/-- **The multilinear factorization.**  Writing `f = x_i P + Q` and setting
`x_i = b` leaves `bitSign b * P + Q`, whose coefficient at a set `S` avoiding
`i` is `f̂(S) + bitSign b * f̂(S ∪ {i})`. -/
theorem restrict_eq (i : Fin n) (b : Bool) (f : Cube n → ℝ) (x : Cube n) :
    restrict i b f x
      = ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin n) => i ∉ S),
          (coeff f S + bitSign b * coeff f (insert i S)) * chi S x := by
  classical
  rw [restrict, ← sum_coeff_mul_chi f (Function.update x i b),
    sum_subsets_split i (fun T => coeff f T * chi T (Function.update x i b))]
  refine Finset.sum_congr rfl fun S hS => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
  rw [chi_update_of_notMem hS, chi_insert hS, Function.update_self,
    chi_update_of_notMem hS]
  ring

/-- A restricted function has no weight on any set containing the restricted
coordinate: it does not depend on that coordinate at all. -/
theorem coeff_restrict_of_mem {i : Fin n} {S : Finset (Fin n)} (hi : i ∈ S) (b : Bool)
    (f : Cube n → ℝ) : coeff (restrict i b f) S = 0 := by
  classical
  rw [show restrict i b f = fun x => ∑ T ∈ Finset.univ.filter
      (fun T : Finset (Fin n) => i ∉ T),
      (coeff f T + bitSign b * coeff f (insert i T)) * chi T x from
    funext (restrict_eq i b f), coeff_sum]
  refine Finset.sum_eq_zero fun T hT => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hT
  rw [coeff_smul, coeff_chi, if_neg (fun h : S = T => hT (h ▸ hi)), mul_zero]

/-- And at a set avoiding it, the two coefficients of `f` are folded
together. -/
theorem coeff_restrict_of_notMem {i : Fin n} {S : Finset (Fin n)} (hi : i ∉ S) (b : Bool)
    (f : Cube n → ℝ) :
    coeff (restrict i b f) S = coeff f S + bitSign b * coeff f (insert i S) := by
  classical
  rw [show restrict i b f = fun x => ∑ T ∈ Finset.univ.filter
      (fun T : Finset (Fin n) => i ∉ T),
      (coeff f T + bitSign b * coeff f (insert i T)) * chi T x from
    funext (restrict_eq i b f), coeff_sum]
  rw [Finset.sum_congr rfl (fun T _ => by rw [coeff_smul, coeff_chi] :
    ∀ T ∈ Finset.univ.filter (fun T : Finset (Fin n) => i ∉ T),
      coeff (fun x => (coeff f T + bitSign b * coeff f (insert i T)) * chi T x) S
        = (coeff f T + bitSign b * coeff f (insert i T)) * (if S = T then 1 else 0))]
  simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq, Finset.mem_filter, Finset.mem_univ,
    true_and, if_pos hi]

end RASPL
end Transformer
