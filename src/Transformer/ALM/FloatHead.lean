/-
# The head's output, at the line the running code returns

`Transformer.ALM.FloatIndex` ends at a key: under the separation condition the
floating-point search returns the key `hullIndex` names
(`fp_hullIndex_key`).  `Transformer.ALM.SoftmaxValue` starts at an index: the
head's output vector is the value stored at `I.ans`, up to `(n-1)e^{-β}·C`.
Between them sits the only claim the machine actually makes — that querying a
stored key gives back the value stored beside it — and neither file made it,
because one never mentions a value and the other never mentions arithmetic.

`fp_head_output` makes it.  Its two conjuncts are the two halves at the same
query: the line the implementation's `lower_bound` lands on carries exactly the
queried key, and the head's output is the value at that line up to the softmax
error.  Nothing is assumed about the query beyond what the running code needs
— a bound on the keys and a separation from the breakpoints, the hypotheses of
`fpProbe_eq_hullProbe`.

On the integer data the primitive stores the separation is not a hypothesis at
all: `half_le_key_dist_mid` shows a stored key is half a unit from every
breakpoint, so `fp_head_output_of_int` asks only that the rounding error stay
below that half unit — one condition on the precision, checkable once for the
machine rather than once per query.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 203-215.
-/

import Transformer.ALM.SoftmaxValue
import Transformer.ALM.FloatLattice

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- In dimension one the index's answer is the hull's own index, by
definition of `hullAns`. -/
theorem hullIndex_ans_one [Nonempty (Fin n)] (K : Fin n → EucSpace 1) (q : EucSpace 1) :
    hullIndex.ans K q = hullIdx (fun j => K j 0) (q 0) := rfl

/-- The scalar keys of an integer family are the coordinates of its lattice
embedding. -/
theorem embInt_one_apply (K : Fin n → ℤ) (j : Fin n) :
    (embInt (fun _ : Fin 1 => K j)) 0 = ((K j : ℝ)) := by simp

private lemma embInt_one_injective {K : Fin n → ℤ} (hinj : Function.Injective K) :
    Function.Injective (fun j => embInt (fun _ : Fin 1 => K j)) := by
  intro a b h
  refine hinj ?_
  have := congrFun (congrArg (fun x : EucSpace 1 => (x : Fin 1 → ℝ)) h) 0
  simp only [embInt_apply] at this
  exact_mod_cast this

/-- **The running search returns the queried key.**  At a stored key the
floating-point `lower_bound` lands on the line carrying that very key: the
separation condition hands the exact probe back
(`Transformer.ALM.FloatIndex`), and exactness pins the exact probe to the
query (`NNIndex.ans_eq_of_query_mem`). -/
theorem fpProbe_key_eq [Nonempty (Fin n)] (F : FPArith) (K : Fin n → ℤ)
    (hinj : Function.Injective K) (i₀ : Fin n) (M : ℝ)
    (hb : ∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1, |sortedKey (fun i => (K i : ℝ)) j| ≤ M)
    (hsep : ∀ j < keyCard (fun i => (K i : ℝ)) - 1,
      F.u * M < |(K i₀ : ℝ) - (sortedKey (fun i => (K i : ℝ)) j
        + sortedKey (fun i => (K i : ℝ)) (j + 1)) / 2|) :
    sortedKey (fun i => (K i : ℝ)) (fpProbe F (fun i => (K i : ℝ)) (K i₀)) = (K i₀ : ℝ) := by
  rw [fp_hullIndex_key F (fun i => (K i : ℝ)) (K i₀) M hb hsep]
  have hans := hullIndex.ans_eq_of_query_mem (fun j => embInt (fun _ : Fin 1 => K j))
    (embInt_one_injective hinj) i₀
  rw [hullIndex_ans_one] at hans
  simp only [embInt_one_apply] at hans
  rw [hans]

/-- **The whole machine, in one statement.**  Query a stored key: the
implementation's own arithmetic lands on the line carrying it, and the softmax
head's output vector is the value stored at that line, within
`(n-1)e^{-β}·C`.  The first conjunct is `Transformer.ALM.FloatIndex`, the
second `Transformer.ALM.SoftmaxValue`, and they are here about one query. -/
theorem fp_head_output {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [Nonempty (Fin n)] (F : FPArith) (β : ℝ) (hβ : 0 ≤ β) (K : Fin n → ℤ)
    (hinj : Function.Injective K) (i₀ : Fin n) (M : ℝ)
    (hb : ∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1, |sortedKey (fun i => (K i : ℝ)) j| ≤ M)
    (hsep : ∀ j < keyCard (fun i => (K i : ℝ)) - 1,
      F.u * M < |(K i₀ : ℝ) - (sortedKey (fun i => (K i : ℝ)) j
        + sortedKey (fun i => (K i : ℝ)) (j + 1)) / 2|)
    (V : Fin n → E) (C : ℝ) (hC : ∀ j, ‖V j - V i₀‖ ≤ C) :
    sortedKey (fun i => (K i : ℝ)) (fpProbe F (fun i => (K i : ℝ)) (K i₀)) = (K i₀ : ℝ) ∧
      ‖(∑ j, (Real.exp (β * score (embInt (fun _ : Fin 1 => K i₀))
              (embInt (fun _ : Fin 1 => K j)))
            / ∑ k, Real.exp (β * score (embInt (fun _ : Fin 1 => K i₀))
              (embInt (fun _ : Fin 1 => K k)))) • V j) - V i₀‖
        ≤ ((n : ℝ) - 1) * Real.exp (-(β * 1)) * C := by
  refine ⟨fpProbe_key_eq F K hinj i₀ M hb hsep, ?_⟩
  have h := head_output_at_index hullIndex β hβ (fun j => fun _ : Fin 1 => K j)
    (fun a b hab => hinj (congrFun hab 0)) i₀ V C hC
  rwa [hullIndex.ans_eq_of_query_mem (fun j => embInt (fun _ : Fin 1 => K j))
    (embInt_one_injective hinj) i₀] at h

/-! ### And on integer data the separation is free -/

/-- **A stored key is half a unit from every breakpoint.**  For an integer
family the sorted keys are integers and the breakpoint of two adjacent ones is
a multiple of `1/2`; a breakpoint lies strictly between its two keys, so it is
never a key itself, and two distinct multiples of `1/2` are `1/2` apart.  The
separation condition of `fpProbe_eq_hullProbe` is therefore not something a
caller has to check on the lookup path: it holds by arithmetic. -/
lemma half_le_key_dist_mid [Nonempty (Fin n)] (K : Fin n → ℤ) (i₀ : Fin n) (j : ℕ) :
    1 / 2 ≤ |(K i₀ : ℝ) - (sortedKey (fun i => (K i : ℝ)) j
      + sortedKey (fun i => (K i : ℝ)) (j + 1)) / 2| := by
  set Ks : Fin n → ℝ := fun i => (K i : ℝ) with hKs
  have hKi : ∀ i, ∃ z : ℤ, Ks i = (z : ℝ) := fun i => ⟨K i, rfl⟩
  have hmono : StrictMono (sortedKey Ks) := strictMono_nat_of_lt_succ (sortedKey_lt_succ Ks)
  obtain ⟨t, -, hteq⟩ := exists_sortedKey_eq Ks i₀
  obtain ⟨za, hza⟩ := sortedKey_int Ks hKi t
  obtain ⟨zj, hzj⟩ := mid_half_int (sortedKey_int Ks hKi) j
  have hstep := sortedKey_lt_succ Ks j
  have hkey : Ks i₀ = (za : ℝ) := by rw [← hteq, hza]
  have hne : Ks i₀ ≠ (sortedKey Ks j + sortedKey Ks (j + 1)) / 2 := by
    rw [← hteq]
    rcases Nat.lt_or_ge t (j + 1) with hlt | hge
    · have hle : sortedKey Ks t ≤ sortedKey Ks j := hmono.monotone (by omega)
      intro h; rw [h] at hle; linarith
    · have hle : sortedKey Ks (j + 1) ≤ sortedKey Ks t := hmono.monotone hge
      intro h; rw [h] at hle; linarith
  rw [hkey, hzj] at hne
  have hzne : 2 * za - zj ≠ 0 := by
    intro h
    refine hne ?_
    have hc : (2 : ℝ) * (za : ℝ) - (zj : ℝ) = 0 := by exact_mod_cast congrArg (fun z : ℤ => (z : ℝ)) h
    linarith
  have hR : (1 : ℝ) ≤ |2 * (za : ℝ) - (zj : ℝ)| := by
    have h1 : (1 : ℤ) ≤ |2 * za - zj| := Int.one_le_abs hzne
    have h2 : ((1 : ℤ) : ℝ) ≤ ((|2 * za - zj| : ℤ) : ℝ) := Int.cast_le.mpr h1
    push_cast at h2
    exact h2
  rw [show (2 : ℝ) * (za : ℝ) - (zj : ℝ) = 2 * ((za : ℝ) - (zj : ℝ) / 2) by ring,
    abs_mul, abs_two] at hR
  rw [show (K i₀ : ℝ) = Ks i₀ from rfl, hkey, hzj]
  linarith

/-- **The whole machine on integer data, with a bound and nothing else.**  The
separation hypothesis of `fp_head_output` is discharged by
`half_le_key_dist_mid`, so all that is left of the arithmetic is one condition
on the precision: the rounding error on the breakpoints must stay below half a
unit of the key lattice.  For long double (`u ≤ 2^-60`) on keys below `2^56`
that error is `2^-4`. -/
theorem fp_head_output_of_int {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [Nonempty (Fin n)] (F : FPArith) (β : ℝ) (hβ : 0 ≤ β) (K : Fin n → ℤ)
    (hinj : Function.Injective K) (i₀ : Fin n) (M : ℝ)
    (hb : ∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1, |sortedKey (fun i => (K i : ℝ)) j| ≤ M)
    (hu : F.u * M < 1 / 2)
    (V : Fin n → E) (C : ℝ) (hC : ∀ j, ‖V j - V i₀‖ ≤ C) :
    sortedKey (fun i => (K i : ℝ)) (fpProbe F (fun i => (K i : ℝ)) (K i₀)) = (K i₀ : ℝ) ∧
      ‖(∑ j, (Real.exp (β * score (embInt (fun _ : Fin 1 => K i₀))
              (embInt (fun _ : Fin 1 => K j)))
            / ∑ k, Real.exp (β * score (embInt (fun _ : Fin 1 => K i₀))
              (embInt (fun _ : Fin 1 => K k)))) • V j) - V i₀‖
        ≤ ((n : ℝ) - 1) * Real.exp (-(β * 1)) * C :=
  fp_head_output F β hβ K hinj i₀ M hb
    (fun j _ => lt_of_lt_of_le hu (half_le_key_dist_mid K i₀ j)) V C hC

/-- The remaining hypotheses are satisfiable for every integer family: the
sorted keys are finitely many, hence bounded, and exact arithmetic meets the
precision condition. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) :
    ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
        |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧ exactArith.u * M < 1 / 2 := by
  obtain ⟨B, hB⟩ := exists_bound_sortedKey (fun i => (K i : ℝ))
  exact ⟨B, hB, by simp [exactArith]⟩

/-- The hypotheses of `fp_head_output` are satisfiable, and the separation one
is no accident: by `half_le_key_dist_mid` a stored key is never within half a
unit of a breakpoint, so with exact arithmetic every integer family meets the
conditions at every one of its own keys. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) (i₀ : Fin n) :
    ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
        |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧
      ∀ j < keyCard (fun i => (K i : ℝ)) - 1,
        exactArith.u * M < |(K i₀ : ℝ) - (sortedKey (fun i => (K i : ℝ)) j
          + sortedKey (fun i => (K i : ℝ)) (j + 1)) / 2| := by
  obtain ⟨B, hB⟩ := exists_bound_sortedKey (fun i => (K i : ℝ))
  exact ⟨B, hB, fun j _ => lt_of_lt_of_le (by simp [exactArith]) (half_le_key_dist_mid K i₀ j)⟩

end ALM
end Transformer
