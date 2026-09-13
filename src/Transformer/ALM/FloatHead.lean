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
`fpProbe_eq_hullProbe`, which the last example discharges for every integer
family at every one of its own keys.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 203-215.
-/

import Transformer.ALM.SoftmaxValue
import Transformer.ALM.FloatIndex

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

/-- The hypotheses are satisfiable, and the separation one is no accident: a
stored key is never a breakpoint, because every breakpoint lies strictly
between two adjacent keys.  So with exact arithmetic every integer family
meets the conditions at every one of its own keys. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) (i₀ : Fin n) :
    ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
        |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧
      ∀ j < keyCard (fun i => (K i : ℝ)) - 1,
        exactArith.u * M < |(K i₀ : ℝ) - (sortedKey (fun i => (K i : ℝ)) j
          + sortedKey (fun i => (K i : ℝ)) (j + 1)) / 2| := by
  set Ks : Fin n → ℝ := fun i => (K i : ℝ) with hKs
  obtain ⟨b, -, hbmax⟩ := Finset.exists_max_image (Finset.range (keyCard Ks))
    (fun j => |sortedKey Ks j|) ⟨0, by simp [keyCard_pos Ks]⟩
  refine ⟨|sortedKey Ks b|, fun j hj => hbmax j (Finset.mem_range.mpr ?_), fun j hj => ?_⟩
  · have hpos := keyCard_pos Ks
    omega
  · have hmono : StrictMono (sortedKey Ks) := strictMono_nat_of_lt_succ (sortedKey_lt_succ Ks)
    obtain ⟨t, -, hteq⟩ := exists_sortedKey_eq Ks i₀
    have hzero : exactArith.u * |sortedKey Ks b| = 0 := by simp [exactArith]
    have hval : ((K i₀ : ℝ)) = sortedKey Ks t := hteq.symm
    rw [hzero, hval]
    rcases Nat.lt_or_ge t (j + 1) with hlt | hge
    · have h₁ : sortedKey Ks t ≤ sortedKey Ks j := hmono.monotone (by omega)
      have h₂ : sortedKey Ks j < sortedKey Ks (j + 1) := hmono (by omega)
      rw [abs_of_neg (by linarith)]
      linarith
    · have h₁ : sortedKey Ks (j + 1) ≤ sortedKey Ks t := hmono.monotone hge
      have h₂ : sortedKey Ks j < sortedKey Ks (j + 1) := hmono (by omega)
      rw [abs_of_pos (by linarith)]
      linarith

end ALM
end Transformer
