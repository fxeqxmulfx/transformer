/-
# The length-free bound in every key dimension

`Transformer.ALM.ScalarSharp` proves that for distinct integer *scalar* keys
the winner's softmax weight is bounded below by a quantity free of the token
count `n`.  The machine's `LookUp` compiles to scalar keys, so that is the
case it needs; but the paraboloid embedding of `Transformer.ALM.Defs` is
stated in every dimension, and the bound should be too.

The scalar proof counts the keys at each distance from the query, which fails
in dimension `m`: the shell `‖x‖² = t` carries `r_m(t)` lattice points, and
`r_m` is unbounded.  `Transformer.ALM.Theta` replaces that count by the
factorization of the whole lattice sum, and what comes out is the scalar bound
raised to the `m`-th power:

  `w i₀ ≥ 1 / (1 + 2 e^{-β}/(1 - e^{-3β}))^m`.

Still no `n`.  The dimension enters as an exponent, so the inverse temperature
needed for an exact simulation grows like `m`, not like the length of the
trace — which is the architectural point `Transformer.ALM.Lattice` makes, now
in the generality the head is defined in.

`softmax_winner_int_sharp_one` checks the specialization: at `m = 1` this is
exactly the bound of `Transformer.ALM.ScalarSharp`, since `x^1 = x`.

* E. Grosswald, *Representations of Integers as Sums of Squares*, Springer
  1985, §1 — the theta function whose `m`-th power appears here.
-/

import Transformer.ALM.Theta
import Transformer.ALM.Basic

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n : ℕ}

/-! ### Integer keys as points of the head's key space -/

/-- An integer vector as a key of the head. -/
noncomputable def embInt (k : Fin m → ℤ) : EucSpace m :=
  WithLp.toLp 2 (fun i => (k i : ℝ))

@[simp] lemma embInt_apply (k : Fin m → ℤ) (i : Fin m) :
    embInt k i = (k i : ℝ) := rfl

/-- **The exact deficiency on the lattice.**  `score_gap` of
`Transformer.ALM.Basic` says the shortfall of a key is its squared distance to
the query; on integer keys that distance is a natural number. -/
theorem score_gap_int (q k : Fin m → ℤ) :
    score (embInt q) (embInt k) - score (embInt q) (embInt q)
      = -((sqNorm (fun i => k i - q i) : ℕ) : ℝ) := by
  have h := score_gap (embInt q) (embInt k)
  rw [norm_sq_eq_sum] at h
  have hco : ∀ i, (embInt k - embInt q) i = (k i : ℝ) - (q i : ℝ) := by
    intro i; simp
  simp only [hco] at h
  rw [sqNorm_cast]
  have hcast : ∑ i, (((k i - q i : ℤ) : ℝ)) ^ 2 = ∑ i, ((k i : ℝ) - (q i : ℝ)) ^ 2 := by
    refine Finset.sum_congr rfl fun i _ => ?_
    push_cast
    ring
  rw [hcast]
  linarith

/-! ### The competitor sum -/

/-- **The competitor sum over a lattice of keys.**  Distinct integer keys give
distinct nonzero difference vectors, so the competitors inject into the
punctured lattice and their total is at most `θ(r)^m - 1`, with
`θ(r) = 1 + 2r/(1 - r³)` and `r = e^{-β}`.

The `-1` is the origin, which the query itself occupies and no competitor
does. -/
theorem sum_exp_lattice_le (β : ℝ) (hβ : 0 < β)
    (K : Fin n → (Fin m → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n) :
    ∑ j ∈ Finset.univ.erase i₀,
        Real.exp (β * (score (embInt (K i₀)) (embInt (K j))
          - score (embInt (K i₀)) (embInt (K i₀))))
      ≤ (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3))) ^ m - 1 := by
  set E := Finset.univ.erase i₀ with hE
  set r := Real.exp (-β) with hr
  have hr0 : 0 < r := Real.exp_pos _
  have hr1 : r < 1 := by rw [hr]; exact Real.exp_lt_one_iff.mpr (by linarith)
  set v : Fin n → (Fin m → ℤ) := fun j i => K j i - K i₀ i with hv
  -- every term is exactly a power of `r`
  have hterm : ∀ j ∈ E, Real.exp (β * (score (embInt (K i₀)) (embInt (K j))
      - score (embInt (K i₀)) (embInt (K i₀)))) = r ^ (sqNorm (v j)) := by
    intro j _
    rw [score_gap_int, hr, ← Real.exp_nat_mul]
    congr 1
    ring
  rw [Finset.sum_congr rfl hterm]
  -- the competitors inject into the punctured lattice
  have hvinj : ∀ a ∈ E, ∀ b ∈ E, v a = v b → a = b := by
    intro a _ b _ hab
    refine hinj (funext fun i => ?_)
    have hi := congrFun hab i
    simp only [hv] at hi
    omega
  have hvne : ∀ j ∈ E, v j ≠ 0 := by
    intro j hj h0
    refine (Finset.ne_of_mem_erase hj) (hinj (funext fun i => ?_))
    have hi := congrFun h0 i
    simp only [hv, Pi.zero_apply] at hi
    omega
  -- a box large enough to hold every difference
  set M := Finset.univ.sup (fun j => Finset.univ.sup (fun i => (v j i).natAbs)) with hM
  set box := Fintype.piFinset (fun _ : Fin m => Finset.Icc (-(M : ℤ)) (M : ℤ)) with hbox
  have hmem : ∀ j, v j ∈ box := by
    intro j
    rw [hbox, Fintype.mem_piFinset]
    intro i
    have h1 : (v j i).natAbs ≤ M := by
      rw [hM]
      exact le_trans (Finset.le_sup (f := fun i => (v j i).natAbs) (Finset.mem_univ i))
        (Finset.le_sup (f := fun j => Finset.univ.sup fun i => (v j i).natAbs)
          (Finset.mem_univ j))
    rw [Finset.mem_Icc]
    omega
  have hsub : E.image v ⊆ box.erase 0 := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := Finset.mem_image.mp hx
    exact Finset.mem_erase.mpr ⟨hvne j hj, hmem j⟩
  have h0mem : (0 : Fin m → ℤ) ∈ box := by
    rw [hbox, Fintype.mem_piFinset]
    intro i
    rw [Finset.mem_Icc]
    simp
  calc ∑ j ∈ E, r ^ (sqNorm (v j))
      = ∑ x ∈ E.image v, r ^ (sqNorm x) :=
        (Finset.sum_image (f := fun x => r ^ (sqNorm x)) hvinj).symm
    _ ≤ ∑ x ∈ box.erase 0, r ^ (sqNorm x) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun x _ _ => pow_nonneg hr0.le _
    _ = (∑ x ∈ box, r ^ (sqNorm x)) - 1 := by
        rw [Finset.sum_erase_eq_sub h0mem, sqNorm_zero, pow_zero]
    _ ≤ (1 + 2 * (r / (1 - r ^ 3))) ^ m - 1 := by
        have := box_sum_le (m := m) hr0.le hr1 M
        rw [← hbox] at this
        linarith

/-! ### The bound -/

/-- **The length-free softmax bound in dimension `m`.**  For distinct integer
keys,

  `w i₀ ≥ 1 / (1 + 2 e^{-β}/(1 - e^{-3β}))^m`,

with no dependence on the number of keys `n`.  This is
`softmax_winner_scalar_int_sharp` of `Transformer.ALM.ScalarSharp` with the
dimension restored. -/
theorem softmax_winner_int_sharp (β : ℝ) (hβ : 0 < β)
    (K : Fin n → (Fin m → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3))) ^ m
      ≤ Real.exp (β * score (embInt (K i₀)) (embInt (K i₀)))
        / ∑ j, Real.exp (β * score (embInt (K i₀)) (embInt (K j))) := by
  have h := softmax_winner_sharp β (fun j => score (embInt (K i₀)) (embInt (K j))) i₀ _
    (sum_exp_lattice_le β hβ K hinj i₀)
  rwa [show (1 : ℝ) + ((1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3))) ^ m - 1)
    = (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3))) ^ m by ring] at h

/-! ### It agrees with the scalar case -/

/-- In dimension one the head's score is the published scalar score. -/
theorem score_embInt_one (q k : Fin 1 → ℤ) :
    score (embInt q) (embInt k) = sScore (q 0) (k 0) := by
  have hinner : inner (𝕜 := ℝ) (embInt k) (embInt q) = (k 0 : ℝ) * (q 0 : ℝ) := by
    simp [PiLp.inner_apply, RCLike.inner_apply, mul_comm]
  have hnorm : ‖embInt k‖ ^ 2 = ((k 0 : ℝ)) ^ 2 := by
    rw [norm_sq_eq_sum, Fin.sum_univ_one, embInt_apply]
  unfold score sScore
  rw [hinner, hnorm]
  ring

/-- **The specialization is the scalar bound.**  At `m = 1` the exponent
disappears and `softmax_winner_int_sharp` is `softmax_winner_scalar_int_sharp`
verbatim — the generalization does not lose the case it generalizes. -/
theorem softmax_winner_int_sharp_one (β : ℝ) (hβ : 0 < β)
    (K : Fin n → (Fin 1 → ℤ)) (hinj : Function.Injective K) (i₀ : Fin n) :
    1 / (1 + 2 * (Real.exp (-β) / (1 - Real.exp (-β) ^ 3)))
      ≤ Real.exp (β * sScore (K i₀ 0) (K i₀ 0))
        / ∑ j, Real.exp (β * sScore (K i₀ 0) (K j 0)) := by
  have h := softmax_winner_int_sharp β hβ K hinj i₀
  rw [pow_one] at h
  simpa [score_embInt_one] using h

/-- The hypotheses are satisfiable: three distinct keys in dimension two. -/
example : Function.Injective (fun j : Fin 3 => fun _ : Fin 2 => (j.val : ℤ)) := by
  intro a b h
  have hab : (a.val : ℤ) = (b.val : ℤ) := congrFun h 0
  exact Fin.ext (by exact_mod_cast hab)

end ALM
end Transformer
