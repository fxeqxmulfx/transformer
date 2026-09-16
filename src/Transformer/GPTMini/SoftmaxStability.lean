/-
# How far the causal softmax can move

`reference/model.py` (`CausalMHA.forward`) turns the scores into weights with
one `softmax` per query position.  Everything downstream of it only ever sees
the weight *vector*, so the quantity that controls the whole head is the ℓ¹
distance between the weight vectors of two different score matrices.

The estimate is elementary and has nothing to do with attention:

  1. if every score moves by at most `r`, the partition function moves by a
     factor in `[e^{-r}, e^{r}]`, so each weight moves by a factor in
     `[e^{-2r}, e^{2r}]` (`causalAttnWeights_le_mul`);
  2. `TotalVariation.l1_le_of_le_mul` turns that into `2 (e^{2r} - 1)` in ℓ¹;
  3. `TotalVariation.l1_le_of_le_exp` trades the multiplicative bound against
     the trivial bound `2` and gets `8 r`, linear in `r` for every `r ≥ 0`.

The second form is the one that matters: `2 (e^{2r} - 1)` is unbounded in `r`
while the ℓ¹ distance it estimates never exceeds `2`, so only `8 r` makes a
head *globally* Lipschitz.  Combined with `QKNormLipschitz.score_lipschitz`,
which bounds `r` by the movement of the normalized queries and keys, this is
what carries the Lipschitz constant of the whole block.
-/

import Transformer.GPTMini.AttentionBounds
import Transformer.GPTMini.TotalVariation

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- **Scores that are `r`-close give weights within a factor `e^{2r}`.**

If `|s_{i,j} - s'_{i,j}| ≤ r` for every key position `j`, then

  `a_{i,j} ≤ e^{2r} · a'_{i,j}`

for every `j` — one factor `e^{r}` from the numerator, one from the partition
function.  The masked positions `j > i` are `0` on both sides.

Source: `reference/model.py` (`CausalMHA.forward`), the `softmax` over
`scores + mask`. -/
theorem causalAttnWeights_le_mul
    {T : ℕ} (alpha eps r : ℝ)
    (q k q' k' : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (hclose : ∀ j : Fin T,
      |preScore cfg alpha eps q k i j - preScore cfg alpha eps q' k' i j| ≤ r)
    (j : Fin T) :
    causalAttnWeights cfg alpha eps q k i j
      ≤ Real.exp (2 * r) * causalAttnWeights cfg alpha eps q' k' i j := by
  classical
  have hpt : ∀ j' : Fin T, Real.exp (preScore cfg alpha eps q k i j')
      ≤ Real.exp r * Real.exp (preScore cfg alpha eps q' k' i j') := by
    intro j'
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [le_of_abs_le (hclose j')])
  have hpt' : ∀ j' : Fin T, Real.exp (preScore cfg alpha eps q' k' i j')
      ≤ Real.exp r * Real.exp (preScore cfg alpha eps q k i j') := by
    intro j'
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [neg_le_of_abs_le (hclose j')])
  unfold causalAttnWeights
  set S := ∑ j' : Fin T,
      (if (j' : ℕ) ≤ (i : ℕ) then Real.exp (preScore cfg alpha eps q k i j') else 0)
    with hSdef
  set S' := ∑ j' : Fin T,
      (if (j' : ℕ) ≤ (i : ℕ) then Real.exp (preScore cfg alpha eps q' k' i j') else 0)
    with hS'def
  have hSpos : 0 < S := by
    rw [hSdef]
    refine Finset.sum_pos' (fun j' _ => ?_) ⟨i, Finset.mem_univ _, ?_⟩
    · split_ifs
      · exact (Real.exp_pos _).le
      · exact le_rfl
    · rw [ite_eq_left (le_refl (i : ℕ))]
      exact Real.exp_pos _
  have hS'pos : 0 < S' := by
    rw [hS'def]
    refine Finset.sum_pos' (fun j' _ => ?_) ⟨i, Finset.mem_univ _, ?_⟩
    · split_ifs
      · exact (Real.exp_pos _).le
      · exact le_rfl
    · rw [ite_eq_left (le_refl (i : ℕ))]
      exact Real.exp_pos _
  have hS'le : S' ≤ Real.exp r * S := by
    rw [hSdef, hS'def, Finset.mul_sum]
    refine Finset.sum_le_sum fun j' _ => ?_
    split_ifs
    · exact hpt' j'
    · simp
  split_ifs with h
  · simp
  · rw [mul_div_assoc', div_le_div_iff₀ hSpos hS'pos]
    calc Real.exp (preScore cfg alpha eps q k i j) * S'
        ≤ (Real.exp r * Real.exp (preScore cfg alpha eps q' k' i j)) * (Real.exp r * S) :=
          mul_le_mul (hpt j) hS'le hS'pos.le (by positivity)
      _ = Real.exp (2 * r) * Real.exp (preScore cfg alpha eps q' k' i j) * S := by
          rw [show (2 : ℝ) * r = r + r from by ring, Real.exp_add]; ring

/-- The hypotheses are satisfiable: a score matrix is `0`-close to itself. -/
example (cfg : Config) (alpha eps : ℝ)
    (q k : Fin 3 → EucSpace cfg.head_dim) (i j : Fin 3) :
    causalAttnWeights cfg alpha eps q k i j
      ≤ Real.exp (2 * 0) * causalAttnWeights cfg alpha eps q k i j :=
  causalAttnWeights_le_mul cfg alpha eps 0 q k q k i (fun _ => by simp) j

/-- **ℓ¹ stability of the causal softmax.**

If every score at query position `i` moves by at most `r ≥ 0`, the whole row
of attention weights moves by at most `2 (e^{2r} - 1)` in ℓ¹.

Source: `reference/model.py` (`CausalMHA.forward`); the proof is
`causalAttnWeights_le_mul` fed to `l1_le_of_le_mul` at `c = e^{2r}`, using
`causalAttnWeights_row_sum` for both rows. -/
theorem causalAttnWeights_l1_le
    {T : ℕ} (alpha eps r : ℝ)
    (q k q' k' : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (hclose : ∀ j : Fin T,
      |preScore cfg alpha eps q k i j - preScore cfg alpha eps q' k' i j| ≤ r) :
    (∑ j : Fin T, |causalAttnWeights cfg alpha eps q k i j
        - causalAttnWeights cfg alpha eps q' k' i j|)
      ≤ 2 * (Real.exp (2 * r) - 1) :=
  l1_le_of_le_mul _ _ (Real.exp (2 * r))
    (fun j => causalAttnWeights_nonneg cfg alpha eps q' k' i j)
    (fun j => causalAttnWeights_le_mul cfg alpha eps r q k q' k' i hclose j)
    (causalAttnWeights_row_sum cfg alpha eps q k i)
    (causalAttnWeights_row_sum cfg alpha eps q' k' i)

/-- The hypotheses are satisfiable: a score matrix is `0`-close to itself. -/
example (cfg : Config) (alpha eps : ℝ)
    (q k : Fin 3 → EucSpace cfg.head_dim) (i : Fin 3) :
    (∑ j : Fin 3, |causalAttnWeights cfg alpha eps q k i j
        - causalAttnWeights cfg alpha eps q k i j|)
      ≤ 2 * (Real.exp (2 * 0) - 1) :=
  causalAttnWeights_l1_le cfg alpha eps 0 q k q k i (fun _ => by simp)

/-- **Linear ℓ¹ stability of the causal softmax.**

If every score at query position `i` moves by at most `r`, the whole row of
attention weights moves by at most `8 r` in ℓ¹ — with no restriction on `r`,
since for large `r` the trivial bound `2` takes over.

Source: `reference/model.py` (`CausalMHA.forward`); `causalAttnWeights_le_mul`
fed to `TotalVariation.l1_le_of_le_exp`. -/
theorem causalAttnWeights_l1_le_linear
    {T : ℕ} (alpha eps r : ℝ)
    (q k q' k' : Fin T → EucSpace cfg.head_dim) (i : Fin T)
    (hclose : ∀ j : Fin T,
      |preScore cfg alpha eps q k i j - preScore cfg alpha eps q' k' i j| ≤ r) :
    (∑ j : Fin T, |causalAttnWeights cfg alpha eps q k i j
        - causalAttnWeights cfg alpha eps q' k' i j|)
      ≤ 8 * r :=
  l1_le_of_le_exp _ _ r ((abs_nonneg _).trans (hclose i))
    (fun j => causalAttnWeights_nonneg cfg alpha eps q k i j)
    (fun j => causalAttnWeights_nonneg cfg alpha eps q' k' i j)
    (fun j => causalAttnWeights_le_mul cfg alpha eps r q k q' k' i hclose j)
    (causalAttnWeights_row_sum cfg alpha eps q k i)
    (causalAttnWeights_row_sum cfg alpha eps q' k' i)

/-- The hypotheses are satisfiable: a score matrix is `0`-close to itself. -/
example (cfg : Config) (alpha eps : ℝ)
    (q k : Fin 3 → EucSpace cfg.head_dim) (i : Fin 3) :
    (∑ j : Fin 3, |causalAttnWeights cfg alpha eps q k i j
        - causalAttnWeights cfg alpha eps q k i j|)
      ≤ 8 * (0 : ℝ) :=
  causalAttnWeights_l1_le_linear cfg alpha eps 0 q k q k i (fun _ => by simp)

end GPTMini
end Transformer
