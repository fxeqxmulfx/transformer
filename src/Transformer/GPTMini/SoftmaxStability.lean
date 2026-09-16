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
  2. two probability vectors related by `a ≤ c · b` pointwise are within
     `2 (c - 1)` in ℓ¹ (`l1_le_of_le_mul`), because `|t| = 2 t⁺ - t` and the
     `-t` part sums to `0`;
  3. `e^{2r} - 1 ≤ 2 r e^{2r}` (`exp_two_mul_sub_one_le`) turns that into a
     bound linear in `r` on any bounded range of `r`.

Together: `Σ_j |a_{i,j} - a'_{i,j}| ≤ 4 r e^{2r}`.  Combined with
`QKNormLipschitz.score_lipschitz`, which bounds `r` by the movement of the
normalized queries and keys, this is what makes one head Lipschitz.
-/

import Transformer.GPTMini.AttentionBounds

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-! ### Two elementary inequalities -/

/-- **ℓ¹ distance of two distributions from a pointwise ratio bound.**

If `a` and `b` are probability vectors with `a j ≤ c · b j` for every `j`,
then `Σ_j |a j - b j| ≤ 2 (c - 1)`.

The proof is the standard total-variation identity `|t| = 2 t⁺ - t`: summing
it, the linear part cancels because both vectors sum to `1`, and the positive
part is bounded by `(c - 1) b j` termwise.

Source: the classical bound on the total variation distance; used here for
the softmax of `reference/model.py` (`CausalMHA.forward`). -/
theorem l1_le_of_le_mul {T : ℕ} (a b : Fin T → ℝ) (c : ℝ)
    (hb : ∀ j, 0 ≤ b j) (hab : ∀ j, a j ≤ c * b j)
    (ha1 : ∑ j, a j = 1) (hb1 : ∑ j, b j = 1) :
    ∑ j, |a j - b j| ≤ 2 * (c - 1) := by
  have habs : ∀ t : ℝ, |t| = 2 * max 0 t - t := by
    intro t
    rcases le_total 0 t with h | h
    · rw [abs_of_nonneg h, max_eq_right h]; ring
    · rw [abs_of_nonpos h, max_eq_left h]; ring
  have hrw : (∑ j, |a j - b j|)
      = 2 * (∑ j, max 0 (a j - b j)) - ((∑ j, a j) - ∑ j, b j) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => habs _
  have hc : (1 : ℝ) ≤ c := by
    have h := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hab j
    rwa [ha1, ← Finset.mul_sum, hb1, mul_one] at h
  have hmax : ∀ j, max 0 (a j - b j) ≤ (c - 1) * b j := by
    intro j
    refine max_le ?_ ?_
    · nlinarith [hb j]
    · nlinarith [hab j]
  have hsum : (∑ j, max 0 (a j - b j)) ≤ (c - 1) * ∑ j, b j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => hmax j
  rw [hb1, mul_one] at hsum
  rw [hrw, ha1, hb1, sub_self, sub_zero]
  linarith

/-- The hypotheses are satisfiable: any probability vector is related to
itself with `c = 1`. -/
example : ∑ _j : Fin 2, |(1 / 2 : ℝ) - 1 / 2| ≤ 2 * ((1 : ℝ) - 1) :=
  l1_le_of_le_mul (fun _ => 1 / 2) (fun _ => 1 / 2) 1
    (fun _ => by norm_num) (fun _ => by norm_num)
    (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])

/-- **`e^{2r} - 1 ≤ 2 r e^{2r}`.**

The convexity bound `x + 1 ≤ e^x` at `x = -2r`, multiplied by `e^{2r} > 0`.
It converts the multiplicative softmax estimate into an additive one. -/
theorem exp_two_mul_sub_one_le (r : ℝ) :
    Real.exp (2 * r) - 1 ≤ 2 * r * Real.exp (2 * r) := by
  have h := Real.add_one_le_exp (-(2 * r))
  have hpos : (0 : ℝ) < Real.exp (2 * r) := Real.exp_pos _
  have hmul : (-(2 * r) + 1) * Real.exp (2 * r)
      ≤ Real.exp (-(2 * r)) * Real.exp (2 * r) :=
    mul_le_mul_of_nonneg_right h hpos.le
  rw [← Real.exp_add, neg_add_cancel, Real.exp_zero] at hmul
  nlinarith

/-! ### The causal softmax -/

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

end GPTMini
end Transformer
