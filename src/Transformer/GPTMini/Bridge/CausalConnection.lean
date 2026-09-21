/-
# Bridge: GPTMini's causal attention matches paper 2411.04990's `eq: csa`

Connects `Transformer.GPTMini.CausalMHA.causalAttnWeights` with the
causal self-attention dynamics `Causal.CSA` of `Transformer.Causal.Basic`.

The first two theorems are about the weights alone: `causalAttnWeights` has
the algebraic form of the `eq: csa` coefficient, with `β = e^{α_h}` and the
QK-normalized vectors in place of `Q x`, `K x`.  They do not mention
`Causal.CSA`, and they cannot put its theorems in scope by themselves: QK-norm
is not linear, so `normL2 eps (q i)` is `Q x_i` for no fixed matrix `Q` in
general, and in the head `q` and `k` are rotated by RoPE at each token's own
position, which no fixed `Q, K` reproduces.

`csa_of_causalAttn` is the bridge proper, in the regime where it holds: tokens
on the unit sphere (so QK-norm is the identity), no RoPE, and value matrix
`V = I`.  A trajectory moved by the projected head output is then a `Causal.CSA`
trajectory at `β = e^{α_h}`, `Q = K = V = I`.
-/

import Transformer.Basic
import Transformer.GPTMini.CausalMHA
import Transformer.Causal.Basic
import Transformer.GPTMini.Bridge.SphereResidence

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- **Structural correspondence.**

The causal weight `a_{i,j}^{(h)}` defined in `causalAttnWeights` has the
same algebraic form as in `Causal.Basic.CSA` (eq: csa):

  `a_{i,j} = exp(β ⟨Q x_i, K x_j⟩) / Σ_{j' ≤ i} exp(β ⟨Q x_i, K x_{j'}⟩)`,

with `β = e^{α_h}` (the per-head learnable inverse-temperature after
QK-norm) and the QK-normalized `q, k` in place of `Q x`, `K x`.

Source: `reference/model.py` (`CausalMHA.forward`) against arXiv:2411.04990,
`eq: csa`. -/
theorem causalAttnWeights_matches_eq_csa
    (cfg : Config)
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (j : ℕ) ≤ (i : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j =
      Real.exp (preScore cfg alpha eps q k i j)
        /
      (∑ j' : Fin T,
          if (j' : ℕ) ≤ (i : ℕ) then
            Real.exp (preScore cfg alpha eps q k i j')
          else 0) := by
  unfold causalAttnWeights
  -- For j ≤ i, the if-then-else collapses to the else branch.
  have : ¬ ((j : ℕ) > (i : ℕ)) := not_lt.mpr hij
  simp [this]

/-- **The weight is the CSA coefficient, at the temperature QK-norm sets.**

`Causal.Basic.CSA` weighs token `j` at position `k` by `Z_k⁻¹ e^{β ⟨Q x_k,
K x_j⟩}`.  This head weighs it by `causalAttnWeights`, and the two are the
same expression once the dictionary is read off: `β = e^{α_h}` is the
per-head inverse temperature, and the vectors CSA writes as `Q x` and `K x`
are the QK-normalized `normL2 eps (q i)` and `normL2 eps (k j)`.

This is a statement about one weight at given vectors, not about a
dynamics: it does not by itself place the head under `Causal.CSA` (see the
module header, and `csa_of_causalAttn` for the regime in which it does).

Source: `reference/model.py` (`CausalMHA.forward`) against arXiv:2411.04990,
`eq: csa`. -/
theorem causalAttnWeights_eq_csa_coeff
    (cfg : Config)
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (j : ℕ) ≤ (i : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j =
      (∑ j' : Fin T,
          if (j' : ℕ) ≤ (i : ℕ) then
            Real.exp (Real.exp alpha *
              inner (𝕜 := ℝ) (normL2 eps (q i)) (normL2 eps (k j')))
          else 0)⁻¹
      *
      Real.exp (Real.exp alpha *
        inner (𝕜 := ℝ) (normL2 eps (q i)) (normL2 eps (k j))) := by
  rw [causalAttnWeights, ite_eq_right (not_lt.mpr hij), div_eq_inv_mul]
  rfl

/-- The causal hypothesis both theorems above take is satisfiable: at
position `2` of a three-token window, `j = 0` is in the past. -/
example : ((0 : Fin 3) : ℕ) ≤ ((2 : Fin 3) : ℕ) := by decide

/-- QK-norm leaves a unit vector alone, for any `eps ≤ 1`. -/
theorem normL2_of_norm_eq_one {d : ℕ} (eps : ℝ) (heps : eps ≤ 1) (x : EucSpace d)
    (hx : ‖x‖ = 1) : normL2 eps x = x := by
  rw [normL2, hx, max_eq_left heps, div_one, one_smul]

/-- **The head, on the sphere, without RoPE and with `V = I`, is `eq: csa`.**
If every token moves by the tangential part of the causal head output — the
weights `causalAttnWeights` of the tokens themselves, value `x_j` — then the
trajectory solves `Causal.CSA` at `β = e^{α}` and `Q = K = V = I`.

`eps ≤ 1` is what makes QK-norm the identity on the sphere; the implementation
uses `eps = 10⁻⁶`.

Source: `reference/model.py` (`CausalMHA.forward`) against arXiv:2411.04990,
`eq: csa`. -/
theorem csa_of_causalAttn (cfg : Config) {n : ℕ} (alpha eps : ℝ) (heps : eps ≤ 1)
    (X : ℝ → SphereTuple cfg.head_dim n)
    (hX : ∀ (t : ℝ) (k : Idx n), HasDerivAt (fun s => (X s k : EucSpace cfg.head_dim))
      (proj cfg.head_dim (X t k : EucSpace cfg.head_dim)
        (∑ j : Idx n, causalAttnWeights cfg alpha eps
            (fun j => (X t j : EucSpace cfg.head_dim))
            (fun j => (X t j : EucSpace cfg.head_dim)) k j
          • (X t j : EucSpace cfg.head_dim))) t) :
    Causal.CSA cfg.head_dim n (Real.exp alpha) 1 1 1 X := by
  intro t k
  have hn : ∀ j : Idx n, normL2 eps (X t j : EucSpace cfg.head_dim) = X t j := fun j =>
    normL2_of_norm_eq_one eps heps _ (mem_sphere_zero_iff_norm.mp (X t j).2)
  convert hX t k using 2
  simp only [one_apply_eq_self, causalAttnWeights, preScore, score, hn,
    Finset.smul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  by_cases hjk : (j : ℕ) ≤ (k : ℕ)
  · simp only [hjk, not_lt.mpr hjk, ↓reduceIte, smul_smul, div_eq_inv_mul]
  · simp only [hjk, lt_of_not_ge hjk, ↓reduceIte, zero_smul, smul_zero]

/-- The hypothesis of `csa_of_causalAttn` is satisfiable: a single token at
rest.  Its causal weight is `1`, so the head output is the token itself, whose
tangential part is `0`. -/
example (cfg : Config) (x : SSphere cfg.head_dim) :
    ∀ (t : ℝ) (k : Idx 1), HasDerivAt (fun _ : ℝ => (x : EucSpace cfg.head_dim))
      (proj cfg.head_dim (x : EucSpace cfg.head_dim)
        (∑ j : Idx 1, causalAttnWeights cfg 0 1e-6
            (fun _ => (x : EucSpace cfg.head_dim))
            (fun _ => (x : EucSpace cfg.head_dim)) k j
          • (x : EucSpace cfg.head_dim))) t := by
  intro t k
  obtain rfl : k = 0 := Subsingleton.elim _ _
  have hw : causalAttnWeights cfg 0 1e-6 (fun _ : Fin 1 => (x : EucSpace cfg.head_dim))
      (fun _ => (x : EucSpace cfg.head_dim)) 0 0 = 1 := by
    simp [causalAttnWeights]
  have hp := proj_smul_self (mem_sphere_zero_iff_norm.mp x.2) 1
  rw [one_smul] at hp
  rw [Fin.sum_univ_one, hw, one_smul, hp]
  exact hasDerivAt_const t _

end Bridge
end GPTMini
end Transformer
