/-
# DASH — finite NDB chains for every dyadic root

arXiv:2602.02016v2, §3.3. For any fixed number of calls, the computed
square-root chain and a final inverse-square-root call converge to
the stated dyadic powers as the iteration count of each call grows.
-/

import Transformer.DASH.Section3_FiniteChaining

open scoped Topology
open Filter

noncomputable section

namespace Transformer.DASH

/-- Every positive dyadic root of an NDB input stays inside `(0,2)`.
Source: arXiv:2602.02016v2, §3.3, chaining calls for powers `1/2^m`. -/
theorem dyadic_scalar_ndb_domain (a : ℝ) (ha : 0 < a) (ha' : a < 2) (m : ℕ) :
    0 < a ^ ((1 / 2 : ℝ) ^ m) ∧ a ^ ((1 / 2 : ℝ) ^ m) < 2 := by
  induction m with
  | zero => simpa using And.intro ha ha'
  | succ m ih =>
    rw [pow_succ, Real.rpow_mul ha.le]
    constructor
    · exact Real.rpow_pos_of_pos ih.1 _
    · rw [← Real.sqrt_eq_rpow]
      apply (Real.sqrt_lt ih.1.le (by norm_num : (0 : ℝ) ≤ 2)).2
      linarith [ih.2]

/-- Dyadic-domain hypotheses have a non-unit example,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 2 := by norm_num

/-- A fixed number of actual finite scalar NDB square-root calls.
Source: arXiv:2602.02016v2, §3.3, repeated numerical solver composition. -/
def ndbFiniteSquareChain (a : ℝ) (iterations : ℕ) : ℕ → ℝ
  | 0 => a
  | m + 1 => (ndbScalar (ndbFiniteSquareChain a iterations m) iterations).1

/-- For any fixed chain length, all finite square-root calls converge
together to the corresponding dyadic power. Errors in intermediate calls
are included through the proved moving-input convergence theorem.
Source: arXiv:2602.02016v2, §3.3, powers `A^(1/2^m)`. -/
theorem ndbFiniteSquareChain_convergence (a : ℝ) (ha : 0 < a) (ha' : a < 2) (m : ℕ) :
    Tendsto (fun k : ℕ => ndbFiniteSquareChain a (k + 1) m) atTop
      (𝓝 (a ^ ((1 / 2 : ℝ) ^ m))) := by
  induction m with
  | zero => simp [ndbFiniteSquareChain]
  | succ m ih =>
    obtain ⟨hpos, hupper⟩ := dyadic_scalar_ndb_domain a ha ha' m
    have h := (ndb_scalar_moving_input (fun k => ndbFiniteSquareChain a (k + 1) m)
      (a ^ ((1 / 2 : ℝ) ^ m)) hpos hupper ih).fst_nhds
    dsimp only at h
    rw [← Real.rpow_mul ha.le, ← pow_succ] at h
    exact h

/-- A positive finite-chain input exists, arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := by norm_num

/-- A final finite inverse-square-root call after `m` finite square-root
calls converges to the negative dyadic power `-1/2^(m+1)`.
Source: arXiv:2602.02016v2, §3.3, the powers available by chaining NDB. -/
theorem ndbFiniteInverseChain_convergence (a : ℝ) (ha : 0 < a) (ha' : a < 2) (m : ℕ) :
    Tendsto (fun k : ℕ => (ndbScalar (ndbFiniteSquareChain a (k + 1) m) (k + 1)).2)
      atTop (𝓝 (a ^ (-((1 / 2 : ℝ) ^ (m + 1))))) := by
  obtain ⟨hpos, hupper⟩ := dyadic_scalar_ndb_domain a ha ha' m
  have h := (ndb_scalar_moving_input (fun k => ndbFiniteSquareChain a (k + 1) m)
    (a ^ ((1 / 2 : ℝ) ^ m)) hpos hupper
    (ndbFiniteSquareChain_convergence a ha ha' m)).snd_nhds
  dsimp only at h
  rw [← Real.rpow_mul ha.le, mul_neg, ← pow_succ] at h
  exact h

/-- The final-inverse-call assumptions are satisfiable,
arXiv:2602.02016v2, §3.3. -/
example : (0 : ℝ) < 3 / 2 ∧ (3 / 2 : ℝ) < 2 := by norm_num

end Transformer.DASH
