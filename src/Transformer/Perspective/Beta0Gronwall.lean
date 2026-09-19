/-
# §4 — `e:approxsphere`, the Grönwall comparison of `SA` with `e:Snonres0`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

One statement: `e:approxsphere`, `‖x_i^β(t) - x_i^0(t)‖ ≤ O(β) e^{3t}`.  It is
proved, from two inputs:

* the `β = 0` drift is `3`-Lipschitz on unit tuples
  (`Perspective.lipschitzOnWith_beta0Field`), which is the `e^{3t}`;
* the `SA` drift differs from it by at most `e^{2β} - 1` at every unit tuple
  (`Perspective.norm_SA_drift_sub_beta0Field_le`), which is the `O(β)`.

Grönwall (`dist_le_of_approx_trajectories_ODE_of_mem`) then compares the exact
trajectory of the `β = 0` field with the approximate one traced by `SA`.
-/

import Transformer.Perspective.Beta0Field
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Calculus.Deriv.Prod

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- Projecting is additive in the vector projected. -/
theorem proj_sub (a u v : EucSpace d) :
    proj d a u - proj d a v = proj d a (u - v) := by
  simp only [proj, inner_sub_right, sub_smul]
  abel

/-- **The `SA` drift is within `e^{2β} - 1` of the `β = 0` drift.**

At a tuple of unit vectors the attention weight `e^{β⟨x_i,x_j⟩}/Z_{β,i}` lies
between `e^{-2β}/n` and `e^{2β}/n`, because the exponent is in `[-β, β]` and the
partition function is in `[n e^{-β}, n e^{β}]`.  Each of the `n` weights is
therefore within `(e^{2β} - 1)/n` of the uniform weight `1/n`, and the two drifts
differ by the projection of a vector of norm at most `e^{2β} - 1`.

Source: arXiv:2312.10794v5, §4, the estimate feeding `e:approxsphere`. -/
theorem norm_SA_drift_sub_beta0Field_le {β : ℝ} (hβ : 0 ≤ β)
    (x : Idx n → EucSpace d) (hx : ∀ j : Idx n, ‖x j‖ = 1) (i : Idx n) :
    ‖proj d (x i)
        ((∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)))⁻¹ •
          ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) • x j)
      - beta0Field d n x i‖
      ≤ Real.exp (2 * β) - 1 := by
  have hn : 0 < n := i.pos
  have hn' : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  set ν : ℝ := ((n : ℝ))⁻¹ with hν
  have hνpos : 0 < ν := by positivity
  have hnν : (n : ℝ) * ν = 1 := by rw [hν, mul_inv_cancel₀ hn'.ne']
  -- the exponents lie in `[-β, β]`
  have hexp : ∀ j : Idx n,
      Real.exp (-β) ≤ Real.exp (β * inner (𝕜 := ℝ) (x i) (x j))
        ∧ Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) ≤ Real.exp β := by
    intro j
    have hcs : |inner (𝕜 := ℝ) (x i) (x j)| ≤ 1 := by
      have := abs_real_inner_le_norm (x i) (x j)
      rwa [hx i, hx j, one_mul] at this
    rw [abs_le] at hcs
    exact ⟨Real.exp_le_exp.mpr (by nlinarith [hcs.1]),
      Real.exp_le_exp.mpr (by nlinarith [hcs.2])⟩
  set Z : ℝ := ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x k)) with hZ
  have hZlb : (n : ℝ) * Real.exp (-β) ≤ Z := by
    have : ∑ _k : Idx n, Real.exp (-β) ≤ Z :=
      Finset.sum_le_sum fun k _ => (hexp k).1
    simpa [mul_comm] using this
  have hZub : Z ≤ (n : ℝ) * Real.exp β := by
    have : Z ≤ ∑ _k : Idx n, Real.exp β :=
      Finset.sum_le_sum fun k _ => (hexp k).2
    simpa [mul_comm] using this
  have hZpos : 0 < Z := lt_of_lt_of_le (by positivity) hZlb
  -- the difference of the two averaged vectors
  have hsplit : (Z⁻¹ • ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) • x j)
      - meanTuple d n x
      = ∑ j : Idx n,
          (Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) - ν) • x j := by
    rw [meanTuple, Finset.smul_sum, Finset.smul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by rw [smul_smul, sub_smul, hν]
  -- each weight is within `(e^{2β} - 1) ν` of the uniform one
  have hE1 : Real.exp (2 * β) * Real.exp (-β) = Real.exp β := by
    rw [← Real.exp_add]; ring_nf
  have hE2 : Real.exp (-(2 * β)) * Real.exp β = Real.exp (-β) := by
    rw [← Real.exp_add]; ring_nf
  have hE3 : (2 : ℝ) ≤ Real.exp (2 * β) + Real.exp (-(2 * β)) := by
    have hprod : Real.exp (2 * β) * Real.exp (-(2 * β)) = 1 := by
      rw [← Real.exp_add]; simp
    nlinarith [sq_nonneg (Real.exp (2 * β) - 1), Real.exp_pos (2 * β)]
  have hweight : ∀ j : Idx n,
      |Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) - ν|
        ≤ (Real.exp (2 * β) - 1) * ν := by
    intro j
    have hub : Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j))
        ≤ Real.exp (2 * β) * ν := by
      rw [inv_mul_eq_div, div_le_iff₀ hZpos]
      have hmid : Real.exp (2 * β) * ν * ((n : ℝ) * Real.exp (-β))
          = (Real.exp (2 * β) * Real.exp (-β)) * ((n : ℝ) * ν) := by ring
      calc Real.exp (β * inner (𝕜 := ℝ) (x i) (x j))
          ≤ Real.exp β := (hexp j).2
        _ = Real.exp (2 * β) * ν * ((n : ℝ) * Real.exp (-β)) := by
            rw [hmid, hE1, hnν, mul_one]
        _ ≤ Real.exp (2 * β) * ν * Z := by
            exact mul_le_mul_of_nonneg_left hZlb (by positivity)
    have hlb : Real.exp (-(2 * β)) * ν
        ≤ Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) := by
      rw [inv_mul_eq_div, le_div_iff₀ hZpos]
      have hmid : Real.exp (-(2 * β)) * ν * ((n : ℝ) * Real.exp β)
          = (Real.exp (-(2 * β)) * Real.exp β) * ((n : ℝ) * ν) := by ring
      calc Real.exp (-(2 * β)) * ν * Z
          ≤ Real.exp (-(2 * β)) * ν * ((n : ℝ) * Real.exp β) := by
            exact mul_le_mul_of_nonneg_left hZub (by positivity)
        _ = Real.exp (-β) := by rw [hmid, hE2, hnν, mul_one]
        _ ≤ Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) := (hexp j).1
    rw [abs_le]
    constructor
    · nlinarith [hνpos, hE3]
    · nlinarith [hνpos]
  -- assemble
  rw [beta0Field, proj_sub, hsplit]
  refine (norm_proj_le (hx i) _).trans ?_
  refine (norm_sum_le _ _).trans ?_
  have hterm : ∀ j : Idx n,
      ‖(Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) - ν) • x j‖
        ≤ (Real.exp (2 * β) - 1) * ν := by
    intro j
    rw [norm_smul, Real.norm_eq_abs, hx j, mul_one]
    exact hweight j
  calc ∑ j : Idx n, ‖(Z⁻¹ * Real.exp (β * inner (𝕜 := ℝ) (x i) (x j)) - ν) • x j‖
      ≤ ∑ _j : Idx n, (Real.exp (2 * β) - 1) * ν :=
        Finset.sum_le_sum fun j _ => hterm j
    _ = (n : ℝ) * ((Real.exp (2 * β) - 1) * ν) := by
        simp [mul_comm]
    _ = Real.exp (2 * β) - 1 := by
        rw [← mul_assoc, mul_comm ((n : ℝ)), mul_assoc, hnν, mul_one]

/-- **Equation (e:approxsphere).** Grönwall bound:

  `‖x_i^β(t) - x_i^0(t)‖ ≤ O(β) e^{3t}`.

**What the source says and what is changed here.**  The quantifier on the
constant is hoisted in front of `β` and in front of the two solutions.  Put
after them — as an earlier form of this statement had it — the inequality says
nothing: for `β > 0` the constant `C = 2/β` already works, since two points of
`𝕊^{d-1}` are never further apart than `2 ≤ 2 e^{3t}`.  The survey's `O(β)` is a
constant depending on `d` and `n` alone, and in fact on neither: `C = 2 e²` is
proved here.  With the quantifier in front the bound has content at every `β`,
including `β = 0`, where it asserts that `e:Snonres0` has at most one solution
from a given initial tuple.

The proof is the survey's.  The `β = 0` drift is `3`-Lipschitz on unit tuples
(`lipschitzOnWith_beta0Field`) — that is the `e^{3t}` — and the `SA` drift
differs from it by at most `e^{2β} - 1` (`norm_SA_drift_sub_beta0Field_le`) —
that is the `O(β)`, through `e^{2β} - 1 ≤ 2 β e^{2β}`.  Grönwall's inequality
for an exact and an approximate trajectory of the same field closes it for
`β ≤ 1`; for `β > 1` the trivial bound `‖x_i^β - x_i^0‖ ≤ 2` is already below
`2 e² β e^{3t}`.

Source: arXiv:2312.10794v5, §4, `e:approxsphere`. -/
theorem solutions_close_at_small_beta :
    ∃ C : ℝ, 0 < C ∧
      ∀ β : ℝ, 0 ≤ β →
        ∀ Xβ X0t : ℝ → SphereTuple d n,
          Xβ 0 = X0t 0 →
          Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
          ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
            ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖
              ≤ C * β * Real.exp (3 * t) := by
  refine ⟨2 * Real.exp 2, by positivity, ?_⟩
  intro β hβ Xβ X0t hinit hSA h0 t ht i
  have hE3t : (1 : ℝ) ≤ Real.exp (3 * t) := Real.one_le_exp (by linarith)
  have hE2 : (1 : ℝ) ≤ Real.exp 2 := Real.one_le_exp (by norm_num)
  rcases le_or_gt β 1 with hβ1 | hβ1
  · -- the Grönwall regime `β ≤ 1`
    have hderf : ∀ s : ℝ,
        HasDerivAt (fun r : ℝ => fun j : Idx n => ((Xβ r j : EucSpace d)))
          (fun j : Idx n =>
            proj d ((Xβ s j : EucSpace d))
              ((partitionSA d n β Xβ s j)⁻¹ • ∑ k : Idx n,
                Real.exp (β * inner (𝕜 := ℝ) ((Xβ s j : EucSpace d))
                    ((Xβ s k : EucSpace d)))
                  • ((Xβ s k : EucSpace d)))) s :=
      fun s => hasDerivAt_pi.mpr fun j => hSA s j
    have hderg : ∀ s : ℝ,
        HasDerivAt (fun r : ℝ => fun j : Idx n => ((X0t r j : EucSpace d)))
          (beta0Field d n (fun j : Idx n => ((X0t s j : EucSpace d)))) s :=
      fun s => hasDerivAt_pi.mpr fun j => h0 s j
    have hεnn : (0 : ℝ) ≤ Real.exp (2 * β) - 1 :=
      sub_nonneg.mpr (Real.one_le_exp (by linarith))
    have key := dist_le_of_approx_trajectories_ODE_of_mem
      (v := fun _ : ℝ => beta0Field d n)
      (s := fun _ : ℝ => unitTuples d n)
      (K := 3)
      (f := fun r : ℝ => fun j : Idx n => ((Xβ r j : EucSpace d)))
      (f' := fun s : ℝ => fun j : Idx n =>
        proj d ((Xβ s j : EucSpace d))
          ((partitionSA d n β Xβ s j)⁻¹ • ∑ k : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) ((Xβ s j : EucSpace d))
                ((Xβ s k : EucSpace d)))
              • ((Xβ s k : EucSpace d))))
      (g := fun r : ℝ => fun j : Idx n => ((X0t r j : EucSpace d)))
      (g' := fun s : ℝ => beta0Field d n (fun j : Idx n => ((X0t s j : EucSpace d))))
      (εf := Real.exp (2 * β) - 1) (εg := 0) (δ := 0) (a := 0) (b := t)
      (hv := fun _ _ => lipschitzOnWith_beta0Field d n)
      (hf := fun s _ => (hderf s).continuousAt.continuousWithinAt)
      (hf' := fun s _ => (hderf s).hasDerivWithinAt)
      (f_bound := by
        intro s _
        rw [dist_pi_le_iff hεnn]
        intro j
        rw [dist_eq_norm]
        exact norm_SA_drift_sub_beta0Field_le d n hβ
          (fun k : Idx n => ((Xβ s k : EucSpace d)))
          (fun k => mem_sphere_zero_iff_norm.mp (Xβ s k).2) j)
      (hfs := fun s _ j => mem_sphere_zero_iff_norm.mp (Xβ s j).2)
      (hg := fun s _ => (hderg s).continuousAt.continuousWithinAt)
      (hg' := fun s _ => (hderg s).hasDerivWithinAt)
      (g_bound := fun s _ => by simp)
      (hgs := fun s _ j => mem_sphere_zero_iff_norm.mp (X0t s j).2)
      (ha := by simp [hinit])
      t (Set.right_mem_Icc.mpr ht)
    have hgb : gronwallBound 0 ((3 : ℝ≥0) : ℝ) ((Real.exp (2 * β) - 1) + 0) (t - 0)
        = (Real.exp (2 * β) - 1) / 3 * (Real.exp (3 * t) - 1) := by
      rw [gronwallBound_of_K_ne_0 (by norm_num)]
      norm_num
    rw [hgb] at key
    have hcoord : ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖
        ≤ dist (fun j : Idx n => ((Xβ t j : EucSpace d)))
            (fun j : Idx n => ((X0t t j : EucSpace d))) := by
      rw [← dist_eq_norm]
      exact dist_le_pi_dist (fun j : Idx n => ((Xβ t j : EucSpace d)))
        (fun j : Idx n => ((X0t t j : EucSpace d))) i
    refine hcoord.trans (key.trans ?_)
    have hexpb : Real.exp (2 * β) - 1 ≤ 2 * β * Real.exp 2 := by
      have h1 : -(2 * β) + 1 ≤ Real.exp (-(2 * β)) := Real.add_one_le_exp _
      have h2 : Real.exp (-(2 * β)) * Real.exp (2 * β) = 1 := by
        rw [← Real.exp_add]; simp
      have h3 : Real.exp (2 * β) ≤ Real.exp 2 := Real.exp_le_exp.mpr (by linarith)
      have h4 : (-(2 * β) + 1) * Real.exp (2 * β)
          ≤ Real.exp (-(2 * β)) * Real.exp (2 * β) :=
        mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
      rw [h2] at h4
      nlinarith [h3, hβ]
    have hbnn : (0 : ℝ) ≤ 2 * β * Real.exp 2 / 3 :=
      div_nonneg (mul_nonneg (by linarith) (Real.exp_pos 2).le) (by norm_num)
    calc (Real.exp (2 * β) - 1) / 3 * (Real.exp (3 * t) - 1)
        ≤ 2 * β * Real.exp 2 / 3 * Real.exp (3 * t) :=
          mul_le_mul (by linarith) (by linarith) (by linarith) hbnn
      _ ≤ 2 * Real.exp 2 * β * Real.exp (3 * t) := by
          nlinarith [mul_nonneg (mul_nonneg hβ (Real.exp_pos 2).le)
            (Real.exp_pos (3 * t)).le]
  · -- `β > 1`: the trivial bound on the sphere already suffices
    have hnx : ‖((Xβ t i : EucSpace d))‖ = 1 := mem_sphere_zero_iff_norm.mp (Xβ t i).2
    have hny : ‖((X0t t i : EucSpace d))‖ = 1 := mem_sphere_zero_iff_norm.mp (X0t t i).2
    have h2 : ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖ ≤ 2 := by
      calc ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖
          ≤ ‖((Xβ t i : EucSpace d))‖ + ‖((X0t t i : EucSpace d))‖ := norm_sub_le _ _
        _ = 2 := by rw [hnx, hny]; norm_num
    refine h2.trans ?_
    have hp1 : (1 : ℝ) ≤ Real.exp 2 * β := by nlinarith
    have hp2 : (1 : ℝ) ≤ Real.exp 2 * β * Real.exp (3 * t) := by nlinarith
    linarith

/-- The hypotheses of `norm_SA_drift_sub_beta0Field_le` are satisfiable: one
token at the base point, at `β = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ ∀ _j : Idx 1, ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
  ⟨le_rfl, fun _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2⟩

end Perspective
end Transformer
