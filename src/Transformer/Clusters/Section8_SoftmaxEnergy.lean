module

public import Transformer.Clusters.Section8_Polytope
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Softmax error in the identity-value energy

The energy derivative `e:derivativenorm`, arXiv:2305.05465v6, §8.1.
We bound the loss of a softmax average relative to any one score by `n/c`,
where `c > 0` is the inverse temperature. At `c = exp (2t)` this gives an
exponentially decaying error in the derivative of `||A z_i||²`, uniformly
over all configurations and all competing tokens.

The source separately estimates positive and negative attention gaps in
Step 2. Here the bound `r exp(-r) ≤ 1` bounds every weighted score regret,
and no estimate of the distance to the candidate set is assumed. Choosing
the self token also gives a global lower bound on the energy derivative.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The elementary exponential bound used for a weighted attention regret.
Source: arXiv:2305.05465v6, §8.1, the softmax estimate in Step 2 of the
proof of `t:Idcase11`. It holds for every real `r`. -/
theorem mul_exp_neg_le_one (r : ℝ) : r * Real.exp (-r) ≤ 1 := by
  have hle : r ≤ Real.exp r := by linarith [Real.add_one_le_exp r]
  calc
    r * Real.exp (-r) ≤ Real.exp r * Real.exp (-r) :=
      mul_le_mul_of_nonneg_right hle (Real.exp_pos _).le
    _ = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]

/-- A score-relative softmax bound for any nonempty indexed row.
Source: arXiv:2305.05465v6, §8.1, Step 2. This generalizes the proof of
`softmaxWeight_le_exp_sub` in `Section7_UnboundedParticles` from `m+1`
tokens to `n` tokens; the reference index witnesses nonemptiness. -/
theorem softmaxWeight_le_exp_difference (u : Idx n → ℝ) (i j : Idx n) :
    Perspective.softmaxWeight u j ≤ Real.exp (u j - u i) := by
  unfold Perspective.softmaxWeight
  have hden : Real.exp (u i) ≤ ∑ k : Idx n, Real.exp (u k) :=
    Finset.single_le_sum (fun k _ => (Real.exp_pos (u k)).le) (Finset.mem_univ i)
  rw [Real.exp_sub]
  exact div_le_div_of_nonneg_left (Real.exp_pos _).le (Real.exp_pos _) hden

/-- A token contributes at most `1/c` to the regret relative to a reference
score. Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm`, Step 2.
The reference need not maximize the scores. -/
theorem softmaxWeight_mul_score_regret_le (u : Idx n → ℝ) (c : ℝ) (hc : 0 < c)
    (i j : Idx n) :
    Perspective.softmaxWeight (fun k => c * u k) j * (u i - u j) ≤ c⁻¹ := by
  rcases le_or_gt (u i) (u j) with hij | hij
  · exact (mul_nonpos_of_nonneg_of_nonpos (Perspective.softmaxWeight_nonneg _ _)
      (sub_nonpos.mpr hij)).trans (inv_pos.mpr hc).le
  · calc
      _ ≤ Real.exp (c * u j - c * u i) * (u i - u j) :=
        mul_le_mul_of_nonneg_right (softmaxWeight_le_exp_difference _ i j)
          (sub_pos.mpr hij).le
      _ ≤ c⁻¹ := by
        rw [show c * u j - c * u i = -(c * (u i - u j)) by ring,
          ← one_div, le_div_iff₀ hc]
        convert mul_exp_neg_le_one (c * (u i - u j)) using 1; ring

/-- The temperature hypothesis is realized by a two-score row at `c=1`;
arXiv:2305.05465v6, §8.1, the estimate of `e:derivativenorm`. -/
example : (0 : ℝ) < 1 ∧ ∃ u : Idx 2 → ℝ, u 0 = 0 ∧ u 1 = 1 :=
  ⟨zero_lt_one, ⟨fun k => k.val, by norm_num, by norm_num⟩⟩

/-- A softmax average lies at most `n/c` below any reference score.
Source: arXiv:2305.05465v6, §8.1, Step 2 of `t:Idcase11`.
This estimate is uniform in the score vector. -/
theorem score_sub_card_div_le_softmax_average (u : Idx n → ℝ) (c : ℝ)
    (hc : 0 < c) (i : Idx n) :
    u i - (n : ℝ) / c ≤ ∑ j : Idx n,
      Perspective.softmaxWeight (fun k => c * u k) j * u j := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hsum := Perspective.sum_softmaxWeight hn (fun k => c * u k)
  have hbound := Finset.sum_le_sum (s := Finset.univ)
    (fun j _ => softmaxWeight_mul_score_regret_le u c hc i j)
  have hid : ∑ j : Idx n, Perspective.softmaxWeight (fun k => c * u k) j * (u i - u j) =
      u i - ∑ j : Idx n, Perspective.softmaxWeight (fun k => c * u k) j * u j := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
  rw [hid, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hbound
  rw [div_eq_mul_inv]
  linarith

/-- The average estimate's sole scalar hypothesis holds for a genuine
nonempty row, at unit inverse temperature. Source: arXiv:2305.05465v6, §8.1. -/
example : (0 : ℝ) < 1 ∧ Nonempty (Idx 2) := ⟨zero_lt_one, ⟨0⟩⟩

/-- **Energy lower bound for the actual rescaled dynamics.** The
derivative of `||A z_i||²` dominates twice every attention gap, up to
`2n exp(-2t)`. Source: arXiv:2305.05465v6, §8.1, `e:derivativenorm`.
The bound is valid at all real times and for every attention operator;
positive definiteness is needed only later to identify candidate limits. -/
theorem IdRescaledDynamics.norm_sq_deriv_lower_bound (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (t : ℝ) (i : Idx n) :
    ∃ v : ℝ, HasDerivAt (fun s => ‖A (Z s i)‖ ^ 2) v t ∧
      ∀ j : Idx n, 2 * (inner (𝕜 := ℝ) (A (Z t i)) (A (Z t j - Z t i)) -
        (n : ℝ) / Real.exp (2 * t)) ≤ v := by
  let u : Idx n → ℝ := fun j => inner (𝕜 := ℝ) (A (Z t i)) (A (Z t j))
  let p : Idx n → ℝ := scaledAttention A (Real.exp (2 * t)) (Z t) i
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hsum : ∑ j : Idx n, p j = 1 := Perspective.sum_softmaxWeight hn _
  have hd := (A.hasFDerivAt.comp_hasDerivAt t (hZ t i)).norm_sq
  refine ⟨2 * ∑ j : Idx n, p j * (u j - u i), hd.congr_deriv ?_, ?_⟩
  · simp only [map_sum, map_smul, inner_sum, real_inner_smul_right, map_sub,
      inner_sub_right]
    rfl
  · intro j
    have haverage := score_sub_card_div_le_softmax_average u (Real.exp (2 * t))
      (Real.exp_pos _) j
    change u j - (n : ℝ) / Real.exp (2 * t) ≤ ∑ k : Idx n, p k * u k at haverage
    have hid : ∑ k : Idx n, p k * (u k - u i) = (∑ k : Idx n, p k * u k) - u i := by
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
    rw [hid, map_sub, inner_sub_right]
    change 2 * (u j - u i - (n : ℝ) / Real.exp (2 * t)) ≤
      2 * ((∑ k : Idx n, p k * u k) - u i)
    linarith

/-- The dynamical hypothesis holds for two coincident stationary unit
tokens, with a genuine time and token index. Source: arXiv:2305.05465v6,
§8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧
    (0 : ℝ) ≤ 1 ∧ Nonempty (Idx 2) :=
  ⟨idRescaledDynamics_const 1 _, zero_le_one, ⟨0⟩⟩

/-- The self-token gap is zero, so the energy derivative is bounded
below by an integrable exponential error. Source: arXiv:2305.05465v6,
§8.1, `e:derivativenorm`. This global bound also covers times near a candidate. -/
theorem IdRescaledDynamics.norm_sq_deriv_ge_neg (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (t : ℝ) (i : Idx n) :
    ∃ v : ℝ, HasDerivAt (fun s => ‖A (Z s i)‖ ^ 2) v t ∧
      -(2 * (n : ℝ) / Real.exp (2 * t)) ≤ v := by
  obtain ⟨v, hv, hbound⟩ := hZ.norm_sq_deriv_lower_bound A Z t i
  refine ⟨v, hv, ?_⟩
  have hself := hbound i
  simp only [sub_self, map_zero, inner_zero_right] at hself
  convert hself using 1
  ring

/-- The global lower bound has an actual stationary two-token instance;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

end Transformer.Clusters
