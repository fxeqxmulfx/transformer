module

public import Transformer.SequenceModel.Attention.Concentration
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Useful attention mass and local retrieval error

Source: `review/flat_spectrum.md`, *Local attention routing*, LR1--LR4.
Each position has its own state. Width is not the number of complete input
tables a model can store. These statements measure one causal attention row:
error of its weighted values, mass reaching a useful set, and a necessary
head-width bound for that mass under the actual source Q/K RMSNorm.
They allow arbitrary rotary phases. They do not construct a learned query,
prove that a useful set is found, or give a multi-layer impossibility result.
The value estimate is before XSA and the output projection.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open scoped BigOperators

/-- LR1: every actual causal softmax coefficient is nonnegative. -/
theorem causalWeight_nonneg {T : ℕ} (scores : Fin T → Fin T → ℝ) (i j : Fin T) :
    0 ≤ causalWeight scores i j := by
  unfold causalWeight
  split_ifs
  · exact div_nonneg (Real.exp_pos _).le (denominator_pos scores i).le
  · exact le_rfl

/-- LR1: the error of the retrieved value is bounded by the attention-weighted
errors of the individual pieces. Scores can depend on the whole visible input. -/
theorem retrieval_error_le {T d : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T)
    (values : Fin T → EucSpace d) (target : EucSpace d) (radius : Fin T → ℝ)
    (herror : ∀ j, ‖values j - target‖ ≤ radius j) :
    ‖(∑ j, causalWeight scores i j • values j) - target‖ ≤
      ∑ j, causalWeight scores i j * radius j := by
  have hsum : (∑ j, causalWeight scores i j • values j) - target =
      ∑ j, causalWeight scores i j • (values j - target) := by
    simp only [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul,
      causalWeight_sum, one_smul]
  rw [hsum]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (causalWeight_nonneg scores i j)]
  exact mul_le_mul_of_nonneg_left (herror j) (causalWeight_nonneg scores i j)

/-- LR1 for a useful set: good pieces have error at most good, the remaining
pieces at most bad. Attention mass on good pieces controls retrieval quality;
the theorem does not assume there is just one correct position. -/
theorem retrieval_error_of_useful_set {T d : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T)
    (values : Fin T → EucSpace d) (target : EucSpace d) (useful : Finset (Fin T))
    (good bad : ℝ) (hgood : ∀ j ∈ useful, ‖values j - target‖ ≤ good)
    (hbad : ∀ j ∉ useful, ‖values j - target‖ ≤ bad) :
    ‖(∑ j, causalWeight scores i j • values j) - target‖ ≤
      (∑ j ∈ useful, causalWeight scores i j) * good +
        (1 - ∑ j ∈ useful, causalWeight scores i j) * bad := by
  classical
  have hr := retrieval_error_le scores i values target (fun j => if j ∈ useful then good else bad)
    (fun j => by split_ifs with h; exact hgood j h; exact hbad j h)
  have hf : Finset.univ.filter (fun j : Fin T => j ∈ useful) = useful := by
    ext j
    simp
  have hid : ∀ j, causalWeight scores i j * (if j ∈ useful then good else bad) =
      (if j ∈ useful then causalWeight scores i j * (good - bad) else 0) +
        causalWeight scores i j * bad := by
    intro j
    split_ifs <;> ring
  simp_rw [hid] at hr
  rw [Finset.sum_add_distrib, ← Finset.sum_filter, hf, ← Finset.sum_mul,
    ← Finset.sum_mul, causalWeight_sum] at hr
  nlinarith

/-- LR2: with scores bounded by B, a set of r useful positions receives at
most r*exp(2B)/N mass, where N is the visible prefix length. The set may
contain future positions; their coefficients remain zero. -/
theorem useful_mass_le {T : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T)
    (useful : Finset (Fin T)) (B : ℝ) (hB : ∀ k ≤ i, |scores i k| ≤ B) :
    ∑ j ∈ useful, causalWeight scores i j ≤
      (useful.card : ℝ) * Real.exp (2 * B) / ((i.val : ℝ) + 1) := by
  calc
    ∑ j ∈ useful, causalWeight scores i j ≤
        ∑ _j ∈ useful, Real.exp (2 * B) / ((i.val : ℝ) + 1) :=
      Finset.sum_le_sum fun j _ => causalWeight_le scores i j B hB
    _ = _ := by rw [Finset.sum_const, nsmul_eq_mul, ← mul_div_assoc]

/-- LR2 with the actual source Q/K normalization and arbitrary RoPE angles. -/
theorem normalized_useful_mass_le {T d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k : Fin T → EucSpace d)
    (i : Fin T) (useful : Finset (Fin T)) :
    ∑ j ∈ useful, causalWeight
      (fun a b => headScore d eps (angle a) (angle b) (q a) (k b)) i j ≤
      (useful.card : ℝ) * Real.exp (2 * Real.sqrt (d : ℝ)) / ((i.val : ℝ) + 1) := by
  apply useful_mass_le
  intro j _
  exact headScore_bound d hd eps heps _ _ _ _

/-- LR3: reaching positive mass mu on r positions requires a logarithmic
score budget, not an exponential width budget. This is necessary only. -/
theorem retrieval_log_budget {T d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k : Fin T → EucSpace d)
    (i : Fin T) (useful : Finset (Fin T)) (hu : useful.Nonempty)
    (mu : ℝ) (hmu : 0 < mu)
    (hmass : mu ≤ ∑ j ∈ useful, causalWeight
      (fun a b => headScore d eps (angle a) (angle b) (q a) (k b)) i j) :
    Real.log (((i.val : ℝ) + 1) * mu / (useful.card : ℝ)) ≤ 2 * Real.sqrt (d : ℝ) := by
  have hn : 0 < (i.val : ℝ) + 1 := by positivity
  have hc : 0 < (useful.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hu
  have hb := hmass.trans (normalized_useful_mass_le hd eps heps angle q k i useful)
  apply (Real.log_le_iff_le_exp (by positivity)).mpr
  apply (div_le_iff₀ hc).mpr
  have h := (le_div_iff₀ hn).mp hb
  nlinarith

/-- LR4: at fixed useful-set size and target mass, the necessary single-row
head-width bound grows as the square of a logarithm of context length.
It is not a sufficient routing construction or a lower bound for every task. -/
theorem retrieval_width_budget {T d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k : Fin T → EucSpace d)
    (i : Fin T) (useful : Finset (Fin T)) (hu : useful.Nonempty)
    (mu : ℝ) (hmu : 0 < mu)
    (hmass : mu ≤ ∑ j ∈ useful, causalWeight
      (fun a b => headScore d eps (angle a) (angle b) (q a) (k b)) i j) :
    (max 0 (Real.log (((i.val : ℝ) + 1) * mu / (useful.card : ℝ)))) ^ 2 ≤ 4 * (d : ℝ) := by
  have hl := retrieval_log_budget hd eps heps angle q k i useful hu mu hmu hmass
  have hs := Real.sqrt_nonneg (d : ℝ)
  have hm := max_le (show (0 : ℝ) ≤ 2 * Real.sqrt (d : ℝ) by positivity) hl
  have hnonneg := le_max_left 0 (Real.log (((i.val : ℝ) + 1) * mu / (useful.card : ℝ)))
  have hsq := Real.sq_sqrt (show (0 : ℝ) ≤ (d : ℝ) by positivity)
  nlinarith

example := retrieval_error_le (fun a b : Fin 3 => (a.val + b.val : ℝ)) 2
  (fun j => WithLp.toLp 2 ![(j.val : ℝ), 1]) (WithLp.toLp 2 ![0, 0])
  (fun j => ‖WithLp.toLp 2 ![(j.val : ℝ), 1] - WithLp.toLp 2 ![0, 0]‖)
  (fun _ => le_rfl)

example := useful_mass_le (fun a b : Fin 3 => if a = b then 1 else -1) 2 {0, 2} 1
  (by intro j hj; split_ifs <;> norm_num)

example := retrieval_error_of_useful_set (fun _ _ : Fin 2 => 0) 1
  (fun j => if j = 0 then (0 : EucSpace 1) else WithLp.toLp 2 ![1]) 0 {0} 0 1
  (by intro j hj; simp only [Finset.mem_singleton] at hj; simp [hj])
  (by intro j hj; simp only [Finset.mem_singleton] at hj; simp [hj, EuclideanSpace.norm_eq])

private def witnessAngles : Fin 3 → Fin 2 → ℝ := fun i k => (i.val * (k.val + 1) : ℝ) / 7
private def witnessQueries : Fin 3 → EucSpace 4 := fun i => WithLp.toLp 2 ![1, (i.val : ℝ), 2, 1]
private def witnessKeys : Fin 3 → EucSpace 4 := fun i => WithLp.toLp 2 ![2, 1, (i.val : ℝ), 3]

example := normalized_useful_mass_le (d := 4) (by decide) 1 (by norm_num)
  witnessAngles witnessQueries witnessKeys 2 {0, 1}

example := retrieval_log_budget (d := 4) (by decide) 1 (by norm_num)
  witnessAngles witnessQueries witnessKeys 2 Finset.univ (by simp) 1 (by norm_num)
  (by exact (causalWeight_sum
    (fun a b => headScore 4 1 (witnessAngles a) (witnessAngles b)
      (witnessQueries a) (witnessKeys b)) 2).ge)

example := retrieval_width_budget (d := 4) (by decide) 1 (by norm_num)
  witnessAngles witnessQueries witnessKeys 2 Finset.univ (by simp) 1 (by norm_num)
  (by exact (causalWeight_sum
    (fun a b => headScore 4 1 (witnessAngles a) (witnessAngles b)
      (witnessQueries a) (witnessKeys b)) 2).ge)

end Transformer.SequenceModel.Attention
