module

public import Transformer.SequenceModel.Attention.Normalization

/-!
# Attention concentration with many variables under actual Q/K RMSNorm

Source: `review/flat_spectrum.md`, *General positional models*, G5. With N
unmasked inputs and head width d, every source attention weight is at most
exp(2*sqrt(d))/N. This upper bound may exceed one and is not an impossibility
of multi-layer computation; it isolates a limitation of one normalized row.
Neither a learned temperature nor rescaling Q/K can bypass it in this source.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel.Attention

open scoped BigOperators

/-- A causal softmax row has a strictly positive denominator: it contains itself. -/
theorem denominator_pos {T : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T) :
    0 < ∑ k ∈ Finset.Iic i, Real.exp (scores i k) := by
  exact Finset.sum_pos' (fun k _ => (Real.exp_pos _).le)
    ⟨i, Finset.mem_Iic.mpr le_rfl, Real.exp_pos _⟩

/-- Causal softmax remains a probability row for every finite tuple length. -/
theorem causalWeight_sum {T : ℕ} (scores : Fin T → Fin T → ℝ) (i : Fin T) :
    ∑ j, causalWeight scores i j = 1 := by
  classical
  have hfilter : Finset.univ.filter (fun j : Fin T => j ≤ i) = Finset.Iic i := by
    ext j
    simp
  simp only [causalWeight]
  rw [← Finset.sum_filter, hfilter, ← Finset.sum_div]
  exact div_self (denominator_pos scores i).ne'

/-- G5: a bounded score row cannot concentrate above exp(2B)/N. -/
theorem causalWeight_le {T : ℕ} (scores : Fin T → Fin T → ℝ) (i j : Fin T)
    (B : ℝ) (hB : ∀ k ≤ i, |scores i k| ≤ B) :
    causalWeight scores i j ≤ Real.exp (2 * B) / ((i.val : ℝ) + 1) := by
  by_cases hij : j ≤ i
  · have hden := denominator_pos scores i
    have hn : 0 < (i.val : ℝ) + 1 := by positivity
    have hlo : ((i.val : ℝ) + 1) * Real.exp (-B) ≤
        ∑ k ∈ Finset.Iic i, Real.exp (scores i k) := by
      have h := Finset.card_nsmul_le_sum (Finset.Iic i)
        (fun k => Real.exp (scores i k)) (Real.exp (-B))
        (fun k hk => Real.exp_le_exp.mpr (neg_le_of_abs_le (hB k (Finset.mem_Iic.mp hk))))
      simpa only [nsmul_eq_mul, Fin.card_Iic, Nat.cast_add, Nat.cast_one] using h
    rw [causalWeight, ite_eq_left hij, div_le_div_iff₀ hden hn]
    calc Real.exp (scores i j) * ((i.val : ℝ) + 1)
        ≤ Real.exp B * ((i.val : ℝ) + 1) :=
          mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr (le_of_abs_le (hB j hij))) hn.le
      _ = Real.exp (2 * B) * (((i.val : ℝ) + 1) * Real.exp (-B)) := by
          have he : Real.exp B = Real.exp (2 * B) * Real.exp (-B) := by
            rw [← Real.exp_add]
            congr 1
            ring
          rw [he]
          ring
      _ ≤ Real.exp (2 * B) * ∑ k ∈ Finset.Iic i, Real.exp (scores i k) :=
          mul_le_mul_of_nonneg_left hlo (Real.exp_pos _).le
  · simp only [causalWeight, ite_eq_right hij]
    positivity

/-- G5 on the actual normalized, arbitrarily rotated query/key logits. -/
theorem normalized_attention_le {T d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k : Fin T → EucSpace d) (i j : Fin T) :
    causalWeight (fun a b => headScore d eps (angle a) (angle b) (q a) (k b)) i j ≤
      Real.exp (2 * Real.sqrt (d : ℝ)) / ((i.val : ℝ) + 1) := by
  exact causalWeight_le _ i j _ (fun b _ => headScore_bound d hd eps heps _ _ _ _)

example := causalWeight_le (fun i j : Fin 5 => if i = j then 1 else -1) 4 2 1
  (by intro k hk; split_ifs <;> norm_num)
example := normalized_attention_le (T := 5) (d := 4) (by decide)
  (1 / 8388608) (by norm_num) (fun i k => (i.val * (k.val + 1) : ℝ) / 7)
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) (fun _ => WithLp.toLp 2 ![2, 1, 4, 3]) 4 2

end Transformer.SequenceModel.Attention
