module

public import Transformer.BAM.Section2_GeneralizedGaussian

/-!
# Lemmas 3 and 4: ALiBi and GGD-BAM with `β > 0` are local

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.2–2.3; proofs in
Appendix B.3–B.4.

`relAttention B f i j` is `p_ij = softmax(qK^⊤ + M_i• + B_i•)_j` for real content scores `f` and
a bias `B` of the relative position. Both lemmas claim `lim_{|j − i| → ∞} p_ij = 0`. Their proofs
treat the content scores `qK^⊤` of "a fixed query" as constants while the bias tends to `−∞`;
the theorems here take content scores bounded by `C`, and the limit is uniform over contexts
and queries: for every `ε > 0` there is a distance `D` past which every weight is below `ε`
(`alibi_local`, `ggd_local`). Without the bound neither lemma holds: content can cancel the bias
(`not_local_of_unbounded`).

Lemma 4 is stated for `α ≥ 0` in §2.3 and `α > 0` in Appendix B.4; at `α = 0` the bias divides
by zero, and the theorem takes `α > 0`. Its proof sets `μ = 0`; the theorem holds for every `μ`.
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {L : ℕ}

/-- The attention weight `p_ij = softmax(qK^⊤ + M_i• + B_i•)_j` of query `i` on key `j`, for real
content scores `f` and a bias `B` of the relative position `j − i` (Appendix B). -/
noncomputable def relAttention (B : ℝ → EReal) (f : Fin L → ℝ) (i j : Fin L) : ℝ :=
  softmax (fun z => (f z : EReal) + causalMask i z + B (relPos i z)) j

@[simp] theorem relPos_self (i : Fin L) : relPos i i = 0 := by simp [relPos]

/-- **Content bounded by `C` cannot outweigh the bias by more than `e^{2C}`:**
`p_ij ≤ e^{2C} e^{B(j − i)} / e^{B(0)}`, comparing key `j` with the query's own key. -/
theorem relAttention_le {B : ℝ → EReal} {C : ℝ} {f : Fin L → ℝ} (hf : ∀ z, |f z| ≤ C)
    (hB : 0 < weight (B 0)) (i j : Fin L) :
    relAttention B f i j ≤ Real.exp (2 * C) * weight (B (relPos i j)) / weight (B 0) := by
  have hnum : weight ((f j : EReal) + causalMask i j + B (relPos i j)) ≤
      Real.exp C * weight (B (relPos i j)) := by
    rw [weight_add, weight_add, weight_coe, weight_causalMask]
    have hw := weight_nonneg (B (relPos i j))
    have hC := Real.exp_le_exp.2 (le_of_abs_le (hf j))
    split_ifs
    · nlinarith
    · simp only [mul_zero, zero_mul]
      exact mul_nonneg (Real.exp_pos C).le hw
  have hden : Real.exp (-C) * weight (B 0) ≤
      partition (fun z => (f z : EReal) + causalMask i z + B (relPos i z)) := by
    refine le_trans ?_ (weight_le_partition _ i)
    rw [weight_add, weight_add, weight_coe, weight_causalMask, ite_eq_left le_rfl, relPos_self,
      mul_one]
    gcongr
    exact neg_le_of_abs_le (hf i)
  calc relAttention B f i j
      ≤ Real.exp C * weight (B (relPos i j)) / (Real.exp (-C) * weight (B 0)) :=
        div_le_div₀ (mul_nonneg (Real.exp_pos C).le (weight_nonneg _)) hnum
          (mul_pos (Real.exp_pos _) hB) hden
    _ = Real.exp (2 * C) * weight (B (relPos i j)) / weight (B 0) := by
        rw [Real.exp_neg, two_mul, Real.exp_add]
        field_simp

/-- The hypotheses of `relAttention_le` hold for zero content and ALiBi of slope `1`. -/
example (i j : Fin 3) :
    relAttention (alibi 1) (fun _ => 0) i j ≤
      Real.exp (2 * 0) * weight (alibi 1 (relPos i j)) / weight (alibi 1 0) :=
  relAttention_le (fun _ => by simp) (by simp [alibi]) i j

/-- **Lemma 3: ALiBi is local attention.** For content scores bounded by `C` and a slope `m > 0`,
`p_ij → 0` as `|j − i| → ∞`, uniformly: past some distance `D` every weight is below `ε`
(§2.2, Appendix B.3). -/
theorem alibi_local {m C ε : ℝ} (hm : 0 < m) (hε : 0 < ε) :
    ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) → D ≤ |relPos i j| →
      relAttention (alibi m) f i j < ε := by
  refine ⟨(2 * C - Real.log ε) / m + 1, fun L f i j hf hd => ?_⟩
  have h := relAttention_le (B := alibi m) hf (by simp [alibi]) i j
  simp only [alibi, weight_coe, abs_zero, mul_zero, neg_zero, Real.exp_zero, div_one] at h
  have hmd := mul_le_mul_of_nonneg_left hd hm.le
  rw [mul_add, mul_div_cancel₀ _ hm.ne', mul_one] at hmd
  calc relAttention (alibi m) f i j ≤ Real.exp (2 * C) * Real.exp (-(m * |relPos i j|)) := h
    _ = Real.exp (2 * C - m * |relPos i j|) := by rw [← Real.exp_add]; ring_nf
    _ < Real.exp (Real.log ε) := Real.exp_lt_exp.2 (by linarith)
    _ = ε := Real.exp_log hε

/-- The hypotheses of `alibi_local` hold for `m = 1`, `ε = 1/2`. -/
example (C : ℝ) : ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) →
    D ≤ |relPos i j| → relAttention (alibi 1) f i j < 1 / 2 :=
  alibi_local one_pos (by norm_num)

/-- **Lemma 4: GGD-BAM with `β > 0` is local attention.** For content scores bounded by `C`,
`α > 0`, `β > 0` and every `μ`, `p_ij → 0` as `|j − i| → ∞`, uniformly (§2.3,
Appendix B.4). -/
theorem ggd_local {μ α β C ε : ℝ} (hα : 0 < α) (hβ : 0 < β) (hε : 0 < ε) :
    ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) → D ≤ |relPos i j| →
      relAttention (ggdBias μ α β) f i j < ε := by
  set K := 2 * C + |(0 - μ) / α| ^ β - Real.log ε
  set T := (max K 0 + 1) ^ β⁻¹
  have hT : 0 < T := by positivity
  have hTβ : T ^ β = max K 0 + 1 := Real.rpow_inv_rpow (by positivity) hβ.ne'
  refine ⟨|μ| + α * T, fun L f i j hf hd => ?_⟩
  have h := relAttention_le (B := ggdBias μ α β) hf
    (by rw [weight_ggdBias (Or.inr hβ.le)]; positivity) i j
  rw [weight_ggdBias (Or.inr hβ.le), weight_ggdBias (Or.inr hβ.le), mul_div_assoc,
    ← Real.exp_sub, ← Real.exp_add] at h
  have hx : T ≤ |(relPos i j - μ) / α| := by
    rw [abs_div, abs_of_pos hα, le_div_iff₀ hα]
    linarith [abs_sub_abs_le_abs_sub (relPos i j) μ]
  have hpow : T ^ β ≤ |(relPos i j - μ) / α| ^ β := Real.rpow_le_rpow hT.le hx hβ.le
  calc relAttention (ggdBias μ α β) f i j ≤ _ := h
    _ < Real.exp (Real.log ε) := Real.exp_lt_exp.2 (by linarith [le_max_left K 0])
    _ = ε := Real.exp_log hε

/-- The hypotheses of `ggd_local` hold for `α = 1`, `β = 1/2`, `ε = 1/2`. -/
example (μ C : ℝ) : ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) →
    D ≤ |relPos i j| → relAttention (ggdBias μ 1 (1 / 2)) f i j < 1 / 2 :=
  ggd_local one_pos (by norm_num) (by norm_num)

/-- **Lemmas 3 and 4 need bounded content.** For every bias `B ≤ 0` with finite values, no
distance makes every weight small: at distance `n`, the content score `log n − B(−n)` of the
farthest key keeps its weight at least `1/2` (Appendix B.3–B.4). -/
theorem not_local_of_unbounded {B : ℝ → EReal} (hB : ∀ x, B x ≤ 0) (hfin : ∀ x, B x ≠ ⊥) :
    ¬ ∀ ε > 0, ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), D ≤ |relPos i j| →
      relAttention B f i j < ε := by
  intro h
  obtain ⟨D, hD⟩ := h (1 / 2) (by norm_num)
  set n := ⌈|D|⌉₊ + 1 with hn_def
  have hn : (1 : ℝ) ≤ n := by simp [n]
  set w := weight (B (relPos (Fin.last n) 0))
  have hw : 0 < w := weight_pos (hfin _) ((hB _).trans_lt EReal.zero_lt_top).ne
  set f : Fin (n + 1) → ℝ := fun z => if z = 0 then Real.log n - Real.log w else 0
  set s : Fin (n + 1) → EReal := fun z => (f z : EReal) + causalMask (Fin.last n) z +
    B (relPos (Fin.last n) z)
  have h0 : weight (s 0) = n := by
    have hf0 : f 0 = Real.log n - Real.log w := by simp [f]
    have hm0 : causalMask (Fin.last n) 0 = 0 := by simp [causalMask]
    simp only [s]
    rw [weight_add, weight_add, weight_coe, hf0, hm0, weight_zero, mul_one, Real.exp_sub,
      Real.exp_log (by linarith), Real.exp_log hw]
    field_simp
    rfl
  have hrest : ∀ k : Fin n, weight (s k.succ) ≤ 1 := fun k => by
    have hfk : f k.succ = 0 := by simp [f, Fin.succ_ne_zero]
    have hmk : causalMask (Fin.last n) k.succ = 0 := by simp [causalMask, Fin.le_last]
    simp only [s]
    rw [hfk, hmk, EReal.coe_zero, zero_add, zero_add]
    exact weight_le_one (hB _)
  have hpart : partition s ≤ 2 * n := by
    rw [partition, Fin.sum_univ_succ, h0]
    have := sum_le_sum fun (k : Fin n) (_ : k ∈ univ) => hrest k
    simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at this
    linarith
  have hd : D ≤ |relPos (Fin.last n) 0| := by
    have : |relPos (Fin.last n) 0| = n := by simp [relPos]
    have hcast : (n : ℝ) = ⌈|D|⌉₊ + 1 := by rw [hn_def]; push_cast; ring
    rw [this, hcast]
    linarith [le_abs_self D, Nat.le_ceil |D|]
  have hlt := hD (n + 1) f (Fin.last n) 0 hd
  have hge : 1 / 2 ≤ relAttention B f (Fin.last n) 0 := by
    rw [relAttention, softmax]
    change 1 / 2 ≤ weight (s 0) / partition s
    rw [h0, le_div_iff₀ (by linarith [weight_le_partition s 0, h0])]
    linarith
  linarith

/-- ALiBi of slope `m > 0` satisfies the hypotheses of `not_local_of_unbounded`. -/
example {m : ℝ} (hm : 0 < m) : ¬ ∀ ε > 0, ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L),
    D ≤ |relPos i j| → relAttention (alibi m) f i j < ε :=
  not_local_of_unbounded (fun x => by
    simp only [alibi]
    exact_mod_cast neg_nonpos.2 (by positivity)) (fun _ => EReal.coe_ne_bot _)

end Transformer.BAM
