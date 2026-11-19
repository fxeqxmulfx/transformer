module

public import Transformer.BAM.Section2_IgnoresLocal

/-!
# Theorem 4: GGD-BAM with `β < 0` fades like NoPE

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.4; proof in Appendix B.7.

Theorem 4 claims `lim_{|j − i| → ∞} p_ij ≠ 0` for every `β < 0` and `α > 0` (`α ≥ 0` in §2.4).
Its proof shows that the bias tends to `0`, so that `p_ij` approaches the NoPE weight
`softmax(qK^⊤ + M_i•)_j`, and concludes that `p_ij` "is not necessarily `0`". Under the causal mask
`|j − i| → ∞` needs `i → ∞`, and with content scores bounded by `C` the `i + 1` keys share the
attention: it fades (`relAttention_le_fade`). The limit is `0`, uniformly, for every `μ`
(`ggd_negative_fades`), as for ALiBi (Lemma 3) and GGD-BAM with `β > 0` (Lemma 4). The theorem is
false.

What its proof shows holds, for any content: past some distance GGD-BAM keeps all but a fraction
`ε` of the NoPE weight, `p_ij ≥ (1 − ε) p^{NoPE}_ij` (`ggd_negative_ge_noPE`), since a bias `≤ 0`
keeps the fraction `e^{b_ij}` of it (`relAttention_ge_noPE`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {L : ℕ}

/-- **Attention fades.** If the content scores are bounded by `C`, the bias is at most `0`, and
its weight is at least `δ > 0` at every distance `≥ K`, then a query `i ≥ K` gives every key at
most `e^{2C} / (δ (i + 1 − K))`: the `i + 1 − K` keys at distance `≥ K` share the attention. -/
theorem relAttention_le_fade {B : ℝ → EReal} {C δ : ℝ} {K : ℕ} {f : Fin L → ℝ}
    (hf : ∀ z, |f z| ≤ C) (hB : ∀ x, B x ≤ 0) (hδ : 0 < δ)
    (hK : ∀ x : ℝ, (K : ℝ) ≤ |x| → δ ≤ weight (B x)) {i : Fin L} (hi : K ≤ (i : ℕ))
    (j : Fin L) : relAttention B f i j ≤ Real.exp (2 * C) / (δ * ((i : ℕ) + 1 - K)) := by
  set s : Fin L → EReal := fun z => (f z : EReal) + causalMask i z + B (relPos i z)
  have hnum : weight (s j) ≤ Real.exp C := by
    simp only [s]
    rw [weight_add, weight_add, weight_coe, weight_causalMask]
    have h2 := weight_le_one (hB (relPos i j))
    have h3 := Real.exp_le_exp.2 (le_of_abs_le (hf j))
    split_ifs
    · nlinarith [weight_nonneg (B (relPos i j)), Real.exp_pos (f j)]
    · simp [(Real.exp_pos C).le]
  set k : Fin L := ⟨(i : ℕ) - K, by omega⟩
  have hterm : ∀ z ∈ Iic k, Real.exp (-C) * δ ≤ weight (s z) := by
    intro z hz
    have hzk' : z ≤ k := Finset.mem_Iic.1 hz
    have hzk : (z : ℕ) ≤ (i : ℕ) - K := hzk'
    have hzi : z ≤ i := by rw [Fin.le_iff_val_le_val]; omega
    have hd : (K : ℝ) ≤ |relPos i z| := by
      have h : ((z : ℕ) : ℝ) + K ≤ (i : ℕ) := by exact_mod_cast (by omega : (z : ℕ) + K ≤ i)
      rw [relPos, abs_of_nonpos (by linarith)]
      linarith
    simp only [s]
    rw [weight_add, weight_add, weight_coe, weight_causalMask, ite_eq_left hzi, mul_one]
    exact mul_le_mul (Real.exp_le_exp.2 (neg_le_of_abs_le (hf z))) (hK _ hd) hδ.le
      (Real.exp_pos _).le
  have hn : (0 : ℝ) < (i : ℕ) + 1 - K := by
    have : (K : ℝ) ≤ (i : ℕ) := by exact_mod_cast hi
    linarith
  have hden : Real.exp (-C) * δ * ((i : ℕ) + 1 - K) ≤ partition s := by
    calc Real.exp (-C) * δ * ((i : ℕ) + 1 - K) = ∑ _z ∈ Iic k, Real.exp (-C) * δ := by
          rw [sum_const, Fin.card_Iic, nsmul_eq_mul]
          simp only [k]
          push_cast [Nat.cast_sub hi]
          ring
      _ ≤ ∑ z ∈ Iic k, weight (s z) := sum_le_sum hterm
      _ ≤ partition s := sum_le_univ_sum_of_nonneg fun z => weight_nonneg _
  calc relAttention B f i j = weight (s j) / partition s := rfl
    _ ≤ Real.exp C / (Real.exp (-C) * δ * ((i : ℕ) + 1 - K)) :=
        div_le_div₀ (Real.exp_pos C).le hnum (by positivity) hden
    _ = Real.exp (2 * C) / (δ * ((i : ℕ) + 1 - K)) := by
        rw [Real.exp_neg, two_mul, Real.exp_add]
        field_simp

/-- The hypotheses of `relAttention_le_fade` hold for NoPE, the zero bias, with `δ = 1`, `K = 0`. -/
example (f : Fin 5 → ℝ) (hf : ∀ z, |f z| ≤ 1) (j : Fin 5) :
    relAttention (fun _ => 0) f 4 j ≤ Real.exp (2 * 1) / (1 * (((4 : Fin 5) : ℕ) + 1 - 0)) := by
  exact_mod_cast relAttention_le_fade (K := 0) hf (fun _ => le_rfl) one_pos (fun _ _ => by simp)
    (Nat.zero_le _) j

/-- **Theorem 4 is false.** For `β < 0`, `α > 0`, every `μ`, and content scores bounded by `C`,
`p_ij → 0` as `|j − i| → ∞`, uniformly over contexts (§2.4, Appendix B.7). -/
theorem ggd_negative_fades {μ α β C ε : ℝ} (hα : 0 < α) (hβ : β < 0) (hε : 0 < ε) :
    ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) → D ≤ |relPos i j| →
      relAttention (ggdBias μ α β) f i j < ε := by
  set K : ℕ := ⌈|μ| + α⌉₊
  refine ⟨K + Real.exp (2 * C + 1) / ε, fun L f i j hf hd => ?_⟩
  rcases lt_or_ge i j with hij | hji
  · rw [relAttention_eq_zero_of_lt _ f hij]
    exact hε
  have hK : ∀ x : ℝ, (K : ℝ) ≤ |x| → Real.exp (-1) ≤ weight (ggdBias μ α β x) := by
    intro x hx
    have h1 : 1 ≤ |(x - μ) / α| := by
      rw [abs_div, abs_of_pos hα, le_div_iff₀ hα, one_mul]
      linarith [abs_sub_abs_le_abs_sub x μ, Nat.le_ceil (|μ| + α)]
    rw [weight_ggdBias (Or.inl (abs_pos.1 (one_pos.trans_le h1)))]
    exact Real.exp_le_exp.2 (neg_le_neg (Real.rpow_le_one_of_one_le_of_nonpos h1 hβ.le))
  have hrel : |relPos i j| ≤ (i : ℕ) := by
    have h : ((j : ℕ) : ℝ) ≤ (i : ℕ) := by exact_mod_cast Fin.le_iff_val_le_val.1 hji
    rw [relPos, abs_of_nonpos (by linarith)]
    linarith [(Nat.cast_nonneg (j : ℕ) : (0 : ℝ) ≤ (j : ℕ))]
  have hq : 0 ≤ Real.exp (2 * C + 1) / ε := by positivity
  have hi : (K : ℝ) + Real.exp (2 * C + 1) / ε ≤ (i : ℕ) := hd.trans hrel
  have hfade := relAttention_le_fade hf (ggdBias_nonpos μ α β) (Real.exp_pos (-1)) hK
    (by exact_mod_cast (by linarith : (K : ℝ) ≤ (i : ℕ))) j
  have hn : Real.exp (2 * C + 1) / ε < (i : ℕ) + 1 - K := by linarith
  have hnpos : (0 : ℝ) < (i : ℕ) + 1 - K := hq.trans_lt hn
  calc relAttention (ggdBias μ α β) f i j
      ≤ Real.exp (2 * C) / (Real.exp (-1) * ((i : ℕ) + 1 - K)) := hfade
    _ = Real.exp (2 * C + 1) / ((i : ℕ) + 1 - K) := by
        rw [Real.exp_neg, Real.exp_add]
        field_simp
    _ < ε := by
        rw [div_lt_iff₀ hnpos]
        rw [div_lt_iff₀ hε] at hn
        linarith

/-- The hypotheses of `ggd_negative_fades` hold for `α = 1`, `β = −1`, `ε = 1/2`. -/
example (μ C : ℝ) : ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), (∀ z, |f z| ≤ C) →
    D ≤ |relPos i j| → relAttention (ggdBias μ 1 (-1)) f i j < 1 / 2 :=
  ggd_negative_fades one_pos (by norm_num) (by norm_num)

/-- **A bias `≤ 0` keeps the fraction `e^{b_ij}` of the NoPE weight**: lowering the other keys'
weights only raises key `j`'s share. -/
theorem relAttention_ge_noPE {B : ℝ → EReal} (hB : ∀ x, B x ≤ 0) (f : Fin L → ℝ) (i j : Fin L) :
    weight (B (relPos i j)) * relAttention (fun _ => 0) f i j ≤ relAttention B f i j := by
  set s₀ : Fin L → EReal := fun z => (f z : EReal) + causalMask i z + 0
  set s : Fin L → EReal := fun z => (f z : EReal) + causalMask i z + B (relPos i z)
  have hw : ∀ z, weight (s z) = weight (s₀ z) * weight (B (relPos i z)) := fun z => by
    simp only [s, s₀, weight_add, weight_zero, mul_one]
  have hZ : partition s ≤ partition s₀ := sum_le_sum fun z _ => by
    rw [hw]
    exact mul_le_of_le_one_right (weight_nonneg _) (weight_le_one (hB _))
  have hZ₀ : 0 < partition s₀ :=
    partition_pos (s := s₀) (z := i) (by simp [s₀, causalMask]) (by simp [s₀, causalMask])
  change weight (B (relPos i j)) * (weight (s₀ j) / partition s₀) ≤ weight (s j) / partition s
  rw [← mul_div_assoc, mul_comm, ← hw]
  rcases (partition_nonneg s).eq_or_lt with h0 | hpos
  · have : weight (s j) = 0 := le_antisymm (h0 ▸ weight_le_partition s j) (weight_nonneg _)
    simp [this]
  · exact div_le_div_of_nonneg_left (weight_nonneg _) hpos hZ

/-- The hypothesis of `relAttention_ge_noPE` holds for ALiBi of slope `1`. -/
example (f : Fin 3 → ℝ) (i j : Fin 3) :
    weight (alibi 1 (relPos i j)) * relAttention (fun _ => 0) f i j ≤
      relAttention (alibi 1) f i j :=
  relAttention_ge_noPE (fun x => by
    simp only [alibi]
    exact_mod_cast neg_nonpos.2 (by positivity)) f i j

/-- **Theorem 4, repaired.** For `β < 0` and `α > 0`, past some distance GGD-BAM keeps all but a
fraction `ε` of the NoPE weight, for every `μ` and all content scores: `p_ij` approaches
`softmax(qK^⊤ + M_i•)_j` from above, up to `ε`, as the proof of Theorem 4 says (§2.4,
Appendix B.7). -/
theorem ggd_negative_ge_noPE {μ α β ε : ℝ} (hα : 0 < α) (hβ : β < 0) (hε : 0 < ε) :
    ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), D ≤ |relPos i j| →
      (1 - ε) * relAttention (fun _ => 0) f i j ≤ relAttention (ggdBias μ α β) f i j := by
  set T := ε ^ β⁻¹
  have hT : 0 < T := Real.rpow_pos_of_pos hε _
  have hTβ : T ^ β = ε := Real.rpow_inv_rpow hε.le hβ.ne
  refine ⟨|μ| + α * T, fun L f i j hd => ?_⟩
  have hx : T ≤ |(relPos i j - μ) / α| := by
    rw [abs_div, abs_of_pos hα, le_div_iff₀ hα]
    linarith [abs_sub_abs_le_abs_sub (relPos i j) μ]
  have hpow : |(relPos i j - μ) / α| ^ β ≤ ε := hTβ ▸ Real.rpow_le_rpow_of_nonpos hT hx hβ.le
  have hw : 1 - ε ≤ weight (ggdBias μ α β (relPos i j)) := by
    rw [weight_ggdBias (Or.inl (abs_pos.1 (hT.trans_le hx)))]
    linarith [Real.add_one_le_exp (-(|(relPos i j - μ) / α| ^ β))]
  exact (mul_le_mul_of_nonneg_right hw (softmax_nonneg _ _)).trans
    (relAttention_ge_noPE (ggdBias_nonpos μ α β) f i j)

/-- The hypotheses of `ggd_negative_ge_noPE` hold for `α = 1`, `β = −1`, `ε = 1/10`. -/
example (μ : ℝ) : ∃ D : ℝ, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), D ≤ |relPos i j| →
    (1 - 1 / 10) * relAttention (fun _ => 0) f i j ≤ relAttention (ggdBias μ 1 (-1)) f i j :=
  ggd_negative_ge_noPE one_pos (by norm_num) (by norm_num)

end Transformer.BAM
