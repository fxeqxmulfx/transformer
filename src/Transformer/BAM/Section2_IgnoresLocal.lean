module

public import Transformer.BAM.Section2_Locality

/-!
# Theorem 3: GGD-BAM with `β < 0` ignores the query's own key

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.4; proof in Appendix B.6.

For `β < 0` the bias `−|(j − i − μ)/α|^β` tends to `−∞` as `j − i → μ` (`tendsto_ggdBias`).
`ggdBias` takes that limit as its value at `j − i = μ`, and a key with that bias gets no attention
(`relAttention_ggd_eq_zero`).

Theorem 3 claims `lim_{|j − i| → 0} p_ij = 0` for every `β < 0`, for `α ≥ 0` in §2.4 and `α > 0`
in Appendix B.6, without fixing `μ`. Positions are integers, so `|j − i| → 0` means `j = i`, as
the proof says. The proof sets `μ = 0`, and so does the theorem (`ggd_ignores_local`), which then
holds for every `α` and all content scores. For `μ ≠ 0` it fails (`not_ggd_ignores_local`).
-/

@[expose] public section

namespace Transformer.BAM

open Filter Topology

variable {L : ℕ}

/-- A key whose bias is `−∞` gets no attention. -/
theorem relAttention_eq_zero {B : ℝ → EReal} (f : Fin L → ℝ) {i j : Fin L}
    (h : B (relPos i j) = ⊥) : relAttention B f i j = 0 := by
  simp [relAttention, softmax, h]

/-- The hypothesis of `relAttention_eq_zero` holds for the GGD-BAM bias with `μ = 0`, `β = −1` at
the query's own key. -/
example (f : Fin 2 → ℝ) : relAttention (ggdBias 0 1 (-1)) f 0 0 = 0 :=
  relAttention_eq_zero f (by rw [relPos_self]; exact ggdBias_self_of_neg (by norm_num))

/-- Under the causal mask a key after the query gets no attention. -/
theorem relAttention_eq_zero_of_lt (B : ℝ → EReal) (f : Fin L → ℝ) {i j : Fin L} (h : i < j) :
    relAttention B f i j = 0 := by
  simp [relAttention, softmax, causalMask, not_le.2 h]

/-- The hypothesis of `relAttention_eq_zero_of_lt` holds for the keys `0 < 1`. -/
example (B : ℝ → EReal) (f : Fin 2 → ℝ) : relAttention B f 0 1 = 0 :=
  relAttention_eq_zero_of_lt B f (by decide)

/-- **For `β < 0`, GGD-BAM ignores the key at relative position `μ`**, whatever its content. -/
theorem relAttention_ggd_eq_zero {μ α β : ℝ} (hβ : β < 0) (f : Fin L → ℝ) {i j : Fin L}
    (h : relPos i j = μ) : relAttention (ggdBias μ α β) f i j = 0 :=
  relAttention_eq_zero f (by rw [h]; exact ggdBias_self_of_neg hβ)

/-- The hypotheses of `relAttention_ggd_eq_zero` hold for `μ = −1`, `β = −2` and the keys `1, 0`. -/
example (f : Fin 2 → ℝ) : relAttention (ggdBias (-1) 3 (-2)) f 1 0 = 0 :=
  relAttention_ggd_eq_zero (by norm_num) f (by simp [relPos])

/-- **For `β < 0` the bias tends to `−∞` as `x → μ`**, where `ggdBias` is `−∞`: the limit of the
proof of Theorem 3, `lim −|α / (j − i)|^{|β|} = −∞` (Appendix B.6). -/
theorem tendsto_ggdBias {μ α β : ℝ} (hα : α ≠ 0) (hβ : β < 0) :
    Tendsto (ggdBias μ α β) (𝓝[≠] μ) (𝓝 ⊥) := by
  have h0 : Tendsto (fun x => (x - μ) / α) (𝓝[≠] μ) (𝓝[≠] 0) := by
    refine tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩
    · have h : Continuous fun x : ℝ => (x - μ) / α := by fun_prop
      have h := h.tendsto μ
      simpa using h.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with x hx
      exact div_ne_zero (sub_ne_zero.2 hx) hα
  have h1 : Tendsto (fun x => |(x - μ) / α| ^ β) (𝓝[≠] μ) atTop :=
    (tendsto_rpow_neg_nhdsGT_zero hβ).comp (tendsto_abs_nhdsNE_zero.comp h0)
  refine (EReal.tendsto_coe_atBot.comp (tendsto_neg_atTop_atBot.comp h1)).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with x hx
  exact (ggdBias_eq (Or.inl (div_ne_zero (sub_ne_zero.2 hx) hα))).symm

/-- The hypotheses of `tendsto_ggdBias` hold for `α = 1`, `β = −1`. -/
example (μ : ℝ) : Tendsto (ggdBias μ 1 (-1)) (𝓝[≠] μ) (𝓝 ⊥) :=
  tendsto_ggdBias one_ne_zero (by norm_num)

/-- **Theorem 3.** GGD-BAM with `μ = 0` ignores local context for every `β < 0`:
`lim_{|j − i| → 0} p_ij = 0`, for every `α` and all content scores (§2.4, Appendix B.6). -/
theorem ggd_ignores_local {α β ε : ℝ} (hβ : β < 0) (hε : 0 < ε) :
    ∃ δ > 0, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), |relPos i j| < δ →
      relAttention (ggdBias 0 α β) f i j < ε := by
  refine ⟨1, one_pos, fun L f i j hd => ?_⟩
  have hij : j = i := by
    by_contra h
    exact absurd (one_le_abs_relPos h) (not_le.2 hd)
  rw [relAttention_ggd_eq_zero hβ f (by rw [hij, relPos_self])]
  exact hε

/-- The hypotheses of `ggd_ignores_local` hold for `β = −1`, `ε = 1/2`. -/
example (α : ℝ) : ∃ δ > 0, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), |relPos i j| < δ →
    relAttention (ggdBias 0 α (-1)) f i j < 1 / 2 :=
  ggd_ignores_local (by norm_num) (by norm_num)

/-- **Theorem 3 needs `μ = 0`.** For `μ ≠ 0` and `α ≠ 0` the bias at the query's own key is
finite, and in a one-token context the query attends to itself with weight `1`, for every `β`
(§2.4, Appendix B.6). -/
theorem not_ggd_ignores_local {μ α β : ℝ} (hμ : μ ≠ 0) (hα : α ≠ 0) :
    ¬ ∀ ε > 0, ∃ δ > 0, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), |relPos i j| < δ →
      relAttention (ggdBias μ α β) f i j < ε := by
  intro h
  obtain ⟨δ, hδ, hD⟩ := h 1 one_pos
  have hlt := hD 1 (fun _ => 0) 0 0 (by simpa using hδ)
  have hb := ggdBias_eq (μ := μ) (α := α) (β := β) (x := 0)
    (Or.inl (div_ne_zero (sub_ne_zero.2 hμ.symm) hα))
  have hone : relAttention (ggdBias μ α β) (fun _ : Fin 1 => 0) 0 0 = 1 := by
    simp only [relAttention, softmax, partition, Fin.sum_univ_one, relPos_self]
    refine div_self ?_
    rw [hb, weight_add, weight_add, weight_coe, weight_causalMask, weight_coe]
    simp [Real.exp_ne_zero]
  linarith

/-- The hypotheses of `not_ggd_ignores_local` hold for `μ = 1`, `α = 1`, `β = −1`. -/
example : ¬ ∀ ε > 0, ∃ δ > 0, ∀ (L : ℕ) (f : Fin L → ℝ) (i j : Fin L), |relPos i j| < δ →
    relAttention (ggdBias 1 1 (-1)) f i j < ε :=
  not_ggd_ignores_local one_ne_zero one_ne_zero

end Transformer.BAM
