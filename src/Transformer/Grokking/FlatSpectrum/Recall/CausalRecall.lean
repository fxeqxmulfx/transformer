module

public import Transformer.Grokking.FlatSpectrum.Recall.CausalFading
public import Transformer.Grokking.FlatSpectrum.Recall.Threshold

/-!
# The causal induction head recalls a key it copies enough

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.CausalFading` shows that the causal induction head fades at a key `q` whose copy
`w = copyWeight s₁ (b + 2q + 2)` at its value token has `s₂ w < 1`. Here is the converse at a
fixed key. At `s₁ ≥ 0` no token but the value token copies `q` more than its key token does,
with `r` (`causalCopy_le_key`), and the token at the position `p` after the value token copies
it with at most `1 / p` (`causalCopy_le_inv`). Splitting the `n = b + 2m + 2` tokens at a
position `P` at or after the value token, the head answers `q` within

    2 C ((P + 1) n^{−s₂ (w − r)} + n^{−(s₂ w − 1 − s₂ / P)})    (`abs_sub_causalInductionHead_le`)

for values bounded by `C`. If `r < w` and `s₂ w > 1`, a large `P` makes both exponents negative,
and from some number of keys on the head answers `q` within every `ε > 0` in every dictionary
(`exists_forall_causalInductionHead_recall`). The threshold `s₂ w = 1` is that of the cyclic
head, with the copy of the key in place of the lead of layer 1, but it holds or fails key by
key: after a beginning-of-sequence token at `s₁ = 2` and `s₂ = 11/10`, the fourth key is
recalled and the first fades.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset Filter Topology

/-- Each position of the causal context holds one token. -/
lemma causalPos_injective (b m : ℕ) : Function.Injective (causalPos b (m := m)) := by
  rintro (j | ⟨i, _ | _⟩) (j' | ⟨i', _ | _⟩) h <;>
    simp only [causalPos, Sum.inl.injEq, Sum.inr.injEq, reduceCtorEq, Prod.mk.injEq, Fin.ext_iff,
      and_true, and_false, Bool.false_eq_true, Bool.true_eq_false] at h ⊢ <;> omega

/-- **No token but the value token copies more than the key token**
(`review/flat_spectrum.md`, SA5): at `s ≥ 0`, every position other than the value token of `c`
copies `c` with at most its copy at the key token of `c`. -/
lemma causalCopy_le_key {s : ℝ} (hs : 0 ≤ s) {b p c : ℕ} (hp : p ≠ b + 2 * c + 1) :
    causalCopy s b p c ≤ causalCopy s b (b + 2 * c) c := by
  rw [causalCopy_key, causalCopy_eq, ite_eq_right hp, mul_zero, zero_add]
  split_ifs with h
  · refine one_div_le_one_div_of_le (by positivity) ?_
    have hb : ((b : ℝ) + 2 * c + 1) ^ s ≤ ((p : ℝ) + 1) ^ s :=
      Real.rpow_le_rpow (by positivity) (by exact_mod_cast (by omega : b + 2 * c + 1 ≤ p + 1)) hs
    have hp' : (b : ℝ) + 2 * c ≤ p := by exact_mod_cast h
    linarith
  · rw [zero_div]
    positivity

/-- The first token after a beginning-of-sequence token. -/
example := causalCopy_le_key zero_le_one (b := 1) (p := 0) (c := 0) (by norm_num)

/-- **After the value token the copy fades** (`review/flat_spectrum.md`, SA5): the token at the
position `p` after the value token of `c` copies `c` with at most `1 / p`. -/
lemma causalCopy_le_inv (s : ℝ) {b p c : ℕ} (hp : b + 2 * c + 1 < p) :
    causalCopy s b p c ≤ 1 / p := by
  rw [causalCopy_of_gt s hp]
  refine one_div_le_one_div_of_le (by exact_mod_cast (by omega : 0 < p)) ?_
  linarith [Real.rpow_pos_of_pos (by positivity : (0 : ℝ) < p + 1) s]

/-- The third token after the first key. -/
example := causalCopy_le_inv 2 (b := 0) (p := 2) (c := 0) (by norm_num)

/-- **The causal head answers within the weight of the other tokens**
(`review/flat_spectrum.md`, SA5): at `s₁, s₂ ≥ 0`, splitting the `n = b + 2m + 2` tokens at a
position `P` at or after the value token of `q`, the head answers `q` within
`2 C ((P + 1) n^{−s₂ (w − r)} + n^{−(s₂ w − 1 − s₂ / P)})` for values bounded by `C`, where
`w = copyWeight s₁ (b + 2q + 2)` is the copy of `q` at its value token and `r` at its key token. -/
theorem abs_sub_causalInductionHead_le {s₁ s₂ : ℝ} (hs₁ : 0 ≤ s₁) (hs₂ : 0 ≤ s₂) {ν : Type*}
    {φ : ν → ℝ} {C : ℝ} (hφ : ∀ x, |φ x| ≤ C) (b : ℕ) {m P : ℕ} (π : Fin (m + 1) → ν)
    (q : Fin (m + 1)) (hP : b + 2 * (q : ℕ) + 1 ≤ P) :
    |φ (π q) - causalInductionHead s₁ s₂ φ b π q| ≤ 2 * C *
      ((P + 1) * (b + 2 * m + 2 : ℝ) ^ (-(s₂ * (copyWeight s₁ (b + 2 * (q : ℕ) + 2) -
          causalCopy s₁ b (b + 2 * (q : ℕ)) q))) +
        (b + 2 * m + 2 : ℝ) ^ (-(s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) - 1 - s₂ / P))) := by
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have hv : ∀ t : Fin b ⊕ Fin (m + 1) × Bool,
      |Sum.elim (fun _ => (0 : ℝ)) (fun t => if t.2 then φ (π t.1) else 0) t| ≤ C := by
    rintro (j | ⟨i, _ | _⟩) <;> simp [hC, hφ]
  have h := abs_sub_logScaleAttention_le_sum_erase (s := s₂)
    (z := fun t => causalCopy s₁ b (causalPos b t) q) hv (.inr (q, true))
  rw [card_causalTokens] at h
  simp only [Sum.elim_inr, causalPos, causalCopy_value, ite_true] at h
  set n : ℝ := (b : ℝ) + 2 * m + 2 with hn
  set r := causalCopy s₁ b (b + 2 * (q : ℕ)) q with hr
  have hn1 : (1 : ℝ) ≤ n := by rw [hn]; linarith [(by positivity : (0 : ℝ) ≤ b + 2 * m)]
  have hP0 : (0 : ℝ) < P := by exact_mod_cast (by omega : 0 < P)
  have hX := Real.rpow_nonneg (by linarith : (0 : ℝ) ≤ n) (s₂ / P)
  have hY := Real.rpow_nonneg (by linarith : (0 : ℝ) ≤ n) (s₂ * r)
  have hR : ∑ t ∈ univ.erase (.inr (q, true)), n ^ (s₂ * causalCopy s₁ b (causalPos b t) q) ≤
      (P + 1) * n ^ (s₂ * r) + n * n ^ (s₂ / P) := by
    calc ∑ t ∈ univ.erase (.inr (q, true)), n ^ (s₂ * causalCopy s₁ b (causalPos b t) q)
        ≤ ∑ t ∈ univ.erase (.inr (q, true)),
            (n ^ (s₂ * r) * (if causalPos b t ≤ P then 1 else 0) + n ^ (s₂ / P)) := by
          refine sum_le_sum fun t ht => ?_
          have ht' : causalPos b t ≠ b + 2 * q + 1 := fun h' => ne_of_mem_erase ht
            (causalPos_injective b m (show causalPos b t = causalPos b (.inr (q, true)) from h'))
          split_ifs with htP
          · have := Real.rpow_le_rpow_of_exponent_le hn1
              (mul_le_mul_of_nonneg_left (causalCopy_le_key hs₁ ht') hs₂)
            linarith
          · rw [mul_zero, zero_add]
            refine Real.rpow_le_rpow_of_exponent_le hn1 ?_
            rw [← mul_one_div]
            refine mul_le_mul_of_nonneg_left ((causalCopy_le_inv s₁ (by omega)).trans ?_) hs₂
            exact one_div_le_one_div_of_le hP0 (by exact_mod_cast (by omega : P ≤ causalPos b t))
      _ ≤ ∑ t, (n ^ (s₂ * r) * (if causalPos b t ≤ P then 1 else 0) + n ^ (s₂ / P)) :=
          sum_le_sum_of_subset_of_nonneg (erase_subset _ _) fun t _ _ => by
            split_ifs <;> nlinarith
      _ = n ^ (s₂ * r) * ((univ.filter fun t => causalPos b t ≤ P).card : ℝ) +
            n * n ^ (s₂ / P) := by
          rw [sum_add_distrib, ← mul_sum, sum_boole, sum_const, card_univ, nsmul_eq_mul,
            card_causalTokens, ← hn]
      _ ≤ n ^ (s₂ * r) * (P + 1) + n * n ^ (s₂ / P) := by
          gcongr
          exact_mod_cast (card_le_card_of_injOn (causalPos b)
            (fun t ht => mem_range.2 (Nat.lt_succ_of_le (mem_filter.1 ht).2))
            (causalPos_injective b m).injOn).trans (card_range _).le
      _ = _ := by ring
  have hn0 : (0 : ℝ) < n := by linarith
  have hW := Real.rpow_pos_of_pos hn0 (s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2))
  refine h.trans (le_of_le_of_eq (mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hR hW.le)
    (by linarith)) ?_)
  rw [show -(s₂ * (copyWeight s₁ (b + 2 * (q : ℕ) + 2) - r)) =
      s₂ * r - s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) by ring,
    show -(s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) - 1 - s₂ / P) =
      1 + s₂ / P - s₂ * copyWeight s₁ (b + 2 * (q : ℕ) + 2) by ring,
    Real.rpow_sub hn0, Real.rpow_sub hn0, Real.rpow_add hn0, Real.rpow_one]
  ring

/-- The fourth of four keys after a beginning-of-sequence token, split at the position `8`. -/
example := abs_sub_causalInductionHead_le (s₁ := 2) (s₂ := 11 / 10) (P := 8) (by norm_num)
  (by norm_num) (φ := fun x : Bool => if x then (1 : ℝ) else -1) (C := 1)
  (fun x => by cases x <;> norm_num) 1 (fun i : Fin 4 => decide (i = 3)) 3 (by decide)

/-- **A key copied above `1 / s₂` is recalled** (`review/flat_spectrum.md`, SA5): at `s₁ ≥ 0`, if
the key `q` is copied at its value token, with `w = copyWeight s₁ (b + 2q + 2)`, more than at its
key token, and `s₂ w > 1`, then for every `ε > 0`, from some number of keys on, the head answers
`q` within `ε` in every dictionary, for values bounded by `C`. -/
theorem exists_forall_causalInductionHead_recall {s₁ s₂ : ℝ} (hs₁ : 0 ≤ s₁) (b q : ℕ)
    (hr : causalCopy s₁ b (b + 2 * q) q < copyWeight s₁ (b + 2 * q + 2))
    (hw : 1 < s₂ * copyWeight s₁ (b + 2 * q + 2)) {ν : Type*} {φ : ν → ℝ} {C : ℝ}
    (hφ : ∀ x, |φ x| ≤ C) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ k : Fin (m + 1), (k : ℕ) = q → ∀ π : Fin (m + 1) → ν,
      |φ (π k) - causalInductionHead s₁ s₂ φ b π k| ≤ ε := by
  have hr0 := causalCopy_nonneg s₁ b (b + 2 * q) q
  generalize hw_def : copyWeight s₁ (b + 2 * q + 2) = w at hr hw
  generalize hr_def : causalCopy s₁ b (b + 2 * q) q = r at hr hr0
  have hs₂ : 0 < s₂ := lt_of_not_ge fun h => by nlinarith
  obtain ⟨N, hN⟩ := exists_nat_gt (s₂ / (s₂ * w - 1))
  have hN0 : (0 : ℝ) < N := (div_pos hs₂ (by linarith)).trans hN
  rw [div_lt_iff₀ (by linarith)] at hN
  have hPN : (N : ℝ) ≤ (max (b + 2 * q + 1) N : ℕ) := by exact_mod_cast le_max_right _ _
  have hδ : 0 < s₂ * w - 1 - s₂ / (max (b + 2 * q + 1) N : ℕ) := by
    have := div_le_div_of_nonneg_left hs₂.le hN0 hPN
    have : s₂ / N < s₂ * w - 1 := (div_lt_iff₀ hN0).2 (by linarith)
    linarith
  have h3 := ((((tendsto_rpow_neg_atTop (mul_pos hs₂ (by linarith : 0 < w - r))).comp
    (tendsto_valuePrefix b)).const_mul ((max (b + 2 * q + 1) N : ℕ) + 1 : ℝ)).add
    ((tendsto_rpow_neg_atTop hδ).comp (tendsto_valuePrefix b))).const_mul (2 * C)
  rw [mul_zero, add_zero, mul_zero] at h3
  obtain ⟨M, hM⟩ := eventually_atTop.1 (h3.eventually (ge_mem_nhds hε))
  refine ⟨M, fun m hm k hk π => ?_⟩
  have hb := abs_sub_causalInductionHead_le hs₁ hs₂.le hφ b π k
    (P := max (b + 2 * q + 1) N) (by rw [hk]; exact le_max_left _ _)
  rw [hk, hw_def, hr_def] at hb
  exact hb.trans (hM m hm)

/-- After a beginning-of-sequence token at `s₁ = 2` and `s₂ = 11/10`, the fourth key is
recalled: `s₂ w = 891/890`. -/
example := exists_forall_causalInductionHead_recall (s₁ := 2) (s₂ := 11 / 10) (by norm_num) 1 3
  (by rw [causalCopy_key, copyWeight]; norm_num) (by rw [copyWeight]; norm_num)
  (φ := fun x : Bool => if x then (1 : ℝ) else -1) (C := 1) (fun x => by cases x <;> norm_num)
  (ε := 1 / 10) (by norm_num)

/-- With the same scales the first key fades: `s₂ w = 9/10`. -/
example : ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), (q : ℕ) = 0 → ∃ π : Fin (m + 1) → Bool,
    1 / 2 < |(if π q then (1 : ℝ) else -1) -
      causalInductionHead 2 (11 / 10) (fun x => if x then 1 else -1) 1 π q| := by
  obtain ⟨M, hM⟩ := exists_lt_causalInductionHead_error (s₂ := 11 / 10) (σ := 9 / 10)
    (ε := 1 / 2) (by norm_num) (by norm_num) (by norm_num) 2 1
  exact ⟨M, fun m hm q hq => hM m hm q (by rw [hq, copyWeight]; norm_num)⟩

end Transformer.Grokking.FlatSpectrum.Recall
