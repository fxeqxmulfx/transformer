module

public import Transformer.MRCR.CountHead

/-!
# Count, then look up: two heads at the scale `s log n` return the needle with the desired index

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37: the query asks for "the
entry with the desired index" among the needles of a key, listed "in the order that they were
presented in the context"; MRCR-V2 of arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, has eight needles; attention at the scale
`s log n` over the `n` visible tokens, as in SSMax, arXiv:2501.19399v1, §2.2. With the rounded
transformers of arXiv:2506.16055v3 this is `Transformer.MRCR.Ordinal`; here the softmax is real.

After a BOS the rows `t < m` carry the keys `κ t` and the values `π t`; the query asks for the
row `t₀` with `κ t₀ = q` and `occ κ t₀ = r + 1`, the needle of the key `q` with `r` needles of
that key before it. Layer 1 (`Transformer.MRCR.countHead`), at the scale `s₁(t) log n` with
`s₁(t) ≥ s`, leaves at every row `t` a value `c_t` within `2^{1 − s}` of `1 / (occ κ t + 1)`.
Layer 2 (`lookupHead`) attends at the query over the BOS and the rows with the raw score

    [κ t = q] − K_r (c_t − 1/(r + 2))²,     K_r = ((r + 2)(r + 3))²,

and `0` at the BOS, and reads the value of the row. The raw score is bilinear in the features
`(e_{κ t}, 1, c_t, c_t²)` of the row and `(e_q, −K_r/(r + 2)², 2 K_r/(r + 2), −K_r)` of the
query; the feedforward layer after layer 1 supplies `c_t²`. The needle scores at least
`1 − K_r 2^{1 − s}`, another row of the key `q` at most `K_r 2^{1 − s}` since its count differs
(`Transformer.MRCR.one_le_lookupWeight_mul_sq`), the other rows and the BOS at most `0`. So the
needle leads by `g = 1 − 2 K_r 2^{1 − s}`, and layer 2 is within `2 C m (m + 1)^{−s₂ g}` of its
value, for values bounded by `C` (`abs_lookupHead_sub_le`). For every bound `R` on the ordinal
and every `s₂ > 1` some `s` makes the two heads return the needle with the desired index to
every accuracy from some number of rows on, whatever the keys and the values
(`exists_forall_lookupHead_recall`). The count at the first rows errs by up to `2^{1 − s}` at
every length, so `s` grows here like `log K_R`. The query token itself, whose raw score would be
`−K_r (c − 1/(r + 2))² ≤ 0`, is left out of layer 2.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Filter Topology Transformer.Grokking.FlatSpectrum.Recall

variable {c : ℕ}

/-- **The lookup head**, layer 2 on layer 1: at the query of the key `q` and the ordinal `r`,
attention at the scale `s₂ log n` over the BOS (`none`) and the rows `t < m`, with the raw score
`[κ t = q] − K_r (c_t − 1/(r + 2))²` at the row `t`, `c_t` the counting head at the scale
`s₁(t) log n`, which may depend on the row as Su's scale does, and `0` at the BOS, reading
`φ (π t)` at the row `t` and `0` at the BOS. -/
noncomputable def lookupHead (s₁ : ℕ → ℝ) (s₂ : ℝ) {ν : Type*} (φ : ν → ℝ) (κ : ℕ → Fin c)
    (π : ℕ → ν) (q : Fin c) (r m : ℕ) : ℝ :=
  logScaleAttention s₂
    (fun u : Option (Fin m) => u.elim 0 fun t =>
      (if κ t = q then 1 else 0) - lookupWeight r * (countHead (s₁ t) κ t - 1 / ((r : ℝ) + 2)) ^ 2)
    fun u => u.elim 0 fun t => φ (π t)

/-- **Count, then look up** (MRCR, arXiv:2409.12640v2, `eval_framework.tex`): if the row `t₀ < m`
is the needle of the key `q` with `r` needles of that key before it, the lookup head at the
scales `s₁(t) log n`, `s₁(t) ≥ s ≥ 1`, and `s₂ log n` is within
`2 C m (m + 1)^{−s₂ (1 − 2 K_r 2^{1 − s})}` of its value, for values bounded by `C`. -/
theorem abs_lookupHead_sub_le {ν : Type*} {s s₂ C : ℝ} {s₁ : ℕ → ℝ} (hs : 1 ≤ s)
    (hs₁ : ∀ t, s ≤ s₁ t) (hs₂ : 0 ≤ s₂) {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) (κ : ℕ → Fin c)
    (π : ℕ → ν) {q : Fin c} {r m t₀ : ℕ}
    (ht₀ : t₀ < m) (hq : κ t₀ = q) (ho : occ κ t₀ = r + 1) :
    |lookupHead s₁ s₂ φ κ π q r m - φ (π t₀)| ≤
      2 * C * m * ((m : ℝ) + 1) ^ (-(s₂ * (1 - 2 * lookupWeight r * 2 ^ (1 - s)))) := by
  set ε : ℝ := 2 ^ (1 - s)
  set K := lookupWeight r
  have hε0 : 0 ≤ ε := Real.rpow_nonneg (by norm_num) _
  have hε1 : ε ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (by linarith)
  have hK0 : 0 ≤ K := sq_nonneg _
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π t₀))
  have hc (t : ℕ) : |countHead (s₁ t) κ t - 1 / ((occ κ t : ℝ) + 1)| ≤ ε :=
    (abs_countHead_sub_le (hs.trans (hs₁ t)) κ t).trans <|
      (Real.rpow_le_rpow_of_nonpos (by norm_num) (by linarith [(t.cast_nonneg : (0 : ℝ) ≤ t)])
        (by linarith [hs₁ t])).trans
      (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith [hs₁ t]))
  have hhalf (t : ℕ) : |1 / ((occ κ t : ℝ) + 1) - 1 / ((r : ℝ) + 2)| ≤ 1 / 2 := by
    have h1 : (1 : ℝ) ≤ occ κ t := by exact_mod_cast one_le_occ κ t
    have h2 : 1 / ((occ κ t : ℝ) + 1) ≤ 1 / 2 :=
      one_div_le_one_div_of_le (by norm_num) (by linarith)
    have h3 : 1 / ((r : ℝ) + 2) ≤ 1 / 2 :=
      one_div_le_one_div_of_le (by norm_num) (by linarith [(r.cast_nonneg : (0 : ℝ) ≤ r)])
    have h4 : 0 < 1 / ((occ κ t : ℝ) + 1) := by positivity
    have h5 : 0 < 1 / ((r : ℝ) + 2) := by positivity
    rw [abs_le]
    constructor <;> linarith
  have htarget : 1 / ((occ κ t₀ : ℝ) + 1) = 1 / ((r : ℝ) + 2) := by rw [ho]; push_cast; ring_nf
  set z : Option (Fin m) → ℝ := fun u => u.elim 0 fun t =>
    (if κ t = q then 1 else 0) - K * (countHead (s₁ t) κ t - 1 / ((r : ℝ) + 2)) ^ 2 with hzdef
  have hz₀ : 1 - K * ε ≤ z (some ⟨t₀, ht₀⟩) := by
    have hd := hc t₀
    rw [htarget] at hd
    have hsq : (countHead (s₁ t₀) κ t₀ - 1 / ((r : ℝ) + 2)) ^ 2 ≤ ε := by
      rw [← sq_abs]
      nlinarith [abs_nonneg (countHead (s₁ t₀) κ t₀ - 1 / ((r : ℝ) + 2))]
    simp only [hzdef, Option.elim, hq, ite_true]
    nlinarith
  have hzq (t : Fin m) (ht : t.val ≠ t₀) (hκ : κ t = q) : z (some t) ≤ K * ε := by
    have hne : occ κ t ≠ r + 1 := by
      rw [← ho]
      rcases Nat.lt_or_gt_of_ne ht with h | h
      · exact (occ_lt_occ h (hκ.trans hq.symm)).ne
      · exact (occ_lt_occ h (hq.trans hκ.symm)).ne'
    have hsep := one_le_lookupWeight_mul_sq hne
    set d := countHead (s₁ t) κ t - 1 / ((occ κ t : ℝ) + 1)
    set e := 1 / ((occ κ t : ℝ) + 1) - 1 / ((r : ℝ) + 2)
    have hd : |d| ≤ ε := hc t
    have he : |e| ≤ 1 / 2 := hhalf t
    have hde : |2 * e * d| ≤ ε := by
      rw [abs_mul, abs_mul, abs_two]
      nlinarith [abs_nonneg e, abs_nonneg d]
    have hlow : e ^ 2 - ε ≤ (countHead (s₁ t) κ t - 1 / ((r : ℝ) + 2)) ^ 2 := by
      have : countHead (s₁ t) κ t - 1 / ((r : ℝ) + 2) = e + d := by ring
      rw [this]
      nlinarith [(abs_le.1 hde).1, sq_nonneg d]
    simp only [hzdef, Option.elim, hκ, ite_true]
    nlinarith [mul_le_mul_of_nonneg_left hlow hK0]
  have hzn (t : Fin m) (hκ : κ t ≠ q) : z (some t) ≤ 0 := by
    simp only [hzdef, Option.elim, hκ, ite_false, zero_sub, neg_nonpos]
    exact mul_nonneg hK0 (sq_nonneg _)
  have hz : ∀ u, u ≠ some ⟨t₀, ht₀⟩ → z u + (1 - 2 * K * ε) ≤ z (some ⟨t₀, ht₀⟩) := by
    rintro (_ | t) hu
    · have : z none = 0 := rfl
      nlinarith [mul_nonneg hK0 hε0]
    · have ht : t.val ≠ t₀ := fun h => hu (congrArg some (Fin.ext h))
      by_cases hκ : κ t = q
      · linarith [hzq t ht hκ]
      · nlinarith [hzn t hκ, mul_nonneg hK0 hε0]
  have h := abs_logScaleAttention_sub_le hs₂ hz (v := fun u => u.elim 0 fun t => φ (π t))
    (C := C) (by rintro (_ | t); exacts [by simpa using hC, hφ _])
  rw [Fintype.card_option, Fintype.card_fin] at h
  push_cast at h
  rwa [add_sub_cancel_right] at h

/-- One key, the first row asked for, two rows, values `±1`, at `s₁ = 2` and `s₂ = 2`. -/
example := abs_lookupHead_sub_le (s := 2) (s₁ := fun _ => 2) (s₂ := 2) (C := 1) (by norm_num)
  (fun _ => le_rfl) (by norm_num) (φ := fun b : Bool => if b then (1 : ℝ) else -1)
  (fun b => by cases b <;> simp) (fun _ => (0 : Fin 1)) (fun _ => true) (q := 0) (r := 0) (m := 2)
  (t₀ := 0) (by norm_num) rfl (by decide)

/-- **Two heads return the needle with the desired index at every length** (MRCR,
arXiv:2409.12640v2, `eval_framework.tex`): for every bound `R` on the ordinal and every `s₂ > 1`,
some `s ≥ 1` makes the lookup head over counts at the scales `s₁(t) ≥ s` return the value of the
needle of the key `q` with `r < R` needles of that key before it, to every accuracy `ε > 0` from
some number of rows on, whatever the keys and the values. -/
theorem exists_forall_lookupHead_recall {ν : Type*} (R : ℕ) {s₂ C : ℝ} (hs₂ : 1 < s₂)
    {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) :
    ∃ s : ℝ, 1 ≤ s ∧ ∀ ε : ℝ, 0 < ε → ∃ M : ℕ, ∀ m ≥ M, ∀ s₁ : ℕ → ℝ, (∀ t, s ≤ s₁ t) →
      ∀ (κ : ℕ → Fin c) (π : ℕ → ν) (q : Fin c) (r t₀ : ℕ), r < R → t₀ < m → κ t₀ = q →
        occ κ t₀ = r + 1 → |lookupHead s₁ s₂ φ κ π q r m - φ (π t₀)| ≤ ε := by
  obtain ⟨N, hN⟩ := pow_unbounded_of_one_lt (4 * lookupWeight R * s₂ / (s₂ - 1))
    (by norm_num : (1 : ℝ) < 2)
  have hN0 : (0 : ℝ) ≤ N := N.cast_nonneg
  refine ⟨(N : ℝ) + 1, by linarith, fun ε hε => ?_⟩
  set e := (s₂ + 1) / 2 with he
  have hx : Tendsto (fun m : ℕ => (m : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have ht : Tendsto (fun m : ℕ => 2 * |C| * ((m : ℝ) + 1) ^ (-(e - 1))) atTop (𝓝 0) := by
    have h := ((tendsto_rpow_neg_atTop (by linarith : 0 < e - 1)).comp hx).const_mul (2 * |C|)
    simpa [Function.comp_def] using h
  obtain ⟨M, hM⟩ := eventually_atTop.1 (ht.eventually (ge_mem_nhds hε))
  refine ⟨M, fun m hm s₁ hs₁ κ π q r t₀ hr ht₀ hq ho => ?_⟩
  have h := abs_lookupHead_sub_le (s := (N : ℝ) + 1) (s₂ := s₂) (by linarith) hs₁ (by linarith)
    hφ κ π ht₀ hq ho
  refine h.trans (le_trans ?_ (hM m hm))
  have hpos : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hP : (0 : ℝ) < 2 ^ N := by positivity
  have hKr : lookupWeight r ≤ lookupWeight R := by
    have hrR : (r : ℝ) ≤ R := by exact_mod_cast hr.le
    exact pow_le_pow_left₀ (by positivity) (by nlinarith) 2
  have hg : e ≤ s₂ * (1 - 2 * lookupWeight r * 2 ^ (1 - ((N : ℝ) + 1))) := by
    have h4 : 4 * lookupWeight R * s₂ < 2 ^ N * (s₂ - 1) := by
      rwa [div_lt_iff₀ (by linarith)] at hN
    have h5 : 2 * lookupWeight r * s₂ / 2 ^ N ≤ (s₂ - 1) / 2 := by
      rw [div_le_iff₀ hP]
      nlinarith [mul_le_mul_of_nonneg_right hKr (by linarith : (0 : ℝ) ≤ s₂)]
    have h6 : s₂ * (1 - 2 * lookupWeight r * 2 ^ (1 - ((N : ℝ) + 1))) =
        s₂ - 2 * lookupWeight r * s₂ / 2 ^ N := by
      rw [show (1 : ℝ) - ((N : ℝ) + 1) = -(N : ℝ) by ring, Real.rpow_neg (by norm_num),
        Real.rpow_natCast]
      field_simp
    rw [h6, he]
    linarith
  have hmono : ((m : ℝ) + 1) ^ (-(s₂ * (1 - 2 * lookupWeight r * 2 ^ (1 - ((N : ℝ) + 1))))) ≤
      ((m : ℝ) + 1) ^ (-e) :=
    Real.rpow_le_rpow_of_exponent_le (by linarith [(m.cast_nonneg : (0 : ℝ) ≤ m)]) (by linarith)
  have hsplit : ((m : ℝ) + 1) ^ (-(e - 1)) = ((m : ℝ) + 1) * ((m : ℝ) + 1) ^ (-e) := by
    rw [show -(e - 1) = 1 + -e by ring, Real.rpow_add hpos, Real.rpow_one]
  have hq1 := Real.rpow_nonneg hpos.le
    (-(s₂ * (1 - 2 * lookupWeight r * 2 ^ (1 - ((N : ℝ) + 1)))))
  rw [hsplit, ← mul_assoc]
  refine mul_le_mul (mul_le_mul (by linarith [le_abs_self C]) (by linarith) (Nat.cast_nonneg _)
    (by positivity)) hmono hq1 (by positivity)

/-- Eight needles, values `±1`, at `s₂ = 2`. -/
example := exists_forall_lookupHead_recall (c := 3) 8 (s₂ := 2) (C := 1) (by norm_num)
  (φ := fun b : Bool => if b then (1 : ℝ) else -1) fun b => by cases b <;> simp

end Transformer.MRCR
