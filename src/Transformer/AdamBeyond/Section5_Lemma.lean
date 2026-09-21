import Transformer.AdamBeyond.Section5_AdamNC
import Transformer.AdamBeyond.AppendixG_Auxiliary
import Transformer.AMSGrad.Section4_MainLemma
import Transformer.AMSGrad.Section1_TheoremA

/-
# Adam and beyond — §5: the lemma of the proof of Theorem 5

The lemma of the appendix, §"Proof of Theorem 5":
`Σ_{t=1}^T α_t ‖V_t^{-1/4} m_t‖² ≤ 2ζ/(1-β₁)² Σᵢ ‖g_{1:T,i}‖₂` for AdamNC
under condition 1 of Theorem 5 (at `α_t`, see `Section5_Regret`).

**The proof carried here.**  The source bounds `m²_{t,i}` by Cauchy–Schwarz,
`m²_{t,i} ≤ (Σ_j β₁^{t-j})(Σ_j β₁^{t-j} g²_{j,i})`, which costs a factor
`1/(1-β₁)`.  Convexity of the square, `m²_t ≤ β_{1,t} m²_{t-1} + (1-β_{1,t}) g²_t`,
gives `m²_{t,i} ≤ Σ_j β₁^{t-j} g²_{j,i}` without it, so the proof gives
`2ζ/(1-β₁)`; the statement keeps the source's `2ζ/(1-β₁)²`, which follows.
Otherwise the proof is the source's: condition 1 turns `α_t/√v_{t,i}` into
`ζ/‖g_{1:t,i}‖₂`, the sum over `t` is exchanged with the geometric weights,
and lem:simple-grad-bound closes.  The hypotheses `0 ≤ β_{2,t} ≤ 1` and
`β₂ = 0` of the source are not needed: `m_t` and `g_t` do not depend on `v_t`,
and condition 1 already forces `v_{t,i} > 0` wherever `g_{1:t,i} ≠ 0`.

Source: arXiv:1904.09237, appendix, §"Proof of Theorem 5", Lemma.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- `Σ_{t=1}^T s_t ≤ (1/(1-c)) Σ_{t=1}^T f_t` for `s = geomSum c f`, `0 ≤ c < 1`,
`f ≥ 0`: the exchange of sums of the proof of the lemma of Theorem 5.
arXiv:1904.09237, appendix, §"Proof of Theorem 5", Lemma. -/
theorem sum_geomSum_le {c : ℝ} (hc : 0 ≤ c) (hc' : c < 1) {f : ℕ → ℝ} (hf : ∀ n, 0 ≤ f n)
    (T : ℕ) : ∑ t ∈ Icc 1 T, geomSum c f t ≤ (∑ t ∈ Icc 1 T, f t) / (1 - c) := by
  have key : ∀ T, (1 - c) * ∑ t ∈ Icc 1 T, geomSum c f t + c * geomSum c f T ≤
      ∑ t ∈ Icc 1 T, f t := by
    intro T
    induction T with
    | zero => simp [geomSum]
    | succ n ih =>
      rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega)]
      simp only [geomSum]
      linarith
  rw [le_div_iff₀ (by linarith), mul_comm]
  linarith [key T, mul_nonneg hc (geomSum_nonneg hc hf T)]

/-- The per-step bound of the lemma of Theorem 5, in the form of its induction:
`m²_{t,i} ≤ ‖g_{1:t,i}‖₂ Σ_{j≤t} β₁^{t-j} g²_{j,i}/‖g_{1:j,i}‖₂`.
arXiv:1904.09237, appendix, §"Proof of Theorem 5", Lemma. -/
theorem m_sq_le_gnorm {S : Setup d} {R : Rule d}
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1) (i : Fin d)
    (n : ℕ) :
    S.m R n i ^ 2 ≤
      S.gnorm R n i * geomSum (S.β₁ 1) (fun t => S.g R t i ^ 2 / S.gnorm R t i) n := by
  have hβ0 : 0 ≤ S.β₁ 1 := (hβ₁ 1 le_rfl).1
  have hf : ∀ t, 0 ≤ S.g R t i ^ 2 / S.gnorm R t i :=
    fun t => div_nonneg (sq_nonneg _) (Real.sqrt_nonneg _)
  induction n with
  | zero => simp [Setup.m, Setup.state, geomSum]
  | succ n ih =>
    obtain ⟨hb0, hb1⟩ := hβ₁ (n + 1) (by omega)
    set b := S.β₁ (n + 1)
    set g := S.g R (n + 1) i
    have hm : S.m R (n + 1) i = b * S.m R n i + (1 - b) * g := rfl
    have hmono : S.gnorm R n i ≤ S.gnorm R (n + 1) i := by
      exact Real.sqrt_le_sqrt (sum_le_sum_of_subset_of_nonneg
        (Icc_subset_Icc_right (by omega)) fun _ _ _ => sq_nonneg _)
    have hg : g ^ 2 = S.gnorm R (n + 1) i * (g ^ 2 / S.gnorm R (n + 1) i) := by
      by_cases h : S.gnorm R (n + 1) i = 0
      · have : g ^ 2 ≤ S.gnorm R (n + 1) i ^ 2 := by
          rw [Setup.gnorm, Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _),
            sum_Icc_succ_top (by omega)]
          linarith [sum_nonneg fun t (_ : t ∈ Icc 1 n) => sq_nonneg (S.g R t i)]
        rw [h] at this ⊢
        rw [zero_mul]
        nlinarith [sq_nonneg g]
      · rw [mul_div_cancel₀ _ h]
    have hs := geomSum_nonneg hβ0 hf n
    calc S.m R (n + 1) i ^ 2 ≤ S.β₁ 1 * S.m R n i ^ 2 + g ^ 2 := by
          rw [hm]
          nlinarith [mul_nonneg (mul_nonneg hb0 (by linarith : 0 ≤ 1 - b))
            (sq_nonneg (S.m R n i - g)), sq_nonneg (S.m R n i), sq_nonneg g]
      _ ≤ S.β₁ 1 * (S.gnorm R (n + 1) i * geomSum (S.β₁ 1)
            (fun t => S.g R t i ^ 2 / S.gnorm R t i) n)
          + S.gnorm R (n + 1) i * (g ^ 2 / S.gnorm R (n + 1) i) := by
          rw [← hg]
          gcongr
          exact ih.trans (mul_le_mul_of_nonneg_right hmono hs)
      _ = _ := by simp only [geomSum]; ring

/-- **The lemma of the proof of Theorem 5.**  Under condition 1 of Theorem 5
(at `α_t`), with `α_t > 0`, `ζ > 0` and `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`,

  `Σ_{t=1}^T α_t ‖V_t^{-1/4} m_t‖² ≤ 2ζ/(1-β₁)² Σᵢ ‖g_{1:T,i}‖₂`.

The source also assumes `β₂ = 0` and `0 ≤ β_{2,t} ≤ 1`, which the proof does
not use.

Source: arXiv:1904.09237, appendix, §"Proof of Theorem 5", Lemma. -/
theorem adamNC_moment_sum {S : Setup d} {β₂ : ℕ → ℝ} (hα : ∀ t, 1 ≤ t → 0 < S.α t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (T : ℕ) {ζ : ℝ} (hζ : 0 < ζ)
    (h₁ : ∀ t ∈ Icc 1 T, ∀ i,
      S.gnorm (adamNCRule β₂) t i / ζ ≤ Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t) :
    ∑ t ∈ Icc 1 T, S.α t *
        ∑ i, S.m (adamNCRule β₂) t i ^ 2 / Real.sqrt (S.vhat (adamNCRule β₂) t i) ≤
      2 * ζ / (1 - S.β₁ 1) ^ 2 * ∑ i, S.gnorm (adamNCRule β₂) T i := by
  set R : Rule d := adamNCRule β₂
  set β := S.β₁ 1
  have hβ0 : 0 ≤ β := (hβ₁ 1 le_rfl).1
  have hB : 0 < 1 - β := by linarith
  set f : Fin d → ℕ → ℝ := fun i t => S.g R t i ^ 2 / S.gnorm R t i
  have hf : ∀ i t, 0 ≤ f i t := fun i t => div_nonneg (sq_nonneg _) (Real.sqrt_nonneg _)
  -- `α_t m²_{t,i}/√v_{t,i} ≤ ζ s_{t,i}`, by condition 1
  have hterm : ∀ t ∈ Icc 1 T, ∀ i,
      S.α t * (S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)) ≤ ζ * geomSum β (f i) t := by
    intro t ht i
    have hat := hα t (mem_Icc.1 ht).1
    have hm := m_sq_le_gnorm (R := R) hβ₁ hβ₁' i t
    have hs := geomSum_nonneg hβ0 (hf i) t
    have hc := h₁ t ht i
    rw [div_le_div_iff₀ hζ hat] at hc
    rcases (sq_nonneg (S.m R t i)).eq_or_lt with h0 | h0
    · rw [← h0, zero_div, mul_zero]; positivity
    have hn : 0 < S.gnorm R t i := by
      by_contra h
      rw [le_antisymm (not_lt.1 h) (Real.sqrt_nonneg _), zero_mul] at hm
      linarith
    have hv : 0 < Real.sqrt (S.vhat R t i) := by nlinarith
    rw [← mul_div_assoc, div_le_iff₀ hv]
    calc S.α t * S.m R t i ^ 2 ≤ S.α t * (S.gnorm R t i * geomSum β (f i) t) := by gcongr
      _ = (S.gnorm R t i * S.α t) * geomSum β (f i) t := by ring
      _ ≤ (Real.sqrt (S.vhat R t i) * ζ) * geomSum β (f i) t := by gcongr
      _ = _ := by ring
  -- Σ_t s_{t,i} ≤ 2‖g_{1:T,i}‖₂/(1-β₁), by lem:simple-grad-bound
  have hsum : ∀ i, ∑ t ∈ Icc 1 T, geomSum β (f i) t ≤ 2 * S.gnorm R T i / (1 - β) := by
    intro i
    refine (sum_geomSum_le hβ0 hβ₁' (hf i) T).trans (div_le_div_of_nonneg_right ?_ hB.le)
    exact sum_div_sqrt_partial_le (fun t => sq_nonneg (S.g R t i)) T
  have hn : ∀ i, 0 ≤ S.gnorm R T i := fun i => Real.sqrt_nonneg _
  calc ∑ t ∈ Icc 1 T, S.α t * ∑ i, S.m R t i ^ 2 / Real.sqrt (S.vhat R t i)
      ≤ ∑ t ∈ Icc 1 T, ∑ i, ζ * geomSum β (f i) t :=
        sum_le_sum fun t ht => by rw [mul_sum]; exact sum_le_sum fun i _ => hterm t ht i
    _ = ζ * ∑ i, ∑ t ∈ Icc 1 T, geomSum β (f i) t := by rw [sum_comm, mul_sum]; simp [mul_sum]
    _ ≤ ζ * ∑ i, 2 * S.gnorm R T i / (1 - β) := by gcongr with i; exact hsum i
    _ = 2 * ζ / (1 - β) * ∑ i, S.gnorm R T i := by
        rw [mul_sum, mul_sum]; exact sum_congr rfl fun i _ => by ring
    _ ≤ 2 * ζ / (1 - β) ^ 2 * ∑ i, S.gnorm R T i := by
        gcongr
        · exact sum_nonneg fun i _ => hn i
        · nlinarith

/-- The hypotheses of the lemma are satisfiable: the zero cost, `α = 1`,
`β_{1,t} = 0`, `β_{2,t} = 1 - 1/t`, `ζ = 1`, `T = 2`; condition 1 holds by
`adamNC_inv_cond`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0
    let R : Rule 1 := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
    (∀ t, 1 ≤ t → 0 < S.α t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ t ∈ Icc 1 2, ∀ i, S.gnorm R t i / 1 ≤ Real.sqrt (S.vhat R t i) / S.α t) := by
  intro S R
  have hc := fun i => adamNC_inv_cond (S := S) rfl one_pos rfl i
  exact ⟨fun t ht => by simp [S, zeroSetup]; omega, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [S, zeroSetup], one_pos, fun t ht i => (hc i).1 t (mem_Icc.mp ht).1⟩

end AdamBeyond
end Transformer
