import Transformer.AMSGrad.Section4_Lemmas
import Transformer.AMSGrad.Section2_Prelim
import Transformer.AMSGrad.Section3_Issue

/-
# AMSGrad — Lemma 4.4

§4 of arXiv:1904.03590v4: the bound on `Σ_t m²_{t,i}/√(t v̂_{t,i})` behind the
corrected convergence theorem of AMSGrad.

**What the source says and what is carried here.**

* The source assumes `γ = β₁/√β₂ ≤ 1`; the bound divides by `1 - γ`, so
  `γ < 1` is assumed instead, together with `0 ≤ β_{1,t} ≤ β₁ < 1` and
  `0 < β₂ < 1`.  The lemma holds for every rule with `v_t ≤ v̂_t`, which is all
  its proof uses of `v̂_t`, and for every `T`: at `T = 0` both sides vanish.

* The proof follows the source with the closed forms of `m_t` and `v_t`
  replaced by their recursions: the per-step bound
  `m²_{t,i}/√(t v̂_{t,i}) ≤ Σ_k γ^{t-k}|g_{k,i}|/((1-β₁)√(1-β₂)√t)` is proved by
  induction on `t` (`moment_le`), the exchange of sums by induction on `T`
  (`sum_geomSum_div_sqrt_le`), and the end is Cauchy–Schwarz with the
  harmonic bound, as in the source.

Source: arXiv:1904.03590v4, §4, Lemma 4.4 and its proof.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- `v_t ≥ 0` for every rule, when `0 ≤ β₂ ≤ 1`. -/
theorem v_nonneg {S : Setup d} (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (R : Rule d) (n : ℕ) :
    0 ≤ (S.state R n).v := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    intro i
    change 0 ≤ S.β₂ * (S.state R n).v i + (1 - S.β₂) * grad (S.f (n + 1)) (S.state R n).x i ^ 2
    exact add_nonneg (mul_nonneg hβ₂ (ih i)) (mul_nonneg (by linarith) (sq_nonneg _))

/-- `s_t = Σ_{k=1}^t c^{t-k} f_k`, through its recursion `s₀ = 0`,
`s_t = c s_{t-1} + f_t`.  arXiv:1904.03590v4, §4, proof of Lemma 4.4. -/
noncomputable def geomSum (c : ℝ) (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => c * geomSum c f n + f (n + 1)

/-- `geomSum c f ≥ 0` for `c ≥ 0`, `f ≥ 0`. -/
theorem geomSum_nonneg {c : ℝ} (hc : 0 ≤ c) {f : ℕ → ℝ} (hf : ∀ n, 0 ≤ f n) (n : ℕ) :
    0 ≤ geomSum c f n := by
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact add_nonneg (mul_nonneg hc ih) (hf _)

/-- `Σ_{t=1}^T s_t/√t ≤ (1/(1-c)) Σ_{t=1}^T f_t/√t` for `s = geomSum c f`,
`0 ≤ c < 1`, `f ≥ 0`: the exchange of sums of the proof of Lemma 4.4.
arXiv:1904.03590v4, §4, proof of Lemma 4.4. -/
theorem sum_geomSum_div_sqrt_le {c : ℝ} (hc : 0 ≤ c) (hc' : c < 1) {f : ℕ → ℝ}
    (hf : ∀ n, 0 ≤ f n) (T : ℕ) :
    ∑ t ∈ Icc 1 T, geomSum c f t / Real.sqrt t ≤ (∑ t ∈ Icc 1 T, f t / Real.sqrt t) / (1 - c) := by
  set u : ℕ → ℝ := fun t => geomSum c f t / Real.sqrt t
  have hu : ∀ t, 0 ≤ u t := fun t => div_nonneg (geomSum_nonneg hc hf t) (Real.sqrt_nonneg _)
  have step : ∀ t, u (t + 1) ≤ c * u t + f (t + 1) / Real.sqrt (t + 1 : ℕ) := by
    intro t
    simp only [u, geomSum, add_div, mul_div_assoc]
    refine add_le_add_left (mul_le_mul_of_nonneg_left ?_ hc) _
    rcases Nat.eq_zero_or_pos t with rfl | ht
    · simp [geomSum]
    · exact div_le_div_of_nonneg_left (geomSum_nonneg hc hf t)
        (Real.sqrt_pos.mpr (by exact_mod_cast ht)) (Real.sqrt_le_sqrt (by push_cast; linarith))
  have key : ∀ T, (1 - c) * ∑ t ∈ Icc 1 T, u t + c * u T ≤
      ∑ t ∈ Icc 1 T, f t / Real.sqrt t := by
    intro T
    induction T with
    | zero => simp [u, geomSum]
    | succ n ih =>
      rw [sum_Icc_succ_top (by omega), sum_Icc_succ_top (by omega)]
      linarith [step n]
  rw [le_div_iff₀ (by linarith), mul_comm]
  linarith [key T, mul_nonneg hc (hu T)]

/-- The per-step bound of the proof of Lemma 4.4, in the form of its
induction: `(1-β₁)√(1-β₂) m²_{t,i} ≤ √v_{t,i} Σ_{k=1}^t γ^{t-k}|g_{k,i}|`.
arXiv:1904.03590v4, §4, proof of Lemma 4.4. -/
theorem moment_le {S : Setup d} {R : Rule d} {β₁ : ℝ}
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (i : Fin d) (n : ℕ) :
    (1 - β₁) * Real.sqrt (1 - S.β₂) * S.m R n i ^ 2 ≤
      Real.sqrt (S.v R n i) * geomSum (β₁ / Real.sqrt S.β₂) (fun t => |S.g R t i|) n := by
  set γ := β₁ / Real.sqrt S.β₂
  have hβ₁0 : 0 ≤ β₁ := (hβ₁ 1 le_rfl).1.trans (hβ₁ 1 le_rfl).2
  have hsq : 0 < Real.sqrt S.β₂ := Real.sqrt_pos.mpr hβ₂
  have hγ0 : 0 ≤ γ := div_nonneg hβ₁0 hsq.le
  have hγ : γ * Real.sqrt S.β₂ = β₁ := div_mul_cancel₀ _ hsq.ne'
  have hK : 0 ≤ (1 - β₁) * Real.sqrt (1 - S.β₂) := mul_nonneg (by linarith) (Real.sqrt_nonneg _)
  induction n with
  | zero => simp [Setup.m, Setup.v, Setup.state, geomSum]
  | succ n ih =>
    obtain ⟨hb0, hb1⟩ := hβ₁ (n + 1) (by omega)
    set b := S.β₁ (n + 1)
    set g := S.g R (n + 1) i
    set s := geomSum γ (fun t => |S.g R t i|) n
    have hs : 0 ≤ s := geomSum_nonneg hγ0 (fun _ => abs_nonneg _) n
    have hv0 : 0 ≤ S.v R n i := v_nonneg hβ₂.le hβ₂'.le R n i
    have hm : S.m R (n + 1) i = b * S.m R n i + (1 - b) * g := rfl
    have hv : S.v R (n + 1) i = S.β₂ * S.v R n i + (1 - S.β₂) * g ^ 2 := rfl
    have hgs : geomSum γ (fun t => |S.g R t i|) (n + 1) = γ * s + |g| := rfl
    have h1 : Real.sqrt S.β₂ * Real.sqrt (S.v R n i) ≤ Real.sqrt (S.v R (n + 1) i) := by
      rw [← Real.sqrt_mul hβ₂.le, hv]
      exact Real.sqrt_le_sqrt (by nlinarith [sq_nonneg g])
    have h2 : Real.sqrt (1 - S.β₂) * |g| ≤ Real.sqrt (S.v R (n + 1) i) := by
      rw [← Real.sqrt_sq_eq_abs, ← Real.sqrt_mul (by linarith), hv]
      exact Real.sqrt_le_sqrt (by nlinarith)
    rw [hm, hgs]
    calc (1 - β₁) * Real.sqrt (1 - S.β₂) * (b * S.m R n i + (1 - b) * g) ^ 2
        ≤ (1 - β₁) * Real.sqrt (1 - S.β₂) * (b * S.m R n i ^ 2 + (1 - b) * g ^ 2) := by
          refine mul_le_mul_of_nonneg_left ?_ hK
          nlinarith [mul_nonneg (mul_nonneg hb0 (by linarith : 0 ≤ 1 - b))
            (sq_nonneg (S.m R n i - g))]
      _ = b * ((1 - β₁) * Real.sqrt (1 - S.β₂) * S.m R n i ^ 2)
          + ((1 - b) * (1 - β₁)) * (Real.sqrt (1 - S.β₂) * |g| * |g|) := by
          rw [mul_assoc (Real.sqrt _), abs_mul_abs_self]; ring
      _ ≤ β₁ * (Real.sqrt (S.v R n i) * s) + 1 * (Real.sqrt (1 - S.β₂) * |g| * |g|) := by
          gcongr; nlinarith
      _ = γ * (Real.sqrt S.β₂ * Real.sqrt (S.v R n i)) * s
          + Real.sqrt (1 - S.β₂) * |g| * |g| := by rw [← hγ]; ring
      _ ≤ γ * Real.sqrt (S.v R (n + 1) i) * s + Real.sqrt (S.v R (n + 1) i) * |g| := by
          gcongr
      _ = Real.sqrt (S.v R (n + 1) i) * (γ * s + |g|) := by ring

/-- The per-step bound of the proof of Lemma 4.4, for a rule with `v_t ≤ v̂_t`:
`m²_{t,i}/√(t v̂_{t,i}) ≤ Σ_{k=1}^t γ^{t-k}|g_{k,i}|/((1-β₁)√(1-β₂)√t)` for `t ≥ 1`.
arXiv:1904.03590v4, §4, proof of Lemma 4.4. -/
theorem term_le {S : Setup d} {R : Rule d} (hR : ∀ t a b, b ≤ R t a b) {β₁ : ℝ}
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (i : Fin d) {t : ℕ} (ht : 1 ≤ t) :
    S.m R t i ^ 2 / Real.sqrt (t * S.vhat R t i) ≤
      geomSum (β₁ / Real.sqrt S.β₂) (fun t => |S.g R t i|) t /
        ((1 - β₁) * Real.sqrt (1 - S.β₂) * Real.sqrt t) := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  have hβ₁0 : 0 ≤ β₁ := (hβ₁ 1 le_rfl).1.trans (hβ₁ 1 le_rfl).2
  have hs : 0 ≤ geomSum (β₁ / Real.sqrt S.β₂) (fun t => |S.g R t i|) (n + 1) :=
    geomSum_nonneg (div_nonneg hβ₁0 (Real.sqrt_nonneg _)) (fun _ => abs_nonneg _) _
  have hK : 0 < (1 - β₁) * Real.sqrt (1 - S.β₂) :=
    mul_pos (by linarith) (Real.sqrt_pos.mpr (by linarith))
  have ht : 0 < Real.sqrt (n + 1 : ℕ) := Real.sqrt_pos.mpr (by positivity)
  have key := moment_le (R := R) hβ₁ hβ₁' hβ₂ hβ₂' i (n + 1)
  have hvv : S.v R (n + 1) i ≤ S.vhat R (n + 1) i := hR (n + 1) (S.vhat R n) (S.v R (n + 1)) i
  rcases (sq_nonneg (S.m R (n + 1) i)).eq_or_lt with hm | hm
  · rw [← hm, zero_div]; positivity
  have hv : 0 < Real.sqrt (S.v R (n + 1) i) := by
    refine (Real.sqrt_nonneg _).lt_of_ne fun h => ?_
    rw [← h, zero_mul] at key
    nlinarith [mul_pos hK hm]
  have hden : Real.sqrt (n + 1 : ℕ) * Real.sqrt (S.v R (n + 1) i) ≤
      Real.sqrt ((n + 1 : ℕ) * S.vhat R (n + 1) i) := by
    rw [← Real.sqrt_mul (by positivity)]
    exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hvv (by positivity))
  rw [div_le_div_iff₀ ((mul_pos ht hv).trans_le hden) (mul_pos hK ht)]
  calc S.m R (n + 1) i ^ 2 * ((1 - β₁) * Real.sqrt (1 - S.β₂) * Real.sqrt (n + 1 : ℕ))
      = Real.sqrt (n + 1 : ℕ) * ((1 - β₁) * Real.sqrt (1 - S.β₂) * S.m R (n + 1) i ^ 2) := by
        ring
    _ ≤ Real.sqrt (n + 1 : ℕ) * (Real.sqrt (S.v R (n + 1) i) *
          geomSum (β₁ / Real.sqrt S.β₂) (fun t => |S.g R t i|) (n + 1)) := by gcongr
    _ = geomSum (β₁ / Real.sqrt S.β₂) (fun t => |S.g R t i|) (n + 1) *
          (Real.sqrt (n + 1 : ℕ) * Real.sqrt (S.v R (n + 1) i)) := by ring
    _ ≤ _ := by gcongr

/-- **Lemma 4.4.**  For a rule with `v_t ≤ v̂_t`, `0 ≤ β_{1,t} ≤ β₁ < 1`,
`0 < β₂ < 1` and `γ = β₁/√β₂ < 1`,

  `Σ_{t=1}^T m²_{t,i}/√(t v̂_{t,i}) ≤ √(ln T + 1)/((1-β₁)√(1-β₂)(1-γ)) ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.03590v4, §4, Lemma 4.4. -/
theorem mainlem {S : Setup d} {R : Rule d} (hR : ∀ t a b, b ≤ R t a b) {β₁ : ℝ}
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ β₁) (hβ₁' : β₁ < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) (T : ℕ) (i : Fin d) :
    ∑ t ∈ Icc 1 T, S.m R t i ^ 2 / Real.sqrt (t * S.vhat R t i) ≤
      Real.sqrt (Real.log T + 1) /
        ((1 - β₁) * Real.sqrt (1 - S.β₂) * (1 - β₁ / Real.sqrt S.β₂)) * S.gnorm R T i := by
  set γ := β₁ / Real.sqrt S.β₂
  set K := (1 - β₁) * Real.sqrt (1 - S.β₂)
  have hβ₁0 : 0 ≤ β₁ := (hβ₁ 1 le_rfl).1.trans (hβ₁ 1 le_rfl).2
  have hγ0 : 0 ≤ γ := div_nonneg hβ₁0 (Real.sqrt_nonneg _)
  have hK : 0 < K := mul_pos (by linarith) (Real.sqrt_pos.mpr (by linarith))
  -- Cauchy–Schwarz and the harmonic bound.
  have hcs : ∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t ≤
      Real.sqrt (Real.log T + 1) * S.gnorm R T i := by
    have h := cauchy_schwarz T (fun t => 1 / Real.sqrt t) (fun t => |S.g R t i|)
    have e1 : ∀ t : ℕ, (1 / Real.sqrt t) ^ 2 = 1 / t := fun t => by
      rw [div_pow, Real.sq_sqrt (Nat.cast_nonneg _), one_pow]
    simp only [e1, sq_abs, one_div_mul_eq_div] at h
    calc ∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t
        ≤ Real.sqrt ((∑ t ∈ Icc 1 T, 1 / (t : ℝ)) * ∑ t ∈ Icc 1 T, S.g R t i ^ 2) :=
          (le_abs_self _).trans (Real.abs_le_sqrt h)
      _ = Real.sqrt (∑ t ∈ Icc 1 T, 1 / (t : ℝ)) * S.gnorm R T i :=
          Real.sqrt_mul (sum_nonneg fun _ _ => by positivity) _
      _ ≤ _ := mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt (harmonic_le T)) (Real.sqrt_nonneg _)
  calc ∑ t ∈ Icc 1 T, S.m R t i ^ 2 / Real.sqrt (t * S.vhat R t i)
      ≤ ∑ t ∈ Icc 1 T, geomSum γ (fun t => |S.g R t i|) t / (K * Real.sqrt t) :=
        sum_le_sum fun t ht => term_le hR hβ₁ hβ₁' hβ₂ hβ₂' i (mem_Icc.mp ht).1
    _ = (∑ t ∈ Icc 1 T, geomSum γ (fun t => |S.g R t i|) t / Real.sqrt t) / K := by
        rw [sum_div]; refine sum_congr rfl fun t _ => ?_; rw [mul_comm K, div_div]
    _ ≤ ((∑ t ∈ Icc 1 T, |S.g R t i| / Real.sqrt t) / (1 - γ)) / K := by
        gcongr; exact sum_geomSum_div_sqrt_le hγ0 hγ (fun _ => abs_nonneg _) T
    _ ≤ (Real.sqrt (Real.log T + 1) * S.gnorm R T i / (1 - γ)) / K := by
        gcongr
    _ = _ := by rw [div_div, div_mul_eq_mul_div, mul_comm K]

/-- The hypotheses of `mainlem` are satisfiable. -/
example :
    let S := zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)
    (∀ t (a b : Vec 1), b ≤ amsgradRule t a b) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ 0) ∧
      (0 : ℝ) < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ 0 / Real.sqrt S.β₂ < 1 :=
  ⟨le_amsgradRule, fun _ _ => ⟨le_rfl, le_rfl⟩, one_pos, by norm_num [zeroSetup],
    by norm_num [zeroSetup], by simp⟩

end AMSGrad
end Transformer
