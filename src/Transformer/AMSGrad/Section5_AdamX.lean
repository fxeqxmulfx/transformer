import Transformer.AMSGrad.Section4_Lemmas

/-
# AdamX — the algorithm and Lemmas 5.2, 5.3

§5 of arXiv:1904.03590v4: AdamX (Algorithm 2), which differs from AMSGrad only
in `v̂_1 = v_1` and `v̂_t = max((1-β_{1,t})²/(1-β_{1,t-1})² v̂_{t-1}, v_t)` for
`t ≥ 2`; Lemma 5.2 (`vtnew`) and Lemma 5.3 (`vt2`).

**What the source says and what is carried here.**

* Lemma 5.2 writes `v̂_t` as a maximum over `1 ≤ s ≤ t`; it is stated as that
  maximum's two properties, an upper bound attained at some `s`.  Its proof
  cancels `1 - β_{1,t-1}`, so it needs `β_{1,t} ≠ 1`.

* Lemma 5.3 needs `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1` and `0 ≤ β₂ ≤ 1`, which the
  source assumes throughout; it holds for every `t`, `v̂₀ = 0` included.

* Lemma 5.4 (`mainlem2`) is Lemma 4.4 for AdamX.  `mainlem` is stated for
  every rule with `v_t ≤ v̂_t`, and AdamX is one, `le_adamXRule`; so Lemma 5.4
  is `mainlem le_adamXRule`, not a separate statement.

* With `β_{1,t} = 0` AdamX is AMSGrad, `adamX_eq_amsgrad`: the counterexample
  to the lower half of Corollary 4.5, `not_cor_lower`, is a run of AdamX too.

Source: arXiv:1904.03590v4, §5, Algorithm 2, Lemmas 5.2–5.4.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- AdamX's `v̂_1 = v_1`, `v̂_t = max((1-β_{1,t})²/(1-β_{1,t-1})² v̂_{t-1}, v_t)`.
arXiv:1904.03590v4, §5, Algorithm 2. -/
noncomputable def adamXRule (β₁ : ℕ → ℝ) : Rule d := fun t a b =>
  if t = 1 then b else (((1 - β₁ t) / (1 - β₁ (t - 1))) ^ 2 • a) ⊔ b

/-- AdamX keeps `v̂_t ≥ v_t`. -/
theorem le_adamXRule (β₁ : ℕ → ℝ) (t : ℕ) (a b : Vec d) : b ≤ adamXRule β₁ t a b := by
  unfold adamXRule; split_ifs
  · exact le_rfl
  · exact le_sup_right

/-- `v̂_1 = v_1` under AdamX. -/
theorem adamX_vhat_one (S : Setup d) (i : Fin d) :
    S.vhat (adamXRule S.β₁) 1 i = S.v (adamXRule S.β₁) 1 i := by
  show (adamXRule S.β₁ 1 (S.vhat (adamXRule S.β₁) 0) (S.v (adamXRule S.β₁) 1)) i = _
  simp only [adamXRule, ↓reduceIte]

/-- `v̂_{t+1} = max((1-β_{1,t+1})²/(1-β_{1,t})² v̂_t, v_{t+1})` under AdamX, `t ≥ 1`. -/
theorem adamX_vhat_succ (S : Setup d) {t : ℕ} (ht : 1 ≤ t) (i : Fin d) :
    S.vhat (adamXRule S.β₁) (t + 1) i =
      max (((1 - S.β₁ (t + 1)) / (1 - S.β₁ t)) ^ 2 * S.vhat (adamXRule S.β₁) t i)
        (S.v (adamXRule S.β₁) (t + 1) i) := by
  show (adamXRule S.β₁ (t + 1) (S.vhat (adamXRule S.β₁) t) (S.v (adamXRule S.β₁) (t + 1))) i = _
  simp only [adamXRule, show t + 1 ≠ 1 by omega, ↓reduceIte, Nat.add_sub_cancel]
  rfl

/-- **Lemma 5.2.**  For `t ≥ 1`, `v̂_t = max_{1≤s≤t} (1-β_{1,t})²/(1-β_{1,s})² v_s`
under AdamX, provided `β_{1,t} ≠ 1`.
Source: arXiv:1904.03590v4, §5, Lemma 5.2. -/
theorem vtnew {S : Setup d} (hβ : ∀ t, 1 ≤ t → S.β₁ t ≠ 1) {t : ℕ} (ht : 1 ≤ t) (i : Fin d) :
    (∀ s ∈ Icc 1 t, ((1 - S.β₁ t) / (1 - S.β₁ s)) ^ 2 * S.v (adamXRule S.β₁) s i ≤
        S.vhat (adamXRule S.β₁) t i) ∧
      ∃ s ∈ Icc 1 t, S.vhat (adamXRule S.β₁) t i =
        ((1 - S.β₁ t) / (1 - S.β₁ s)) ^ 2 * S.v (adamXRule S.β₁) s i := by
  have hne : ∀ t, 1 ≤ t → 1 - S.β₁ t ≠ 0 := fun t ht h => hβ t ht (by linarith)
  induction t, ht using Nat.le_induction with
  | base =>
    have h1 : ((1 - S.β₁ 1) / (1 - S.β₁ 1)) ^ 2 = 1 := by rw [div_self (hne 1 le_rfl), one_pow]
    refine ⟨fun s hs => ?_, 1, by simp, by rw [h1, one_mul, adamX_vhat_one]⟩
    obtain rfl : s = 1 := by simp at hs; omega
    rw [h1, one_mul, adamX_vhat_one]
  | succ t ht ih =>
    set q := ((1 - S.β₁ (t + 1)) / (1 - S.β₁ t)) ^ 2
    have hq : ∀ s, q * ((1 - S.β₁ t) / (1 - S.β₁ s)) ^ 2 = ((1 - S.β₁ (t + 1)) / (1 - S.β₁ s)) ^ 2 :=
      fun s => by rw [← mul_pow, div_mul_div_cancel₀ (hne t ht)]
    have h1 : ((1 - S.β₁ (t + 1)) / (1 - S.β₁ (t + 1))) ^ 2 = 1 := by
      rw [div_self (hne (t + 1) (by omega)), one_pow]
    rw [adamX_vhat_succ S ht]
    refine ⟨fun s hs => ?_, ?_⟩
    · rcases (show s ≤ t ∨ s = t + 1 by simp at hs; omega) with hst | rfl
      · have := mul_le_mul_of_nonneg_left (ih.1 s (by simp at hs ⊢; omega)) (sq_nonneg
          ((1 - S.β₁ (t + 1)) / (1 - S.β₁ t)))
        rw [← mul_assoc, hq] at this
        exact this.trans (le_max_left _ _)
      · rw [h1, one_mul]; exact le_max_right _ _
    · rcases le_total (q * S.vhat (adamXRule S.β₁) t i) (S.v (adamXRule S.β₁) (t + 1) i)
        with h | h
      · exact ⟨t + 1, by simp, by rw [max_eq_right h, h1, one_mul]⟩
      · obtain ⟨s, hs, e⟩ := ih.2
        refine ⟨s, by simp at hs ⊢; omega, ?_⟩
        rw [max_eq_left h, e, ← mul_assoc, hq]

/-- **Lemma 5.3.**  Under AdamX, `√v̂_{t,i} ≤ G∞/(1 - β₁)`.
Source: arXiv:1904.03590v4, §5, Lemma 5.3. -/
theorem vt2 {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (t : ℕ) (i : Fin d) :
    Real.sqrt (S.vhat (adamXRule S.β₁) t i) ≤ G / (1 - S.β₁ 1) := by
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  have hb : 0 < 1 - S.β₁ 1 := by linarith
  have hK : 0 ≤ G / (1 - S.β₁ 1) := div_nonneg hG hb.le
  rcases Nat.eq_zero_or_pos t with rfl | ht
  · show Real.sqrt 0 ≤ _
    rw [Real.sqrt_zero]; exact hK
  obtain ⟨s, hs, e⟩ := (vtnew (fun t ht => (lt_of_le_of_lt (hβ₁ t ht).2 hβ₁').ne) ht i).2
  have hs1 : 1 ≤ s := (Finset.mem_Icc.mp hs).1
  have hr0 : 0 ≤ (1 - S.β₁ t) / (1 - S.β₁ s) :=
    div_nonneg (by linarith [(hβ₁ t ht).2]) (by linarith [(hβ₁ s hs1).2])
  have hr : (1 - S.β₁ t) / (1 - S.β₁ s) ≤ 1 / (1 - S.β₁ 1) := by
    rw [div_le_div_iff₀ (by linarith [(hβ₁ s hs1).2]) hb]
    nlinarith [(hβ₁ t ht).1, (hβ₁ s hs1).2]
  have hv := v_le hS hβ₂ hβ₂' (adamXRule S.β₁) s i
  have h : S.vhat (adamXRule S.β₁) t i ≤ (G / (1 - S.β₁ 1)) ^ 2 := by
    have h2 : (G / (1 - S.β₁ 1)) ^ 2 = (1 / (1 - S.β₁ 1)) ^ 2 * G ^ 2 := by ring
    rw [e, h2]
    exact mul_le_mul (pow_le_pow_left₀ hr0 hr 2) hv.2 hv.1 (sq_nonneg _)
  calc Real.sqrt (S.vhat (adamXRule S.β₁) t i) ≤ Real.sqrt ((G / (1 - S.β₁ 1)) ^ 2) :=
        Real.sqrt_le_sqrt h
    _ = G / (1 - S.β₁ 1) := Real.sqrt_sq hK

/-- The hypotheses of `vtnew` and `vt2` are satisfiable. -/
example :
    let S := zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (∀ t, 1 ≤ t → S.β₁ t ≠ 1) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧ S.β₁ 1 < 1 ∧ 0 ≤ S.β₂ ∧ S.β₂ ≤ 1 :=
  ⟨isOnlineConvex_zero _ _ _, fun _ _ => by simp [zeroSetup], fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup]⟩

/-! ### With `β_{1,t} = 0`, AdamX is AMSGrad -/

/-- `v_t ≥ 0` for every rule, when `0 ≤ β₂ ≤ 1`. -/
theorem v_nonneg {S : Setup d} (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (R : Rule d) (n : ℕ) :
    0 ≤ (S.state R n).v := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    intro i
    change 0 ≤ S.β₂ * (S.state R n).v i + (1 - S.β₂) * grad (S.f (n + 1)) (S.state R n).x i ^ 2
    exact add_nonneg (mul_nonneg hβ₂ (ih i)) (mul_nonneg (by linarith) (sq_nonneg _))

/-- With `β_{1,t} = 0` and `0 ≤ β₂ ≤ 1`, the runs of AdamX and AMSGrad coincide.
arXiv:1904.03590v4, §5, Algorithm 2. -/
theorem adamX_eq_amsgrad {S : Setup d} (hβ : ∀ t, S.β₁ t = 0) (hβ₂ : 0 ≤ S.β₂)
    (hβ₂' : S.β₂ ≤ 1) (n : ℕ) : S.state (adamXRule S.β₁) n = S.state amsgradRule n := by
  have key : ∀ (s : State d) (t : ℕ), 0 ≤ s.v →
      (∀ b : Vec d, 0 ≤ b → adamXRule S.β₁ t s.vhat b = amsgradRule t s.vhat b) →
      S.step (adamXRule S.β₁) t s = S.step amsgradRule t s := by
    intro s t hv h
    have hb : 0 ≤ S.β₂ • s.v + (1 - S.β₂) • grad (S.f t) s.x ^ 2 := fun i => by
      simp only [Pi.add_apply, Pi.smul_apply, Pi.pow_apply, smul_eq_mul, Pi.zero_apply]
      exact add_nonneg (mul_nonneg hβ₂ (hv i)) (mul_nonneg (by linarith) (sq_nonneg _))
    simp only [Setup.step]
    rw [h _ hb]
  induction n with
  | zero => rfl
  | succ n ih =>
    show S.step _ (n + 1) (S.state _ n) = S.step _ (n + 1) (S.state _ n)
    rw [ih]
    refine key _ _ (v_nonneg hβ₂ hβ₂' _ n) fun b hb => ?_
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp only [adamXRule, amsgradRule, ↓reduceIte]
      exact (sup_eq_right.mpr hb).symm
    · simp only [adamXRule, amsgradRule, hβ, show n + 1 ≠ 1 by omega, ↓reduceIte, sub_zero,
        div_one, one_pow, one_smul]

end AMSGrad
end Transformer
