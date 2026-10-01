import Transformer.AMSGrad.Section1_TheoremA
import Mathlib.Analysis.SpecificLimits.Normed

/-
# AMSGrad — Lemmas 4.2 and 4.3

§4 of arXiv:1904.03590v4: the two lemmas on `v̂_t` behind the corrected
convergence theorem of AMSGrad.

**What the source says and what is carried here.**

* Lemma 4.2 (`vt`), `√v̂_t ≤ G∞`, uses that every iterate lies in `F`, so that
  the gradient bound applies to every `g_t`.  That is `x_mem`: the projection
  keeps the run in `F`.  The lemma needs only `0 ≤ β₂ ≤ 1`, not the source's
  `0 < β₂ < 1`, and holds for every `t`, `v̂₀ = 0` included.

* Lemma 4.3 (`t_0`) is stated once per setting of `β_{1,t}`, `t_0_lambda` and
  `t_0_inv`, both proved.  For `β_{1,t} = β₁/t` any `t₀ ≥ 3` works.

Source: arXiv:1904.03590v4, §4, Lemmas 4.2 and 4.3.
-/

open Filter Topology

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-! ### The run stays in `F` and `v̂_t` stays below `G²` -/

/-- Every state of the run carries a point of `F`. -/
theorem state_x_mem {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (R : Rule d) (n : ℕ) : (S.state R n).x ∈ F := by
  cases n with
  | zero => exact hS.x₁_mem
  | succ n => exact hS.proj.mem _ _ fun i => Real.sqrt_nonneg _

/-- Every iterate `x_t` lies in `F`. -/
theorem x_mem {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (R : Rule d) (t : ℕ) : S.x R t ∈ F :=
  state_x_mem hS R _

/-- `0 ≤ v_{t,i} ≤ G∞²`, for every rule. -/
theorem v_le {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (R : Rule d) (n : ℕ) (i : Fin d) :
    0 ≤ (S.state R n).v i ∧ (S.state R n).v i ≤ G ^ 2 := by
  induction n with
  | zero => exact ⟨le_rfl, sq_nonneg _⟩
  | succ n ih =>
    have hg := hS.grad_le (n + 1) _ (state_x_mem hS R n) i
    have hg2 : grad (S.f (n + 1)) (S.state R n).x i ^ 2 ≤ G ^ 2 := by
      rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hg 2
    change 0 ≤ S.β₂ * (S.state R n).v i + (1 - S.β₂) * grad (S.f (n + 1)) (S.state R n).x i ^ 2
      ∧ S.β₂ * (S.state R n).v i + (1 - S.β₂) * grad (S.f (n + 1)) (S.state R n).x i ^ 2 ≤ G ^ 2
    obtain ⟨h0, h1⟩ := ih
    have h1' : 0 ≤ 1 - S.β₂ := by linarith
    constructor
    · positivity
    · nlinarith [mul_le_mul_of_nonneg_left h1 hβ₂, mul_le_mul_of_nonneg_left hg2 h1']

/-- `v̂_{t+1,i} = max(v̂_{t,i}, v_{t+1,i})` under AMSGrad. -/
theorem vhat_succ (S : Setup d) (n : ℕ) (i : Fin d) :
    S.vhat amsgradRule (n + 1) i = max (S.vhat amsgradRule n i) (S.v amsgradRule (n + 1) i) :=
  rfl

/-- `v̂_{t,i}` is non-decreasing under AMSGrad. -/
theorem vhat_le_succ (S : Setup d) (n : ℕ) (i : Fin d) :
    S.vhat amsgradRule n i ≤ S.vhat amsgradRule (n + 1) i := by
  rw [vhat_succ]; exact le_max_left _ _

/-- `v̂_{t,i} ≥ 0` under AMSGrad. -/
theorem vhat_nonneg (S : Setup d) (n : ℕ) (i : Fin d) : 0 ≤ S.vhat amsgradRule n i := by
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact ih.trans (vhat_le_succ S n i)

/-- **Lemma 4.2.**  Under AMSGrad, `√v̂_{t,i} ≤ G∞`.
Source: arXiv:1904.03590v4, §4, Lemma 4.2. -/
theorem vt {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (hβ₂ : 0 ≤ S.β₂) (hβ₂' : S.β₂ ≤ 1) (t : ℕ) (i : Fin d) :
    Real.sqrt (S.vhat amsgradRule t i) ≤ G := by
  have hG : 0 ≤ G := (abs_nonneg _).trans (hS.grad_le 0 _ hS.x₁_mem i)
  have h : S.vhat amsgradRule t i ≤ G ^ 2 := by
    induction t with
    | zero => exact sq_nonneg _
    | succ n ih => exact max_le ih (v_le hS hβ₂ hβ₂' _ (n + 1) i).2
  calc Real.sqrt (S.vhat amsgradRule t i) ≤ Real.sqrt (G ^ 2) := Real.sqrt_le_sqrt h
    _ = G := Real.sqrt_sq hG

/-- The hypotheses of `vt` are satisfiable. -/
example : IsOnlineConvex (zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2))
      (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨isOnlineConvex_zero _ _ _, by norm_num, by norm_num⟩

/-! ### Lemma 4.3 -/

/-- `√(s u')/e ≤ √((s + 1)u)/c` from `s c² ≤ (s + 1)e²` and `0 ≤ u' ≤ u`. -/
theorem sqrt_div_le {s u' u c e : ℝ} (hs : 0 ≤ s) (hu' : 0 ≤ u') (hu : u' ≤ u) (hc : 0 < c)
    (he : 0 < e) (h : s * c ^ 2 ≤ (s + 1) * e ^ 2) :
    Real.sqrt (s * u') / e ≤ Real.sqrt ((s + 1) * u) / c := by
  have h0 : 0 ≤ u := hu'.trans hu
  have h1 : Real.sqrt (s * u') ≤ Real.sqrt (s * u) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hu hs)
  have e1 : Real.sqrt (s * u) * c = Real.sqrt (s * u * c ^ 2) := by
    rw [Real.sqrt_mul (by positivity : 0 ≤ s * u) (c ^ 2), Real.sqrt_sq hc.le]
  have e2 : Real.sqrt ((s + 1) * u) * e = Real.sqrt ((s + 1) * u * e ^ 2) := by
    rw [Real.sqrt_mul (by positivity : 0 ≤ (s + 1) * u) (e ^ 2), Real.sqrt_sq he.le]
  have h2 : Real.sqrt (s * u) * c ≤ Real.sqrt ((s + 1) * u) * e := by
    rw [e1, e2]; exact Real.sqrt_le_sqrt (by nlinarith [mul_le_mul_of_nonneg_left h h0])
  rw [div_le_div_iff₀ he hc]
  nlinarith [mul_le_mul_of_nonneg_right h1 hc.le]

/-- Lemma 4.3 for any `β_{1,t} < 1` satisfying `n(1 - β_{1,n+1})² ≤ (n + 1)(1 - β_{1,n})²`
from some `t₀` on. -/
theorem t_0_of_key {S : Setup d} (hb : ∀ t, S.β₁ t < 1) {t₀ : ℕ}
    (hkey : ∀ n, t₀ ≤ n → (n : ℝ) * (1 - S.β₁ (n + 1)) ^ 2 ≤ ((n : ℝ) + 1) * (1 - S.β₁ n) ^ 2)
    (t : ℕ) (ht : t₀ < t) (i : Fin d) :
    Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat amsgradRule (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
      Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (1 - S.β₁ t) := by
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := ⟨t - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  push_cast
  exact sqrt_div_le (Nat.cast_nonneg _) (vhat_nonneg S n i) (vhat_le_succ S n i)
    (by linarith [hb (n + 1)]) (by linarith [hb n]) (hkey n (by omega))

/-- **Lemma 4.3, `β_{1,t} = β₁λ^{t-1}`.**  There is `t₀` with
`√((t-1)v̂_{t-1,i})/(1 - β_{1,t-1}) ≤ √(t v̂_{t,i})/(1 - β_{1,t})` for all `t > t₀`.
Source: arXiv:1904.03590v4, §4, Lemma 4.3. -/
theorem t_0_lambda {S : Setup d} {β₁ lam : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam)
    (hl' : lam < 1) (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1)) :
    ∃ t₀, 1 ≤ t₀ ∧ ∀ t, t₀ < t → ∀ i : Fin d,
      Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat amsgradRule (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
        Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (1 - S.β₁ t) := by
  have hlim : Tendsto (fun n : ℕ => 2 * ((n : ℝ) * lam ^ n + 2 * lam ^ n)) atTop (𝓝 0) := by
    simpa using ((tendsto_self_mul_const_pow_of_lt_one hl.le hl').add
      ((tendsto_pow_atTop_nhds_zero_of_lt_one hl.le hl').const_mul 2)).const_mul 2
  obtain ⟨N, hN⟩ := eventually_atTop.mp (hlim.eventually (gt_mem_nhds one_pos))
  have hlt : ∀ t, S.β₁ t < 1 := fun t => by
    rw [hb]; nlinarith [pow_le_one₀ hl.le hl'.le (n := t - 1)]
  refine ⟨N + 1, by omega, fun t ht i => t_0_of_key hlt (fun n hn => ?_) t ht i⟩
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have hk := hN k (by omega)
  simp only [hb, show k + 1 + 1 - 1 = k + 1 by omega, show k + 1 - 1 = k by omega, pow_succ]
  push_cast
  set p := lam ^ k
  have hp0 : 0 ≤ p := by positivity
  have hp1 : p ≤ 1 := pow_le_one₀ hl.le hl'.le
  have ha : β₁ * p ≤ p := mul_le_of_le_one_left hp0 hβ₁'.le
  have hc : (1 - β₁ * (p * lam)) ^ 2 ≤ 1 := by
    have : 0 ≤ β₁ * (p * lam) := by positivity
    have : β₁ * (p * lam) ≤ 1 := by nlinarith
    nlinarith
  have h1 : ((k : ℝ) + 1) * (1 - β₁ * (p * lam)) ^ 2 ≤ (k : ℝ) + 1 := by
    nlinarith [mul_le_mul_of_nonneg_left hc (by positivity : (0 : ℝ) ≤ k + 1)]
  have h2 : ((k : ℝ) + 1 + 1) * (1 - 2 * (β₁ * p)) ≤ ((k : ℝ) + 1 + 1) * (1 - β₁ * p) ^ 2 :=
    mul_le_mul_of_nonneg_left (by nlinarith [sq_nonneg (β₁ * p)]) (by positivity)
  have h3 : ((k : ℝ) + 1 + 1) * (β₁ * p) ≤ ((k : ℝ) + 1 + 1) * p :=
    mul_le_mul_of_nonneg_left ha (by positivity)
  nlinarith

/-- **Lemma 4.3, `β_{1,t} = β₁/t`.**  `t₀ = 3` works.
Source: arXiv:1904.03590v4, §4, Lemma 4.3. -/
theorem t_0_inv {S : Setup d} {β₁ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t) :
    ∃ t₀, 1 ≤ t₀ ∧ ∀ t, t₀ < t → ∀ i : Fin d,
      Real.sqrt (((t - 1 : ℕ) : ℝ) * S.vhat amsgradRule (t - 1) i) / (1 - S.β₁ (t - 1)) ≤
        Real.sqrt ((t : ℝ) * S.vhat amsgradRule t i) / (1 - S.β₁ t) := by
  have hlt : ∀ t, S.β₁ t < 1 := fun t => by
    rw [hb]
    rcases Nat.eq_zero_or_pos t with rfl | ht
    · simp
    · exact (div_le_self hβ₁ (by exact_mod_cast ht)).trans_lt hβ₁'
  refine ⟨3, by omega, fun t ht i => t_0_of_key hlt (fun n hn => ?_) t ht i⟩
  have hn3 : (3 : ℝ) ≤ n := by exact_mod_cast hn
  rw [hb, hb]
  push_cast
  have e1 : 1 - β₁ / ((n : ℝ) + 1) = ((n : ℝ) + 1 - β₁) / ((n : ℝ) + 1) := by
    field_simp
  have e2 : 1 - β₁ / (n : ℝ) = ((n : ℝ) - β₁) / n := by
    field_simp
  rw [e1, e2, div_pow, div_pow, mul_div_assoc', mul_div_assoc',
    div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ n) (by linarith : (0 : ℝ) ≤ n))
      (sq_nonneg (β₁ - 1)), mul_nonneg (by linarith : (0 : ℝ) ≤ n) (sq_nonneg (3 * β₁ - 1)),
    mul_nonneg hβ₁ (by linarith : (0 : ℝ) ≤ 1 - β₁), pow_le_pow_left₀ (by norm_num) hn3 3,
    pow_le_pow_left₀ (by norm_num) hn3 4]

/-- The hypotheses of `t_0_lambda` and `t_0_inv` are satisfiable. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    ∀ t : ℕ, (zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)).β₁ t = 0 * (1 / 2) ^ (t - 1) ∧
      (zeroSetup (d := 1) (fun _ => 1) (fun _ => 0) (1 / 2)).β₁ t = 0 / t :=
  ⟨le_rfl, one_pos, by norm_num, by norm_num, fun _ => ⟨by simp [zeroSetup], by simp [zeroSetup]⟩⟩

end AMSGrad
end Transformer
