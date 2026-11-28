module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Bounds

/-! The literal Argand AdamW in width. In the mean-field parametrization a
width unit receives its gradient divided by the width N. Adam is invariant
under rescaling the gradient together with the floor, so the literal
`AdamW.run` with floor 10⁻⁸ on the flat parameters Fin N × P is the
N-particle `system` with floor N · 10⁻⁸; its moments are the system's
divided by N and N². Clipping does not act once the squared gradient norm
is at most 1, which holds for per-particle gradients bounded by C when
|P| C² ≤ N. A floor that does not grow with N would instead make the
direction vanish like 10|m|/eps: the literal family has no nondegenerate
limit unless the floor is held at its value N · 10⁻⁸ for one width. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Finset Transformer.AMSGradW.Kernite

variable {P K : Type*}

/-- Adam is invariant under rescaling the gradient together with the floor. -/
theorem direction_scale {s : ℝ} (hs : 0 < s) (eps : ℝ) (t : ℕ) (m v : ℝ) :
    direction eps t (m / s) (v / s^2) = direction (s * eps) t m v := by
  have hc : Real.sqrt (1 - (95/100 : ℝ)^(t+1)) ≠ 0 :=
    (Real.sqrt_pos.2 (sub_pos.2 (pow_lt_one₀ (by norm_num) (by norm_num) (Nat.succ_ne_zero t)))).ne'
  unfold direction denominator
  rw [Real.sqrt_div' v (by positivity), Real.sqrt_sq hs.le]
  have hden : firstBias t * (Real.sqrt v / s / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) + eps) =
      firstBias t * (Real.sqrt v / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) + s * eps) / s := by
    field_simp
  rw [hden, div_div_div_cancel_right₀ hs.ne']

/-- With a positive floor the direction is at most 10|m|/eps. -/
theorem abs_direction_le_floor {eps : ℝ} (he : 0 < eps) (t : ℕ) (m v : ℝ) :
    |direction eps t m v| ≤ 10 * |m| / eps := by
  have hb := firstBias_lower t
  have hq : 0 ≤ Real.sqrt v / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) := by positivity
  have hd : eps / 10 ≤ denominator eps t v := by
    unfold denominator
    nlinarith
  have hd0 : 0 < denominator eps t v := lt_of_lt_of_le (by positivity) hd
  rw [direction, abs_div, abs_of_pos hd0, div_le_div_iff₀ hd0 he]
  nlinarith [abs_nonneg m]

/-- Clipping does not act on a gradient of squared norm at most 1. -/
theorem clip_of_sum_sq_le {ι : Type*} [Fintype ι] (g : ι → ℝ) (h : ∑ i, g i ^ 2 ≤ 1) :
    clip g = g := by
  funext i
  have hn : ‖(WithLp.toLp 2 g : Space ι)‖ ≤ 1 := by
    rw [EuclideanSpace.norm_eq]
    exact Real.sqrt_le_one.2 (by simpa using h)
  rw [clip, max_eq_right hn, div_one]

/-- The flat parameter vector of N particles. -/
def flatten {N : ℕ} (w : Fin N → P → ℝ) : Fin N × P → ℝ := fun q => w q.1 q.2

/-- The literal gradient in the mean-field parametrization: particle i
receives its per-particle gradient divided by the width. -/
noncomputable def literalGrad (A : Training P K) (N t : ℕ) (W : Fin N × P → ℝ) : Fin N × P → ℝ :=
  fun q => A.grad t (fun p => W (q.1, p)) (meanField A.field fun j p => W (j, p)) q.2 / N

/-- Bounded per-particle gradients have squared norm at most 1 once |P| C² ≤ N. -/
theorem sum_sq_literalGrad_le [Fintype P] (A : Training P K) {N : ℕ} (hN : 0 < N) {C : ℝ}
    (hG : ∀ t θ Φ p, |A.grad t θ Φ p| ≤ C) (hw : Fintype.card P * C^2 ≤ N) (t : ℕ)
    (W : Fin N × P → ℝ) : ∑ q, literalGrad A N t W q ^ 2 ≤ 1 := by
  have hN' : (0:ℝ) < N := Nat.cast_pos.2 hN
  have hq (q : Fin N × P) : literalGrad A N t W q ^ 2 ≤ C^2 / N^2 := by
    rw [literalGrad, div_pow]
    have := abs_le.1 (hG t (fun p => W (q.1, p)) (meanField A.field fun j p => W (j, p)) q.2)
    exact div_le_div_of_nonneg_right (sq_le_sq' this.1 this.2) (by positivity)
  calc ∑ q, literalGrad A N t W q ^ 2 ≤ ∑ _q : Fin N × P, C^2 / N^2 := sum_le_sum fun q _ => hq q
    _ = Fintype.card P * C^2 / N := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul]
        push_cast
        field_simp
    _ ≤ 1 := (div_le_one hN').2 hw

theorem run_succ_first (rate : ℕ → ℝ) {ι : Type*} [Fintype ι] (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (t : ℕ) (i : ι) :
    (AdamW.run rate grad initial (t+1)).first i = 9/10 * (AdamW.run rate grad initial t).first i +
      1/10 * clip (grad t (AdamW.run rate grad initial t).weights) i := rfl

theorem run_succ_second (rate : ℕ → ℝ) {ι : Type*} [Fintype ι] (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (t : ℕ) (i : ι) :
    (AdamW.run rate grad initial (t+1)).second i = 95/100 * (AdamW.run rate grad initial t).second i +
      5/100 * (clip (grad t (AdamW.run rate grad initial t).weights) i)^2 := rfl

theorem run_succ_weights (rate : ℕ → ℝ) {ι : Type*} [Fintype ι] (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (t : ℕ) (i : ι) :
    (AdamW.run rate grad initial (t+1)).weights i =
      (1 - rate t / 10) * (AdamW.run rate grad initial t).weights i - rate t *
        AdamW.direction t ((AdamW.run rate grad initial (t+1)).first i)
          ((AdamW.run rate grad initial (t+1)).second i) := rfl

/-- The literal AdamW run with floor 10⁻⁸ is the N-particle system with
floor N · 10⁻⁸ while clipping does not act. -/
theorem run_literal [Fintype P] (A : Training P K) {N : ℕ} (hN : 0 < N)
    (heps : A.eps = N / 100000000) (hclip : ∀ t W, ∑ q, literalGrad A N t W q ^ 2 ≤ 1)
    (init : Fin N → P → ℝ) (t : ℕ) (i : Fin N) (p : P) :
    (AdamW.run A.rate (literalGrad A N) (flatten init) t).weights (i, p) =
        (system A init t i).weights p ∧
      (AdamW.run A.rate (literalGrad A N) (flatten init) t).first (i, p) =
        (system A init t i).first p / N ∧
      (AdamW.run A.rate (literalGrad A N) (flatten init) t).second (i, p) =
        (system A init t i).second p / N^2 := by
  have hN' : (0:ℝ) < N := Nat.cast_pos.2 hN
  have he : (N:ℝ) * (1/100000000) = A.eps := by rw [heps]; ring
  induction t generalizing i p with
  | zero => simp [AdamW.run, system, flatten]
  | succ t ih =>
    have hw : (AdamW.run A.rate (literalGrad A N) (flatten init) t).weights =
        flatten fun j => (system A init t j).weights := funext fun q => (ih q.1 q.2).1
    have hg (i : Fin N) (p : P) :
        clip (literalGrad A N t (AdamW.run A.rate (literalGrad A N) (flatten init) t).weights) (i, p) =
          A.grad t (system A init t i).weights (meanField A.field fun j => (system A init t j).weights) p
            / N := by
      rw [clip_of_sum_sq_le _ (hclip t _), hw]
      rfl
    have h1 : (AdamW.run A.rate (literalGrad A N) (flatten init) (t+1)).first (i, p) =
        (system A init (t+1) i).first p / N := by
      rw [run_succ_first, hg, (ih i p).2.1, system_succ]
      change _ = (9/10 * (system A init t i).first p + 1/10 * A.grad t (system A init t i).weights
        (meanField A.field fun j => (system A init t j).weights) p) / N
      ring
    have h2 : (AdamW.run A.rate (literalGrad A N) (flatten init) (t+1)).second (i, p) =
        (system A init (t+1) i).second p / N^2 := by
      rw [run_succ_second, hg, (ih i p).2.2, system_succ]
      change _ = (95/100 * (system A init t i).second p + 5/100 * (A.grad t (system A init t i).weights
        (meanField A.field fun j => (system A init t j).weights) p)^2) / N^2
      ring
    refine ⟨?_, h1, h2⟩
    rw [run_succ_weights, h1, h2, (ih i p).1, ← direction_literal, direction_scale hN', he]
    rfl

section Examples

/-- A training with zero gradient and the literal floor of width 2. -/
noncomputable def quietTraining : Training (Fin 1) (Fin 1) :=
  ⟨2 / 100000000, fun _ => 1 / 1000, fun _ _ _ _ => 0, fun _ _ => 0⟩

example := direction_scale (s := 2) two_pos 1 0 1 1
example := abs_direction_le_floor one_pos 0 1 0
example := clip_of_sum_sq_le (ι := Fin 1) 0 (by simp)
example := sum_sq_literalGrad_le quietTraining (N := 2) two_pos (C := 0)
  (fun _ _ _ _ => by simp [quietTraining]) (by norm_num) 3 0
example := run_literal quietTraining (N := 2) two_pos (by norm_num [quietTraining])
  (fun t W => sum_sq_literalGrad_le quietTraining two_pos (C := 0)
    (fun _ _ _ _ => by simp [quietTraining]) (by norm_num) t W) (fun _ => 0) 3 0 0

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
