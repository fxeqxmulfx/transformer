module

public import Transformer.Perspective.SyncKernelIntegral
public import Transformer.Perspective.SyncKernelExp

/-!
# Bounds on `f_β''` from pieces of `cos x`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §5, proof of `cor:conjecture`.

The criterion `τ_β ∫ |f_β'''|₊ < 4` of §5 follows from bounds `L ≤ f_β'' ≤ U` on
`[0, π]` (`integral_posPart_f_β'''_le`). With `c = cos x` and `s = sin x ≥ 0`,

  `f_β''(x) = s (β² s² - 3βc - 1) e^{β(c - 1)}`  (`f_β''_eq_mul`).

On a piece `c ∈ [c₁, c₂] ⊆ [0, 1]` with `β ∈ [β₁, β₂]`: `s² ≤ 1 - c₁²`,
`1 + 3βc - β² s² ≤ 1 + 3β₂c₂ - β₁²(1 - c₂²)` and `e^{β(c - 1)} ≤ e^{-β₁(1 - c₂)}`,
where `e^y` is at least its Taylor polynomial `expPoly y` of degree 5
(`taylor_five_le_exp`). So `-f_β'' ≤ A` on the piece once an inequality
between numbers holds (`NegPieceOK`, `neg_f_β''_le_of_negPieceOK`). Likewise
`f_β'' ≤ U` on a piece `-c ∈ [u₁, u₂]` (`PosPieceOK`, `f_β''_le_of_posPieceOK`). The pieces between
consecutive points of a list cover `[0, 1]` (`exists_piece_of_isChain`).

These bounds take the place of the functions `g¹`, `g²` and `g_β` of
`lemma:beta1`, …, `lemma:beta_0.33`.
-/

@[expose] public section

open Real

namespace Transformer
namespace Perspective

/-- The Taylor polynomial of degree 5 of `exp` at `0`. -/
noncomputable def expPoly (y : ℝ) : ℝ :=
  1 + y + y ^ 2 / 2 + y ^ 3 / 6 + y ^ 4 / 24 + y ^ 5 / 120

theorem expPoly_le_exp {y : ℝ} (hy : 0 ≤ y) : expPoly y ≤ Real.exp y :=
  taylor_five_le_exp hy

theorem one_le_expPoly {y : ℝ} (hy : 0 ≤ y) : 1 ≤ expPoly y := by
  rw [expPoly]
  linarith [pow_nonneg hy 2, pow_nonneg hy 3, pow_nonneg hy 4, pow_nonneg hy 5]

/-- `f_β''(x) = sin x (β² sin² x - 3β cos x - 1) e^{β(cos x - 1)}`. -/
theorem f_β''_eq_mul (β x : ℝ) : f_β'' β x = Real.sin x *
    (β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1) * Real.exp (β * (Real.cos x - 1)) := by
  rw [f_β'']
  ring

/-- The common step of the piece bounds: `s R E ≤ B` from `0 ≤ s`, `s² ≤ σ`,
`R ≤ ρ`, `0 < E`, `E p ≤ 1` and `ρ ≤ 0 ∨ σ ρ² ≤ B² p²`. -/
theorem mul_mul_le_of_sq_le {s R E σ ρ p B : ℝ} (hs : 0 ≤ s) (hσ : s ^ 2 ≤ σ) (hR : R ≤ ρ)
    (hE : 0 < E) (hp : 0 ≤ p) (hEp : E * p ≤ 1) (hB : 0 ≤ B)
    (h : ρ ≤ 0 ∨ σ * ρ ^ 2 ≤ B ^ 2 * p ^ 2) : s * R * E ≤ B := by
  rcases le_or_gt R 0 with hR0 | hR0
  · nlinarith [mul_nonneg hs hE.le]
  have h' := h.resolve_left (not_le.mpr (hR0.trans_le hR))
  have hR2 : R ^ 2 ≤ ρ ^ 2 := pow_le_pow_left₀ hR0.le hR 2
  have hsq : (s * R * E) ^ 2 ≤ B ^ 2 := by
    calc (s * R * E) ^ 2 = s ^ 2 * R ^ 2 * E ^ 2 := by ring
      _ ≤ σ * ρ ^ 2 * E ^ 2 := mul_le_mul_of_nonneg_right
          (mul_le_mul hσ hR2 (sq_nonneg _) ((sq_nonneg s).trans hσ)) (sq_nonneg _)
      _ ≤ B ^ 2 * p ^ 2 * E ^ 2 := by gcongr
      _ = B ^ 2 * (E * p) ^ 2 := by ring
      _ ≤ B ^ 2 * 1 := by gcongr; nlinarith [mul_nonneg hE.le hp]
      _ = B ^ 2 := mul_one _
  exact (le_abs_self _).trans (abs_le_of_sq_le_sq hsq hB)

/-- The numerical condition for `-f_β'' ≤ A` on a piece `cos x ∈ [c₁, c₂] ⊆ [0, 1]`
when `β ∈ [β₁, β₂]`. -/
def NegPieceOK (β₁ β₂ A c₁ c₂ : ℝ) : Prop :=
  0 ≤ c₁ ∧ c₂ ≤ 1 ∧ (1 - c₁ ^ 2) * (1 + 3 * β₂ * c₂ - β₁ ^ 2 * (1 - c₂ ^ 2)) ^ 2 ≤
    A ^ 2 * expPoly (β₁ * (1 - c₂)) ^ 2

/-- The numerical condition for `f_β'' ≤ U` on a piece `-cos x ∈ [u₁, u₂]`, `0 ≤ u₁`,
when `β ∈ [β₁, β₂]`. -/
def PosPieceOK (β₁ β₂ U u₁ u₂ : ℝ) : Prop :=
  0 ≤ u₁ ∧ (β₂ ^ 2 * (1 - u₁ ^ 2) + 3 * β₂ * u₂ - 1 ≤ 0 ∨
    (1 - u₁ ^ 2) * (β₂ ^ 2 * (1 - u₁ ^ 2) + 3 * β₂ * u₂ - 1) ^ 2 ≤
      U ^ 2 * expPoly (β₁ * (1 + u₁)) ^ 2)

/-- **`-f_β'' ≤ A` on a piece of `cos x ≥ 0`.** -/
theorem neg_f_β''_le_of_negPieceOK {β β₁ β₂ A c₁ c₂ x : ℝ} (hβ₁ : 0 ≤ β₁)
    (hβ : β ∈ Set.Icc β₁ β₂) (hA : 0 ≤ A) (hx : 0 ≤ Real.sin x)
    (hc : Real.cos x ∈ Set.Icc c₁ c₂) (h : NegPieceOK β₁ β₂ A c₁ c₂) : -f_β'' β x ≤ A := by
  obtain ⟨hc₁, hc₂, h⟩ := h
  have hs2 := Real.sin_sq x
  have hβ0 : 0 ≤ β := hβ₁.trans hβ.1
  have hc0 : 0 ≤ Real.cos x := hc₁.trans hc.1
  have hy : 0 ≤ β₁ * (1 - c₂) := mul_nonneg hβ₁ (by linarith)
  rw [show -f_β'' β x = Real.sin x * (1 + 3 * β * Real.cos x - β ^ 2 * Real.sin x ^ 2) *
      Real.exp (β * (Real.cos x - 1)) by rw [f_β''_eq_mul]; ring]
  refine mul_mul_le_of_sq_le hx (σ := 1 - c₁ ^ 2)
    (by rw [hs2]; nlinarith [mul_le_mul hc.1 hc.1 hc₁ hc0]) (ρ := 1 + 3 * β₂ * c₂ -
    β₁ ^ 2 * (1 - c₂ ^ 2)) ?_ (Real.exp_pos _) (zero_le_one.trans (one_le_expPoly hy)) ?_ hA
    (Or.inr h)
  · have h1 : β * Real.cos x ≤ β₂ * c₂ := mul_le_mul hβ.2 hc.2 hc0 (hβ0.trans hβ.2)
    have h2 : β₁ ^ 2 * (1 - c₂ ^ 2) ≤ β ^ 2 * Real.sin x ^ 2 :=
      mul_le_mul (pow_le_pow_left₀ hβ₁ hβ.1 2)
        (by rw [hs2]; nlinarith [mul_le_mul hc.2 hc.2 hc0 (hc0.trans hc.2)])
        (by nlinarith [mul_le_mul hc₂ hc₂ (hc0.trans hc.2) zero_le_one]) (sq_nonneg _)
    linarith
  · calc Real.exp (β * (Real.cos x - 1)) * expPoly (β₁ * (1 - c₂))
        ≤ Real.exp (-(β₁ * (1 - c₂))) * expPoly (β₁ * (1 - c₂)) := by
          gcongr ?_ * _
          · exact zero_le_one.trans (one_le_expPoly hy)
          · refine Real.exp_le_exp.mpr ?_
            nlinarith [mul_le_mul hβ.1 (by linarith [hc.2] : 1 - c₂ ≤ 1 - Real.cos x)
              (by linarith) hβ0]
      _ ≤ Real.exp (-(β₁ * (1 - c₂))) * Real.exp (β₁ * (1 - c₂)) := by
          gcongr
          exact expPoly_le_exp hy
      _ = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]

/-- **`f_β'' ≤ U` on a piece of `cos x ≤ 0`.** -/
theorem f_β''_le_of_posPieceOK {β β₁ β₂ U u₁ u₂ x : ℝ} (hβ₁ : 0 ≤ β₁)
    (hβ : β ∈ Set.Icc β₁ β₂) (hU : 0 ≤ U) (hx : 0 ≤ Real.sin x)
    (hu : -Real.cos x ∈ Set.Icc u₁ u₂) (h : PosPieceOK β₁ β₂ U u₁ u₂) : f_β'' β x ≤ U := by
  obtain ⟨hu₁, h⟩ := h
  have hs2 := Real.sin_sq x
  have hβ0 : 0 ≤ β := hβ₁.trans hβ.1
  have hu0 : 0 ≤ -Real.cos x := hu₁.trans hu.1
  have hy : 0 ≤ β₁ * (1 + u₁) := mul_nonneg hβ₁ (by linarith)
  rw [f_β''_eq_mul]
  refine mul_mul_le_of_sq_le hx (σ := 1 - u₁ ^ 2)
    (by rw [hs2]; nlinarith [mul_le_mul hu.1 hu.1 hu₁ hu0]) (ρ := β₂ ^ 2 *
    (1 - u₁ ^ 2) + 3 * β₂ * u₂ - 1) ?_ (Real.exp_pos _) (zero_le_one.trans (one_le_expPoly hy))
    ?_ hU h
  · have h1 : β * -Real.cos x ≤ β₂ * u₂ := mul_le_mul hβ.2 hu.2 hu0 (hβ0.trans hβ.2)
    have h2 : β ^ 2 * Real.sin x ^ 2 ≤ β₂ ^ 2 * (1 - u₁ ^ 2) :=
      mul_le_mul (pow_le_pow_left₀ hβ0 hβ.2 2)
        (by rw [hs2]; nlinarith [mul_le_mul hu.1 hu.1 hu₁ hu0]) (sq_nonneg _) (sq_nonneg _)
    linarith
  · calc Real.exp (β * (Real.cos x - 1)) * expPoly (β₁ * (1 + u₁))
        ≤ Real.exp (-(β₁ * (1 + u₁))) * expPoly (β₁ * (1 + u₁)) := by
          gcongr ?_ * _
          · exact zero_le_one.trans (one_le_expPoly hy)
          · refine Real.exp_le_exp.mpr ?_
            nlinarith [mul_le_mul hβ.1 (by linarith [hu.1] : 1 + u₁ ≤ 1 - Real.cos x)
              (by linarith) hβ0]
      _ ≤ Real.exp (-(β₁ * (1 + u₁))) * Real.exp (β₁ * (1 + u₁)) := by
          gcongr
          exact expPoly_le_exp hy
      _ = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]

/-- `-f_β'' ≤ 1` where `cos x ≤ 0`, for `β ≥ 0`: there `1 + 3β cos x - β² sin² x ≤ 1`. -/
theorem neg_f_β''_le_one {β x : ℝ} (hβ : 0 ≤ β) (hx : 0 ≤ Real.sin x)
    (hc : Real.cos x ≤ 0) : -f_β'' β x ≤ 1 := by
  rw [show -f_β'' β x = Real.sin x * (1 + 3 * β * Real.cos x - β ^ 2 * Real.sin x ^ 2) *
      Real.exp (β * (Real.cos x - 1)) by rw [f_β''_eq_mul]; ring]
  refine mul_mul_le_of_sq_le hx (Real.sin_sq_le_one x) (ρ := 1)
    (by nlinarith [sq_nonneg (β * Real.sin x)])
    (Real.exp_pos _) zero_le_one ?_ zero_le_one (Or.inr (by norm_num))
  rw [mul_one, Real.exp_le_one_iff]
  nlinarith [Real.cos_le_one x]

/-- `f_β'' ≤ 0` where `cos x ≥ 0`, for `0 ≤ β ≤ 1`: there
`β² sin² x - 3β cos x - 1 ≤ β² - 1 ≤ 0`. -/
theorem f_β''_nonpos {β x : ℝ} (hβ : 0 ≤ β) (hβ1 : β ≤ 1) (hx : 0 ≤ Real.sin x)
    (hc : 0 ≤ Real.cos x) : f_β'' β x ≤ 0 := by
  rw [f_β''_eq_mul]
  refine mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonneg_of_nonpos hx ?_) (Real.exp_pos _).le
  nlinarith [Real.sin_sq_le_one x, mul_nonneg hβ hc, mul_le_mul hβ1 hβ1 hβ zero_le_one]

/-- **Pieces cover.**  If `P` holds between consecutive points of `a :: l ++ [b]`,
every `c ∈ [a, b]` lies in a piece `[c₁, c₂]` with `P c₁ c₂`. -/
theorem exists_piece_of_isChain {P : ℝ → ℝ → Prop} :
    ∀ (l : List ℝ) {a b : ℝ}, List.IsChain P (a :: (l ++ [b])) → ∀ c ∈ Set.Icc a b,
      ∃ c₁ c₂, P c₁ c₂ ∧ c ∈ Set.Icc c₁ c₂
  | [], a, b, h, c, hc => ⟨a, b, List.rel_of_isChain_cons_cons h, hc⟩
  | d :: l, a, b, h, c, hc => by
    rw [List.cons_append, List.isChain_cons_cons] at h
    rcases le_or_gt c d with hcd | hcd
    · exact ⟨a, d, h.1, hc.1, hcd⟩
    · exact exists_piece_of_isChain l h.2 c ⟨hcd.le, hc.2⟩

/-- The hypotheses of the bounds above are satisfiable: `β = 0` at `x = π/2`, the
pieces `[0, 1]`, and `y = 0` for `expPoly`. -/
example : expPoly 0 ≤ Real.exp 0 ∧ -f_β'' 0 (π / 2) ≤ 1 ∧ f_β'' 0 (π / 2) ≤ 1 ∧
    f_β'' 0 0 ≤ 0 ∧ (0 : ℝ) * 0 * 1 ≤ 1 ∧ ∃ c₁ c₂ : ℝ, c₁ ≤ c₂ ∧ (1 / 2 : ℝ) ∈ Set.Icc c₁ c₂ :=
  ⟨expPoly_le_exp le_rfl, neg_f_β''_le_of_negPieceOK (c₁ := 0) (c₂ := 1) le_rfl
    ⟨le_rfl, le_rfl⟩ zero_le_one (by simp) (by simp) (by norm_num [NegPieceOK, expPoly]),
    f_β''_le_of_posPieceOK (u₁ := 0) (u₂ := 1) le_rfl ⟨le_rfl, le_rfl⟩ zero_le_one (by simp)
      (by simp) (by norm_num [PosPieceOK]),
    f_β''_nonpos le_rfl zero_le_one (by simp) (by simp),
    mul_mul_le_of_sq_le le_rfl le_rfl le_rfl one_pos le_rfl (by norm_num) zero_le_one
      (Or.inl le_rfl), exists_piece_of_isChain [] (P := (· ≤ ·)) (a := 0) (b := 1)
      (List.isChain_pair.mpr zero_le_one) _ ⟨by norm_num, by norm_num⟩⟩

/-- The hypotheses of `neg_f_β''_le_one` and `one_le_expPoly` are satisfiable. -/
example : -f_β'' 0 π ≤ 1 ∧ 1 ≤ expPoly 0 :=
  ⟨neg_f_β''_le_one le_rfl (by simp) (by simp), one_le_expPoly le_rfl⟩

end Perspective
end Transformer
