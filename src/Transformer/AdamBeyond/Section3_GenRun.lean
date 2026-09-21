import Transformer.AdamBeyond.Section3_Run

/-
# Adam and beyond — §3: the run behind Theorem 2

The one-dimensional problem of the proof of Theorem 2: `F = [-1, 1]`,
`f_t(x) = C x` for `t mod C = 1` and `-x` otherwise, Adam with `β_{1,t} = β₁`,
`β₂` and `α_t = α/√t`, started at `x₁ = 1`, preceded here by a warm-up of `C`
steps with `f_t = 0`, during which nothing moves (`gen_warmup`).  Along the whole
run `-1 ≤ m_t ≤ C` and `0 ≤ v_t ≤ C²` (`gen_bounds`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 2.
-/

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- The slope of `f_t`: `0` during the warm-up `t ≤ C`, then `C` if `t mod C = 1`,
else `-1`. -/
noncomputable def genSlope (C : ℕ) (t : ℕ) : ℝ :=
  if t ≤ C then 0 else if t % C = 1 then C else -1

/-- The problem of Theorem 2, with the warm-up.
arXiv:1904.09237, Appendix, proof of Theorem 2. -/
noncomputable def genSetup (C : ℕ) (β₁ β₂ α : ℝ) : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun t x => genSlope C t * x 0, fun t => α / Real.sqrt t,
    fun _ => β₁, β₂, fun _ => 1⟩

theorem abs_genSlope_le {C : ℕ} (hC : 1 ≤ C) (t : ℕ) : |genSlope C t| ≤ C := by
  have : (1 : ℝ) ≤ C := by exact_mod_cast hC
  unfold genSlope; split_ifs
  · simp
  · rw [abs_of_nonneg (by linarith)]
  · rw [abs_neg, abs_one]; exact this

/-- The problem is an online convex one, with `D = 2`, `G = C`. -/
theorem isOnlineConvex_gen {C : ℕ} (hC : 1 ≤ C) (β₁ β₂ α : ℝ) :
    IsOnlineConvex (genSetup C β₁ β₂ α) (Set.Icc (fun _ => -1) (fun _ => 1)) 2 C where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [genSetup], fun _ => le_rfl⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [genSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul _).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by
    change |grad (fun x : Vec 1 => genSlope C t * x 0) x i| ≤ C
    rw [grad_linear]
    exact abs_genSlope_le hC t

section Run

variable (C : ℕ) (β₁ β₂ α : ℝ)

/-- `m_n`. -/
noncomputable def genM (n : ℕ) : ℝ := ((genSetup C β₁ β₂ α).state adamRule n).m 0
/-- `v_n`. -/
noncomputable def genV (n : ℕ) : ℝ := ((genSetup C β₁ β₂ α).state adamRule n).v 0
/-- `x_{n+1}`. -/
noncomputable def genX (n : ℕ) : ℝ := ((genSetup C β₁ β₂ α).state adamRule n).x 0

theorem genM_succ (n : ℕ) :
    genM C β₁ β₂ α (n + 1) = β₁ * genM C β₁ β₂ α n + (1 - β₁) * genSlope C (n + 1) := by
  simp [genM, Setup.state, Setup.step, genSetup, grad_linear]

theorem genV_succ (n : ℕ) :
    genV C β₁ β₂ α (n + 1) = β₂ * genV C β₁ β₂ α n + (1 - β₂) * genSlope C (n + 1) ^ 2 := by
  simp [genV, Setup.state, Setup.step, genSetup, grad_linear]

theorem genX_succ (n : ℕ) :
    genX C β₁ β₂ α (n + 1) = max (-1) (min 1 (genX C β₁ β₂ α n -
      α / Real.sqrt (n + 1 : ℕ) * (genM C β₁ β₂ α (n + 1) / Real.sqrt (genV C β₁ β₂ α (n + 1))))) := by
  simp [genX, genM, genV, Setup.state, Setup.step, genSetup, grad_linear, boxProj, adamRule, vsqrt]

theorem genX_zero : genX C β₁ β₂ α 0 = 1 := by simp [genX, Setup.state, genSetup]

theorem genM_zero : genM C β₁ β₂ α 0 = 0 := by simp [genM, Setup.state]

theorem genV_zero : genV C β₁ β₂ α 0 = 0 := by simp [genV, Setup.state]

theorem genSlope_mem {C : ℕ} (hC : 1 ≤ C) (t : ℕ) :
    -1 ≤ genSlope C t ∧ genSlope C t ≤ C ∧ genSlope C t ^ 2 ≤ (C : ℝ) ^ 2 := by
  have := abs_genSlope_le hC t
  have h1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
  refine ⟨?_, (le_abs_self _).trans this, ?_⟩
  · unfold genSlope; split_ifs <;> linarith
  · rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) this 2

/-- The warm-up: for `n ≤ C` nothing has moved, `m_n = v_n = 0` and `x_{n+1} = 1`. -/
theorem gen_warmup {n : ℕ} (hn : n ≤ C) :
    genM C β₁ β₂ α n = 0 ∧ genV C β₁ β₂ α n = 0 ∧ genX C β₁ β₂ α n = 1 := by
  induction n with
  | zero => exact ⟨genM_zero .., genV_zero .., genX_zero ..⟩
  | succ n ih =>
    obtain ⟨hm, hv, hx⟩ := ih (by omega)
    have hs : genSlope C (n + 1) = 0 := by simp [genSlope, hn]
    have hm' : genM C β₁ β₂ α (n + 1) = 0 := by rw [genM_succ, hm, hs]; ring
    refine ⟨hm', by rw [genV_succ, hv, hs]; ring, ?_⟩
    rw [genX_succ, hx, hm']
    norm_num

/-- The iterate stays in `[-1, 1]`. -/
theorem genX_le_one (n : ℕ) : genX C β₁ β₂ α n ≤ 1 := by
  cases n with
  | zero => rw [genX_zero]
  | succ n => rw [genX_succ]; exact max_le (by norm_num) (min_le_left _ _)

theorem neg_one_le_genX (n : ℕ) : -1 ≤ genX C β₁ β₂ α n := by
  cases n with
  | zero => rw [genX_zero]; norm_num
  | succ n => rw [genX_succ]; exact le_max_left _ _

end Run

/-- Along the whole run, `-1 ≤ m_n ≤ C` and `0 ≤ v_n ≤ C²`. -/
theorem gen_bounds {C : ℕ} (hC : 1 ≤ C) {β₁ β₂ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ ≤ 1)
    (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (α : ℝ) (n : ℕ) :
    -1 ≤ genM C β₁ β₂ α n ∧ genM C β₁ β₂ α n ≤ C ∧
      0 ≤ genV C β₁ β₂ α n ∧ genV C β₁ β₂ α n ≤ (C : ℝ) ^ 2 := by
  have h1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
  induction n with
  | zero => simp [genM_zero, genV_zero]
  | succ n ih =>
    obtain ⟨a, b, c, e⟩ := ih
    obtain ⟨s1, s2, s3⟩ := genSlope_mem hC (n + 1)
    rw [genM_succ, genV_succ]
    refine ⟨by nlinarith, by nlinarith, by positivity, by nlinarith⟩

/-- The hypotheses are satisfiable: `C = 2`, `β₁ = 0`, `β₂ = 1/2`. -/
example : (1 : ℕ) ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨by norm_num, le_rfl, by norm_num, by norm_num, by norm_num⟩

end AdamBeyond
end Transformer
