module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Transformer.CRASP.Grams

/-!
# Positional logits read the last tokens: rounding and sums

In no paper; the construction is `Transformer.CRASP.Grams`, for the rounded transformers of
arXiv:2506.16055v3, Appendix B.1 (`def:transformer`, Equation `eq:att`). At precision `p` the
integers from `-2^(p-1)` to `2^(p-1) - 1` are exact (`val_round_intCast`), so from `-4` to `3`
when `p ≥ 3` (`val_round_int`). Rounding is down, so `round(exp z) = 0` when `z < 0`
(`val_round_exp_neg`), and `round(exp z · x) = 0` when `z ≤ -1` and `0 ≤ x ≤ 2`, since `2 < e`
(`val_round_exp_mul`).

A reading layer gives the position `j` a logit `z j` for the query at `i`. When it is `0` at `i`
and, if the gram of `i - 1` differs from that of `i`, at `i - 1`, and at most `-1` at the other
positions up to `i`, a sum of rounded weights times values in `[0, 2]` keeps only those one or two
positions (`sum_read`). `Transformer.CRASP.GramsRead` computes the reading layers from these sums.
`ReadsPrev L b` collects what the construction asks of a logit `L` with the dot product `b`, and
ALiBi of a slope `a ≥ 1` meets it with `b = a` (`readsPrev_alibi`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u
variable {σ : Type u} {n p : ℕ}

/-- At precision `p` the integers from `-2^(p-1)` to `2^(p-1) - 1` are exact. -/
theorem val_round_intCast {x : ℝ} (z : ℤ) (hz : x = z) (h₁ : -2 ^ (p - 1) ≤ z)
    (h₂ : z < 2 ^ (p - 1)) : (Fx.round p 0 x).val = x := by
  subst hz
  rw [Fx.val, Fx.m_round, pow_zero, mul_one, Int.floor_intCast, Fx.clamp_eq_self h₁ h₂]
  simp

example : (Fx.round 2 0 (-2 : ℝ)).val = -2 :=
  val_round_intCast (-2) (by norm_num) (by norm_num) (by norm_num)

/-- At precision `p ≥ 3` the integers from `-4` to `3` are exact. -/
theorem val_round_int (hp : 3 ≤ p) {x : ℝ} (z : ℤ) (hz : x = z) (h₁ : -4 ≤ z) (h₂ : z < 4) :
    (Fx.round p 0 x).val = x := by
  have h4 : (4 : ℤ) ≤ 2 ^ (p - 1) := by
    calc (4 : ℤ) = 2 ^ 2 := by norm_num
      _ ≤ 2 ^ (p - 1) := pow_le_pow_right₀ (by norm_num) (by omega)
  exact val_round_intCast z hz (by omega) (by omega)

/-- The hypotheses of `val_round_int` hold for `3` at precision `3`. -/
example : (Fx.round 3 0 3).val = 3 :=
  val_round_int le_rfl 3 (by norm_num) (by norm_num) (by norm_num)

/-- A natural number below `2^(p-1)` is exact at precision `p`. -/
theorem val_round_natCast {b : ℕ} (hb : b < 2 ^ (p - 1)) : (Fx.round p 0 (b : ℝ)).val = b := by
  have hb' : (b : ℤ) < 2 ^ (p - 1) := by exact_mod_cast hb
  have h0 : (0 : ℤ) ≤ b := Int.natCast_nonneg b
  exact val_round_intCast (b : ℤ) (by simp) (by
    have : (0 : ℤ) < 2 ^ (p - 1) := by positivity
    omega) hb'

/-- The hypothesis of `val_round_natCast` holds for `5` at precision `4`. -/
example : (Fx.round 4 0 (5 : ℕ)).val = (5 : ℕ) := val_round_natCast (by norm_num)

/-- `round(exp z) = 0` when `z < 0`. -/
theorem val_round_exp_neg {z : ℝ} (hz : z < 0) : (Fx.round p 0 (Real.exp z)).val = 0 := by
  have hf : ⌊Real.exp z * 2 ^ 0⌋ = 0 := by
    rw [pow_zero, mul_one, Int.floor_eq_zero_iff]
    exact ⟨(Real.exp_pos z).le, Real.exp_lt_one_iff.mpr hz⟩
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  rw [Fx.val, Fx.m_round, hf, Fx.clamp_eq_self (by omega) hp]
  simp

/-- The hypothesis of `val_round_exp_neg` holds for `z = -1`. -/
example : (Fx.round 3 0 (Real.exp (-1))).val = 0 := val_round_exp_neg (by norm_num)

/-- `round(exp z · x) = 0` when `z ≤ -1` and `0 ≤ x ≤ 2`, since `2 e⁻¹ < 1`. -/
theorem val_round_exp_mul {z x : ℝ} (hz : z ≤ -1) (h₀ : 0 ≤ x) (h₂ : x ≤ 2) :
    (Fx.round p 0 (Real.exp z * x)).val = 0 := by
  have he : Real.exp z ≤ Real.exp (-1) := Real.exp_le_exp.mpr hz
  have h1 : Real.exp (-1) * 2 < 1 := by
    have h := Real.exp_one_gt_d9
    rw [Real.exp_neg, inv_mul_lt_iff₀ (Real.exp_pos 1)]
    linarith
  have hx0 : 0 ≤ Real.exp z * x := mul_nonneg (Real.exp_pos z).le h₀
  have hx1 : Real.exp z * x < 1 :=
    lt_of_le_of_lt (mul_le_mul he h₂ h₀ (Real.exp_pos _).le) h1
  have hf : ⌊Real.exp z * x * 2 ^ 0⌋ = 0 := by
    rw [pow_zero, mul_one, Int.floor_eq_zero_iff]
    exact ⟨hx0, hx1⟩
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  rw [Fx.val, Fx.m_round, hf, Fx.clamp_eq_self (by omega) hp]
  simp

/-- The hypotheses of `val_round_exp_mul` hold for `z = -1` and `x = 2`. -/
example : (Fx.round 3 0 (Real.exp (-1) * 2)).val = 0 :=
  val_round_exp_mul le_rfl (by norm_num) le_rfl

/-- **What the construction asks of the logit `L`.** A reading layer, whose dot product is `b`
between differing grams and `0` between equal ones, sends the query position and, if its gram
differs, the previous one to `0`, and the other positions to at most `-1`. A running layer, whose
dot product is `-1`, sends the keys of each query all below `0` or all to `0`. -/
structure ReadsPrev (L : ℕ → ℕ → ℝ → ℝ) (b : ℕ) : Prop where
  self : ∀ i, L i i 0 = 0
  prev : ∀ i, 0 < i → L i (i - 1) b = 0
  prevSame : ∀ i, 0 < i → L i (i - 1) 0 ≤ -1
  far : ∀ i j, j + 2 ≤ i → L i j 0 ≤ -1 ∧ L i j b ≤ -1
  run : ∀ i, (∀ j ≤ i, L i j (-1) < 0) ∨ ∀ j ≤ i, L i j (-1) = 0

/-- **ALiBi**, `eq:alibi`, of a slope `a ≥ 1` meets `ReadsPrev` with the dot product `a`. -/
theorem readsPrev_alibi {a : ℕ} (ha : 1 ≤ a) : ReadsPrev (alibiLogit a) a where
  self i := by simp [alibiLogit]
  prev i hi := by
    simp only [alibiLogit]
    rw [Nat.cast_sub (by omega : 1 ≤ i)]
    ring
  prevSame i hi := by
    have h1 : (1 : ℝ) ≤ a := by exact_mod_cast ha
    simp only [alibiLogit]
    rw [Nat.cast_sub (by omega : 1 ≤ i), Nat.cast_one, sub_sub_cancel, mul_one]
    linarith
  far i j hij := by
    have h1 : (1 : ℝ) ≤ a := by exact_mod_cast ha
    have h2 : (2 : ℝ) ≤ (i : ℝ) - j := by
      have : (j : ℝ) + 2 ≤ i := by exact_mod_cast hij
      linarith
    have h3 : (a : ℝ) * 2 ≤ a * ((i : ℝ) - j) := mul_le_mul_of_nonneg_left h2 (by linarith)
    simp only [alibiLogit]
    constructor <;> linarith
  run i := Or.inl fun j hj => by
    have h0 : (0 : ℝ) ≤ (i : ℝ) - j := by
      have : (j : ℝ) ≤ i := by exact_mod_cast hj
      linarith
    have h1 : (0 : ℝ) ≤ a * ((i : ℝ) - j) := mul_nonneg (Nat.cast_nonneg a) h0
    simp only [alibiLogit]
    linarith

/-- The hypothesis of `readsPrev_alibi` holds for the slope `1`. -/
example : ReadsPrev (alibiLogit (1 : ℕ)) 1 := readsPrev_alibi le_rfl

variable [DecidableEq σ]

/-- **A reading layer keeps one or two positions**: the query position `i` and, if its gram
differs, `i - 1`, where the logit is `0`. Every other position up to `i` has a logit of at most
`-1` and contributes `0`. -/
theorem sum_read (c : ℕ → Gram σ n) {N : ℕ} (i : Fin N) (z : ℕ → ℝ) (g : Fin N → ℝ)
    (hg : ∀ j, 0 ≤ g j ∧ g j ≤ 2 ∧ (Fx.round p 0 (g j)).val = g j) (hself : z i.val = 0)
    (hprev : 0 < i.val → c (i.val - 1) ≠ c i.val → z (i.val - 1) = 0)
    (hsame : 0 < i.val → c (i.val - 1) = c i.val → z (i.val - 1) ≤ -1)
    (hfar : ∀ j, j + 2 ≤ i.val → z j ≤ -1) :
    ∑ j ∈ RTfr.masked i, (Fx.round p 0 (Real.exp (z j.val) * g j)).val =
      g i + if h : 0 < i.val ∧ c (i.val - 1) ≠ c i.val then
        g ⟨i.val - 1, by have := i.isLt; omega⟩ else 0 := by
  have hs : (Fx.round p 0 (Real.exp (z i.val) * g i)).val = g i := by
    rw [hself, Real.exp_zero, one_mul, (hg i).2.2]
  have hzero : ∀ j : Fin N, z j.val ≤ -1 →
      (Fx.round p 0 (Real.exp (z j.val) * g j)).val = 0 := fun j hj =>
    val_round_exp_mul hj (hg j).1 (hg j).2.1
  by_cases hc : 0 < i.val ∧ c (i.val - 1) ≠ c i.val
  · rw [dite_eq_left hc]
    let i' : Fin N := ⟨i.val - 1, by have := i.isLt; omega⟩
    have hp : (Fx.round p 0 (Real.exp (z i'.val) * g i')).val = g i' := by
      rw [show z i'.val = 0 from hprev hc.1 hc.2, Real.exp_zero, one_mul, (hg i').2.2]
    rw [Finset.sum_eq_add_of_mem i i' (RTfr.self_mem_masked i)
      (Finset.mem_filter.mpr ⟨Finset.mem_univ _, Fin.le_def.mpr (Nat.sub_le _ _)⟩) (fun h => by
        have := congrArg Fin.val h; simp [i'] at this; omega), hs, hp]
    intro j hj hji
    have hle : j ≤ i := by simpa [RTfr.masked] using hj
    refine hzero j (hfar j.val ?_)
    have h1 : j.val ≠ i.val := fun h => hji.1 (Fin.ext h)
    have h2 : j.val ≠ i.val - 1 := fun h => hji.2 (Fin.ext h)
    have := Fin.le_def.mp hle
    omega
  · rw [dite_eq_right hc, add_zero, Finset.sum_eq_single_of_mem i (RTfr.self_mem_masked i), hs]
    intro j hj hji
    have hle : j ≤ i := by simpa [RTfr.masked] using hj
    refine hzero j ?_
    have hlt : j.val < i.val := lt_of_le_of_ne (Fin.le_def.mp hle) fun h => hji (Fin.ext h)
    by_cases hj1 : j.val + 1 = i.val
    · rw [show j.val = i.val - 1 by omega]
      refine hsame (by omega) ?_
      by_contra hne
      exact hc ⟨by omega, hne⟩
    · exact hfar j.val (by omega)

/-- The hypotheses of `sum_read` hold for the weights `1` of a word of one position. -/
example := sum_read (p := 3) (fun _ => (fun _ => none : Gram Bool 0)) (0 : Fin 1) (fun _ => 0)
  (fun _ => 1) (fun _ => ⟨zero_le_one, by norm_num,
    val_round_int le_rfl 1 (by norm_num) (by norm_num) (by norm_num)⟩) rfl
  (fun h => absurd h (lt_irrefl 0)) (fun h => absurd h (lt_irrefl 0)) (fun j hj => by omega)

end Transformer.CRASP.Grams
