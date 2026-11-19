module

public import Transformer.RoPE.BaseBounds.Section4_LongTermDecay
public import Transformer.RoPE.RoFormer.Section3_Rotary

/-!
# §3.3 Properties of RoPE: long-term decay and linear attention

Su et al., *RoFormer*, arXiv:2104.09864v5, §3.3 and §3.4.3, Eqs. (17)–(19) and (35)–(37).

* `score_eq_re_sum`: Eq. (35), `(R_m q)ᵀ(R_n k) = Re[∑_i h_i e^{i(m−n)θ_i}]` with the complex
  products `h_i = q_[2i:2i+1] k*_[2i:2i+1]`;
* `abs_score_le`: Eqs. (36)–(37). Abel's summation with `h_{d/2} = 0` and `S_0 = 0` gives
  `|(R_m q)ᵀ(R_n k)| ≤ M ∑_{j=1}^{d/2} |S_j|` whenever `|h_{i+1} − h_i| ≤ M`, the source's
  `max_i |h_{i+1} − h_i|` included. Men et al. restate this bound as their Eq. (7); the summation
  by parts is `BaseBounds.abs_score_le`. That `(1/(d/2)) ∑ |S_i|` decays "as shown in Figure 2"
  is read off a plot and is not formalized;
* `rotary_dotProduct`, `linear_numerator_eq`: Eq. (19). Each weight of RoPE's linear attention is
  `φ(q_m)ᵀ R_{n−m} φ(k_n)`, and the numerator is `(R_m φ(q_m))ᵀ ∑_n R_n φ(k_n) v_nᵀ`, so the
  key–value summary is computed once, as in linear attention;
* `exists_linear_term_neg`: "the summation in the numerator could contain negative terms". With
  nonnegative features, RoPE's first frequency `θ_0 = 1` and `m − n = 2`, the numerator's weight
  is `cos 2 < 0` while the denominator's `φ(q_m)ᵀ φ(k_n)` is `1`.
-/

@[expose] public section

open scoped BigOperators
open Matrix

namespace Transformer.RoPE.RoFormer

variable {p : ℕ}

/-- The `i`-th pair `(x_{2i+1}, x_{2i+2})` of `x`, and `0` from `i = d/2` on. -/
def pairOf (x : Fin 2 × Fin p → ℝ) (i : ℕ) : ℝ × ℝ :=
  if h : i < p then (x (0, ⟨i, h⟩), x (1, ⟨i, h⟩)) else 0

/-- The frequencies `θ_i`, and `0` from `i = d/2` on. -/
def freqOf (θ : Fin p → ℝ) (i : ℕ) : ℝ :=
  if h : i < p then θ ⟨i, h⟩ else 0

/-- The RoPE score is the sum of the pair scores at the relative position `m − n`. -/
theorem score_eq_sum_pairScore (θ : Fin p → ℝ) (q k : Fin 2 × Fin p → ℝ) (m n : ℝ) :
    (rotary θ m *ᵥ q) ⬝ᵥ (rotary θ n *ᵥ k) =
      ∑ i ∈ Finset.range p, pairScore (pairOf q i) (pairOf k i) (freqOf θ i) (m - n) := by
  rw [rotary_mulVec, rotary_mulVec,
    ← Fin.sum_univ_eq_sum_range (fun i => pairScore (pairOf q i) (pairOf k i) (freqOf θ i) (m - n))]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [pairOf, freqOf, dite_eq_left i.isLt, Fin.eta, pairScore, rotateHalf]
  rw [show θ i * (m - n) = m * θ i - n * θ i by ring, Real.cos_sub, Real.sin_sub]
  simp only [Fin.isValue, ite_true, Fin.one_eq_zero_iff, OfNat.ofNat_ne_one, ite_false]
  ring

/-- Eq. (35): `(R_m q)ᵀ(R_n k) = Re[∑_i h_i e^{i(m−n)θ_i}]`, `h_i = q_[2i:2i+1] k*_[2i:2i+1]`. -/
theorem score_eq_re_sum (θ : Fin p → ℝ) (q k : Fin 2 × Fin p → ℝ) (m n : ℝ) :
    (rotary θ m *ᵥ q) ⬝ᵥ (rotary θ n *ᵥ k) =
      (∑ i ∈ Finset.range p, BaseBounds.pairProduct p (pairOf q) (pairOf k) i *
        Complex.exp ((freqOf θ i * (m - n) : ℝ) * Complex.I)).re := by
  rw [score_eq_sum_pairScore, Complex.re_sum]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [BaseBounds.pairScore_eq_re, BaseBounds.pairProduct, ite_eq_left (Finset.mem_range.mp hi)]

/-- Eqs. (36)–(37): with `h_{d/2} = 0` and `|h_{i+1} − h_i| ≤ M` for `i < d/2`,
`|(R_m q)ᵀ(R_n k)| ≤ M ∑_{j=1}^{d/2} |S_j|`, `S_j = ∑_{i<j} e^{i(m−n)θ_i}`. -/
theorem abs_score_le (θ : Fin p → ℝ) (q k : Fin 2 × Fin p → ℝ) (m n : ℝ) {M : ℝ}
    (hM : ∀ i < p, ‖BaseBounds.pairProduct p (pairOf q) (pairOf k) i -
      BaseBounds.pairProduct p (pairOf q) (pairOf k) (i + 1)‖ ≤ M) :
    |(rotary θ m *ᵥ q) ⬝ᵥ (rotary θ n *ᵥ k)| ≤
      M * ∑ j ∈ Finset.range p, ‖BaseBounds.phaseSum (freqOf θ) (m - n) (j + 1)‖ := by
  rw [score_eq_sum_pairScore]
  exact BaseBounds.abs_score_le p (pairOf q) (pairOf k) (freqOf θ) (m - n) hM

example (θ : Fin 64 → ℝ) (q k : Fin 2 × Fin 64 → ℝ) (m n : ℝ) :
    |(rotary θ m *ᵥ q) ⬝ᵥ (rotary θ n *ᵥ k)| ≤
      (∑ i ∈ Finset.range 64, ‖BaseBounds.pairProduct 64 (pairOf q) (pairOf k) i -
        BaseBounds.pairProduct 64 (pairOf q) (pairOf k) (i + 1)‖) *
      ∑ j ∈ Finset.range 64, ‖BaseBounds.phaseSum (freqOf θ) (m - n) (j + 1)‖ :=
  abs_score_le θ q k m n fun _ hi =>
    Finset.single_le_sum (f := fun i => ‖BaseBounds.pairProduct 64 (pairOf q) (pairOf k) i -
      BaseBounds.pairProduct 64 (pairOf q) (pairOf k) (i + 1)‖)
      (fun _ _ => norm_nonneg _) (Finset.mem_range.mpr hi)

/-- Eq. (19): each weight `(R_m φ(q_m))ᵀ(R_n φ(k_n))` is `φ(q_m)ᵀ R_{n−m} φ(k_n)`. -/
theorem rotary_dotProduct (θ : Fin p → ℝ) (a b : Fin 2 × Fin p → ℝ) (m n : ℝ) :
    (rotary θ m *ᵥ a) ⬝ᵥ (rotary θ n *ᵥ b) = a ⬝ᵥ (rotary θ (n - m) *ᵥ b) := by
  rw [mulVec_dotProduct, mulVec_mulVec, rotary_transpose_mul]

/-- Eq. (19): the numerator `∑_n (R_m φ(q_m))ᵀ(R_n φ(k_n)) v_n` is
`(R_m φ(q_m))ᵀ ∑_n R_n φ(k_n) v_nᵀ`; the summary `∑_n R_n φ(k_n) v_nᵀ` does not depend on `m`. -/
theorem linear_numerator_eq {N e : ℕ} (θ : Fin p → ℝ) (a : Fin 2 × Fin p → ℝ)
    (b : Fin N → Fin 2 × Fin p → ℝ) (v : Fin N → Fin e → ℝ) (pos : Fin N → ℝ) (m : ℝ) :
    ∑ j, ((rotary θ m *ᵥ a) ⬝ᵥ (rotary θ (pos j) *ᵥ b j)) • v j =
      (rotary θ m *ᵥ a) ᵥ* ∑ j, vecMulVec (rotary θ (pos j) *ᵥ b j) (v j) := by
  rw [vecMul_sum]
  simp only [vecMul_vecMulVec]

/-- §3.3: "the summation in the numerator could contain negative terms". The features
`φ(q_m) = φ(k_n) = e_1` are nonnegative and `φ(q_m)ᵀφ(k_n) = 1`, but with RoPE's frequency
`θ_0 = 10000^0 = 1` of a head with `d = 2` and `m − n = 2` the numerator's weight is `cos 2 < 0`. -/
theorem exists_linear_term_neg :
    ∃ a b : Fin 2 × Fin 1 → ℝ, 0 ≤ a ∧ 0 ≤ b ∧ a ⬝ᵥ b = 1 ∧
      (rotary (fun i => frequency 10000 2 i) 2 *ᵥ a) ⬝ᵥ
        (rotary (fun i => frequency 10000 2 i) 0 *ᵥ b) < 0 := by
  refine ⟨fun ri => if ri = (0, 0) then 1 else 0, fun ri => if ri = (0, 0) then 1 else 0,
    fun ri => by dsimp only; split_ifs <;> norm_num,
    fun ri => by dsimp only; split_ifs <;> norm_num,
    by simp [dotProduct], ?_⟩
  rw [rotary_zero, one_mulVec, rotary_mulVec]
  have h0 : frequency 10000 2 0 = 1 := by simp [frequency]
  simp [dotProduct, rotateHalf]
  rw [h0, mul_one]
  exact Real.cos_neg_of_pi_div_two_lt_of_lt (by linarith [Real.pi_lt_d2])
    (by linarith [Real.pi_gt_three])

end Transformer.RoPE.RoFormer
