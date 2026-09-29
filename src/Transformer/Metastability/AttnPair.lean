/-
# Metastability — the velocity of a token with arbitrary attention weights

The estimates of §2 of arXiv:2410.06833v1 behind `eq: ze.equation`, stated about one
configuration `x` of unit vectors and one row `a = (a_k)` of nonnegative weights, with no
dynamics and no particular model.  `Metastability.PairVelocity` proves the same estimates
for the `SA` weights `e^{β⟨x_i,x_k⟩}/Z_i`; here the weights are arbitrary, subject only to
the two properties the proof uses: `a_k ≥ 0`, and `a_k ≤ E` for every `k` outside the
index set `I` of the cap.  That is what lets `SA` and `USA` share one proof
(`Metastability.AttnFlow`).

* `inner_attnVel` — `⟨Proj_{x_i} Σ_k a_k x_k, y⟩ = Σ_k a_k (⟨x_k,y⟩ - ⟨x_i,x_k⟩⟨x_i,y⟩)`;
* `bracket_ge_neg_one` — each bracket is at least `-1` for unit `x_i, x_k, y`;
* `cap_velocity_ge` — a token that is the minimum of `⟨·, w⟩` over `I` loses at most
  `(n-1) E` of `⟨x_i, w⟩` per unit time;
* `pair_velocity_ge` — the `k = j` term carries the gain `a_j (1 - ρ²)`, the rest of `I`
  contributes nonnegatively, and the tokens outside `I` cost at most `(n-1) E`.

The leakage count is `(n-1)`, not the paper's `n`: the index `i` itself is in `I`.
-/

import Transformer.Basic

open scoped BigOperators InnerProductSpace
open Real

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- **The velocity `Proj_{x_i} (Σ_k a_k x_k)`, resolved against a vector `y`:**

  `⟨Proj_{x_i} Σ_k a_k x_k, y⟩ = Σ_k a_k (⟨x_k, y⟩ - ⟨x_i, x_k⟩ ⟨x_i, y⟩)`. -/
theorem inner_attnVel (x : Idx n → EucSpace d) (a : Idx n → ℝ) (i : Idx n)
    (y : EucSpace d) :
    ⟪proj d (x i) (∑ k, a k • x k), y⟫_ℝ
      = ∑ k, a k * (⟪x k, y⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, y⟫_ℝ) := by
  have h1 : ∑ k, a k * (⟪x k, y⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, y⟫_ℝ)
      = (∑ k, a k * ⟪x k, y⟫_ℝ) - (∑ k, a k * ⟪x i, x k⟫_ℝ) * ⟪x i, y⟫_ℝ := by
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [h1]
  simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_smul_right, sum_inner,
    inner_sum]

/-- **A bracket is at least `-1`.**  For unit vectors `x_i, x_k, y`,
`⟨x_k, y⟩ - ⟨x_i, x_k⟩⟨x_i, y⟩ = ⟨x_k, y - ⟨x_i, y⟩ x_i⟩ ≥ -‖y - ⟨x_i, y⟩ x_i‖ ≥ -1`. -/
theorem bracket_ge_neg_one {xi xk y : EucSpace d} (hi : ‖xi‖ = 1) (hk : ‖xk‖ = 1)
    (hy : ‖y‖ = 1) : -1 ≤ ⟪xk, y⟫_ℝ - ⟪xi, xk⟫_ℝ * ⟪xi, y⟫_ℝ := by
  have hsq : ‖y - ⟪xi, y⟫_ℝ • xi‖ ^ 2 = 1 - ⟪xi, y⟫_ℝ ^ 2 := by
    rw [norm_sub_sq_real, norm_smul, real_inner_smul_right, hi, hy, Real.norm_eq_abs, mul_pow,
      sq_abs, real_inner_comm xi y]
    ring
  have hle : ‖y - ⟪xi, y⟫_ℝ • xi‖ ≤ 1 := by
    nlinarith [norm_nonneg (y - ⟪xi, y⟫_ℝ • xi), sq_nonneg ⟪xi, y⟫_ℝ]
  have h := abs_real_inner_le_norm xk (y - ⟪xi, y⟫_ℝ • xi)
  rw [hk, one_mul, inner_sub_right, real_inner_smul_right, real_inner_comm xi xk] at h
  linarith [(abs_le.mp (h.trans hle)).1]

/-- The number of indices outside a set containing `i` is at most `n - 1`. -/
theorem card_compl_le (I : Finset (Idx n)) (i : Idx n) (hi : i ∈ I) :
    ((Iᶜ).card : ℝ) ≤ (n : ℝ) - 1 := by
  have h1 : 1 ≤ I.card := Finset.card_pos.2 ⟨i, hi⟩
  have h2 : I.card + Iᶜ.card = n := by simp [Finset.card_add_card_compl]
  have h3 : (1 : ℝ) ≤ I.card := by exact_mod_cast h1
  have h4 : (I.card : ℝ) + Iᶜ.card = n := by exact_mod_cast h2
  linarith

/-- A sum over `Idx n` is at least its part over `I`, minus `E` for each index outside `I`,
when the terms inside `I` are nonnegative and those outside are at least `-E`. -/
theorem sum_ge_far (f : Idx n → ℝ) (I : Finset (Idx n)) (E : ℝ)
    (hfar : ∀ k, k ∉ I → -E ≤ f k) :
    ∑ k ∈ I, f k - E * ((Iᶜ).card : ℝ) ≤ ∑ k, f k := by
  have h := Finset.sum_add_sum_compl I f
  have h2 : ∑ _k ∈ Iᶜ, (-E) ≤ ∑ k ∈ Iᶜ, f k :=
    Finset.sum_le_sum fun k hk => hfar k (Finset.mem_compl.1 hk)
  rw [Finset.sum_const, nsmul_eq_mul] at h2
  linarith

/-- **A token loses at most `(n-1) E` of `⟨x_i, w⟩` per unit time.**

If `x_i` attains the minimum of `⟨x_k, w⟩` over `k ∈ I`, that minimum is nonnegative,
and every token outside `I` has weight at most `E` in the row `a` of `i`, then

  `⟨Proj_{x_i} Σ_k a_k x_k, w⟩ ≥ -(n-1) E`.

A `k ∈ I` contributes `a_k (⟨x_k,w⟩ - ⟨x_i,x_k⟩η) ≥ a_k η (1 - ⟨x_i,x_k⟩) ≥ 0`, with
`η = ⟨x_i,w⟩`; a `k ∉ I` contributes at least `-a_k ≥ -E`, by `bracket_ge_neg_one`.

Source: arXiv:2410.06833v1, §2, Step 1 of the direct proof (cap containment). -/
theorem cap_velocity_ge (x : Idx n → EucSpace d) (a : Idx n → ℝ) (I : Finset (Idx n))
    (i : Idx n) (hi : i ∈ I) (w : EucSpace d) (hw : ‖w‖ = 1) (hx : ∀ k, ‖x k‖ = 1)
    (E : ℝ) (hE : 0 ≤ E) (ha0 : ∀ k, 0 ≤ a k) (hfar : ∀ k, k ∉ I → a k ≤ E)
    (hη : 0 ≤ ⟪x i, w⟫_ℝ) (hmin : ∀ k ∈ I, ⟪x i, w⟫_ℝ ≤ ⟪x k, w⟫_ℝ) :
    -(((n : ℝ) - 1) * E) ≤ ⟪proj d (x i) (∑ k, a k • x k), w⟫_ℝ := by
  rw [inner_attnVel]
  have hle1 : ∀ k, ⟪x i, x k⟫_ℝ ≤ 1 := fun k => by
    have h := abs_real_inner_le_norm (x i) (x k)
    rw [hx i, hx k, one_mul] at h
    exact (abs_le.mp h).2
  have hfar' := sum_ge_far
    (fun k => a k * (⟪x k, w⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, w⟫_ℝ)) I E (fun k hk => by
      have hb := bracket_ge_neg_one (hx i) (hx k) hw
      nlinarith [mul_nonneg (ha0 k) (by linarith : 0 ≤ ⟪x k, w⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, w⟫_ℝ + 1),
        hfar k hk])
  have hI : 0 ≤ ∑ k ∈ I, a k * (⟪x k, w⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, w⟫_ℝ) :=
    Finset.sum_nonneg fun k hk => mul_nonneg (ha0 k) (by
      nlinarith [hmin k hk, hle1 k, mul_nonneg hη (sub_nonneg.2 (hle1 k))])
  have hcard := card_compl_le I i hi
  nlinarith [mul_le_mul_of_nonneg_left hcard hE]

/-- **The pair estimate for arbitrary weights.**

Let `ρ = ⟨x_i, x_j⟩` be the minimum of `⟨x_k, x_l⟩` over `I × I`, and nonnegative.  If the
weight `a_j` of `j` in the row of `i` is at least `m`, and every token outside `I` has
weight at most `E`, then

  `⟨Proj_{x_i} Σ_k a_k x_k, x_j⟩ ≥ m (1 - ρ²) - (n-1) E`.

The `k = j` term contributes `a_j (1 - ρ²)`; every other `k ∈ I` contributes
`a_k (⟨x_k, x_j⟩ - ⟨x_i, x_k⟩ρ) ≥ a_k ρ (1 - ⟨x_i, x_k⟩) ≥ 0`, which is where `ρ ≥ 0` is
used; a `k ∉ I` contributes at least `-E`.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation`. -/
theorem pair_velocity_ge (x : Idx n → EucSpace d) (a : Idx n → ℝ) (I : Finset (Idx n))
    (i j : Idx n) (hi : i ∈ I) (hj : j ∈ I) (hx : ∀ k, ‖x k‖ = 1) (E m ρ : ℝ)
    (hE : 0 ≤ E) (ha0 : ∀ k, 0 ≤ a k) (hfar : ∀ k, k ∉ I → a k ≤ E)
    (haj : m ≤ a j) (hρ0 : 0 ≤ ρ) (hρij : ρ = ⟪x i, x j⟫_ℝ)
    (hmin : ∀ k ∈ I, ∀ l ∈ I, ρ ≤ ⟪x k, x l⟫_ℝ) :
    m * (1 - ρ ^ 2) - ((n : ℝ) - 1) * E ≤ ⟪proj d (x i) (∑ k, a k • x k), x j⟫_ℝ := by
  rw [inner_attnVel]
  have hle1 : ∀ k, ⟪x i, x k⟫_ℝ ≤ 1 := fun k => by
    have h := abs_real_inner_le_norm (x i) (x k)
    rw [hx i, hx k, one_mul] at h
    exact (abs_le.mp h).2
  have hρ1 : ρ ≤ 1 := hρij ▸ hle1 j
  have hjj : ⟪x j, x j⟫_ℝ - ⟪x i, x j⟫_ℝ * ⟪x i, x j⟫_ℝ = 1 - ρ ^ 2 := by
    rw [real_inner_self_eq_norm_mul_norm, hx j, ← hρij]; ring
  have hfar' := sum_ge_far
    (fun k => a k * (⟪x k, x j⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, x j⟫_ℝ)) I E (fun k hk => by
      have hb := bracket_ge_neg_one (hx i) (hx k) (hx j)
      nlinarith [mul_nonneg (ha0 k)
        (by linarith : 0 ≤ ⟪x k, x j⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, x j⟫_ℝ + 1), hfar k hk])
  have hterm : ∀ k ∈ I, 0 ≤ a k * (⟪x k, x j⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, x j⟫_ℝ) := fun k hk =>
    mul_nonneg (ha0 k) (by
      rw [← hρij]
      nlinarith [hmin k hk j hj, hle1 k, mul_nonneg hρ0 (sub_nonneg.2 (hle1 k))])
  have hsingle : a j * (1 - ρ ^ 2) ≤
      ∑ k ∈ I, a k * (⟪x k, x j⟫_ℝ - ⟪x i, x k⟫_ℝ * ⟪x i, x j⟫_ℝ) := by
    have h := Finset.single_le_sum hterm hj
    rwa [hjj] at h
  have hgain : m * (1 - ρ ^ 2) ≤ a j * (1 - ρ ^ 2) :=
    mul_le_mul_of_nonneg_right haj (by nlinarith)
  have hcard := card_compl_le I i hi
  nlinarith [mul_le_mul_of_nonneg_left hcard hE]

end Metastability
end Transformer
