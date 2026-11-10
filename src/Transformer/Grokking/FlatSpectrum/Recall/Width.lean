module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Recall.Basic

/-!
# Recall takes one unit of cut width for each key

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR4.
Let `F q π = g q + ∑_{r < d} a_r(q) b_r(π)` have cut width `d`, `m = |κ|` keys, `c = |ν|` values,
and `μ` the mean of the readout `φ`. Then

    (m − d) c^{m−1} ∑_v (φ(v) − μ)² ≤ ∑_{q, π} (φ(π q) − F q π)²     (`HasCutWidth.mul_le_sum_sq`):

the mean squared error is at least `(1 − d / m)` of the variance of `φ`.

Proof. Subtract its mean over `π` from each `b_r`: `F q π = γ q + X q π` with
`X q π = ∑_r a_r(q) (b_r(π) − β_r)`. For each `q` the numbers `φ(π q) − μ − X q π` sum to `0`
over `π`, so removing the constant `γ q − μ` from them adds to their sum of squares
(`sum_sq_le_sum_sub_sq`). What remains is `‖K − A B‖_S2²` for the recall kernel `K` of `φ − μ`,
flat with squared singular value `c^{m−1} ∑_v (φ(v) − μ)²`, and a product `A B` through width
`d`: at least `(m − d)` times that value (`IsFlat.mul_sub_card_le_frobeniusSq_sub`).

So a readout that recalls a non-constant `φ` exactly has `d ≥ m`
(`HasCutWidth.card_le_of_recall`). One-hot keys reach `d = m` (`hasCutWidth_recall`): the cut
width of recall is the number of keys (`hasCutWidth_recall_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- Removing a constant from numbers of sum zero adds to their sum of squares. -/
lemma sum_sq_le_sum_sub_sq {ι : Type*} [Fintype ι] {y : ι → ℝ} (hy : ∑ i, y i = 0) (δ : ℝ) :
    ∑ i, y i ^ 2 ≤ ∑ i, (y i - δ) ^ 2 :=
  calc ∑ i, y i ^ 2 ≤ ∑ i, ((y i - δ) ^ 2 + 2 * δ * y i) :=
        Finset.sum_le_sum fun i _ => by nlinarith [sq_nonneg δ]
    _ = ∑ i, (y i - δ) ^ 2 := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, hy, mul_zero, add_zero]

example := sum_sq_le_sum_sub_sq (y := ![1, -1]) (by simp [Fin.sum_univ_two]) 3

/-- Some constants `β_r` center the context features: `∑_x (b_r(x) − β_r) = 0`. -/
lemma exists_centering {C : Type*} [Fintype C] {d : ℕ} (b : C → Fin d → ℝ) :
    ∃ β : Fin d → ℝ, ∀ r, ∑ x, (b x r - β r) = 0 := by
  refine ⟨fun r => (∑ x, b x r) / Fintype.card C, fun r => ?_⟩
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rcases Nat.eq_zero_or_pos (Fintype.card C) with h0 | h0
  · have := Fintype.card_eq_zero_iff.1 h0
    simp
  · rw [mul_div_cancel₀ _ (Nat.cast_ne_zero.2 h0.ne'), sub_self]

/-- **The error of a readout of cut width `d` on recall** (`review/flat_spectrum.md`, AR4): with
`μ` the mean of `φ`, the squared error over all queries and dictionaries is at least
`(m − d) c^{m−1} ∑_v (φ(v) − μ)²`. -/
theorem HasCutWidth.mul_le_sum_sq {d : ℕ} {F : κ → (κ → ν) → ℝ} (hF : HasCutWidth d F)
    {φ : ν → ℝ} {μ : ℝ} (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - d) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν, (φ (π q) - F q π) ^ 2 := by
  classical
  obtain ⟨g, a, b, hF⟩ := hF
  obtain ⟨β, hβ⟩ := exists_centering b
  have hrow (q : κ) : ∑ π : κ → ν, (φ (π q) - μ - ∑ r, a q r * (b π r - β r)) ^ 2 ≤
      ∑ π : κ → ν, (φ (π q) - F q π) ^ 2 := by
    have hy : ∑ π : κ → ν, (φ (π q) - μ - ∑ r, a q r * (b π r - β r)) = 0 := by
      rw [Finset.sum_sub_distrib, sum_apply q fun v => φ v - μ, hμ, mul_zero, zero_sub,
        Finset.sum_comm, neg_eq_zero]
      exact Finset.sum_eq_zero fun r _ => by rw [← Finset.mul_sum, hβ r, mul_zero]
    refine (sum_sq_le_sum_sub_sq hy (g q + ∑ r, a q r * β r - μ)).trans (le_of_eq ?_)
    refine Finset.sum_congr rfl fun π _ => ?_
    rw [hF q π]
    congr 1
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  have hgram := recallKernel_mul_conjTranspose (κ := κ) (φ := fun v => ((φ v - μ : ℝ) : ℂ))
    (by rw [← Complex.ofReal_sum, hμ, Complex.ofReal_zero])
  simp only [Complex.norm_real, Real.norm_eq_abs, sq_abs] at hgram
  have hc0 : 0 ≤ (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, (φ v - μ) ^ 2 :=
    mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _) (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  rcases hc0.eq_or_lt with hc | hc
  · rw [← hc, mul_zero]
    exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hle : Fintype.card κ ≤ Fintype.card (κ → ν) := by
    have h1 := rank_le_card_width (recallKernel (κ := κ) fun v => ((φ v - μ : ℝ) : ℂ))
    rwa [rank_eq_card_of_mul_conjTranspose_self hc.ne' hgram] at h1
  have hcap := IsFlat.mul_sub_card_le_frobeniusSq_sub hc (Or.inr hgram)
    (Matrix.of fun q r => (a q r : ℂ)) (Matrix.of fun r (π : κ → ν) => ((b π r - β r : ℝ) : ℂ))
  rw [min_eq_left hle, Fintype.card_fin] at hcap
  refine (le_of_eq (mul_comm _ _)).trans (hcap.trans
    (le_trans (le_of_eq ?_) (Finset.sum_le_sum fun q _ => hrow q)))
  simp only [frobeniusSq, Matrix.sub_apply, Matrix.mul_apply, recallKernel, Matrix.of_apply]
  refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun π _ => ?_
  rw [show ((φ (π q) - μ : ℝ) : ℂ) - ∑ r, (a q r : ℂ) * ((b π r - β r : ℝ) : ℂ) =
      ((φ (π q) - μ - ∑ r, a q r * (b π r - β r) : ℝ) : ℂ) by push_cast; rfl,
    Complex.norm_real, Real.norm_eq_abs, sq_abs]

omit [Fintype ν] in
/-- **Recall has cut width `d` for every `d` at least the number of keys**
(`review/flat_spectrum.md`, AR4): one-hot keys, `a_r(q) = [r = ι q]` and `b_r(π) = φ(π k)` for
`r = ι k`. -/
theorem hasCutWidth_recall {d : ℕ} (hd : Fintype.card κ ≤ d) (φ : ν → ℝ) :
    HasCutWidth d fun (q : κ) (π : κ → ν) => φ (π q) := by
  obtain ⟨ι⟩ : Nonempty (κ ↪ Fin d) :=
    Function.Embedding.nonempty_of_card_le (by simpa using hd)
  refine ⟨fun _ => 0, fun q r => if r = ι q then 1 else 0,
    fun π r => ∑ k, if ι k = r then φ (π k) else 0, fun q π => ?_⟩
  simp [ι.injective.eq_iff]

/-- **Exact recall takes cut width at least the number of keys** (`review/flat_spectrum.md`,
AR4): a readout of cut width `d` that returns a non-constant `φ` of the bound value for every
query and dictionary has `d ≥ m`. -/
theorem HasCutWidth.card_le_of_recall {d : ℕ} {F : κ → (κ → ν) → ℝ} (hF : HasCutWidth d F)
    {φ : ν → ℝ} {v v' : ν} (hφ : φ v ≠ φ v') (h : ∀ q π, F q π = φ (π q)) :
    Fintype.card κ ≤ d := by
  have hν : (Fintype.card ν : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fintype.card_pos_iff.2 ⟨v⟩).ne'
  have hμ : ∑ w, (φ w - (∑ w, φ w) / Fintype.card ν) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_div_cancel₀ _ hν, sub_self]
  obtain ⟨w, hw⟩ : ∃ w, φ w ≠ (∑ w, φ w) / Fintype.card ν := by
    by_contra hall
    push Not at hall
    exact hφ ((hall v).trans (hall v').symm)
  have hpos : 0 < (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ w, (φ w - (∑ w, φ w) / Fintype.card ν) ^ 2 :=
    mul_pos (pow_pos (Nat.cast_pos.2 (Fintype.card_pos_iff.2 ⟨v⟩)) _)
      (Finset.sum_pos' (fun _ _ => sq_nonneg _)
        ⟨w, Finset.mem_univ w, sq_pos_iff.2 (sub_ne_zero.2 hw)⟩)
  have hb := hF.mul_le_sum_sq hμ
  rw [show ∑ q, ∑ π : κ → ν, (φ (π q) - F q π) ^ 2 = 0 by simp [h]] at hb
  have hsub : (Fintype.card κ : ℝ) - d ≤ 0 := by
    by_contra hlt
    exact absurd hb (not_le.2 (mul_pos (not_le.1 hlt) hpos))
  exact_mod_cast sub_nonpos.1 hsub

/-- **The cut width of recall is the number of keys** (`review/flat_spectrum.md`, AR4). -/
theorem hasCutWidth_recall_iff {d : ℕ} {φ : ν → ℝ} {v v' : ν} (hφ : φ v ≠ φ v') :
    HasCutWidth d (fun (q : κ) (π : κ → ν) => φ (π q)) ↔ Fintype.card κ ≤ d :=
  ⟨fun h => h.card_le_of_recall hφ fun _ _ => rfl, fun h => hasCutWidth_recall h φ⟩

/-- Three keys with values `±1`: recall has cut width `3` and no smaller one. -/
example (d : ℕ) : HasCutWidth d (fun (q : Fin 3) (π : Fin 3 → Fin 2) => (![1, -1] : Fin 2 → ℝ)
    (π q)) ↔ 3 ≤ d := by
  simpa using hasCutWidth_recall_iff (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℝ)) (v := 0)
    (v' := 1) (by norm_num)

example := (hasCutWidth_recall (κ := Fin 3) (d := 3) (by simp)
  (![1, -1] : Fin 2 → ℝ)).mul_le_sum_sq (φ := ![1, -1]) (μ := 0) (by simp [Fin.sum_univ_two])

example := (hasCutWidth_recall (κ := Fin 3) (d := 3) (by simp)
  (![1, -1] : Fin 2 → ℝ)).card_le_of_recall (v := 0) (v' := 1) (by norm_num) fun _ _ => rfl

end Transformer.Grokking.FlatSpectrum.Recall
