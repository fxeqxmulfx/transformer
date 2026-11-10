module

public import Transformer.Grokking.FlatSpectrum.Recall.Width

/-!
# A flat kernel inside a task prices the cut width

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), item BL4, the
mechanism of item AR4 of *Associative recall has a flat spectrum* for any task: Arora et al.,
arXiv:2312.04927v1, §3, state recall alone.

Let a readout `F q x = g q + ∑_{r < d} a_r(q) b_r(x)` have cut width `d`. Choose queries
`q_i`, `i ∈ m`, and contexts `x_j`, `j ∈ n`, and let `K` be a real flat kernel on `m × n` with
squared singular value `c > 0`. The table `F(q_i, x_j)` is a product through width `d + 1`, the
term `g` being one more column, so by the capacity bound of flat kernels

    c (min(|m|, |n|) − (d + 1)) ≤ ∑_{i, j} (K_{ij} − F(q_i, x_j))²
                                                    (`HasCutWidth.mul_le_sum_sq_of_isFlat`).

If the rows of `K` sum to zero, the term `g` buys nothing and `d` replaces `d + 1`
(`HasCutWidth.mul_le_sum_sq_of_isFlat_of_sum_eq_zero`): subtract from each `b_r` its mean over
the chosen contexts, as in AR4. A readout that equals `K` on the chosen pairs has
`min(|m|, |n|) ≤ d + 1`, and `≤ d` for rows of sum zero (`HasCutWidth.card_le_of_isFlat`,
`HasCutWidth.card_le_of_isFlat_of_sum_eq_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix

variable {Q C m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **A flat kernel inside a task prices the cut width** (`review/flat_spectrum.md`, BL4): on
queries `q_i` and contexts `x_j` that carry a flat kernel `K` with squared singular value `c`,
a readout of cut width `d` has squared error at least `c (min(|m|, |n|) − (d + 1))`. -/
theorem HasCutWidth.mul_le_sum_sq_of_isFlat {d : ℕ} {F : Q → C → ℝ} (hF : HasCutWidth d F)
    (q : m → Q) (x : n → C) {K : Matrix m n ℝ} {c : ℝ} (hc : 0 < c)
    (hK : IsFlat (K.map ((↑) : ℝ → ℂ)) c) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - (d + 1 : ℕ)) ≤
      ∑ i, ∑ j, (K i j - F (q i) (x j)) ^ 2 := by
  obtain ⟨g, a, b, hF⟩ := hF
  have hcap := hK.mul_sub_card_le_frobeniusSq_sub hc
    (Matrix.of fun i (r : Fin (d + 1)) =>
      ((Fin.snoc (α := fun _ => ℝ) (a (q i)) (g (q i)) r : ℝ) : ℂ))
    (Matrix.of fun (r : Fin (d + 1)) j => ((Fin.snoc (α := fun _ => ℝ) (b (x j)) 1 r : ℝ) : ℂ))
  rw [Fintype.card_fin] at hcap
  refine hcap.trans (le_of_eq ?_)
  simp only [frobeniusSq, Matrix.sub_apply, Matrix.mul_apply, Matrix.map_apply, Matrix.of_apply]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [show (K i j : ℂ) - ∑ r : Fin (d + 1),
      ((Fin.snoc (α := fun _ => ℝ) (a (q i)) (g (q i)) r : ℝ) : ℂ) *
        ((Fin.snoc (α := fun _ => ℝ) (b (x j)) 1 r : ℝ) : ℂ) =
      ((K i j - F (q i) (x j) : ℝ) : ℂ) by
    rw [hF, Fin.sum_univ_castSucc]
    simp only [Fin.snoc_castSucc, Fin.snoc_last]
    push_cast
    ring, Complex.norm_real, Real.norm_eq_abs, sq_abs]

/-- **With rows of sum zero the term of the query alone buys nothing**
(`review/flat_spectrum.md`, BL4): the squared error is at least `c (min(|m|, |n|) − d)`. -/
theorem HasCutWidth.mul_le_sum_sq_of_isFlat_of_sum_eq_zero {d : ℕ} {F : Q → C → ℝ}
    (hF : HasCutWidth d F) (q : m → Q) (x : n → C) {K : Matrix m n ℝ} {c : ℝ} (hc : 0 < c)
    (hK : IsFlat (K.map ((↑) : ℝ → ℂ)) c) (h0 : ∀ i, ∑ j, K i j = 0) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - (d : ℕ)) ≤
      ∑ i, ∑ j, (K i j - F (q i) (x j)) ^ 2 := by
  obtain ⟨g, a, b, hF⟩ := hF
  obtain ⟨β, hβ⟩ := exists_centering fun j => b (x j)
  have hrow (i : m) : ∑ j, (K i j - ∑ r, a (q i) r * (b (x j) r - β r)) ^ 2 ≤
      ∑ j, (K i j - F (q i) (x j)) ^ 2 := by
    have hy : ∑ j, (K i j - ∑ r, a (q i) r * (b (x j) r - β r)) = 0 := by
      rw [Finset.sum_sub_distrib, h0 i, zero_sub, Finset.sum_comm, neg_eq_zero]
      exact Finset.sum_eq_zero fun r _ => by rw [← Finset.mul_sum, hβ r, mul_zero]
    refine (sum_sq_le_sum_sub_sq hy (g (q i) + ∑ r, a (q i) r * β r)).trans (le_of_eq ?_)
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hF]
    congr 1
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  have hcap := hK.mul_sub_card_le_frobeniusSq_sub hc
    (Matrix.of fun i r => (a (q i) r : ℂ)) (Matrix.of fun r j => ((b (x j) r - β r : ℝ) : ℂ))
  rw [Fintype.card_fin] at hcap
  refine hcap.trans (le_trans (le_of_eq ?_) (Finset.sum_le_sum fun i _ => hrow i))
  simp only [frobeniusSq, Matrix.sub_apply, Matrix.mul_apply, Matrix.map_apply, Matrix.of_apply]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [show (K i j : ℂ) - ∑ r, (a (q i) r : ℂ) * ((b (x j) r - β r : ℝ) : ℂ) =
      ((K i j - ∑ r, a (q i) r * (b (x j) r - β r) : ℝ) : ℂ) by push_cast; rfl,
    Complex.norm_real, Real.norm_eq_abs, sq_abs]

/-- **A readout that returns a flat kernel on chosen pairs has cut width `≥ min(|m|, |n|) − 1`**
(`review/flat_spectrum.md`, BL4). -/
theorem HasCutWidth.card_le_of_isFlat {d : ℕ} {F : Q → C → ℝ} (hF : HasCutWidth d F)
    (q : m → Q) (x : n → C) {K : Matrix m n ℝ} {c : ℝ} (hc : 0 < c)
    (hK : IsFlat (K.map ((↑) : ℝ → ℂ)) c) (h : ∀ i j, F (q i) (x j) = K i j) :
    min (Fintype.card m) (Fintype.card n) ≤ d + 1 := by
  have hb := hF.mul_le_sum_sq_of_isFlat q x hc hK
  rw [show ∑ i, ∑ j, (K i j - F (q i) (x j)) ^ 2 = 0 by simp [h]] at hb
  have hsub : ((min (Fintype.card m) (Fintype.card n) : ℕ) : ℝ) - (d + 1 : ℕ) ≤ 0 :=
    nonpos_of_mul_nonpos_right hb hc
  exact_mod_cast sub_nonpos.1 hsub

/-- **With rows of sum zero, cut width `≥ min(|m|, |n|)`** (`review/flat_spectrum.md`, BL4). -/
theorem HasCutWidth.card_le_of_isFlat_of_sum_eq_zero {d : ℕ} {F : Q → C → ℝ}
    (hF : HasCutWidth d F) (q : m → Q) (x : n → C) {K : Matrix m n ℝ} {c : ℝ} (hc : 0 < c)
    (hK : IsFlat (K.map ((↑) : ℝ → ℂ)) c) (h0 : ∀ i, ∑ j, K i j = 0)
    (h : ∀ i j, F (q i) (x j) = K i j) : min (Fintype.card m) (Fintype.card n) ≤ d := by
  have hb := hF.mul_le_sum_sq_of_isFlat_of_sum_eq_zero q x hc hK h0
  rw [show ∑ i, ∑ j, (K i j - F (q i) (x j)) ^ 2 = 0 by simp [h]] at hb
  have hsub : ((min (Fintype.card m) (Fintype.card n) : ℕ) : ℝ) - (d : ℕ) ≤ 0 :=
    nonpos_of_mul_nonpos_right hb hc
  exact_mod_cast sub_nonpos.1 hsub

/-- The identity of size two is flat: a readout of cut width `1` that returns it. -/
example : min (Fintype.card (Fin 2)) (Fintype.card (Fin 2)) ≤ 1 + 1 :=
  HasCutWidth.card_le_of_isFlat (F := fun (q x : Fin 2) => if q = x then (1 : ℝ) else 0)
    ⟨fun q => if q = 0 then 1 else 0, fun q _ => if q = 0 then -1 else 1,
      fun x _ => if x = 0 then 0 else 1, fun q x => by fin_cases q <;> fin_cases x <;> simp⟩
    id id (K := 1) one_pos (by simpa using isFlat_one (n := Fin 2))
    fun i j => by simp [Matrix.one_apply]

/-- One query and the values `±1`: the kernel `(1, −1)` is flat and its row sums to zero. -/
example : min (Fintype.card (Fin 1)) (Fintype.card (Fin 2)) ≤ 1 :=
  HasCutWidth.card_le_of_isFlat_of_sum_eq_zero
    (F := fun (_ : Fin 1) (x : Fin 2) => if x = 0 then (1 : ℝ) else -1)
    ⟨fun _ => 0, fun _ _ => 1, fun x _ => if x = 0 then 1 else -1, fun q x => by simp⟩
    id id (K := Matrix.of fun _ x => if x = 0 then 1 else -1) (c := 2) two_pos
    (Or.inr (by
      ext i j
      simp [Matrix.mul_apply, Fin.sum_univ_two, Subsingleton.elim i j]
      norm_num))
    (fun i => by simp [Fin.sum_univ_two]) fun i j => by simp

end Transformer.Grokking.FlatSpectrum.Recall
