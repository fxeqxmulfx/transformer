module

public import Transformer.BAM.Section2_BayesianAttention
public import Transformer.BAM.SectionA_Notation

/-!
# §2.2 Positional encodings as priors: the causal mask and ALiBi

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, §2.2; proofs in Appendix B.1–B.2.

A context has `L` tokens. Key `j : Fin L` is the paper's token `x_{j+1}` (the paper counts from
`1`), so the paper's `Uniform(1, i, j)` is `uniform 1 (i + 1) (j + 1)` here; relative positions
`j − i` (`relPos`) are the same in both countings.

* Lemma 1: the causal mask is the uniform prior, `softmax(M_i•)_j = Uniform(1, i, j)`
  (`softmax_causalMask`); with content scores `f`, `p_ij = p(f)_j · Uniform(1, i, j) / Z`
  (`softmax_add_causalMask`).
* Lemma 2: ALiBi, `a_ij = −m|j − i|`, is the prior `Uniform(1, i, j) · Laplace(0, 1/m, j − i)`.
  Its proof drops the normalizer `Σ_z (m/2) e^{−m|z − i|}` "since `Z_ALiBi` accounts for the
  normalization of the whole expression". The statement holds with that normalizer
  (`softmax_causalMask_add_alibi`, `softmax_add_alibi`) and fails without it
  (`softmax_causalMask_add_alibi_ne`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {L : ℕ}

/-- The relative position `j − i` of key `j` to query `i`. -/
noncomputable def relPos (i j : Fin L) : ℝ := (j : ℕ) - (i : ℕ)

/-- The causal mask row of query `i`: `M_ij = 0` for `j ≤ i` and `−∞` otherwise
(Appendix B.1). -/
noncomputable def causalMask (i j : Fin L) : EReal := if j ≤ i then 0 else ⊥

/-- The ALiBi bias `a_ij = −m|j − i|` at the relative position `x = j − i` (Appendix B.2). -/
noncomputable def alibi (m x : ℝ) : EReal := ((-(m * |x|) : ℝ) : EReal)

theorem weight_causalMask (i j : Fin L) : weight (causalMask i j) = if j ≤ i then 1 else 0 := by
  unfold causalMask
  split_ifs <;> simp

/-- The causal window of query `i` holds `i + 1` keys. -/
theorem partition_causalMask (i : Fin L) : partition (causalMask i) = (i : ℕ) + 1 := by
  simp [partition, weight_causalMask, sum_boole, filter_ge_eq_Iic]

/-- `Uniform(1, i, j)` in the paper's positions: `1/i` for `j ≤ i`, else `0`. -/
theorem uniform_causal (i j : Fin L) :
    uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) = if j ≤ i then 1 / ((i : ℕ) + 1 : ℝ) else 0 := by
  have h : (1 ≤ ((j : ℕ) : ℤ) + 1 ∧ ((j : ℕ) : ℤ) + 1 ≤ ((i : ℕ) : ℤ) + 1) ↔ j ≤ i := by
    rw [Fin.le_iff_val_le_val]
    omega
  unfold uniform
  split_ifs with h1 h2 h2
  · push_cast
    ring_nf
  · exact absurd (h.1 h1) h2
  · exact absurd (h.2 h2) h1
  · rfl

/-- **Lemma 1.** The causal mask is a uniform prior: its position marginal `softmax(M_i•)_j` is
`Uniform(1, i, j)`, `1/i` for `j ≤ i` and `0` otherwise (§2.2, Appendix B.1). -/
theorem softmax_causalMask (i j : Fin L) :
    softmax (causalMask i) j = uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) := by
  rw [softmax, partition_causalMask, weight_causalMask, uniform_causal]
  split_ifs <;> simp

/-- **Lemma 1, with content.** `softmax(f + M_i•)_j = p(f)_j · Uniform(1, i, j) / Z` with
`Z = Σ_z p(f)_z · Uniform(1, i, z)` (§2.2, Appendix B.1). -/
theorem softmax_add_causalMask (f : Fin L → EReal) (i j : Fin L) :
    softmax (f + causalMask i) j =
      softmax f j * uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) /
        ∑ z, softmax f z * uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) := by
  simp only [softmax_add, overlap, softmax_causalMask]

/-- The softmax of scores whose weights are proportional to a prior `π` is `π` normalized. -/
theorem softmax_eq_of_proportional {ι : Type*} [Fintype ι] {g : ι → EReal} {π : ι → ℝ} {c : ℝ}
    (hc : c ≠ 0) (hπ : ∀ z, π z = c * weight (g z)) (j : ι) :
    softmax g j = π j / ∑ z, π z := by
  simp only [softmax, partition, hπ, ← mul_sum, mul_div_mul_left _ _ hc]

/-- The hypotheses of `softmax_eq_of_proportional` hold for the prior `3 e^{g}`. -/
example (g : Fin 2 → EReal) (j : Fin 2) :
    softmax g j = 3 * weight (g j) / ∑ z, 3 * weight (g z) :=
  softmax_eq_of_proportional (π := fun z => 3 * weight (g z)) three_ne_zero (fun _ => rfl) j

/-- ALiBi's prior: `Uniform(1, i, z) · Laplace(0, 1/m, z − i) = m / (2(i + 1)) · e^{M_iz + a_iz}`
(Appendix B.2). -/
theorem uniform_mul_laplace {m : ℝ} (hm : 0 < m) (i z : Fin L) :
    uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * laplace 0 (1 / m) (relPos i z) =
      m / (2 * ((i : ℕ) + 1)) * weight (causalMask i z + alibi m (relPos i z)) := by
  have he : -|relPos i z - 0| / (1 / m) = -(m * |relPos i z|) := by
    rw [sub_zero]
    field_simp
  rw [weight_add, weight_causalMask, uniform_causal, alibi, weight_coe, laplace, he]
  split_ifs
  · field_simp
  · simp

/-- **Lemma 2, the position marginal.** With ALiBi of slope `m > 0`, the position marginal
`softmax(M_i• + A_i•)_j` is `Uniform(1, i, j) · Laplace(0, 1/m, j − i)`, normalized over the
context (§2.2, Appendix B.2). The paper states the equation without the normalizer; see
`softmax_causalMask_add_alibi_ne`. -/
theorem softmax_causalMask_add_alibi {m : ℝ} (hm : 0 < m) (i j : Fin L) :
    softmax (fun z => causalMask i z + alibi m (relPos i z)) j =
      uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * laplace 0 (1 / m) (relPos i j) /
        ∑ z : Fin L, uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * laplace 0 (1 / m) (relPos i z) :=
  softmax_eq_of_proportional (by positivity) (uniform_mul_laplace hm i) j

/-- The hypothesis of `softmax_causalMask_add_alibi` holds for the slope `m = 1/2`. -/
example (i j : Fin 4) :
    softmax (fun z => causalMask i z + alibi (1 / 2) (relPos i z)) j =
      uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * laplace 0 (1 / (1 / 2)) (relPos i j) /
        ∑ z : Fin 4, uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * laplace 0 (1 / (1 / 2)) (relPos i z) :=
  softmax_causalMask_add_alibi (by norm_num) i j

/-- **Lemma 2.** ALiBi is the prior `Uniform(1, i, j) · Laplace(0, 1/m, j − i)`:
`softmax(qK^⊤ + M_i• + A_i•)_j = p(f)_j · Uniform(1, i, j) · Laplace(0, 1/m, j − i) / Z_ALiBi`
with `Z_ALiBi = Σ_z p(f)_z · Uniform(1, i, z) · Laplace(0, 1/m, z − i)` (§2.2,
Appendix B.2). -/
theorem softmax_add_alibi {m : ℝ} (hm : 0 < m) (f : Fin L → EReal) (i j : Fin L) :
    softmax (fun z => f z + causalMask i z + alibi m (relPos i z)) j =
      softmax f j * (uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * laplace 0 (1 / m) (relPos i j)) /
        ∑ z, softmax f z *
          (uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * laplace 0 (1 / m) (relPos i z)) := by
  have hfun : (fun z => f z + causalMask i z + alibi m (relPos i z)) =
      f + fun z => causalMask i z + alibi m (relPos i z) := by
    funext z
    simp [add_assoc]
  rw [hfun]
  exact softmax_add_eq_of_proportional (by positivity) (uniform_mul_laplace hm i) j

/-- The hypothesis of `softmax_add_alibi` holds for the slope `m = 1/2`. -/
example (f : Fin 4 → EReal) (i j : Fin 4) :
    softmax (fun z => f z + causalMask i z + alibi (1 / 2) (relPos i z)) j =
      softmax f j * (uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) * laplace 0 (1 / (1 / 2)) (relPos i j)) /
        ∑ z, softmax f z *
          (uniform 1 ((i : ℕ) + 1) ((z : ℕ) + 1) * laplace 0 (1 / (1 / 2)) (relPos i z)) :=
  softmax_add_alibi (by norm_num) f i j

/-- **Lemma 2 without its normalizer is false.** Lemma 2 reads
`p(g_pos) = Uniform(1, i, j) · Laplace(0, 1/m, j − i)`. For one token and `m = 1` the position
marginal is `1`, while `Uniform(1, 1, 1) · Laplace(0, 1, 0) = 1/2` (§2.2, Appendix B.2). -/
theorem softmax_causalMask_add_alibi_ne :
    softmax (fun z => causalMask (0 : Fin 1) z + alibi 1 (relPos 0 z)) 0 ≠
      uniform 1 1 1 * laplace 0 (1 / 1) (relPos (0 : Fin 1) 0) := by
  norm_num [softmax, partition, causalMask, alibi, relPos, uniform, laplace]

end Transformer.BAM
