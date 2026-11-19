module

public import Mathlib.Analysis.Convex.Radon
public import Transformer.Grokking.FlatSpectrum.Recall.Threshold
public import Transformer.MHC.PeriRMS.Scores

/-!
# A raw gap above one fits at most `d + 1` keys

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3, and for the diff-gpt model of `Transformer.MHC.PeriRMS`.

At the scale `s log n`, raw scores that put every queried key ahead of every other key by a
fixed `g` recall at every length once `s g > 1` (`exists_forall_logScaleAttention_recall`).
Whether such a gap exists is a question about the geometry of the raw scores. Let the raw score
of the query of the key `i` at the key `j` be `⟨u_i, v_j⟩`, with `‖u_i‖ ≤ a` and `‖v_j‖ ≤ b`
in a space of dimension `d`. If every key leads every other key by more than `a b` at its own
query, the keys are affinely independent (`affineIndependent_of_inner_gap`), so there are at most
`d + 1` of them (`card_le_finrank_add_one_of_inner_gap`).

Proof. If the keys are affinely dependent, Radon's theorem splits them into two parts whose
convex hulls share a point `x` (`Convex.radon_partition`). A key `v_i` of the first part leads
every key of the second part, hence `x`, by more than `a b`, so `‖v_i − x‖ > b ≥ ‖v_i‖` and
`⟨x, v_i⟩ < ‖x‖² / 2`. That open half-space is convex, so it holds `x` as well, a convex
combination of the first part: `‖x‖² < ‖x‖² / 2`, which is absurd.

The diff-gpt model takes `s = 1`, so SA5 asks it for a raw gap above `1`. Its raw score is
`(1/d) ⟨q̂, k̂⟩` with `‖q̂‖, ‖k̂‖ ≤ √d` after Q/K RMS and rotary phases, for the head width `d`
(`Transformer.MHC.PeriRMS.normalized_norm`). So a raw gap above `1` holds for at most `d + 1`
keys at once (`card_le_of_rawScore_gap`), whatever the training.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.MHC.PeriRMS

/-- A half-space `⟨a, y⟩ < r` of a real inner product space is convex. -/
lemma convex_inner_lt {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (a : E)
    (r : ℝ) : Convex ℝ {y : E | inner ℝ a y < r} :=
  convex_halfSpace_lt ⟨fun x y => inner_add_right a x y, fun c x => inner_smul_right a x c⟩ r

/-- **Keys with a gap above `a b` are affinely independent** (`review/flat_spectrum.md`, SA5):
raw scores `⟨u_i, v_j⟩` of queries `‖u_i‖ ≤ a` and keys `‖v_j‖ ≤ b`. If every key leads every
other key by more than `a b` at its own query, the keys are affinely independent. -/
theorem affineIndependent_of_inner_gap {E ι : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {u v : ι → E} {a b : ℝ} (hu : ∀ i, ‖u i‖ ≤ a) (hv : ∀ i, ‖v i‖ ≤ b)
    (h : ∀ i j, j ≠ i → inner ℝ (u i) (v j) + a * b < inner ℝ (u i) (v i)) :
    AffineIndependent ℝ v := by
  by_contra hind
  obtain ⟨I, x, hxI, hxJ⟩ := Convex.radon_partition hind
  have hfar (i : ι) (hi : i ∈ I) : inner ℝ x (v i) < ‖x‖ ^ 2 / 2 := by
    have hsub : v '' Iᶜ ⊆ {y | inner ℝ (u i) y < inner ℝ (u i) (v i) - a * b} := by
      rintro _ ⟨j, hj, rfl⟩
      have := h i j (by rintro rfl; exact hj hi)
      show inner ℝ (u i) (v j) < _
      linarith
    have hx : inner ℝ (u i) x < inner ℝ (u i) (v i) - a * b :=
      convexHull_min hsub (convex_inner_lt _ _) hxJ
    have hcs := real_inner_le_norm (u i) (v i - x)
    rw [inner_sub_right] at hcs
    have ha : 0 ≤ a := (norm_nonneg _).trans (hu i)
    have hb : 0 ≤ b := (norm_nonneg _).trans (hv i)
    have h1 : b < ‖v i - x‖ := by
      by_contra hle
      rw [not_lt] at hle
      linarith [mul_le_mul (hu i) hle (norm_nonneg _) ha]
    have h2 := norm_sub_sq_real (v i) x
    rw [real_inner_comm x (v i)] at h2
    nlinarith [mul_self_lt_mul_self hb h1, mul_self_le_mul_self (norm_nonneg (v i)) (hv i)]
  have hsub : v '' I ⊆ {y | inner ℝ x y < ‖x‖ ^ 2 / 2} := by
    rintro _ ⟨i, hi, rfl⟩
    exact hfar i hi
  have hx : inner ℝ x x < ‖x‖ ^ 2 / 2 := convexHull_min hsub (convex_inner_lt _ _) hxI
  rw [real_inner_self_eq_norm_sq] at hx
  linarith [sq_nonneg ‖x‖]

/-- Two keys `±1` on the line lead each other by `2`. -/
example := affineIndependent_of_inner_gap (u := ![(1 : ℝ), -1]) (v := ![1, -1]) (a := 1) (b := 1)
  (fun i => by fin_cases i <;> simp) (fun i => by fin_cases i <;> simp)
  (fun i j hji => by fin_cases i <;> fin_cases j <;> simp_all)

/-- **A gap above `a b` fits at most `d + 1` keys** (`review/flat_spectrum.md`, SA5): raw scores
`⟨u_i, v_j⟩` of queries `‖u_i‖ ≤ a` and keys `‖v_j‖ ≤ b` in a space of dimension `d`. If every
key leads every other key by more than `a b` at its own query, there are at most `d + 1` keys. -/
theorem card_le_finrank_add_one_of_inner_gap {E ι : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [Fintype ι] {u v : ι → E} {a b : ℝ}
    (hu : ∀ i, ‖u i‖ ≤ a) (hv : ∀ i, ‖v i‖ ≤ b)
    (h : ∀ i j, j ≠ i → inner ℝ (u i) (v j) + a * b < inner ℝ (u i) (v i)) :
    Fintype.card ι ≤ Module.finrank ℝ E + 1 :=
  (affineIndependent_of_inner_gap hu hv h).card_le_finrank_succ.trans
    (Nat.add_le_add_right (Submodule.finrank_le _) 1)

/-- Two keys `±1` on the line lead each other by `2`: the bound `1 + 1` is reached. -/
example : Fintype.card (Fin 2) ≤ Module.finrank ℝ ℝ + 1 :=
  card_le_finrank_add_one_of_inner_gap (u := ![1, -1]) (v := ![1, -1]) (a := 1) (b := 1)
    (fun i => by fin_cases i <;> simp) (fun i => by fin_cases i <;> simp)
    (fun i j hji => by fin_cases i <;> fin_cases j <;> simp_all)

/-- **In the diff-gpt model a raw gap above `1` holds for at most `d + 1` keys**
(`review/flat_spectrum.md`, SA5, for the scores of `Transformer.MHC.PeriRMS.score_bound`): the
query of each key `i` has the rotary phases `a i` and the key `j` has `b j`; the raw score is
`(1/d) ⟨q̂, k̂⟩` after Q/K RMS. If the key of each `i` leads every other key by more than `1` at
the query of `i`, there are at most `d + 1` keys. -/
theorem card_le_of_rawScore_gap {d : ℕ} {ι : Type*} [Fintype ι] (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (a b : ι → Fin (d / 2) → ℝ) (q k : ι → EucSpace d)
    (h : ∀ i j, j ≠ i →
      (1 / (d : ℝ)) * inner ℝ (rotatedQK eps (a i) (q i)) (rotatedQK eps (b j) (k j)) + 1 <
        (1 / (d : ℝ)) * inner ℝ (rotatedQK eps (a i) (q i)) (rotatedQK eps (b i) (k i))) :
    Fintype.card ι ≤ d + 1 := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have h' := card_le_finrank_add_one_of_inner_gap (u := fun i => rotatedQK eps (a i) (q i))
    (v := fun j => rotatedQK eps (b j) (k j)) (a := Real.sqrt d) (b := Real.sqrt d)
    (fun i => normalized_norm eps heps hd (a i) (q i))
    (fun j => normalized_norm eps heps hd (b j) (k j)) fun i j hji => by
      have hij := h i j hji
      rw [Real.mul_self_sqrt hd'.le,
        ← mul_lt_mul_iff_of_pos_left (show (0 : ℝ) < 1 / d by positivity)]
      calc 1 / (d : ℝ) * (inner ℝ (rotatedQK eps (a i) (q i)) (rotatedQK eps (b j) (k j)) + d)
          = 1 / (d : ℝ) * inner ℝ (rotatedQK eps (a i) (q i)) (rotatedQK eps (b j) (k j)) + 1 := by
            field_simp
        _ < _ := hij
  rwa [finrank_euclideanSpace_fin] at h'

/-- At head width `1` there is no pair of channels to rotate: the rotary map is the identity. -/
lemma rotateHead_one (angle : Fin (1 / 2) → ℝ) (x : EucSpace 1) :
    Transformer.SequenceModel.rotateHead 1 angle x = x := by
  ext i
  fin_cases i
  rfl

/-- At head width `1` and `eps = 1/2` the keys `±e` with the queries `±e` have the raw scores
`±2/3`, so each leads the other by `4/3 > 1`: two keys, the bound `1 + 1`. -/
example : Fintype.card (Fin 2) ≤ 1 + 1 := by
  have hneg (x : EucSpace 1) (a : Fin (1 / 2) → ℝ) :
      rotatedQK (1 / 2) a (-x) = -rotatedQK (1 / 2) a x := by
    simp only [rotatedQK, rotateHead_one, GPTMini.rmsNormEps, norm_neg, smul_neg]
  have hr : ‖rotatedQK (1 / 2 : ℝ) (fun _ => 0) (EuclideanSpace.single 0 1 : EucSpace 1)‖ ^ 2 =
      2 / 3 := by
    rw [rotatedQK, rotateHead_one, GPTMini.rmsNormEps, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (by positivity), mul_pow, div_pow, Real.sq_sqrt (by positivity),
      Real.sq_sqrt (by positivity)]
    simp
    norm_num
  refine card_le_of_rawScore_gap (1 / 2) (by norm_num) one_pos (fun _ _ => 0) (fun _ _ => 0)
    ![EuclideanSpace.single 0 1, -EuclideanSpace.single 0 1]
    ![EuclideanSpace.single 0 1, -EuclideanSpace.single 0 1] fun i j hji => ?_
  fin_cases i <;> fin_cases j <;> simp at hji
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one, hneg,
      inner_neg_left, inner_neg_right, neg_neg, real_inner_self_eq_norm_sq, hr]
    norm_num

end Transformer.Grokking.FlatSpectrum.Recall
