module

public import Mathlib.Analysis.Convex.Segment
public import Transformer.Grokking.FlatSpectrum.Recall.SoftmaxRecall

/-!
# Softmax recall needs dimension two

Source: `review/flat_spectrum.md`, *Functions of n variables* (added), item NV4: the lower side
of SA2.

A score linear in the key never peaks inside the hull of the other keys: if the projection of the
key `k` is a convex combination of the projections of the others, its score is at most the
largest of theirs, at every query. So its weight is at most that of another key, at most half of
the sum, and the dictionary with the value `1` at `k` and `−1` elsewhere gets an attention `≤ 0`
at the query of `k`, whatever the scale (`softmaxAttention_indicator_nonpos_of_mem_convexHull`).

In dimension at most one, among three keys the one in the middle lies between the other two
(`exists_mem_convexHull_compl_of_le_one`). So over `m ≥ 3` keys no projections of dimension one
recall every dictionary of values `±1` by the sign of the attention, and SA2 recalls them in
dimension two: two is the least dimension (`isLeast_dim_softmaxAttention_recall_sign`). SA2
puts the keys on a parabola, where none lies in the hull of the others.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- The attention to the dictionary that is `1` at `k` and `−1` elsewhere, as weights: the
weight of `k` minus that of the other keys. -/
lemma sum_exp_mul_indicator {ι : Type*} [Fintype ι] [DecidableEq ι] (s : ι → ℝ) (k : ι) :
    ∑ j, Real.exp (s j) * (if j = k then 1 else -1) =
      Real.exp (s k) - ∑ j ∈ univ.erase k, Real.exp (s j) := by
  rw [← add_sum_erase _ _ (mem_univ k), sub_eq_add_neg, ← sum_neg_distrib]
  congr 1
  · simp
  · exact sum_congr rfl fun j hj => by simp [ne_of_mem_erase hj]

example (s : Fin 2 → ℝ) := sum_exp_mul_indicator s 0

/-- **A key inside the hull of the others keeps at most half of the weight**
(`review/flat_spectrum.md`, NV4): if the projection `kf k` of the key `k` lies in the convex hull
of the projections of the other keys, then at every query `a` the dictionary with the value `1`
at `k` and `−1` elsewhere gets an attention `≤ 0`. -/
theorem softmaxAttention_indicator_nonpos_of_mem_convexHull {ι : Type*} [Fintype ι]
    [DecidableEq ι] {d : ℕ} (a : Fin d → ℝ) {kf : ι → Fin d → ℝ} {k : ι}
    (hk : kf k ∈ convexHull ℝ (kf '' {k}ᶜ)) :
    softmaxAttention (fun j => ∑ r, a r * kf j r) (fun j => if j = k then 1 else -1) ≤ 0 := by
  set s : ι → ℝ := fun j => ∑ r, a r * kf j r
  have hne : (univ.erase k).Nonempty := by
    by_contra h
    have h0 : kf '' {k}ᶜ = ∅ := Set.image_eq_empty.2 (Set.eq_empty_of_forall_notMem fun j hj =>
      h ⟨j, mem_erase.2 ⟨hj, mem_univ j⟩⟩)
    rw [h0, convexHull_empty] at hk
    exact hk
  obtain ⟨l, hl, hmax⟩ := exists_max_image (univ.erase k) s hne
  have hlin : IsLinearMap ℝ fun x : Fin d → ℝ => ∑ r, a r * x r :=
    ⟨fun x y => by simp [mul_add, sum_add_distrib], fun c x => by simp [mul_sum, mul_left_comm]⟩
  have hsk : s k ≤ s l := convexHull_min (t := {x | ∑ r, a r * x r ≤ s l})
    (by rintro _ ⟨j, hj, rfl⟩; exact hmax j (mem_erase.2 ⟨hj, mem_univ j⟩))
    (convex_halfSpace_le hlin _) hk
  have hl' : Real.exp (s l) ≤ ∑ j ∈ univ.erase k, Real.exp (s j) :=
    single_le_sum (f := fun j => Real.exp (s j)) (fun j _ => (Real.exp_pos _).le) hl
  rw [softmaxAttention, sum_exp_mul_indicator]
  exact div_nonpos_of_nonpos_of_nonneg (by linarith [Real.exp_le_exp.2 hsk])
    (sum_nonneg fun j _ => (Real.exp_pos _).le)

/-- Two keys with one projection: the first keeps at most half of the weight. -/
example := softmaxAttention_indicator_nonpos_of_mem_convexHull (ι := Fin 2) (d := 1) 1
  (kf := fun _ => 0) (k := 0) (subset_convexHull ℝ _ ⟨1, by simp, rfl⟩)

/-- **In dimension at most one, a key lies between two others** (`review/flat_spectrum.md`,
NV4): among three or more points of dimension `d ≤ 1`, one is in the convex hull of the rest. -/
theorem exists_mem_convexHull_compl_of_le_one {ι : Type*} [Fintype ι] {d : ℕ} (hd : d ≤ 1)
    (hι : 3 ≤ Fintype.card ι) (kf : ι → Fin d → ℝ) : ∃ k, kf k ∈ convexHull ℝ (kf '' {k}ᶜ) := by
  classical
  have : Subsingleton (Fin d) := ⟨fun x y => Fin.ext (by omega)⟩
  set b : ι → ℝ := fun j => ∑ r, kf j r
  have hb (j : ι) (r : Fin d) : kf j r = b j := (Fintype.sum_subsingleton _ r).symm
  have hne : (univ : Finset ι).Nonempty := univ_nonempty_iff.2 (Fintype.card_pos_iff.1 (by omega))
  obtain ⟨i₀, -, hi₀⟩ := exists_min_image univ b hne
  obtain ⟨j₀, -, hj₀⟩ := exists_max_image univ b hne
  obtain ⟨k, hki, hkj⟩ : ∃ k, k ≠ i₀ ∧ k ≠ j₀ := by
    by_contra h
    push Not at h
    have hsub : (univ : Finset ι) ⊆ {i₀, j₀} := fun k _ => by
      by_cases hk : k = i₀
      · simp [hk]
      · simp [h k hk]
    have := (card_le_card hsub).trans card_le_two
    rw [card_univ] at this
    omega
  refine ⟨k, segment_subset_convexHull (s := kf '' {k}ᶜ)
    (Set.mem_image_of_mem kf (Set.mem_compl_singleton_iff.2 hki.symm))
    (Set.mem_image_of_mem kf (Set.mem_compl_singleton_iff.2 hkj.symm)) ?_⟩
  have hk : b k ∈ segment ℝ (b i₀) (b j₀) := by
    rw [segment_eq_Icc (hi₀ j₀ (mem_univ _))]
    exact ⟨hi₀ k (mem_univ _), hj₀ k (mem_univ _)⟩
  obtain ⟨α, β, hα, hβ, hαβ, hab⟩ := hk
  exact ⟨α, β, hα, hβ, hαβ, funext fun r => by simpa [hb] using hab⟩

example := exists_mem_convexHull_compl_of_le_one (ι := Fin 3) le_rfl (by simp)
  fun j _ => (j : ℝ)

/-- **Two is the least dimension of softmax recall by sign** (`review/flat_spectrum.md`, NV4,
with SA2): over `m ≥ 3` keys, projections of dimension `d` whose softmax attention answers every
dictionary of values `±1` with its sign exist for `d = 2` and for no smaller `d`. -/
theorem isLeast_dim_softmaxAttention_recall_sign {m : ℕ} (hm : 3 ≤ m) :
    IsLeast {d | ∃ (qf kf : Fin m → Fin d → ℝ), ∀ (π : Fin m → Bool) (q : Fin m),
      0 < softmaxAttention (fun j => ∑ r, qf q r * kf j r) (fun j => if π j then 1 else -1) ↔
        π q = true} 2 := by
  refine ⟨exists_softmaxAttention_recall_sign m, fun d hd => ?_⟩
  obtain ⟨qf, kf, h⟩ := hd
  by_contra hd2
  obtain ⟨k, hk⟩ := exists_mem_convexHull_compl_of_le_one (by omega) (by simpa using hm) kf
  have h1 := (h (fun j => decide (j = k)) k).2 (by simp)
  have h2 := softmaxAttention_indicator_nonpos_of_mem_convexHull (qf k) hk
  simp only [decide_eq_true_eq] at h1
  linarith

example := isLeast_dim_softmaxAttention_recall_sign (m := 3) le_rfl

end Transformer.Grokking.FlatSpectrum.Recall
