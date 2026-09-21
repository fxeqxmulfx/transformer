import Transformer.Modes.Section1_Sketch

/-
# The number of modes of a Gaussian KDE — `lem:scale-space`

§4.2 of arXiv:2412.09080v3, `sec:tail`: `lem:scale-space`, the bound on the
modes far from the origin by the sample points beyond them.

**What the source says and what is carried here.**

* `lem:scale-space` is stated for `β > 0` and `n ≥ 1`.  The source leaves both
  implicit, and both are needed: for `β ≤ 0` or `n = 0` the KDE is identically
  `0`, every point is a mode, and the count is infinite.  Its proof rests on
  Carreira-Perpiñán–Williams, Theorem 2, which is not in Mathlib; the lemma is
  not proved here.

* "By symmetry, the same estimate holds for modes in `(-∞, -a)`" is proved,
  `modeCount_kde_Iio_le`, from the `(a, ∞)` case applied to the reflected
  sample.

Source: arXiv:2412.09080v3, §4.2, `lem:scale-space`.
-/

open Real
open scoped ENNReal

namespace Transformer
namespace Modes

variable {n : ℕ}

open Classical in
/-- The number of sample points in `S`. -/
noncomputable def countIn (X : Idx n → ℝ) (S : Set ℝ) : ℕ :=
  (Finset.univ.filter fun i => X i ∈ S).card

/-- **Lemma (lem:scale-space).**  For `a > 0`, the number of modes of `P̂_n` in
`(a, ∞)` is at most `|{i : Xᵢ ≥ a}|`.

Not proved here; the source deduces it from Carreira-Perpiñán–Williams,
Theorem 2.  See the module docstring for `β > 0` and `n ≥ 1`.

Source: arXiv:2412.09080v3, `lem:scale-space`. -/
theorem scale_space {β : ℝ} (hβ : 0 < β) (hn : 0 < n) {a : ℝ} (ha : 0 < a)
    (X : Idx n → ℝ) : modeCount (kde β X) (Set.Ioi a) ≤ countIn X (Set.Ici a) := by
  sorry

/-- The hypotheses of `scale_space` are satisfiable. -/
example : (0 : ℝ) < 1 ∧ 0 < 1 ∧ (0 : ℝ) < 1 := ⟨one_pos, one_pos, one_pos⟩

/-! ### Symmetry -/

/-- Reflecting the sample reflects the KDE. -/
theorem kde_neg (β : ℝ) (X : Idx n → ℝ) (t : ℝ) : kde β (-X) t = kde β X (-t) := by
  unfold kde
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  congr 2
  simp only [Pi.neg_apply]
  ring

/-- A local maximum of `f ∘ (-·)` at `-s` is one of `f` at `s`. -/
theorem isLocalMax_comp_neg_iff (f : ℝ → ℝ) (s : ℝ) :
    IsLocalMax (fun t => f (-t)) (-s) ↔ IsLocalMax f s := by
  constructor
  · intro h
    have := h.comp_continuous (g := fun t : ℝ => -t) continuous_neg.continuousAt
    simpa [Function.comp_def] using this
  · intro h
    exact (neg_neg s ▸ h).comp_continuous (g := fun t : ℝ => -t) continuous_neg.continuousAt

/-- The modes of `P̂_n` in `(-∞, -a)` are the reflections of those of the
KDE of the reflected sample in `(a, ∞)`. -/
theorem modeCount_kde_Iio (β : ℝ) (X : Idx n → ℝ) (a : ℝ) :
    modeCount (kde β X) (Set.Iio (-a)) = modeCount (kde β (-X)) (Set.Ioi a) := by
  have h : modeSet (kde β X) (Set.Iio (-a)) =
      (fun t : ℝ => -t) '' modeSet (kde β (-X)) (Set.Ioi a) := by
    ext s
    simp only [modeSet, Set.mem_ofPred_eq, Set.mem_image, Set.mem_Iio, Set.mem_Ioi]
    constructor
    · rintro ⟨hs, hmax⟩
      refine ⟨-s, ⟨by linarith, ?_⟩, neg_neg s⟩
      simpa [funext (kde_neg β X)] using (isLocalMax_comp_neg_iff (kde β X) s).2 hmax
    · rintro ⟨t, ⟨ht, hmax⟩, rfl⟩
      refine ⟨by linarith, ?_⟩
      rw [funext (kde_neg β X)] at hmax
      exact (isLocalMax_comp_neg_iff (kde β X) (-t)).1 (by simpa using hmax)
  rw [modeCount, modeCount, h, neg_injective.encard_image]

/-- **`lem:scale-space`, "by symmetry".**  If the `(a, ∞)` case holds for every
sample, the number of modes in `(-∞, -a)` is at most `|{i : Xᵢ ≤ -a}|`.

Source: arXiv:2412.09080v3, `lem:scale-space`, second sentence. -/
theorem modeCount_kde_Iio_le {β a : ℝ}
    (hss : ∀ X : Idx n → ℝ, modeCount (kde β X) (Set.Ioi a) ≤ countIn X (Set.Ici a))
    (X : Idx n → ℝ) : modeCount (kde β X) (Set.Iio (-a)) ≤ countIn X (Set.Iic (-a)) := by
  rw [modeCount_kde_Iio]
  refine (hss (-X)).trans_eq ?_
  unfold countIn
  congr 3
  ext i
  simp [le_neg]

/-- With one sample, the only mode is the sample itself. -/
theorem eq_of_isLocalMax_kde_one {β : ℝ} (hβ : 0 < β) (X : Idx 1 → ℝ) {t : ℝ}
    (h : IsLocalMax (kde β X) t) : t = X 0 := by
  set C := Real.sqrt β / ((1 : ℕ) * Real.sqrt (2 * π))
  have hC : 0 < C := by positivity
  have hk : kde β X = fun s => C * Real.exp (-(β / 2) * (s - X 0) ^ 2) := by
    funext s
    simp [kde, C]
  have hd : HasDerivAt (fun s => C * Real.exp (-(β / 2) * (s - X 0) ^ 2))
      (C * (Real.exp (-(β / 2) * (t - X 0) ^ 2) * (-(β / 2) * (2 * (t - X 0) ^ 1 * 1)))) t :=
    (((((hasDerivAt_id t).sub_const (X 0)).pow 2).const_mul (-(β / 2))).exp).const_mul C
  rw [hk] at h
  have h0 := h.hasDerivAt_eq_zero hd
  simp only [pow_one, mul_one, mul_eq_zero, hC.ne', (Real.exp_pos _).ne', neg_eq_zero,
    div_eq_zero_iff, hβ.ne', two_ne_zero, sub_eq_zero, false_or, or_false] at h0
  exact h0

/-- `lem:scale-space` holds for one sample: its hypothesis, as carried by
`modeCount_kde_Iio_le` and `expectedModes_compl_le`, is satisfiable. -/
theorem scale_space_one {β : ℝ} (hβ : 0 < β) (a : ℝ) (X : Idx 1 → ℝ) :
    modeCount (kde β X) (Set.Ioi a) ≤ countIn X (Set.Ici a) := by
  have hsub : modeSet (kde β X) (Set.Ioi a)
      ⊆ X '' ↑(Finset.univ.filter fun i => X i ∈ Set.Ici a) := by
    rintro t ⟨ht, hmax⟩
    have h := eq_of_isLocalMax_kde_one hβ X hmax
    refine ⟨0, ?_, h.symm⟩
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, Set.mem_Ici]
    exact h ▸ (le_of_lt ht)
  have h1 := (Set.encard_le_encard hsub).trans (Set.encard_image_le _ _)
  rw [Set.encard_coe_eq_coe_finsetCard] at h1
  simpa [modeCount, countIn] using (ENat.toENNReal_le.mpr h1)

/-- The hypothesis of `modeCount_kde_Iio_le` is satisfiable. -/
example : ∀ X : Idx 1 → ℝ, modeCount (kde 1 X) (Set.Ioi 1) ≤ countIn X (Set.Ici 1) :=
  scale_space_one one_pos 1

end Modes
end Transformer
