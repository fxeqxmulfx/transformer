module

public import Transformer.Grokking.FlatSpectrum.Training.MaxMargin

/-!
# Trained products: the minimizers of `L + λ ‖·‖_S1`, without their factors

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE5; item WD2.
A trained fit is a factorization of least weight-decayed loss among those of every width, and
the statements about sequences of trained fits take one column type for the whole sequence.
The product of a trained fit minimizes `L + λ ‖·‖_S1` over all matrices (item WD2); call such
a matrix a trained product (`IsNuclearMin`). For `λ ≥ 0`:

* the product of a trained fit of any width is one (`IsDecayedMin.isNuclearMin`);
* every matrix is `U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1` at the width of its columns
  (`exists_mul_eq_weightDecay_eq`, Lemma 1), so a trained product is the product of a trained
  fit of that width (`IsNuclearMin.exists_isDecayedMin`, `isNuclearMin_iff`);
* hence what holds for the products of every sequence of trained fits of that width holds for
  every sequence of trained products (`forall_seq_isNuclearMin`), whatever the widths of the
  fits they come from.

Trained products of the cross-entropy exist for `λ > 0` (`exists_isNuclearMin_ceLoss`), and
item XE5 holds for them: `IsNuclearMin.ceLoss_add_le`, `IsNuclearMin.ceLoss_le_smul`,
`IsNuclearMin.log_le_mul_exp` and `IsNuclearMin.sub_log_le_gap`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {lam : ℝ} {X : Matrix m n ℂ}

/-- A trained product: a matrix that minimizes `L + λ ‖·‖_S1` over all matrices, item WD2. -/
def IsNuclearMin (L : Matrix m n ℂ → ℝ) (lam : ℝ) (X : Matrix m n ℂ) : Prop :=
  ∀ Y : Matrix m n ℂ, L X + lam * nuclearNorm X ≤ L Y + lam * nuclearNorm Y

omit [Fintype k] in
/-- Lemma 1 at the width of the columns: `X = U Vᴴ` with `½ (‖U‖² + ‖V‖²) = ‖X‖_S1`. -/
lemma exists_mul_eq_weightDecay_eq (X : Matrix m n ℂ) :
    ∃ (U : Matrix m n ℂ) (V : Matrix n n ℂ), U * Vᴴ = X ∧
      (frobeniusSq U + frobeniusSq V) / 2 = nuclearNorm X := by
  obtain ⟨hfac, hU, hV⟩ := factor_of_diagonalization (d := singularValues X)
    (fun i => Real.sqrt_nonneg _)
    (eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self X))
    (gram_diagonalization X)
  exact ⟨_, _, hfac.symm, by rw [hU, hV, nuclearNorm]; ring⟩

/-- **The product of a trained fit is a trained product** (`review/flat_spectrum.md`, item
WD2), for a fit of any width and `λ ≥ 0`. -/
theorem IsDecayedMin.isNuclearMin (hlam : 0 ≤ lam) {U : Matrix m k ℂ} {V : Matrix n k ℂ}
    (h : IsDecayedMin L lam U V) : IsNuclearMin L lam (U * Vᴴ) :=
  h.add_nuclearNorm_le hlam

omit [Fintype k] in
/-- **A trained product is the product of a trained fit of the width of its columns**
(`review/flat_spectrum.md`, item XE5), for `λ ≥ 0`. -/
theorem IsNuclearMin.exists_isDecayedMin (hlam : 0 ≤ lam) (h : IsNuclearMin L lam X) :
    ∃ (U : Matrix m n ℂ) (V : Matrix n n ℂ), U * Vᴴ = X ∧ IsDecayedMin L lam U V := by
  obtain ⟨U, V, hX, hw⟩ := exists_mul_eq_weightDecay_eq X
  refine ⟨U, V, hX, fun r U' V' => ?_⟩
  have h1 := h (U' * V'ᴴ)
  have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul U' V' rfl) hlam
  unfold decayedLoss
  rw [hX, hw]
  linarith

omit [Fintype k] in
/-- Trained products are the products of the trained fits of the width of the columns. -/
theorem isNuclearMin_iff (hlam : 0 ≤ lam) :
    IsNuclearMin L lam X ↔
      ∃ (U : Matrix m n ℂ) (V : Matrix n n ℂ), U * Vᴴ = X ∧ IsDecayedMin L lam U V :=
  ⟨fun h => h.exists_isDecayedMin hlam, fun ⟨_, _, hX, h⟩ => hX ▸ h.isNuclearMin hlam⟩

omit [Fintype k] in
/-- **Statements about sequences of trained fits hold for sequences of trained products**
(`review/flat_spectrum.md`, item XE5): what holds for the products of all trained fits of the
width of the columns along `λ_j ≥ 0` holds for all trained products along `λ_j`. -/
theorem forall_seq_isNuclearMin {lams : ℕ → ℝ} (hlam : ∀ j, 0 ≤ lams j)
    {P : (ℕ → Matrix m n ℂ) → Prop}
    (hP : ∀ (Us : ℕ → Matrix m n ℂ) (Vs : ℕ → Matrix n n ℂ),
      (∀ j, IsDecayedMin L (lams j) (Us j) (Vs j)) → P fun j => Us j * (Vs j)ᴴ)
    {X : ℕ → Matrix m n ℂ} (hX : ∀ j, IsNuclearMin L (lams j) (X j)) : P X := by
  choose Us Vs hUV hmin using fun j => (hX j).exists_isDecayedMin (hlam j)
  have e : (fun j => Us j * (Vs j)ᴴ) = X := funext hUV
  exact e ▸ hP Us Vs hmin

variable [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C} {μ' : ℝ}
  {X' : Matrix m n ℂ}

omit [Fintype k] in
/-- **A trained product against a scaled competitor** (`review/flat_spectrum.md`, item XE5):
for `λ ≥ 0`, `‖X'‖_S1 ≤ 1` and `t ≥ 0`, `L(X) + λ ‖X‖_S1 ≤ L(t X') + λ t`. -/
theorem IsNuclearMin.ceLoss_add_le (hlam : 0 ≤ lam) (h : IsNuclearMin (ceLoss A y) lam X)
    (hX' : nuclearNorm X' ≤ 1) {t : ℝ} (ht : 0 ≤ t) :
    ceLoss A y X + lam * nuclearNorm X ≤ ceLoss A y (t • X') + lam * t := by
  obtain ⟨U, V, rfl, hmin⟩ := h.exists_isDecayedMin hlam
  exact hmin.ceLoss_add_le hlam hX' ht

omit [Fintype k] in
/-- **A trained product has no more loss than the competitor at its own norm**
(`review/flat_spectrum.md`, item XE5): `L(X) ≤ L(‖X‖_S1 X')` for `λ ≥ 0` and `‖X'‖_S1 ≤ 1`. -/
theorem IsNuclearMin.ceLoss_le_smul (hlam : 0 ≤ lam) (h : IsNuclearMin (ceLoss A y) lam X)
    (hX' : nuclearNorm X' ≤ 1) : ceLoss A y X ≤ ceLoss A y (nuclearNorm X • X') := by
  obtain ⟨U, V, rfl, hmin⟩ := h.exists_isDecayedMin hlam
  exact hmin.ceLoss_le_smul hlam hX'

omit [Fintype k] in
/-- **The gaps of a trained product against the margin of a competitor**
(`review/flat_spectrum.md`, item XE5): if `‖X'‖_S1 ≤ 1` and every gap of `X'` is at least
`μ'`, every gap `g` of a trained product has
`log (1 + exp (−g)) ≤ |S| (|C| − 1) exp (−‖X‖_S1 μ')`. -/
theorem IsNuclearMin.log_le_mul_exp (hlam : 0 ≤ lam) (h : IsNuclearMin (ceLoss A y) lam X)
    (hX' : nuclearNorm X' ≤ 1) (hμ' : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c) {s : S} {c : C}
    (hc : c ≠ y s) :
    Real.log (1 + Real.exp (-(gap A y X s c))) ≤
      Fintype.card S * (Fintype.card C - 1) * Real.exp (-(nuclearNorm X * μ')) := by
  obtain ⟨U, V, rfl, hmin⟩ := h.exists_isDecayedMin hlam
  exact hmin.log_le_mul_exp hlam hX' hμ' hc

omit [Fintype k] in
/-- **A trained product has the margin of the competitor up to a constant**
(`review/flat_spectrum.md`, item XE5): with `κ = |S| (|C| − 1)`, `‖X'‖_S1 ≤ 1` and every gap of
`X'` at least `μ'`, once `log (2κ) ≤ ‖X‖_S1 μ'` every gap of a trained product is at least
`‖X‖_S1 μ' − log (2κ)`. -/
theorem IsNuclearMin.sub_log_le_gap (hlam : 0 ≤ lam) (h : IsNuclearMin (ceLoss A y) lam X)
    (hX' : nuclearNorm X' ≤ 1) (hμ' : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c)
    (hB : Real.log (2 * (Fintype.card S * (Fintype.card C - 1))) ≤ nuclearNorm X * μ')
    {s : S} {c : C} (hc : c ≠ y s) :
    nuclearNorm X * μ' - Real.log (2 * (Fintype.card S * (Fintype.card C - 1))) ≤
      gap A y X s c := by
  obtain ⟨U, V, rfl, hmin⟩ := h.exists_isDecayedMin hlam
  exact hmin.sub_log_le_gap hlam hX' hμ' hB hc

variable [DecidableEq m]

omit [Fintype k] in
/-- **Trained products of the cross-entropy exist** for every weight decay `λ > 0`
(`review/flat_spectrum.md`, items XE3 and XE5). -/
theorem exists_isNuclearMin_ceLoss (A : S → C → Matrix m n ℂ) (y : S → C) (hlam : 0 < lam) :
    ∃ X : Matrix m n ℂ, IsNuclearMin (ceLoss A y) lam X := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_ceLoss (k := n) A y hlam (Or.inr le_rfl)
  exact ⟨_, h.isNuclearMin hlam.le⟩

omit [Fintype k] in
/-- Trained products of the cross-entropy exist along every sequence `λ_j > 0`. -/
lemma exists_seq_isNuclearMin_ceLoss (A : S → C → Matrix m n ℂ) (y : S → C) {lams : ℕ → ℝ}
    (hpos : ∀ j, 0 < lams j) :
    ∃ X : ℕ → Matrix m n ℂ, ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j) :=
  ⟨_, fun j => (exists_isNuclearMin_ceLoss A y (hpos j)).choose_spec⟩

/-! The witness: the linear loss `−Re X` of a `1 × 1` matrix with `λ = 1`, whose trained fit
`1 · 1` gives the trained product `1`, and the sample `signProbe` of item XE3. -/

example := exists_mul_eq_weightDecay_eq (1 : Matrix (Fin 1) (Fin 1) ℂ)

/-- The product of the trained fit `1 · 1` is a trained product, and so a product again. -/
example := (isDecayedMin_linear_one.isNuclearMin zero_le_one).exists_isDecayedMin zero_le_one

example {X : Matrix (Fin 1) (Fin 1) ℂ} := isNuclearMin_iff
  (L := fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing (-1) X) (X := X) zero_le_one

/-- A statement about every sequence of trained fits, passed to the trained products. -/
example {X : ℕ → Matrix (Fin 1) (Fin 1) ℂ}
    (hX : ∀ j, IsNuclearMin (ceLoss signProbe fun _ => true) 1 (X j)) :
    ∀ j, ceLoss signProbe (fun _ => true) (X j) ≤
      ceLoss signProbe (fun _ => true) (nuclearNorm (X j) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) :=
  forall_seq_isNuclearMin (lams := fun _ => 1) (fun _ => zero_le_one)
    (P := fun X => ∀ j, ceLoss signProbe (fun _ => true) (X j) ≤
      ceLoss signProbe (fun _ => true) (nuclearNorm (X j) • (1 : Matrix (Fin 1) (Fin 1) ℂ)))
    (fun _ _ h j => (h j).ceLoss_le_smul zero_le_one nuclearNorm_one_fin_one.le) hX

example := exists_seq_isNuclearMin_ceLoss signProbe (fun _ => true)
  (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity

/-- Item XE5 for a trained product of the witness, against the competitor `1`. -/
example : ∃ X : Matrix (Fin 1) (Fin 1) ℂ,
    ceLoss signProbe (fun _ => true) X + 1 * nuclearNorm X ≤
        ceLoss signProbe (fun _ => true) ((3 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) + 1 * 3 ∧
      ceLoss signProbe (fun _ => true) X ≤
        ceLoss signProbe (fun _ => true) (nuclearNorm X • (1 : Matrix (Fin 1) (Fin 1) ℂ)) ∧
      Real.log (1 + Real.exp (-(gap signProbe (fun _ => true) X 0 false))) ≤
        (Fintype.card (Fin 1) : ℝ) * (Fintype.card Bool - 1) *
          Real.exp (-(nuclearNorm X * 2)) := by
  obtain ⟨X, h⟩ := exists_isNuclearMin_ceLoss signProbe (fun _ => true) one_pos
  exact ⟨X, h.ceLoss_add_le zero_le_one nuclearNorm_one_fin_one.le (by norm_num),
    h.ceLoss_le_smul zero_le_one nuclearNorm_one_fin_one.le,
    h.log_le_mul_exp zero_le_one nuclearNorm_one_fin_one.le le_gap_signProbe_one (by simp)⟩

/-- The margin bound, for the trained products whose norm has passed `log (2κ) / μ'`. -/
example {X : Matrix (Fin 1) (Fin 1) ℂ} (h : IsNuclearMin (ceLoss signProbe fun _ => true) 1 X)
    (hB : Real.log (2 * ((Fintype.card (Fin 1) : ℝ) * (Fintype.card Bool - 1))) ≤
      nuclearNorm X * 2) :=
  h.sub_log_le_gap zero_le_one nuclearNorm_one_fin_one.le le_gap_signProbe_one hB
    (s := 0) (c := false) (by simp)

end Transformer.Grokking.FlatSpectrum.Training
