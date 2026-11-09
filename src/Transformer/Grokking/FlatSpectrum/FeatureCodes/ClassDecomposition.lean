module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ObservableGeometry

/-!
# Supervised class subspaces and their energy

Source: `review/flat_spectrum.md`, *Class-contrast discovery and persistence*,
CG3. Centering within observed-answer and unused-class groups gives an exact
orthogonal energy split. A group of `n+1` readout rows has only `n` independent
centered directions, with explicit coefficients; energy concentration in
those directions is a separate empirical and dynamical question.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset
variable {C D K : Type*} [Fintype C] [Fintype D] [Fintype K]

/-- The scalar class mean, used coordinatewise for readout matrices, CG3. -/
def classMean (x : C → ℝ) : ℝ := (∑ c, x c) / Fintype.card C

omit [Fintype D] [Fintype K] in
/-- A computed nonempty class mean removes exactly the constant direction,
CG3. This is not an assumed centering condition. -/
theorem class_center_sum_zero [Nonempty C] (x : C → ℝ) :
    (∑ c, (x c - classMean x)) = 0 := by
  simp only [sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul, classMean]
  rw [← mul_div_assoc, mul_div_cancel_left₀ _ (by exact_mod_cast Fintype.card_ne_zero)]
  ring

omit [Fintype D] [Fintype K] in
/-- Moving the center separates within-group scatter from mean movement,
CG3. The cross term vanishes because the actual mean was subtracted. -/
theorem class_translation_energy [Nonempty C] (x : C → ℝ) (μ : ℝ) :
    (∑ c, (x c - μ) ^ 2) = (∑ c, (x c - classMean x) ^ 2) +
      (Fintype.card C : ℝ) * (classMean x - μ) ^ 2 := by
  calc
    _ = ∑ c, ((x c - classMean x) ^ 2 +
        (2 * (classMean x - μ)) * (x c - classMean x) + (classMean x - μ) ^ 2) := by
      apply sum_congr rfl
      intro c hc
      ring
    _ = _ := by
      simp only [sum_add_distrib, ← mul_sum, sum_const, card_univ,
        nsmul_eq_mul, class_center_sum_zero, mul_zero, add_zero]

omit [Fintype K] in
/-- The exact three-part class energy split, CG3. Set μ to the overall mean
to obtain active contrasts, unused contrasts and between-group energy. -/
theorem class_split_energy [Nonempty C] [Nonempty D] (x : C → ℝ) (y : D → ℝ) (μ : ℝ) :
    (∑ c : Sum C D, (Sum.elim x y c - μ) ^ 2) =
      (∑ c, (x c - classMean x) ^ 2) + (∑ d, (y d - classMean y) ^ 2) +
        ((Fintype.card C : ℝ) * (classMean x - μ) ^ 2 +
          (Fintype.card D : ℝ) * (classMean y - μ) ^ 2) := by
  rw [Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr]
  rw [class_translation_energy x μ, class_translation_energy y μ]
  ring

omit [Fintype D] [Fintype K] in
/-- Pairwise products of zero-sum coordinates equal the centered product
energy with the exact factor `2*card`, CG3. -/
theorem zero_sum_pair_product (x y : C → ℝ) (hx : ∑ c, x c = 0) (hy : ∑ c, y c = 0) :
    (∑ c, ∑ d, (x c - x d) * (y c - y d)) =
      2 * (Fintype.card C : ℝ) * ∑ c, x c * y c := by
  have inner (c : C) : (∑ d, (x c - x d) * (y c - y d)) =
      (Fintype.card C : ℝ) * (x c * y c) + ∑ d, x d * y d := by
    calc
      _ = ∑ d, (x c * y c - x c * y d - x d * y c + x d * y d) := by
        apply sum_congr rfl
        intro d hd
        ring
      _ = _ := by
        rw [sum_add_distrib, sum_sub_distrib, sum_sub_distrib]
        simp only [sum_const, card_univ, nsmul_eq_mul, ← mul_sum, ← sum_mul,
          hx, hy, mul_zero, zero_mul, sub_zero]
  simp_rw [inner]
  rw [sum_add_distrib, ← mul_sum, sum_const]
  simp only [card_univ, nsmul_eq_mul]
  ring

omit [Fintype D] [Fintype K] in
/-- Centered covariance is the average pairwise product, CG3. This converts
the pairwise CE damping theorem into damping of the whole unused group. -/
theorem centered_pair_product [Nonempty C] (x y : C → ℝ) :
    (∑ c, (x c - classMean x) * (y c - classMean y)) =
      (∑ c, ∑ d, (x c - x d) * (y c - y d)) / (2 * Fintype.card C) := by
  have h := zero_sum_pair_product (fun c => x c - classMean x)
    (fun c => y c - classMean y) (class_center_sum_zero x) (class_center_sum_zero y)
  have hd (c d : C) : (x c - classMean x - (x d - classMean x)) *
      (y c - classMean y - (y d - classMean y)) = (x c - x d) * (y c - y d) := by ring
  simp_rw [hd] at h
  rw [h, mul_div_cancel_left₀ _ (mul_ne_zero (by norm_num)
    (by exact_mod_cast Fintype.card_ne_zero))]

omit [Fintype C] [Fintype D] [Fintype K] in
/-- Explicit `n`-direction reconstruction of `n+1` centered readout rows,
CG3. For 33 observed answers this supplies 32 generating differences. -/
theorem centered_rows_from_anchor {n : ℕ} (E : Fin (n + 1) → K → ℝ) (c : Fin (n + 1)) (k : K) :
    centeredReadout E c k = ∑ i : Fin n,
      ((if c = i.succ then 1 else 0) - 1 / (n + 1 : ℝ)) * (E i.succ k - E 0 k) := by
  classical
  have hn : (n + 1 : ℝ) ≠ 0 := by positivity
  refine Fin.cases ?_ (fun j => ?_) c
  · have hz (i : Fin n) : (0 : Fin (n+1)) ≠ i.succ := (Fin.succ_ne_zero i).symm
    simp only [centeredReadout, readoutMean, Fintype.card_fin, Fin.sum_univ_succ,
      ite_eq_right (hz _), zero_sub, neg_mul, sum_neg_distrib, ← mul_sum,
      sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul]
    push_cast
    field_simp
    ring
  · simp only [centeredReadout, readoutMean, Fintype.card_fin, Fin.sum_univ_succ,
      Fin.succ_inj, sub_mul, ite_mul, one_mul, zero_mul, sum_sub_distrib,
      sum_ite_eq, mem_univ, ite_true, ← mul_sum, sum_const, card_univ,
      nsmul_eq_mul]
    push_cast
    field_simp
    ring

example := class_split_energy (fun c : Bool => if c then 3 else 1)
  (fun c : Bool => if c then 9 else 5) (9 / 2)

example := zero_sum_pair_product (fun c : Bool => if c then 1 else -1)
  (fun c : Bool => if c then 2 else -2) (by norm_num) (by norm_num)

example := centered_rows_from_anchor (K := Fin 1)
  (fun (c : Fin 3) _ => (c.val : ℝ) ^ 2) 1 0

/-- CG6: all centered readout energy can lie in the active contrast space
while a query has the wrong strict winner. Concentration needs a separate
alignment and margin argument. -/
theorem active_concentration_insufficient_witness :
    let E : Sum Bool (Fin 1) → Fin 1 → ℝ :=
      Sum.elim (fun c _ => if c then 1 else -1) (fun _ _ => 0)
    let x : Fin 1 → ℝ := fun _ => -1
    (∑ c : Bool, (E (.inl c) 0 - classMean (fun d => E (.inl d) 0)) ^ 2) = 2 ∧
      (∑ c, (centeredReadout E c 0) ^ 2) = 2 ∧
      linearScore (E (.inl true)) x < linearScore (E (.inl false)) x := by
  norm_num [classMean, centeredReadout, readoutMean, linearScore, Fintype.sum_sum_type]

end Transformer.Grokking.FlatSpectrum.FeatureCodes
