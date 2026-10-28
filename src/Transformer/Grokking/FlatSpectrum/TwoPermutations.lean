module

public import Transformer.Grokking.FlatSpectrum.OnesRecovery
public import Transformer.Grokking.FlatSpectrum.CyclePoincare

/-!
# Two permutations whose ratio is one cycle recover the all-ones matrix

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let `σ, τ`
be permutations of an `N`-set with `τ σ⁻¹` a single `N`-cycle, written `τ σ⁻¹ (e l) = e (l + 1)`
for a bijection `e : ℤ/N ≃ n`. The certificate is `Y = (P_σ + P_τ) / 2`: it lives on the two
graphs and has unit row and column sums. With `w = v − mean v`,
`(Y − Q) v = (w ∘ σ + w ∘ τ) / 2`, so by the parallelogram law
`‖(Y − Q) v‖² = ‖w‖² − D / 4`, `D = ∑_i |w (σ i) − w (τ i)|²`, and the Poincaré inequality
along the cycle gives `‖w‖² ≤ N² D`; hence `‖(Y − Q) v‖² ≤ (1 − 1/(4N²)) ‖v‖²`. Entries
`r_i + s_j` vanishing on both graphs force `s` constant along the cycle, hence everywhere.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The average `(P_σ + P_τ) / 2` of two permutation matrices. -/
noncomputable def permAvg (σ τ : Equiv.Perm n) : Matrix n n ℂ :=
  of fun i j => (if σ i = j then (1 / 2 : ℂ) else 0) + if τ i = j then 1 / 2 else 0

lemma permAvg_mulVec (σ τ : Equiv.Perm n) (v : n → ℂ) (i : n) :
    (permAvg σ τ *ᵥ v) i = (v (σ i) + v (τ i)) / 2 := by
  simp only [permAvg, mulVec, dotProduct, of_apply, add_mul, ite_mul, zero_mul, sum_add_distrib,
    sum_ite_eq, mem_univ, ↓reduceIte]
  ring

lemma sum_permAvg_row (σ τ : Equiv.Perm n) (i : n) : ∑ j, permAvg σ τ i j = 1 := by
  simp only [permAvg, of_apply, sum_add_distrib, sum_ite_eq, mem_univ, ↓reduceIte]
  norm_num

lemma sum_permAvg_col (σ τ : Equiv.Perm n) (j : n) : ∑ i, permAvg σ τ i j = 1 := by
  have h (π : Equiv.Perm n) : ∑ i, (if π i = j then (1 / 2 : ℂ) else 0) = 1 / 2 :=
    (Equiv.sum_comp π fun k => if k = j then (1 / 2 : ℂ) else 0).trans (by simp)
  simp only [permAvg, of_apply, sum_add_distrib, h]
  norm_num

omit [Fintype n] in
/-- `Y = (P_σ + P_τ) / 2` vanishes off every `Ω` that contains both graphs. -/
lemma permAvg_eq_zero {σ τ : Equiv.Perm n} {Ω : Set (n × n)} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hτ : ∀ i, (i, τ i) ∈ Ω) (i j : n) (hij : (i, j) ∉ Ω) : permAvg σ τ i j = 0 := by
  have h1 : σ i ≠ j := fun h => hij (h ▸ hσ i)
  have h2 : τ i ≠ j := fun h => hij (h ▸ hτ i)
  simp [permAvg, h1, h2]

example : permAvg (Equiv.refl (Fin 3)) (Equiv.swap 0 1) 0 2 = 0 :=
  permAvg_eq_zero (Ω := {ij | ij.2 = ij.1 ∨ ij.2 = Equiv.swap 0 1 ij.1}) (fun _ => Or.inl rfl)
    (fun _ => Or.inr rfl) 0 2 (by simp only [Set.mem_ofPred_eq, Equiv.swap_apply_left]; decide)

variable {N : ℕ} [NeZero N]

omit [Fintype n] [DecidableEq n] in
/-- If `τ σ⁻¹` is one `N`-cycle, entries `r_i + s_j` that vanish on the graphs of `σ` and `τ`
vanish everywhere. -/
lemma add_eq_zero_of_cycle {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) {r s : n → ℂ}
    (hσ : ∀ i, r i + s (σ i) = 0) (hτ : ∀ i, r i + s (τ i) = 0) (i j : n) : r i + s j = 0 := by
  have hstep (l : ZMod N) : s (e (l + 1)) = s (e l) := by
    have h1 := hσ (σ.symm (e l))
    have h2 := hτ (σ.symm (e l))
    rw [Equiv.apply_symm_apply] at h1
    rw [he] at h2
    linear_combination h2 - h1
  have hconst (t : ℕ) : s (e t) = s (e 0) := by
    induction t with
    | zero => simp
    | succ t ih => rw [Nat.cast_succ, hstep, ih]
  have hs (k : n) : s k = s (e 0) := by
    have h := hconst (e.symm k).val
    rwa [ZMod.natCast_zmod_val, Equiv.apply_symm_apply] at h
  have h := hσ i
  rw [hs] at h ⊢
  exact h

example := add_eq_zero_of_cycle (N := 2) (σ := Equiv.refl (Fin 2)) (τ := Equiv.swap 0 1)
  (Equiv.refl _) (by decide) (r := fun _ => 1) (s := fun _ => -1) (fun _ => by norm_num)
  (fun _ => by norm_num)

/-- `‖(Y − Q) v‖² ≤ (1 − 1/(4N²)) ‖v‖²` for `Y = (P_σ + P_τ) / 2` when `τ σ⁻¹` is one
`N`-cycle. -/
theorem sum_norm_sq_mulVec_permAvg_sub_le {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (v : n → ℂ) :
    ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2 ≤
      (1 - 1 / (4 * (N : ℝ) ^ 2)) * ∑ i, ‖v i‖ ^ 2 := by
  have hcard : Fintype.card n = N := by rw [← Fintype.card_congr e, ZMod.card]
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hN' : (Fintype.card n : ℂ) ≠ 0 := by rw [hcard]; exact_mod_cast NeZero.ne N
  set m : ℂ := (Fintype.card n : ℂ)⁻¹ * ∑ j, v j with hm
  set w : n → ℂ := fun j => v j - m with hw
  have hwsum : ∑ j, w j = 0 := by
    simp only [hw, sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul, hm]
    field_simp
    ring
  have hentry (i : n) : ((permAvg σ τ - meanProj n) *ᵥ v) i = (w (σ i) + w (τ i)) / 2 := by
    have hQ : (meanProj n *ᵥ v) i = m := by simp [meanProj, mulVec, dotProduct, hm, mul_sum]
    rw [sub_mulVec, Pi.sub_apply, permAvg_mulVec, hQ, hw]
    ring
  have hpar (i : n) : ‖(w (σ i) + w (τ i)) / 2‖ ^ 2 =
      (‖w (σ i)‖ ^ 2 + ‖w (τ i)‖ ^ 2) / 2 - ‖w (σ i) - w (τ i)‖ ^ 2 / 4 := by
    have h := parallelogram_law_with_norm ℝ (w (σ i)) (w (τ i))
    rw [norm_div, Complex.norm_ofNat, div_pow]
    linarith
  have hcomp (π : Equiv.Perm n) : ∑ i, ‖w (π i)‖ ^ 2 = ∑ i, ‖w i‖ ^ 2 :=
    Equiv.sum_comp π fun i => ‖w i‖ ^ 2
  have hS : ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2 =
      ∑ i, ‖w i‖ ^ 2 - (∑ i, ‖w (σ i) - w (τ i)‖ ^ 2) / 4 := by
    simp only [hentry, hpar, sum_sub_distrib, ← sum_div, sum_add_distrib, hcomp]
    ring
  have hD : ∑ i, ‖w i‖ ^ 2 ≤ N * ((N : ℝ) - 1) * ∑ i, ‖w (σ i) - w (τ i)‖ ^ 2 := by
    have hu : ∑ l, w (e l) = 0 := (Equiv.sum_comp e w).trans hwsum
    have h := sum_norm_sq_le_cycle hu
    have h1 : ∑ l, ‖w (e l)‖ ^ 2 = ∑ i, ‖w i‖ ^ 2 := Equiv.sum_comp e fun i => ‖w i‖ ^ 2
    have h2 := Equiv.sum_comp (e.trans σ.symm) fun i => ‖w (σ i) - w (τ i)‖ ^ 2
    simp only [Equiv.trans_apply, Equiv.apply_symm_apply, he] at h2
    have h3 : ∑ l, ‖w (e (l + 1)) - w (e l)‖ ^ 2 = ∑ l, ‖w (e l) - w (e (l + 1))‖ ^ 2 :=
      sum_congr rfl fun l _ => by rw [norm_sub_rev]
    rw [h1, h3, h2] at h
    exact h
  have hwv : ∑ i, ‖w i‖ ^ 2 ≤ ∑ i, ‖v i‖ ^ 2 := by
    have h := sum_norm_sq_add_const hwsum m
    simp only [hw, sub_add_cancel] at h
    rw [h]
    exact le_add_of_nonneg_right (by positivity)
  set W2 := ∑ i, ‖w i‖ ^ 2
  set D := ∑ i, ‖w (σ i) - w (τ i)‖ ^ 2
  have hD0 : 0 ≤ D := sum_nonneg fun _ _ => sq_nonneg _
  have hW2 : W2 ≤ (N : ℝ) ^ 2 * D := by nlinarith
  have hε : W2 / (4 * (N : ℝ) ^ 2) ≤ D / 4 := by
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  have hε1 : 1 / (4 * (N : ℝ) ^ 2) ≤ 1 := by
    rw [div_le_one (by positivity)]
    nlinarith [show (1 : ℝ) ≤ N from by exact_mod_cast Nat.one_le_iff_ne_zero.mpr (NeZero.ne N)]
  rw [hS]
  calc W2 - D / 4 ≤ (1 - 1 / (4 * (N : ℝ) ^ 2)) * W2 := by
        rw [sub_mul, one_mul, one_div_mul_eq_div]
        linarith
    _ ≤ (1 - 1 / (4 * (N : ℝ) ^ 2)) * ∑ i, ‖v i‖ ^ 2 :=
        mul_le_mul_of_nonneg_left hwv (by linarith)

example := sum_norm_sq_mulVec_permAvg_sub_le (N := 2) (σ := Equiv.refl (Fin 2))
  (τ := Equiv.swap 0 1) (Equiv.refl _) (by decide) ![1, 0]

/-- `c (Y − Q)` is a contraction for `c = √(1 + 1/(4N²))`. -/
theorem isContraction_permAvg {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) :
    IsContraction ((√(1 + 1 / (4 * (N : ℝ) ^ 2)) : ℂ) • (permAvg σ τ - meanProj n)) := by
  refine isContraction_of_forall_sum_norm_sq_le fun v => ?_
  have h := sum_norm_sq_mulVec_permAvg_sub_le e he v
  have hV : 0 ≤ ∑ i, ‖v i‖ ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  simp only [smul_mulVec, Pi.smul_apply, smul_eq_mul, norm_mul, Complex.norm_real,
    Real.norm_of_nonneg (Real.sqrt_nonneg _), mul_pow, Real.sq_sqrt (by positivity :
      (0 : ℝ) ≤ 1 + 1 / (4 * (N : ℝ) ^ 2)), ← mul_sum]
  set ε := 1 / (4 * (N : ℝ) ^ 2)
  have h2 := mul_le_mul_of_nonneg_left h (by positivity : (0 : ℝ) ≤ 1 + ε)
  nlinarith [mul_nonneg (sq_nonneg ε) hV]

example := isContraction_permAvg (N := 2) (σ := Equiv.refl (Fin 2)) (τ := Equiv.swap 0 1)
  (Equiv.refl _) (by decide)

/-- **Two permutations recover `J`**, `review/flat_spectrum.md`, *Exact recovery of a rank-one
kernel*: if `Ω` contains the graphs of `σ` and `τ` and `τ σ⁻¹` is one `N`-cycle, the all-ones
matrix is the unique minimum of the nuclear norm among the matrices equal to it on `Ω`. -/
theorem isMinOn_onesMatrix_iff_of_cycle {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) {Ω : Set (n × n)} (hσ : ∀ i, (i, σ i) ∈ Ω)
    (hτ : ∀ i, (i, τ i) ∈ Ω) {X : Matrix n n ℂ}
    (hX : ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
    IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = onesMatrix n ij.1 ij.2} X ↔
      X = onesMatrix n := by
  have : Nonempty n := ⟨e 0⟩
  have hc : 1 < √(1 + 1 / (4 * (N : ℝ) ^ 2)) := by
    rw [Real.lt_sqrt zero_le_one, one_pow]
    have : (0 : ℝ) < 1 / (4 * (N : ℝ) ^ 2) := by
      have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
      positivity
    linarith
  exact isMinOn_onesMatrix_iff (permAvg_eq_zero hσ hτ) (sum_permAvg_row σ τ)
    (sum_permAvg_col σ τ) hc (isContraction_permAvg e he)
    (fun r s h => add_eq_zero_of_cycle e he (fun i => h _ (hσ i)) fun i => h _ (hτ i)) hX

example := (isMinOn_onesMatrix_iff_of_cycle (N := 2) (σ := Equiv.refl (Fin 2))
  (τ := Equiv.swap 0 1) (Equiv.refl _) (by decide) (Ω := Set.univ) (fun _ => Set.mem_univ _)
  (fun _ => Set.mem_univ _) (X := onesMatrix (Fin 2)) fun _ _ => rfl).mpr rfl

end Transformer.Grokking.FlatSpectrum
