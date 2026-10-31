module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Margins
public import Transformer.Grokking.FlatSpectrum.RankRecovery

/-!
# From arbitrary-rank matrix recovery to joint classification

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*, item 5.
The ideal learner minimizes the sum of the feature kernels' nuclear norms while matching
every observed feature value. Unique recovery of each kernel gives unique recovery of the
whole bank, and a jointly separating answer code then gives correct classification.
The existing arbitrary-rank certificates supply the component recovery hypotheses.
This objective and its global optimality remain explicit; neither is attributed to the
finite iterates or weight decay of the nonlinear Rotate90 model.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix
open scoped BigOperators

variable {m n R : Type*} [Fintype m] [Fintype n] [Fintype R] [DecidableEq n]

/-- **Joint recovery**, source item 5. The unweighted sum of kernel nuclear norms has
exactly the true bank as its minimum if each component has that unique minimum. -/
theorem bank_minimum_iff {K X : R → Matrix m n ℂ} {Ω : Set (m × n)}
    (hrec : ∀ r, ∀ Y ∈ {Y' : Matrix m n ℂ | ∀ ij ∈ Ω, Y' ij.1 ij.2 = K r ij.1 ij.2},
      IsMinOn nuclearNorm {Y' | ∀ ij ∈ Ω, Y' ij.1 ij.2 = K r ij.1 ij.2} Y ↔ Y = K r)
    (hfit : ∀ r, ∀ ij ∈ Ω, X r ij.1 ij.2 = K r ij.1 ij.2) :
    IsMinOn (fun Y => ∑ r, nuclearNorm (Y r))
      {Y : R → Matrix m n ℂ | ∀ r, ∀ ij ∈ Ω, Y r ij.1 ij.2 = K r ij.1 ij.2} X ↔ X = K := by
  have hmin r : IsMinOn nuclearNorm
      {Y' | ∀ ij ∈ Ω, Y' ij.1 ij.2 = K r ij.1 ij.2} (K r) :=
    (hrec r (K r) (fun _ _ => rfl)).mpr rfl
  constructor
  · intro hX
    have hlow r : nuclearNorm (K r) ≤ nuclearNorm (X r) :=
      (isMinOn_iff.mp (hmin r)) _ (hfit r)
    have hsum : ∑ r, (nuclearNorm (X r) - nuclearNorm (K r)) = 0 := by
      apply le_antisymm
      · rw [Finset.sum_sub_distrib]
        have h := (isMinOn_iff.mp hX) K (fun _ _ _ => rfl)
        linarith
      · exact Finset.sum_nonneg fun r _ => sub_nonneg.mpr (hlow r)
    have heq r : nuclearNorm (X r) = nuclearNorm (K r) :=
      sub_eq_zero.mp ((Finset.sum_eq_zero_iff_of_nonneg
        (fun r _ => sub_nonneg.mpr (hlow r))).mp hsum r (Finset.mem_univ _))
    funext r
    apply (hrec r (X r) (hfit r)).mp
    refine isMinOn_iff.mpr fun Y hY => ?_
    rw [heq]
    exact (isMinOn_iff.mp (hmin r)) Y hY
  · rintro rfl
    refine isMinOn_iff.mpr fun Y hY => ?_
    exact Finset.sum_le_sum fun r _ => (isMinOn_iff.mp (hmin r)) (Y r) (hY r)

/-- **Recovery implies classification**, source item 5. Kernels may have any ranks and
individual coordinates may collide. Global minimum, observed agreement, and joint code
separation are all hypotheses; the conclusion concerns every input, including unseen ones. -/
theorem classification_of_bank_minimum {A : Type*} {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (C : A → R → ℂ) (hC : ∀ a r, Complex.normSq (C a r) = 1)
    (hsep : ∀ a b, a ≠ b → ∃ r, 0 < w r ∧ C a r ≠ C b r)
    (f : m → n → A) {X : R → Matrix m n ℂ} {Ω : Set (m × n)}
    (hrec : ∀ r, ∀ Y ∈ {Y' : Matrix m n ℂ | ∀ ij ∈ Ω, Y' ij.1 ij.2 = C (f ij.1 ij.2) r},
      IsMinOn nuclearNorm {Y' | ∀ ij ∈ Ω, Y' ij.1 ij.2 = C (f ij.1 ij.2) r} Y ↔
        Y = fun i j => C (f i j) r)
    (hfit : ∀ r, ∀ ij ∈ Ω, X r ij.1 ij.2 = C (f ij.1 ij.2) r)
    (hmin : IsMinOn (fun Y => ∑ r, nuclearNorm (Y r))
      {Y : R → Matrix m n ℂ | ∀ r, ∀ ij ∈ Ω, Y r ij.1 ij.2 = C (f ij.1 ij.2) r} X) :
    ∀ i j c, c ≠ f i j →
      score w C (fun r => X r i j) c < score w C (fun r => X r i j) (f i j) := by
  have hX := (bank_minimum_iff (K := fun r i j => C (f i j) r)
    (X := X) (Ω := Ω) hrec hfit).mp hmin
  subst X
  intro i j c hc
  exact (codeScore_lt_self_iff hw C hC (f i j) c).mpr (hsep _ _ hc.symm)

/-- The existing **arbitrary-rank dual certificates** establish joint recovery, source
item 5. Each kernel may have its own singular vectors and contraction certificate. -/
theorem bank_minimum_iff_of_certificates {q : Type*} [Fintype q] [DecidableEq m]
    [DecidableEq q] {K X W : R → Matrix m n ℂ} {U : R → Matrix m q ℂ}
    {V : R → Matrix n q ℂ} {Ω : Set (m × n)} {c : ℝ}
    (hU : ∀ r, (U r)ᴴ * U r = 1) (hV : ∀ r, (V r)ᴴ * V r = 1)
    (hKU : ∀ r, U r * (U r)ᴴ * K r = K r)
    (hK : ∀ r, ((U r * (V r)ᴴ)ᴴ * K r).trace = nuclearNorm (K r))
    (hY : ∀ r i j, (i, j) ∉ Ω → (U r * (V r)ᴴ + W r) i j = 0)
    (hWU : ∀ r, (U r)ᴴ * W r = 0) (hWV : ∀ r, W r * V r = 0)
    (hc : 1 < c) (hW : ∀ r, IsContraction ((c : ℂ) • W r))
    (hT : ∀ r (A : Matrix n q ℂ) (B : Matrix m q ℂ),
      (∀ ij ∈ Ω, (U r * Aᴴ + B * (V r)ᴴ) ij.1 ij.2 = 0) → U r * Aᴴ + B * (V r)ᴴ = 0)
    (hfit : ∀ r, ∀ ij ∈ Ω, X r ij.1 ij.2 = K r ij.1 ij.2) :
    IsMinOn (fun Y => ∑ r, nuclearNorm (Y r))
      {Y : R → Matrix m n ℂ | ∀ r, ∀ ij ∈ Ω, Y r ij.1 ij.2 = K r ij.1 ij.2} X ↔ X = K := by
  apply bank_minimum_iff (hfit := hfit)
  intro r Y hYfit
  exact isMinOn_iff_of_certificate (hU r) (hV r) (hKU r) (hK r) (hY r) (hWU r)
    (hWV r) hc (hW r) (hT r) hYfit

/-- Full observation supplies a concrete recovery witness for every matrix, source item 5's
examples. This elementary lemma is used only to check nonempty theorem hypotheses. -/
theorem full_observation_minimum_iff (K Y : Matrix m n ℂ)
    (hY : ∀ ij ∈ (Set.univ : Set (m × n)), Y ij.1 ij.2 = K ij.1 ij.2) :
    IsMinOn nuclearNorm {Y' | ∀ ij ∈ (Set.univ : Set (m × n)),
      Y' ij.1 ij.2 = K ij.1 ij.2} Y ↔ Y = K := by
  have h : Y = K := by ext i j; exact hY (i, j) (Set.mem_univ _)
  subst Y
  refine ⟨fun _ => rfl, fun _ => isMinOn_iff.mpr fun Y hY => ?_⟩
  have h : Y = K := by ext i j; exact hY (i, j) (Set.mem_univ _)
  rw [h]

example := full_observation_minimum_iff (1 : Matrix (Fin 2) (Fin 2) ℂ) 1 (fun _ _ => rfl)

example := bank_minimum_iff (K := fun _ : Bool => (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (X := fun _ => 1) (Ω := Set.univ)
  (fun _ Y hY => full_observation_minimum_iff _ Y hY) (fun _ _ _ => rfl)

example := classification_of_bank_minimum
  (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) binaryCode binaryCode_unit
  (fun a b hab => (codeScore_lt_self_iff
    (fun _ : Bool => (by norm_num : (0 : ℝ) ≤ 1 / 2)) binaryCode binaryCode_unit a b).mp
      ((unique_classification_iff_injective
        (fun _ : Bool => (by norm_num : (0 : ℝ) < 1 / 2)) binaryCode binaryCode_unit).mpr
          binaryCode_injective a b hab))
  (fun a b : Bool => (a, b)) (X := fun r a b => binaryCode (a, b) r) (Ω := Set.univ)
  (fun r Y hY => full_observation_minimum_iff (fun a b => binaryCode (a, b) r) Y hY)
  (fun _ _ _ => rfl)
  ((bank_minimum_iff (K := fun r a b => binaryCode (a, b) r) (Ω := Set.univ)
    (fun r Y hY => full_observation_minimum_iff (fun a b => binaryCode (a, b) r) Y hY)
    (fun _ _ _ => rfl)).mpr rfl)

/-- A rank-two bank satisfies the actual certificates: two copies of the identity with
full observation, orthonormal singular vectors, and zero transverse certificate. -/
example := bank_minimum_iff_of_certificates (R := Bool) (q := Fin 2)
  (K := fun _ => (1 : Matrix (Fin 2) (Fin 2) ℂ)) (X := fun _ => 1)
  (U := fun _ => 1) (V := fun _ => 1) (W := fun _ => 0) (Ω := Set.univ)
  (fun _ => by simp) (fun _ => by simp) (fun _ => by simp)
  (fun _ => by
    have hn := nuclearNorm_eq_of_diagonalization
      (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (W := 1) (d := fun _ => 1)
      (fun _ => zero_le_one) (by simp) (by simp)
    rw [hn]
    norm_num)
  (fun _ _ _ h => absurd (Set.mem_univ _) h) (fun _ => by simp) (fun _ => by simp)
  one_lt_two (fun _ => by rw [smul_zero]; exact isContraction_zero)
  (fun _ _ _ h => by ext i j; exact h (i, j) (Set.mem_univ _)) (fun _ _ _ => rfl)

end Transformer.Grokking.FlatSpectrum.FeatureCodes
