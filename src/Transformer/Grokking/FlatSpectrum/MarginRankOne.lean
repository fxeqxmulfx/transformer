module

public import Transformer.Grokking.FlatSpectrum.MarginLargest
public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery

/-!
# The failure margin for every rank-one table with entries of modulus one

Source: `review/flat_spectrum.md`, *The failure margin*, item 4 (added). Rescaling by
`diag a · X · diag b` with `|a i| = |b j| = 1` keeps the nuclear norm (item D.4,
`nuclearNorm_diagonal_mul_mul_diagonal`), so the completion norms of `a bᵀ` from any `Ω` are those
of `J` (`image_nuclearNorm_vecMulVec`). Every `u vᵀ` with `|u i v j| = 1` is such an `a bᵀ`
(`exists_norm_eq_one_of_vecMulVec`), so its completion norms are those of `J` too
(`image_nuclearNorm_vecMulVec_of_norm`), and every bound of the failure margin, with the pattern
that attains it, carries over from `J` (`margin_vecMulVec_of_onesMatrix`). So for every `N × N`
table `u vᵀ` with entries of modulus one, whose nuclear norm is `N`: some completion from a
pattern without a permutation graph is shorter than `‖u vᵀ‖_S1` by more than `1/(2N)`
(`exists_nuclearNorm_le_sqrt_vecMulVec_of_not_perm`); for `a + b = N + 1` with `a ≤ b ≤ a + 1`,
the largest least completion norm over these patterns is `√(a (a − 1)) + √(b (b − 1))`
(`failureMargin_vecMulVec`): `√(N² − 1)` for odd `N` (`failureMargin_odd_vecMulVec`) and
`√2 + √6` for `N = 4` (`failureMargin_four_vecMulVec`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **Rescaling keeps the completion norms**: for `|a i| = |b j| = 1`, the nuclear norms of the
matrices equal to `a bᵀ` on `Ω` are those of the matrices equal to `J` on `Ω`. -/
theorem image_nuclearNorm_vecMulVec {a b : n → ℂ} (ha : ∀ i, ‖a i‖ = 1) (hb : ∀ j, ‖b j‖ = 1)
    (Ω : Set (n × n)) :
    nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2} =
      nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2} := by
  have hsa (i : n) : star (a i) * a i = 1 := by
    rw [Complex.star_def, Complex.conj_mul', ha i]
    norm_num
  have hsb (j : n) : star (b j) * b j = 1 := by
    rw [Complex.star_def, Complex.conj_mul', hb j]
    norm_num
  have ha' (i : n) : ‖star a i‖ = 1 := by rw [Pi.star_apply, norm_star, ha i]
  have hb' (j : n) : ‖star b j‖ = 1 := by rw [Pi.star_apply, norm_star, hb j]
  ext x
  constructor
  · rintro ⟨X, hX, rfl⟩
    refine ⟨diagonal (star a) * X * diagonal (star b), fun ij hij => ?_,
      nuclearNorm_diagonal_mul_mul_diagonal ha' hb' X⟩
    rw [mul_diagonal, diagonal_mul, Pi.star_apply, Pi.star_apply, hX ij hij, vecMulVec_apply,
      onesMatrix, of_apply]
    linear_combination (star (b ij.2) * b ij.2) * hsa ij.1 + hsb ij.2
  · rintro ⟨Y, hY, rfl⟩
    refine ⟨diagonal a * Y * diagonal b, fun ij hij => ?_,
      nuclearNorm_diagonal_mul_mul_diagonal ha hb Y⟩
    rw [mul_diagonal, diagonal_mul, hY ij hij, vecMulVec_apply, onesMatrix, of_apply, mul_one]

example := image_nuclearNorm_vecMulVec (n := Fin 2) (a := fun _ => -1) (b := fun _ => Complex.I)
  (by simp) (by simp) ∅

/-- **The completion norms of a rank-one table with entries of modulus one are those of `J`**:
if every `|u i v j| = 1`, the nuclear norms of the matrices equal to `u vᵀ` on `Ω` are those of
the matrices equal to `J` on `Ω`. -/
theorem image_nuclearNorm_vecMulVec_of_norm {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1)
    (Ω : Set (n × n)) :
    nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2} =
      nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2} := by
  rcases isEmpty_or_nonempty n with hn | hn
  · congr 1
    ext X
    exact iff_of_true (fun ij => (IsEmpty.false ij.1).elim) (fun ij => (IsEmpty.false ij.1).elim)
  · obtain ⟨a, b, ha, hb, hab⟩ := exists_norm_eq_one_of_vecMulVec h
    rw [← hab]
    exact image_nuclearNorm_vecMulVec ha hb Ω

example := image_nuclearNorm_vecMulVec_of_norm (n := Fin 2) (u := fun _ => Complex.I)
  (v := fun _ => -Complex.I) (fun _ _ => by simp [vecMulVec_apply]) ∅

/-- **The failure margins of `J` hold for every rank-one table with entries of modulus one**: if
every pattern without a permutation graph has a completion of `J` of norm at most `c`, and `c` is
the least completion norm of `J` from one of them, the same holds for `u vᵀ` with every
`|u i v j| = 1`. -/
theorem margin_vecMulVec_of_onesMatrix {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) {c : ℝ}
    (hJ : (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
        ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) ∧ nuclearNorm X ≤ c) ∧
      ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
        IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = onesMatrix n ij.1 ij.2}) c) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
        ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2) ∧ nuclearNorm X ≤ c) ∧
      ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
        IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2}) c := by
  obtain ⟨h₁, Ω, hΩ, hleast⟩ := hJ
  refine ⟨fun Ω' hΩ' => ?_, Ω, hΩ, by rwa [image_nuclearNorm_vecMulVec_of_norm h]⟩
  obtain ⟨X, hX, hle⟩ := h₁ Ω' hΩ'
  obtain ⟨Y, hY, hYX⟩ : nuclearNorm X ∈
      nuclearNorm '' {X | ∀ ij ∈ Ω', X ij.1 ij.2 = vecMulVec u v ij.1 ij.2} := by
    rw [image_nuclearNorm_vecMulVec_of_norm h]
    exact ⟨X, hX, rfl⟩
  exact ⟨Y, hY, hYX ▸ hle⟩

example := margin_vecMulVec_of_onesMatrix (n := Fin 3) (u := fun _ => Complex.I)
  (v := fun _ => -Complex.I) (fun _ _ => by simp [vecMulVec_apply]) (failureMargin_odd (by decide))

/-- **The margin for every table of rank one with entries of modulus one**: if `Ω` contains no
permutation graph, some `X` equal to `u vᵀ` on `Ω` has `‖X‖_S1 ≤ √(N² − 1)` and
`‖X‖_S1 < ‖u vᵀ‖_S1 − 1/(2N)`. -/
theorem exists_nuclearNorm_le_sqrt_vecMulVec_of_not_perm {u v : n → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) {Ω : Set (n × n)}
    (hΩ : ¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) :
    ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2) ∧
      nuclearNorm X ≤ √((Fintype.card n : ℝ) ^ 2 - 1) ∧
      nuclearNorm X < nuclearNorm (vecMulVec u v) - 1 / (2 * Fintype.card n) := by
  have : Nonempty n := not_isEmpty_iff.mp fun _ => hΩ ⟨1, fun i => (IsEmpty.false i).elim⟩
  obtain ⟨a, b, ha, hb, hab⟩ := exists_norm_eq_one_of_vecMulVec h
  obtain ⟨X, hX, hle⟩ := exists_nuclearNorm_le_sqrt_of_not_perm hΩ
  obtain ⟨Y, hY, hYX⟩ : nuclearNorm X ∈
      nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2} := by
    rw [image_nuclearNorm_vecMulVec_of_norm h]
    exact ⟨X, hX, rfl⟩
  rw [← hab, nuclearNorm_vecMulVec_of_norm_eq_one ha hb, hab]
  exact ⟨Y, hY, hYX ▸ hle⟩

example := exists_nuclearNorm_le_sqrt_vecMulVec_of_not_perm (n := Fin 2) (u := fun _ => 1)
  (v := fun _ => 1) (fun _ _ => by simp [vecMulVec_apply]) (Ω := ∅) fun ⟨_, h⟩ => h 0

/-- **The largest margin for every table of rank one with entries of modulus one**:
`failureMargin` for `u vᵀ` in place of `J`. -/
theorem failureMargin_vecMulVec [Nonempty n] {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1)
    {a b : ℕ} (hab : a + b = Fintype.card n + 1) (hba : a ≤ b) (hb : b ≤ a + 1) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2) ∧
        nuclearNorm X ≤ √((a : ℝ) * (a - 1)) + √((b : ℝ) * (b - 1))) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2})
        (√((a : ℝ) * (a - 1)) + √((b : ℝ) * (b - 1))) :=
  margin_vecMulVec_of_onesMatrix h (failureMargin Fintype.card_pos hab hba hb)

example := failureMargin_vecMulVec (n := Fin 4) (u := fun _ => Complex.I)
  (v := fun _ => -Complex.I) (fun _ _ => by simp [vecMulVec_apply]) (a := 2) (b := 3) (by simp)
  (by norm_num) (by norm_num)

/-- **The worst failure for odd `N`, for every table of rank one with entries of modulus one**:
`failureMargin_odd` for `u vᵀ` in place of `J`. -/
theorem failureMargin_odd_vecMulVec {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1)
    (hN : Odd (Fintype.card n)) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2) ∧
        nuclearNorm X ≤ √((Fintype.card n : ℝ) ^ 2 - 1)) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2})
        (√((Fintype.card n : ℝ) ^ 2 - 1)) :=
  margin_vecMulVec_of_onesMatrix h (failureMargin_odd hN)

example := failureMargin_odd_vecMulVec (n := Fin 3) (u := fun _ => -1) (v := fun _ => Complex.I)
  (fun _ _ => by simp [vecMulVec_apply]) (by decide)

/-- **The worst failure for `N = 4`, for every table of rank one with entries of modulus one**:
`failureMargin_four` for `u vᵀ` in place of `J`. -/
theorem failureMargin_four_vecMulVec {u v : n → ℂ} (h : ∀ i j, ‖vecMulVec u v i j‖ = 1)
    (hN : Fintype.card n = 4) :
    (∀ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) →
      ∃ X : Matrix n n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2) ∧
        nuclearNorm X ≤ √2 + √6) ∧
    ∃ Ω : Set (n × n), (¬ ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) ∧
      IsLeast (nuclearNorm '' {X | ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec u v ij.1 ij.2})
        (√2 + √6) :=
  margin_vecMulVec_of_onesMatrix h (failureMargin_four hN)

example := failureMargin_four_vecMulVec (n := Fin 4) (u := fun _ => Complex.I)
  (v := fun _ => Complex.I) (fun _ _ => by simp [vecMulVec_apply]) (by simp)

end Transformer.Grokking.FlatSpectrum
