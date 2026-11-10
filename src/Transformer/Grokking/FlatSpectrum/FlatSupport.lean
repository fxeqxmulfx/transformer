module

public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Transformer.Grokking.FlatSpectrum.RankMinimum

/-!
# Theorem A for a flat kernel of any shape, read off its support

Source: `review/flat_spectrum.md`, *Flat kernels of any shape* (added), item R2.

Theorem A at full row rank (`isMinOn_iff_of_full_row_rank`): if `K = U diag(s) Vᴴ` with `s > 0`
and `U` square, then `Uᴴ W = 0` forces `W = 0` in `isMinOn_iff_exists_certificate`, and `K` is a
minimum of M exactly when `U Vᴴ` vanishes off `Ω`. This mirrors `isMinOn_iff_of_full_rank`, where
`V` is square.

For a flat kernel `U Vᴴ = K / √c`: take `U = K / √c`, `V = I` if `Kᴴ K = c I`, and `U = I`,
`V = Kᴴ / √c` if `K Kᴴ = c I`. So a flat `K` is a minimum of M from `Ω` exactly when every unseen
cell is a zero of `K`: `Ω` contains the support of `K` (`IsFlat.isMinOn_iff`). A flat kernel is
not a minimum from an `Ω` that misses a nonzero cell (`IsFlat.not_isMinOn_of_ne_zero`), so one
without zero entries is a minimum from no proper `Ω` (`IsFlat.not_isMinOn`); for a square kernel
this is `not_isMinOn_of_singularValues_eq`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **Theorem A at full row rank** (`review/flat_spectrum.md`, R2): if `K = U diag(s) Vᴴ` with
`s > 0` and `U` square, `K` is a minimum of the nuclear norm among the matrices equal to it on
`Ω` exactly when `U Vᴴ` vanishes off `Ω`: `Uᴴ W = 0` forces `W = 0`. -/
theorem isMinOn_iff_of_full_row_rank {U : Matrix m m ℂ} {V : Matrix n m ℂ} (hU : Uᴴ * U = 1)
    (hV : Vᴴ * V = 1) {s : m → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∀ i j, (i, j) ∉ Ω → (U * Vᴴ) i j = 0 := by
  rw [isMinOn_iff_exists_certificate hU hV hs hK]
  have hU' : U * Uᴴ = 1 := mul_eq_one_comm.1 hU
  constructor
  · rintro ⟨W, hUW, -, -, hΩ⟩ i j hij
    have hW : W = 0 := by
      rw [← Matrix.one_mul W, ← hU', Matrix.mul_assoc, hUW, Matrix.mul_zero]
    simpa [hW] using hΩ i j hij
  · intro hΩ
    exact ⟨0, Matrix.mul_zero _, Matrix.zero_mul _, isContraction_zero,
      fun i j hij => by simpa using hΩ i j hij⟩

/-- The `1 × 2` kernel `(2, 0)` is a minimum of M from its first cell. -/
example := (isMinOn_iff_of_full_row_rank (U := (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (V := Matrix.of ![![1], ![0]]) (by simp)
  (by ext i j; simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply, Subsingleton.elim i j])
  (s := fun _ => 2) (fun _ => two_pos) rfl (Ω := {(0, 0)})).2 fun i j hij => by
    fin_cases i; fin_cases j
    · exact absurd rfl hij
    · simp [Matrix.mul_apply]

/-- **Theorem A for a flat kernel** (`review/flat_spectrum.md`, R2): a flat `K` of any shape is
a minimum of the nuclear norm among the matrices equal to it on `Ω` exactly when every unseen
cell is a zero of `K`. -/
theorem IsFlat.isMinOn_iff {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    {Ω : Set (m × n)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K ↔
      ∀ i j, (i, j) ∉ Ω → K i j = 0 := by
  have h0 : ((√c : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.2 (Real.sqrt_ne_zero'.2 hc)
  rcases h with h | h
  · have hK : K = (((√c : ℝ) : ℂ))⁻¹ • K * diagonal (fun _ : n => ((√c : ℝ) : ℂ)) *
        (1 : Matrix n n ℂ)ᴴ := by
      rw [conjTranspose_one, Matrix.mul_one, ← smul_one_eq_diagonal, Matrix.mul_smul,
        Matrix.mul_one, smul_smul, mul_inv_cancel₀ h0, one_smul]
    rw [isMinOn_iff_of_full_rank (conjTranspose_smul_mul_smul_eq_one hc h) (by simp)
      (fun _ => Real.sqrt_pos.2 hc) hK]
    simp [h0]
  · have hK : K = (1 : Matrix m m ℂ) * diagonal (fun _ : m => ((√c : ℝ) : ℂ)) *
        ((((√c : ℝ) : ℂ))⁻¹ • Kᴴ)ᴴ := by
      rw [Matrix.one_mul, ← smul_one_eq_diagonal, Matrix.smul_mul, Matrix.one_mul,
        conjTranspose_smul, conjTranspose_conjTranspose, smul_smul]
      simp [h0]
    rw [isMinOn_iff_of_full_row_rank (by simp) (conjTranspose_smul_mul_smul_eq_one hc
      (by rwa [conjTranspose_conjTranspose])) (fun _ => Real.sqrt_pos.2 hc) hK]
    simp [h0]

/-- The identity is a minimum of M exactly from the sets of cells that contain its diagonal. -/
example {Ω : Set (Fin 2 × Fin 2)} :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = (1 : Matrix (Fin 2) (Fin 2) ℂ) ij.1 ij.2}
      1 ↔ ∀ i j, (i, j) ∉ Ω → (1 : Matrix (Fin 2) (Fin 2) ℂ) i j = 0 :=
  isFlat_one.isMinOn_iff one_pos

/-- **A flat kernel is not a minimum of M when a nonzero cell is unseen**
(`review/flat_spectrum.md`, R2). -/
theorem IsFlat.not_isMinOn_of_ne_zero {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω) (hK : K i j ≠ 0) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K :=
  fun hmin => hK ((h.isMinOn_iff hc).1 hmin i j hij)

/-- The identity is not a minimum of M when no cell is seen. -/
example := isFlat_one.not_isMinOn_of_ne_zero (n := Fin 2) one_pos (Ω := ∅) (i := 0) (j := 0)
  (by simp) (by simp)

/-- **A flat kernel without zeros is never a minimum of M** (`review/flat_spectrum.md`, R2): for
every proper `Ω` some matrix equal to `K` on `Ω` has a smaller nuclear norm. -/
theorem IsFlat.not_isMinOn {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    (hK : ∀ i j, K i j ≠ 0) {Ω : Set (m × n)} (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K := by
  obtain ⟨⟨a, b⟩, hab⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  exact h.not_isMinOn_of_ne_zero hc hab (hK a b)

/-- With no cell seen, the `a³ + a b` kernel for `p = 3` is not a minimum of M. -/
example := (isFlat_cubicKernel 3).not_isMinOn (by norm_num)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

/-- The `1 × 2` kernel `(1, 1)` is a minimum of M from no proper set of cells. -/
example {Ω : Set (Fin 1 × Fin 2)} (hΩ : Ω ≠ Set.univ) :=
  isFlat_row_ones.not_isMinOn two_pos (fun i j => by fin_cases i; fin_cases j <;> simp) hΩ

/-- **M does not recover a flat kernel without zeros** (`review/flat_spectrum.md`, R2), from any
proper `Ω`. -/
theorem IsFlat.not_forall_isMinOn_iff {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    (hK : ∀ i j, K i j ≠ 0) {Ω : Set (m × n)} (hΩ : Ω ≠ Set.univ) :
    ¬ ∀ X ∈ {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix m n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K) :=
  fun hrec => h.not_isMinOn hc hK hΩ ((hrec K fun _ _ => rfl).mpr rfl)

example := (isFlat_cubicKernel 3).not_forall_isMinOn_iff (by norm_num)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

end Transformer.Grokking.FlatSpectrum
