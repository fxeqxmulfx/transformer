module

public import Transformer.Grokking.FlatSpectrum.NuclearBounds
public import Transformer.Grokking.FlatSpectrum.Training.Vanishing

/-!
# Theorems A and E for training

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD7, with
Theorem A and Theorem E. A trained fit is a factorization of least weight-decayed squared loss
among those of every width, where a convergent run ends.

* Theorem A: for a square `K` without zero entries whose singular values are all equal and a
  proper `Ω`, the trained products stay a fixed distance below `‖K‖_S1`, for every weight decay
  `λ > 0` and every width (`exists_nuclearNorm_add_le_of_singularValues_eq`). Training never
  returns such a kernel, however small the weight decay.
* If `K` is the only matrix of least nuclear norm among those equal to it on `Ω`, the trained
  products converge to `K` as `λ → 0⁺` (`tendsto_mul_conjTranspose_of_forall_isMinOn`): they
  are bounded by `‖K‖_S1` and their loss vanishes, so every limit point equals `K` on `Ω` with
  nuclear norm at most `‖K‖_S1` (`tendsto_of_fitLoss_tendsto_zero`).
* Theorem E: for `u vᵀ` with `|u_i v_j| = 1` and cells `Ω` that connect the rows with the
  columns and contain a permutation, the trained products converge to `u vᵀ`
  (`tendsto_mul_conjTranspose_vecMulVec_of_connected`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

/-- **Theorem A for training** (`review/flat_spectrum.md`, Theorem A with item WD7): if the
singular values of a square `K` are all equal, no entry of `K` vanishes and `Ω` is proper, then
some `δ > 0` has `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1` for the trained fits of every weight decay `λ > 0` and
every width. -/
theorem exists_nuclearNorm_add_le_of_singularValues_eq {K : Matrix n n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (lam : ℝ), 0 < lam → ∀ (r : ℕ) (U V : Matrix n (Fin r) ℂ),
      IsDecayedMin (fitLoss Ω K) lam U V → nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K :=
  exists_nuclearNorm_add_le_of_not_isMinOn (not_isMinOn_of_singularValues_eq h hK hΩ)

/-- With no cell seen, training never returns the `a³ + a b` kernel for `p = 3`. -/
example := exists_nuclearNorm_add_le_of_singularValues_eq (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- Matrices with entries bounded by `c` form a compact set. -/
lemma isCompact_setOf_norm_apply_le (c : ℝ) :
    IsCompact {X : Matrix m n ℂ | ∀ i j, ‖X i j‖ ≤ c} := by
  have h : IsCompact (Set.pi Set.univ fun _ : m => Set.pi Set.univ fun _ : n =>
      Metric.closedBall (0 : ℂ) c) :=
    isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_closedBall _ _
  have e : {X : Matrix m n ℂ | ∀ i j, ‖X i j‖ ≤ c} = Set.pi Set.univ fun _ : m =>
      Set.pi Set.univ fun _ : n => Metric.closedBall (0 : ℂ) c := by
    ext X
    exact ⟨fun h i _ j _ => mem_closedBall_zero_iff.mpr (h i j), fun h i j =>
      mem_closedBall_zero_iff.mp (h i (Set.mem_univ i) j (Set.mem_univ j))⟩
  rw [e]
  exact h

/-- **Vanishing loss below `‖K‖_S1` forces convergence to a recovered kernel**
(`review/flat_spectrum.md`, item WD7): if `K` is the only matrix of least nuclear norm among
those equal to it on `Ω`, a sequence with `‖P_j‖_S1 ≤ ‖K‖_S1` whose squared loss on `Ω` tends to
`0` converges to `K`. -/
theorem tendsto_of_fitLoss_tendsto_zero {Ω : Set (m × n)} {K : Matrix m n ℂ}
    (hrec : ∀ X ∈ {X : Matrix m n ℂ | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K))
    {P : ℕ → Matrix m n ℂ} (hloss : Tendsto (fun j => fitLoss Ω K (P j)) atTop (𝓝 0))
    (hnorm : ∀ j, nuclearNorm (P j) ≤ nuclearNorm K) : Tendsto P atTop (𝓝 K) := by
  have hKmin := (hrec K fun _ _ => rfl).mpr rfl
  have : FirstCountableTopology (Matrix m n ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → n → ℂ))
  refine tendsto_of_subseq_tendsto fun ns hns => ?_
  obtain ⟨X, -, φ, hφ, hX⟩ := (isCompact_setOf_norm_apply_le (nuclearNorm K)).tendsto_subseq
    (x := fun j => P (ns j)) fun j i l => (norm_apply_le_nuclearNorm _ i l).trans (hnorm _)
  have hXS : X ∈ {X : Matrix m n ℂ | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} :=
    fitLoss_eq_zero_iff.mp (tendsto_nhds_unique (((continuous_fitLoss Ω K).tendsto X).comp hX)
      (hloss.comp (hns.comp hφ.tendsto_atTop)))
  have hXle : nuclearNorm X ≤ nuclearNorm K := by
    obtain ⟨Z, hZ, htr⟩ := exists_isContraction_trace_eq X
    have h1 : Tendsto (fun j => pairing Z (P (ns (φ j)))) atTop (𝓝 (pairing Z X)) :=
      ((continuous_pairing (f := fun _ : Matrix m n ℂ => Z) (g := fun Y => Y) continuous_const
        continuous_id).tendsto X).comp hX
    have h2 : pairing Z X = nuclearNorm X := by rw [pairing, htr, Complex.ofReal_re]
    rw [h2] at h1
    exact le_of_tendsto' h1 fun j =>
      ((Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hZ _)).trans (hnorm _)
  have hXK : X = K := (hrec X hXS).mp
    (isMinOn_iff.mpr fun Y hY => hXle.trans (isMinOn_iff.mp hKmin Y hY))
  exact ⟨φ, hXK ▸ hX⟩

/-- **Trained products converge to a kernel that M recovers** (`review/flat_spectrum.md`,
item WD7): if `K` is the only matrix of least nuclear norm among those equal to it on `Ω`, the
weight decays `λ_j > 0` tend to `0` and `(U_j, V_j)` are trained fits for the squared loss on
`Ω`, then `U_j V_jᴴ → K`. -/
theorem tendsto_mul_conjTranspose_of_forall_isMinOn {Ω : Set (m × n)} {K : Matrix m n ℂ}
    (hrec : ∀ X ∈ {X : Matrix m n ℂ | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} X ↔ X = K))
    {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}
    (hmin : ∀ j, IsDecayedMin (fitLoss Ω K) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 K) := by
  have hbound : Tendsto (fun j => lams j * nuclearNorm K) atTop (𝓝 0) := by
    simpa using hlam.mul_const (nuclearNorm K)
  exact tendsto_of_fitLoss_tendsto_zero hrec
    (tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hbound
      (fun j => fitLoss_nonneg _ _ _)
      fun j => (hmin j).fitLoss_le (hpos j).le (X' := K) fun _ _ => rfl)
    fun j => (hmin j).nuclearNorm_le (hpos j) (X' := K) fun _ _ => rfl

/-- **Theorem E for training** (`review/flat_spectrum.md`, Theorem E with item WD7): for
`u vᵀ` with `|u_i v_j| = 1` and cells `Ω` that connect the rows with the columns and contain the
graph of a permutation, the trained products converge to `u vᵀ` as the weight decay tends to
`0`. -/
theorem tendsto_mul_conjTranspose_vecMulVec_of_connected [Nonempty n] {u v : n → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) {Ω : Set (n × n)} (hconn : (patternGraph Ω).Connected)
    (hperm : ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω) {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) {Us Vs : ℕ → Matrix n k ℂ}
    (hmin : ∀ j, IsDecayedMin (fitLoss Ω (vecMulVec u v)) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 (vecMulVec u v)) :=
  tendsto_mul_conjTranspose_of_forall_isMinOn
    ((forall_isMinOn_vecMulVec_iff_connected_of_entries h).mpr ⟨hconn, hperm⟩) hpos hlam hmin

/-! The witness: the `1 × 1` kernel `1 = u vᵀ` with `u = v = 1`, seen on its one cell, and the
shrinking family of `isDecayedMin_shrink`, whose products `(1 − s_j)²` tend to `1`. -/

private lemma witness_staircase : staircase 0 = Set.univ :=
  Set.eq_univ_of_forall fun ⟨i, j⟩ => by
    have hij : j = i := Fin.ext (by have := i.isLt; have := j.isLt; omega)
    rw [hij]
    exact Or.inl ⟨i, rfl⟩

private lemma witness_kernel :
    vecMulVec (fun _ : Fin 1 => (1 : ℂ)) (fun _ => 1) = (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
  ext i j
  rw [Subsingleton.elim j i]
  simp [vecMulVec_apply]

private lemma witness_rec : ∀ X ∈ {X : Matrix (Fin 1) (Fin 1) ℂ |
    ∀ ij ∈ (Set.univ : Set (Fin 1 × Fin 1)),
      X ij.1 ij.2 = (1 : Matrix (Fin 1) (Fin 1) ℂ) ij.1 ij.2},
    (IsMinOn nuclearNorm {X : Matrix (Fin 1) (Fin 1) ℂ | ∀ ij ∈ (Set.univ : Set (Fin 1 × Fin 1)),
      X ij.1 ij.2 = (1 : Matrix (Fin 1) (Fin 1) ℂ) ij.1 ij.2} X ↔ X = 1) := by
  have hS : ∀ X ∈ {X : Matrix (Fin 1) (Fin 1) ℂ | ∀ ij ∈ (Set.univ : Set (Fin 1 × Fin 1)),
      X ij.1 ij.2 = (1 : Matrix (Fin 1) (Fin 1) ℂ) ij.1 ij.2}, X = 1 := fun X hX => by
    ext i j
    exact hX (i, j) (Set.mem_univ _)
  exact fun X hX => ⟨fun _ => hS X hX, fun h => isMinOn_iff.mpr fun Y hY => by rw [h, hS Y hY]⟩

example := tendsto_of_fitLoss_tendsto_zero witness_rec (P := fun _ => 1)
  (by simp [fitLoss, mask, frobeniusSq]) fun _ => le_rfl

example := tendsto_mul_conjTranspose_of_forall_isMinOn witness_rec shrink_pos tendsto_shrink
  isDecayedMin_shrink

example := tendsto_mul_conjTranspose_vecMulVec_of_connected (u := fun _ : Fin 1 => (1 : ℂ))
  (v := fun _ => 1) (by simp [vecMulVec_apply]) (staircase_connected 0)
  ⟨Equiv.refl _, fun i => Or.inl ⟨i, rfl⟩⟩ shrink_pos tendsto_shrink
  (Us := fun j => (1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (Vs := fun j => (1 - 1 / ((j : ℝ) + 1)) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) fun j => by
    rw [witness_staircase, witness_kernel]
    exact isDecayedMin_shrink j

end Transformer.Grokking.FlatSpectrum.Training
