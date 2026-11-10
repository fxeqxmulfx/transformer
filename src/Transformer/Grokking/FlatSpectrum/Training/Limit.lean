module

public import Mathlib.LinearAlgebra.Matrix.Charpoly.Coeff
public import Transformer.Grokking.FlatSpectrum.Training.Continuity
public import Transformer.Grokking.FlatSpectrum.Training.Growth

/-!
# A run cannot settle where the lifted gradient has a negative direction

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD4.
Let `W_{t+1} = (1 − η M_t) W_t` with `η > 0`, every `1 − η M_t` positive definite, `M_t → M∞`,
`W_t` bounded and `W_0` of full row rank. Then `M∞` is positive semidefinite
(`posSemidef_of_tendsto`).

If not, some `x` has `xᴴ M∞ x < 0`. A small shift `S = M∞ + c` is invertible
(`exists_pos_lt_det_add_smul_ne_zero`) and still negative at `x`, and
`1 − η M∞ = a − η S` with `a = 1 + η c > 1`. As `W_0` has a right inverse and the steps are
invertible, every `W_t` has full row rank, so from a late time `t₁` on the columns
`Z_j = W_{t₁+j} z` start at `Z_0 = x` for a suitable `z`. They solve the same recursion and stay
bounded, while `−⟨Z_j, S Z_j⟩` grows like `a^{2j}` once `M_t` is close to `M∞`
(`exists_pos_forall_pairing_nonneg`, from `pairing_nonneg_of_bounded`): a contradiction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexOrder

variable {p k : Type*} [Fintype p] [Fintype k] [DecidableEq p]

/-- A small positive real shift makes a square matrix invertible: its characteristic
polynomial has finitely many roots. -/
lemma exists_pos_lt_det_add_smul_ne_zero (A : Matrix p p ℂ) {δ : ℝ} (hδ : 0 < δ) :
    ∃ c : ℝ, 0 < c ∧ c < δ ∧ (A + c • (1 : Matrix p p ℂ)).det ≠ 0 := by
  have hinj : Set.InjOn (fun c : ℝ => (c : ℂ)) (Set.Ioo 0 δ) :=
    fun _ _ _ _ h => Complex.ofReal_injective h
  obtain ⟨z, ⟨c, hc, rfl⟩, hz⟩ :=
    ((Set.Ioo_infinite hδ).image hinj).exists_notMem_finset (-A).charpoly.roots.toFinset
  refine ⟨c, hc.1, hc.2, fun hdet => hz ?_⟩
  rw [Multiset.mem_toFinset, Polynomial.mem_roots (charpoly_monic _).ne_zero,
    Polynomial.IsRoot.def, eval_charpoly, sub_neg_eq_add, add_comm, scalar_apply,
    ← smul_one_eq_diagonal, Complex.coe_smul]
  exact hdet

example := exists_pos_lt_det_add_smul_ne_zero (0 : Matrix (Fin 2) (Fin 2) ℂ) one_pos

/-- **The limit of the lifted gradients is positive semidefinite**
(`review/flat_spectrum.md`, item WD4): if `W_{t+1} = (1 − η M_t) W_t` with `η > 0`, every
`1 − η M_t` positive definite, `M_t → M∞` Hermitian, `W_t` bounded and `W_0` of full row rank,
then `M∞` is positive semidefinite. -/
theorem posSemidef_of_tendsto {M : ℕ → Matrix p p ℂ} {Minf : Matrix p p ℂ}
    (hMinf : Minf.IsHermitian) (hM : Tendsto M atTop (𝓝 Minf)) {η : ℝ} (hη : 0 < η)
    (hstep : ∀ t, (1 - η • M t).PosDef) {W : ℕ → Matrix p k ℂ}
    (hW : ∀ t, W (t + 1) = (1 - η • M t) * W t) (hfull : Function.Surjective (W 0).mulVec)
    {B : ℝ} (hB : ∀ t, frobeniusSq (W t) ≤ B) : Minf.PosSemidef := by
  obtain ⟨R₀, hR₀⟩ := mulVec_surjective_iff_exists_right_inverse.mp hfull
  have hP : ∀ t, (W t * R₀).det ≠ 0 := by
    intro t
    induction t with
    | zero => rw [hR₀, det_one]; exact one_ne_zero
    | succ t ih =>
      rw [hW t, Matrix.mul_assoc, det_mul]
      exact mul_ne_zero (hstep t).det_pos.ne' ih
  have hT : ∀ Z : Matrix p Unit ℂ, 0 ≤ pairing Z ((1 - η • Minf) * Z) := fun Z =>
    ge_of_tendsto' (((continuous_pairing continuous_const
      ((continuous_const.sub (continuous_id.const_smul η)).matrix_mul continuous_const)).tendsto
        Minf).comp hM) fun t => pairing_nonneg_of_posSemidef (hstep t).posSemidef Z
  by_contra hneg
  obtain ⟨Z0, hZ0⟩ : ∃ Z0 : Matrix p Unit ℂ, pairing Z0 (Minf * Z0) < 0 := by
    by_contra hall
    exact hneg (posSemidef_of_pairing_nonneg hMinf fun Z => not_lt.mp fun h => hall ⟨Z, h⟩)
  have hN1 : 0 < frobeniusSq Z0 + 1 := by linarith [frobeniusSq_nonneg Z0]
  obtain ⟨c, hc0, hcδ, hdet⟩ := exists_pos_lt_det_add_smul_ne_zero Minf
    (div_pos (neg_pos.mpr hZ0) hN1)
  have hQ : pairing Z0 ((Minf + c • 1) * Z0) < 0 := by
    rw [Matrix.add_mul, Matrix.smul_mul, Matrix.one_mul, pairing_add_right, pairing_smul_right,
      pairing_self]
    have h1 := (lt_div_iff₀ hN1).mp hcδ
    nlinarith [frobeniusSq_nonneg Z0]
  have hSH : (Minf + c • (1 : Matrix p p ℂ)).IsHermitian := hMinf.add (by
    rw [IsHermitian, conjTranspose_smul, conjTranspose_one, star_trivial])
  have hTeq : (1 + η * c) • (1 : Matrix p p ℂ) - η • (Minf + c • 1) = 1 - η • Minf := by module
  obtain ⟨ε, hε0, hgain⟩ := exists_pos_forall_pairing_nonneg (q := Unit) hSH hdet
    (a := 1 + η * c) (by nlinarith [mul_pos hη hc0]) hη (fun Z => by rw [hTeq]; exact hT Z)
  have h0 : Tendsto (fun t => frobeniusSq (M t - Minf)) atTop (𝓝 0) := by
    have h := (continuous_frobeniusSq.tendsto (Minf - Minf)).comp (hM.sub_const Minf)
    rw [sub_self, show frobeniusSq (0 : Matrix p p ℂ) = 0 by simp [frobeniusSq]] at h
    exact h
  obtain ⟨t₁, ht₁⟩ := eventually_atTop.mp (h0.eventually_lt_const (pow_pos hε0 2))
  obtain ⟨Zc, hZc⟩ : ∃ Zc : Matrix k Unit ℂ, W t₁ * Zc = Z0 :=
    ⟨R₀ * ((W t₁ * R₀)⁻¹ * Z0), by
      rw [← Matrix.mul_assoc, ← Matrix.mul_assoc,
        mul_nonsing_inv _ (isUnit_iff_ne_zero.mpr (hP t₁)), Matrix.one_mul]⟩
  have key : 0 ≤ pairing (W (t₁ + 0) * Zc) ((Minf + c • 1) * (W (t₁ + 0) * Zc)) := by
    refine hgain (fun j => M (t₁ + j) - Minf) (fun j => W (t₁ + j) * Zc) (B * frobeniusSq Zc)
      (fun j => (ht₁ _ (Nat.le_add_right _ _)).le) (fun j => ?_) fun j =>
        (frobeniusSq_mul_le _ _).trans (mul_le_mul_of_nonneg_right (hB _) (frobeniusSq_nonneg _))
    change W (t₁ + j + 1) * Zc = ((1 + η * c) • 1 - η • (Minf + c • 1) - η • (M (t₁ + j) - Minf)) *
      (W (t₁ + j) * Zc)
    rw [hW (t₁ + j), Matrix.mul_assoc, hTeq]
    congr 1
    module
  rw [Nat.add_zero, hZc] at key
  exact absurd key (not_le.mpr hQ)

/-- The recursion `W ← ½ W`, from `M_t = 1` and `η = ½`: every hypothesis holds, and the limit
`M∞ = 1` is positive semidefinite. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℂ).PosSemidef := by
  have hhalf : (1 : Matrix (Fin 1) (Fin 1) ℂ) - (1 / 2 : ℝ) • 1 = (1 / 2 : ℝ) • 1 := by module
  refine posSemidef_of_tendsto (M := fun _ => 1) (η := 1 / 2) (B := 1)
    (W := fun t => ((1 / 2 : ℝ) ^ t) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) isHermitian_one
    tendsto_const_nhds (by norm_num) (fun _ => ?_) (fun t => ?_) (fun v => ⟨v, by simp⟩)
    fun t => ?_
  · rw [hhalf]
    refine posDef_of_pairing_pos (by
      rw [IsHermitian, conjTranspose_smul, conjTranspose_one, star_trivial]) fun W hW => ?_
    rw [Matrix.smul_mul, Matrix.one_mul, pairing_smul_right, pairing_self]
    exact mul_pos (by norm_num) (lt_of_le_of_ne (frobeniusSq_nonneg W) fun h0 =>
      hW (eq_zero_of_frobeniusSq_eq_zero h0.symm))
  · rw [hhalf, Matrix.smul_mul, Matrix.one_mul, smul_smul, pow_succ']
  · rw [frobeniusSq_smul, show frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 by
      simp [frobeniusSq], mul_one]
    exact pow_le_one₀ (by positivity) (pow_le_one₀ (by norm_num) (by norm_num))

end Transformer.Grokking.FlatSpectrum.Training
