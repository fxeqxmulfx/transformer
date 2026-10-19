module

public import Transformer.Landscape.PhaseModel.Basic
public import Mathlib.Algebra.Order.ToIntervalMod

/-!
# Phase model — integer rows and periods

For an integer row `a`, `⟪a, v⟫ = Σ_i a_i v_i` (`inner_intVec`), a multiple
of `2π` when `v` is a period (`exists_inner_intVec_eq`), so the periods
`2πℤⁿ` leave a phase energy with integer rows invariant
(`phaseEnergy_add_of_mem_periods`). A relation `⟪a, v⟫ ∈ G`, for an additive
subgroup `G` of `ℝ`, passes from rows to their integer combinations
(`inner_intVec_mem_of_mem_span`). The periods form a closed set
(`isClosed_periods`), and every point has a translate by a period in the
compact cube `[0, 2π]ⁿ` (`isCompact_cube`, `exists_add_mem_cube`). The
inner product with the row of `x / y` is `ψ_x - ψ_y - ψ_{x-y}`
(`inner_intVec_divisionRow`), which gives `divisionEnergy_apply`.
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- The inner product with an integer row: `⟪a, v⟫ = Σ_i a_i v_i`. -/
theorem inner_intVec (a : Fin n → ℤ) (v : EucSpace n) :
    ⟪intVec a, v⟫ = ∑ i, (a i : ℝ) * v.ofLp i := by
  simp [intVec, PiLp.inner_apply, mul_comm]

/-- The inner product of an integer row with a period is a multiple of `2π`. -/
theorem exists_inner_intVec_eq (a : Fin n → ℤ) {v : EucSpace n} (hv : v ∈ periods n) :
    ∃ k : ℤ, ⟪intVec a, v⟫ = k * (2 * π) := by
  choose k hk using hv
  refine ⟨∑ i, a i * k i, ?_⟩
  rw [inner_intVec]
  push_cast
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun i _ => by rw [hk i]; ring

/-- The squared norm of an integer row: `‖a‖² = Σ_i a_i²`. -/
theorem norm_intVec_sq (a : Fin n → ℤ) : ‖intVec a‖ ^ 2 = ∑ i, (a i : ℝ) ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_intVec]
  simp [intVec, sq]

/-- The inner product with an integer row is additive in the row. -/
theorem inner_intVec_add (a b : Fin n → ℤ) (v : EucSpace n) :
    ⟪intVec (a + b), v⟫ = ⟪intVec a, v⟫ + ⟪intVec b, v⟫ := by
  simp only [inner_intVec, Pi.add_apply, Int.cast_add, add_mul, Finset.sum_add_distrib]

/-- The inner product with an integer row is homogeneous in the row. -/
theorem inner_intVec_zsmul (c : ℤ) (a : Fin n → ℤ) (v : EucSpace n) :
    ⟪intVec (c • a), v⟫ = c * ⟪intVec a, v⟫ := by
  simp only [inner_intVec, Pi.smul_apply, smul_eq_mul, Int.cast_mul, mul_assoc, Finset.mul_sum]

/-- **Relations pass to integer combinations of rows**: if `⟪a, v⟫` lies in
an additive subgroup `G` of `ℝ` for every row `a ∈ A`, it does for every row
in the `ℤ`-span of `A`. -/
theorem inner_intVec_mem_of_mem_span (G : AddSubgroup ℝ) {A : Set (Fin n → ℤ)}
    {v : EucSpace n} (hA : ∀ a ∈ A, ⟪intVec a, v⟫ ∈ G) {a : Fin n → ℤ}
    (ha : a ∈ Submodule.span ℤ A) : ⟪intVec a, v⟫ ∈ G := by
  induction ha using Submodule.span_induction with
  | mem a ha => exact hA a ha
  | zero => simp [inner_intVec]
  | add a b _ _ ha hb => rw [inner_intVec_add]; exact G.add_mem ha hb
  | smul c a _ ha => rw [inner_intVec_zsmul, ← zsmul_eq_mul]; exact G.zsmul_mem ha c

/-- **The periods leave a phase energy with integer rows invariant.** -/
theorem phaseEnergy_add_of_mem_periods {R : Type*} [Fintype R] (a : R → Fin n → ℤ)
    {v : EucSpace n} (hv : v ∈ periods n) (ψ : EucSpace n) :
    phaseEnergy (fun r => intVec (a r)) (ψ + v) = phaseEnergy (fun r => intVec (a r)) ψ :=
  phaseEnergy_add _ (fun r => exists_inner_intVec_eq (a r) hv) ψ

/-- The periods form a closed set: the points whose coordinates all have
cosine `1`. -/
theorem isClosed_periods (n : ℕ) : IsClosed (periods n) := by
  have h : periods n = ⋂ i, (fun v : EucSpace n => cos (v.ofLp i)) ⁻¹' {1} := by
    ext v
    simp only [periods, Set.mem_iInter, Set.mem_preimage, Set.mem_singleton_iff,
      Set.mem_ofPred_eq, cos_eq_one_iff]
    exact forall_congr' fun i => exists_congr fun k => eq_comm
  rw [h]
  exact isClosed_iInter fun i =>
    isClosed_singleton.preimage (continuous_cos.comp (PiLp.continuous_apply 2 _ i))

/-- The cube `[0, 2π]ⁿ` is compact. -/
theorem isCompact_cube (n : ℕ) : IsCompact (cube n) := by
  have h : cube n =
      (EuclideanSpace.equiv (Fin n) ℝ) ⁻¹' Set.Icc (fun _ => 0) fun _ => 2 * π := by
    ext v
    simp only [cube, Set.mem_ofPred_eq, Set.mem_preimage, Set.mem_Icc, Pi.le_def, ← forall_and]
    rfl
  rw [h]
  exact (EuclideanSpace.equiv (Fin n) ℝ).toHomeomorph.isCompact_preimage.2 isCompact_Icc

/-- **Every point has a translate by a period in the cube**: subtract from
each coordinate the multiple of `2π` below it. -/
theorem exists_add_mem_cube (ψ : EucSpace n) : ∃ v ∈ periods n, ψ + v ∈ cube n := by
  refine ⟨WithLp.toLp 2 fun i => (-toIcoDiv two_pi_pos 0 (ψ.ofLp i) : ℤ) * (2 * π),
    fun i => ⟨-toIcoDiv two_pi_pos 0 (ψ.ofLp i), rfl⟩, fun i => ?_⟩
  have h := toIcoMod_mem_Ico two_pi_pos 0 (ψ.ofLp i)
  rw [zero_add, ← self_sub_toIcoDiv_zsmul, zsmul_eq_mul] at h
  simp only [WithLp.ofLp_add, Pi.add_apply, Int.cast_neg, neg_mul, ← sub_eq_add_neg]
  exact Set.Ico_subset_Icc_self h

/-- The inner product with the row of `x / y` is `ψ_x - ψ_y - ψ_{x-y}`. -/
theorem inner_intVec_divisionRow (p : Fin n × Fin n) (ψ : EucSpace n) :
    ⟪intVec (divisionRow p), ψ⟫ = ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2) := by
  simp [inner_intVec, divisionRow, Pi.single_apply, sub_mul, Finset.sum_sub_distrib]

/-- **The energy of a set `S` of rows of the division table** is
`Σ_{(x, y) ∈ S} (1 - cos (ψ_x - ψ_y - ψ_{x-y}))`. -/
theorem divisionEnergy_apply (S : Finset (Fin n × Fin n)) (ψ : EucSpace n) :
    divisionEnergy S ψ =
      ∑ p ∈ S, (1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2))) := by
  rw [divisionEnergy, phaseEnergy, ← Finset.sum_coe_sort S]
  simp only [inner_intVec_divisionRow]

/-- The hypothesis of `exists_inner_intVec_eq` and
`phaseEnergy_add_of_mem_periods` holds for the point with every coordinate
`2π`, and that of `inner_intVec_mem_of_mem_span` for `v = 0` and the trivial
subgroup. -/
example (n : ℕ) : (WithLp.toLp 2 fun _ : Fin n => 2 * π) ∈ periods n ∧
    ∀ A : Set (Fin n → ℤ), ∀ a ∈ A, ⟪intVec a, (0 : EucSpace n)⟫ ∈ (⊥ : AddSubgroup ℝ) :=
  ⟨fun _ => ⟨1, by simp⟩, fun _ _ _ => by simp⟩

end Transformer.Landscape.PhaseModel
