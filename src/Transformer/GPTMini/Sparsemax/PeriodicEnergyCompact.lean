import Transformer.GPTMini.Sparsemax.PeriodicEnergyTopology

/-!
# Compact learned edge/output domains and minimum attainment

New existence guarantee following arXiv:1602.02068v2, Eq. (1) and §2.5.
Every output coordinate obeys `Z[i,d]^2 <= energy`, from a genuine two-by-two
principal PSD submatrix and the probability-score diagonal bound. Together
with the incident edge bounds this places the entire closed joint domain
inside a compact coordinate box. The tied domain is compact as well.

Every nonempty domain therefore attains the ordinary squared-output minimum
for arbitrary finite observation tables, including conflicting or unattainable
answers. Nonemptiness is the only remaining existence premise. No prescribed
minimum, sparse support, eigenbasis or numerical optimization oracle is used.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A genuine output channel forces the coupling energy to be nonnegative.
Source: the lower diagonal block of the new PSD restriction following sparsemax Eq. (1). -/
theorem periodicEnergy_energy_nonneg {N D : ℕ} (floor budget energy : ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ periodicEnergyDomain N D floor budget energy) (d : Fin D) : 0 ≤ energy := by
  have h := hx.2.2.diag_nonneg (i := Sum.inr d)
  change 0 ≤ energy * (if d = d then (1 : ℝ) else 0) at h
  simpa only [ite_true, mul_one] using h

/-- The nonconstant ordinary-answer witness inhabits every energy-sign premise. -/
example : (0 : ℝ) ≤ 6 :=
  periodicEnergy_energy_nonneg (3 / 4) (1 / 8) 6 _ (taskPeriodicEnergyPair_mem 0) (0 : Fin 1)

/-- Every learned output coordinate has a dictionary-size independent squared energy bound.
Source: a two-by-two principal PSD determinant after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergy_output_sq_bound {N D : ℕ} (floor budget energy : ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 0 ≤ floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (i : Fin (N + 1)) (d : Fin D) : (x.2 i d) ^ 2 ≤ energy := by
  let m : Fin 2 → Fin (N + 1) ⊕ Fin D := fun k => if k = 0 then Sum.inl i else Sum.inr d
  have hd := (hx.2.2.submatrix m).det_nonneg
  norm_num [Matrix.det_fin_two, Matrix.submatrix_apply, m, energyCoupledMemoryMatrix,
    localMemoryGram_scores, Matrix.fromBlocks, Matrix.transpose_apply, Matrix.one_apply] at hd
  have hb : memoryGramScores (localMemoryCore x.1) i i ≤ 1 := by
    have h := Finset.single_le_sum
      (fun j _ => incidentMemory_scores_nonneg floor x.1 hf hx.1.1 i j) (Finset.mem_univ i)
    rw [localMemoryCore_scores_rowSum] at h
    exact h
  have he := periodicEnergy_energy_nonneg floor budget energy x hx d
  have hu := mul_le_mul_of_nonneg_right hb he
  nlinarith

/-- Nonconstant learned original-value predictions inhabit every coordinate-energy premise. -/
example : (taskEnergyTarget 1 0 0) ^ 2 ≤ 6 :=
  periodicEnergy_output_sq_bound (3 / 4) (1 / 8) 6 _ (by norm_num)
    (taskPeriodicEnergyPair_mem 1) 0 0

/-- Every learned output coordinate lies in an explicit compact scalar interval.
Source: the PSD coordinate-energy bound following sparsemax Eq. (1). -/
theorem periodicEnergy_output_interval {N D : ℕ} (floor budget energy : ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 0 ≤ floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (i : Fin (N + 1)) (d : Fin D) :
    -( |energy| + 1) ≤ x.2 i d ∧ x.2 i d ≤ |energy| + 1 := by
  have hz := periodicEnergy_output_sq_bound floor budget energy x hf hx i d
  have ha := le_abs_self energy
  have hn := abs_nonneg energy
  constructor
  · nlinarith [sq_nonneg (x.2 i d + 1)]
  · nlinarith [sq_nonneg (x.2 i d - 1)]

/-- Genuine nonzero learned outputs inhabit the compact-interval premises. -/
example : -( |(6 : ℝ)| + 1) ≤ taskEnergyTarget 0 2 0 ∧ taskEnergyTarget 0 2 0 ≤ |(6 : ℝ)| + 1 :=
  periodicEnergy_output_interval (3 / 4) (1 / 8) 6 _ (by norm_num)
    (taskPeriodicEnergyPair_mem 0) 2 0

/-- The entire finite learned edge/output energy domain is compact.
Source: the genuine PSD coordinate bounds and closed constraints after sparsemax Eq. (1). -/
theorem periodicEnergyDomain_compact (N D : ℕ) (floor budget energy : ℝ)
    (hf : 0 ≤ floor) : IsCompact (periodicEnergyDomain N D floor budget energy) := by
  have ht : IsCompact (Set.Icc (0 : Fin N → ℝ) (fun _ => 1 - floor)) :=
    CompactIccSpace.isCompact_Icc
  -- Use scalar coordinate boxes for rectangular learned output tables.
  have hZ : IsCompact ((Set.Icc (-( |energy| + 1)) (|energy| + 1)).matrix :
      Set (Matrix (Fin (N + 1)) (Fin D) ℝ)) :=
    (CompactIccSpace.isCompact_Icc : IsCompact (Set.Icc (-( |energy| + 1)) (|energy| + 1))).matrix
  apply (ht.prod hZ).of_isClosed_subset
    (periodicEnergyDomain_closed N D floor budget energy)
  intro x hx
  refine ⟨⟨hx.1.1.1, ?_⟩, ?_⟩
  · intro e
    exact (incidentMemoryWeightDomain_coordinateBound floor x.1 hx.1.1 e).2
  · intro i d
    exact periodicEnergy_output_interval floor budget energy x hf hx i d

/-- A nonsingleton ordinary-answer domain inhabits the joint compactness premise. -/
example : IsCompact (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6) :=
  periodicEnergyDomain_compact _ _ _ _ _ (by norm_num)

/-- The full affine-sharing domain is compact without an assumed optimization solution.
Source: the closed architectural restriction of the compact energy domain after sparsemax Eq. (1). -/
theorem outputTiedEnergyDomain_compact (N D : ℕ) (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D) (hf : 0 ≤ floor) :
    IsCompact (outputTiedEnergyDomain N D floor budget energy offset gain channel) :=
  (periodicEnergyDomain_compact N D floor budget energy hf).of_isClosed_subset
    (outputTiedEnergyDomain_closed N D floor budget energy offset gain channel) (fun _ hx => hx.1)

/-- Nonzero affine sharing and both changing attention supports inhabit the compactness premise. -/
example : IsCompact (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) :=
  outputTiedEnergyDomain_compact _ _ _ _ _ _ _ _ (by norm_num)

/-- Any nonempty genuine joint energy domain attains ordinary error for arbitrary finite answers.
Source: compactness and continuous actual prediction error after sparsemax §2.5. -/
theorem periodicEnergySquaredError_exists_min {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (target : Matrix (Fin R) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (periodicEnergyDomain N D floor budget energy).Nonempty) :
    ∃ x ∈ periodicEnergyDomain N D floor budget energy,
      IsMinOn (periodicEnergySquaredError code target) (periodicEnergyDomain N D floor budget energy) x :=
  (periodicEnergyDomain_compact N D floor budget energy (by linarith)).exists_isMinOn hn
    (periodicEnergySquaredError_continuousOn floor budget energy code target hf)

/-- A complete nonidentity/nonconstant witness supplies the existence premise. -/
example : ∃ x ∈ periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6,
    IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
      (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6) x :=
  periodicEnergySquaredError_exists_min _ _ _ _ _ (by norm_num)
    ⟨_, taskPeriodicEnergyPair_mem 0⟩

/-- Each complete learned output row has a bound depending on answer dimension, not prototype count.
Source: coordinate-energy bounds from the PSD coupling after sparsemax Eq. (1). -/
theorem periodicEnergy_output_row_sq_bound {N D : ℕ} (floor budget energy : ℝ)
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 0 ≤ floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (i : Fin (N + 1)) : (∑ d, (x.2 i d) ^ 2) ≤ (D : ℝ) * energy := by
  calc
    _ ≤ ∑ _ : Fin D, energy := Finset.sum_le_sum
      (fun d _ => periodicEnergy_output_sq_bound floor budget energy x hf hx i d)
    _ = (D : ℝ) * energy := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- A nonconstant original-value regression fit inhabits every complete-row norm premise. -/
example : (∑ d, (taskEnergyTarget 1 0 d) ^ 2) ≤ (1 : ℝ) * 6 := by
  simpa only [Nat.cast_one] using periodicEnergy_output_row_sq_bound (3 / 4) (1 / 8) 6 _ (by norm_num)
    (taskPeriodicEnergyPair_mem 1) 0

end Transformer.GPTMini.Sparsemax
