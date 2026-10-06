import Transformer.GPTMini.Sparsemax.PeriodicMemoryAttention
import Transformer.GPTMini.Sparsemax.EnergyMemoryIdentification

/-!
# Joint constant-width attention and original common-value learning

New architecture after arXiv:1602.02068v2, Eq. (1). Intrinsic parameters
are the learned path edges and a single output-coordinate table. Bounded
width-three Q/K and the structural local mask produce actual attention B.
Original common values are decoded by its proved inverse, and categorical
observation queries select the corresponding genuine learned query rows.

The old PSD energy coupling is unchanged on B and Z, without independent
norm additions. Its affine auxiliary lift is only a certificate device;
the physical embedding Gram is the explicit width-three one. Probability
mixtures of differently masked query rows are not asserted to commute with
this new mask. Registered and nearest categorical observations do apply.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Convex joint edges/output domain with the earlier genuine value-energy restriction.
Source: the new fixed-width memory chart after arXiv:1602.02068v2, Eq. (1). -/
def periodicEnergyDomain (N D : ℕ) (floor budget energy : ℝ) :
    Set ((Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {x | ((x.1, (0 : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)), x.2) ∈
    energyCoupledMemoryDomain N D 1 floor budget energy}

/-- All intrinsic learned edges and output coordinates retain a jointly convex domain.
Source: the new bounded fixed-width chart after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergyDomain_convex (N D : ℕ) (floor budget energy : ℝ) :
    Convex ℝ (periodicEnergyDomain N D floor budget energy) := by
  intro x hx y hy a b ha hb hab
  have h := energyCoupledMemoryDomain_convex N D 1 floor budget energy hx hy ha hb hab
  change ((a • x.1 + b • y.1, 0), a • x.2 + b • y.2) ∈
    energyCoupledMemoryDomain N D 1 floor budget energy
  change ((a • x.1 + b • y.1,
    a • (0 : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ) + b • 0), a • x.2 + b • y.2) ∈
      energyCoupledMemoryDomain N D 1 floor budget energy at h
  simpa only [smul_zero, add_zero] using h

/-- One globally decoded original common value table, using actual width-three attention.
Source: the new true sparsemax inverse chart after arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryValues {N D : ℕ} (t : Fin N → ℝ)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) : Matrix (Fin (N + 1)) (Fin D) ℝ :=
  (periodicMemoryAttention t)⁻¹ * Z

/-- Actual categorical query attention multiplied by the globally learned common values.
Source: the new bounded fixed-width `attn @ v` architecture after sparsemax Eq. (1). -/
def periodicEnergyForward {R N D : ℕ} (code : Fin R → Fin (N + 1))
    (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) : Matrix (Fin R) (Fin D) ℝ :=
  Matrix.of (fun r => periodicMemoryAttention x.1 (code r)) * periodicMemoryValues x.1 x.2

/-- The original actual attention and common-value decoder give exactly the coded output table.
Source: the proved inverse of genuine width-three sparsemax after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergyForward_eq {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy) :
    periodicEnergyForward code x = Matrix.of (fun r => x.2 (code r)) := by
  have ht : x.1 ∈ incidentMemoryWeightDomain N floor := hx.1.1
  have hi := periodicMemoryAttention_det_unit floor x.1 hf ht
  unfold periodicEnergyForward periodicMemoryValues
  rw [← oneHotContextCodes_mul code (periodicMemoryAttention x.1), Matrix.mul_assoc,
    Matrix.mul_nonsing_inv_cancel_left _ _ hi, oneHotContextCodes_mul]

/-- The fixed-width domain has a complete nonempty zero-budget instance at every memory size.
Source: the genuine zero-value endpoint after arXiv:1602.02068v2, Eq. (1). -/
theorem zero_mem_periodicEnergyDomain (N D : ℕ) (floor : ℝ) (hf : floor ≤ 1) :
    ((0 : Fin N → ℝ), (0 : Matrix (Fin (N + 1)) (Fin D) ℝ)) ∈
      periodicEnergyDomain N D floor 0 1 :=
  zero_mem_energyCoupledMemoryDomain N D 1 floor (by norm_num) hf

/-- Four prototypes inhabit every zero-budget feasibility premise. -/
example : ((0 : Fin 3 → ℝ), (0 : Matrix (Fin 4) (Fin 1) ℝ)) ∈
    periodicEnergyDomain 3 1 (3 / 4) 0 1 :=
  zero_mem_periodicEnergyDomain _ _ _ (by norm_num)

/-- Both nonidentity task witnesses remain feasible with physical embedding width exactly three.
Source: the new restriction of the energy-coupled examples after arXiv:1602.02068v2, Eq. (1). -/
theorem taskPeriodicEnergyPair_mem (edge : Fin 2) :
    ((taskEnergyParameters edge).1, taskEnergyTarget edge) ∈
      periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6 := by
  refine ⟨⟨(taskEnergyParameters_mem edge).1, ?_⟩, taskEnergyParameters_budget edge, ?_⟩
  · intro x
    norm_num
  · rw [energyCoupledMemoryMatrix_edges_invariant 6
      ((taskEnergyParameters edge).1, 0) (taskEnergyParameters edge) (taskEnergyTarget edge) rfl]
    exact taskEnergyMatrix_posSemidef edge

/-- Nonconstant answers fit through actual fixed-width attention and the original common values. -/
example : periodicEnergyForward (fun j : Fin 3 => j)
    ((taskEnergyParameters 0).1, taskEnergyTarget 0) = taskEnergyTarget 0 :=
  periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num) (taskPeriodicEnergyPair_mem 0)

/-- Actual fixed-width predictions are affine jointly in learned edges and common output coordinates.
Source: the new genuine constant-width inverse chart after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergyForward_affine {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1))
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hx : x ∈ periodicEnergyDomain N D floor budget energy)
    (hy : y ∈ periodicEnergyDomain N D floor budget energy) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    periodicEnergyForward code (a • x + b • y) =
      a • periodicEnergyForward code x + b • periodicEnergyForward code y := by
  have hm := periodicEnergyDomain_convex N D floor budget energy hx hy ha hb hab
  rw [periodicEnergyForward_eq floor budget energy code _ hf hm,
    periodicEnergyForward_eq floor budget energy code x hf hx,
    periodicEnergyForward_eq floor budget energy code y hf hy]
  ext r d
  rfl

/-- Opposite learned edge allocations and nonconstant outputs inhabit all midpoint premises. -/
example : periodicEnergyForward (fun j : Fin 3 => j)
    ((1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1)) =
    (1 / 2 : ℝ) • periodicEnergyForward (fun j : Fin 3 => j)
      ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
      (1 / 2 : ℝ) • periodicEnergyForward (fun j : Fin 3 => j)
        ((taskEnergyParameters 1).1, taskEnergyTarget 1) :=
  periodicEnergyForward_affine (3 / 4) (1 / 8) 6 _ _ _ (by norm_num)
    (taskPeriodicEnergyPair_mem 0) (taskPeriodicEnergyPair_mem 1)
    _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Every convex output criterion stays jointly convex with constant-width learned Q/K and values.
Source: the new restricted sparsemax memory chart after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergyObjective_convex {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (periodicEnergyDomain N D floor budget energy)
      (fun x => objective (periodicEnergyForward code x)) := by
  refine ⟨periodicEnergyDomain_convex N D floor budget energy, ?_⟩
  intro x hx y hy a b ha hb hab
  change objective (periodicEnergyForward code (a • x + b • y)) ≤
    a • objective (periodicEnergyForward code x) + b • objective (periodicEnergyForward code y)
  rw [periodicEnergyForward_affine floor budget energy code x y hf hx hy a b ha hb hab]
  exact hl.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A nonconstant ordinary squared criterion inhabits every joint-convexity premise. -/
example : ConvexOn ℝ (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6)
    (fun x => (periodicEnergyForward (fun j : Fin 3 => j) x 0 0) ^ 2) := by
  apply periodicEnergyObjective_convex (3 / 4) (1 / 8) 6 _ (fun Y => (Y 0 0) ^ 2) (by norm_num)
  have hs : ConvexOn ℝ Set.univ (fun z : ℝ => z ^ 2) := (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro A _ B _ a b ha hb hab
  simpa only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] using
    hs.2 (Set.mem_univ (A 0 0)) (Set.mem_univ (B 0 0)) ha hb hab

/-- Arbitrarily many registered zero outputs are genuine predictions with width-three embeddings. -/
example : periodicEnergyForward (fun j : Fin 8 => j)
    ((0 : Fin 7 → ℝ), (0 : Matrix (Fin 8) (Fin 2) ℝ)) = 0 :=
  periodicEnergyForward_eq (3 / 4) 0 1 _ _ (by norm_num)
    (zero_mem_periodicEnergyDomain _ _ _ (by norm_num))

end Transformer.GPTMini.Sparsemax
