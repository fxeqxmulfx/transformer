module

public import Transformer.FrankWolfe.Section5_Process
public import Mathlib.Probability.Kernel.IonescuTulcea.Traj

/-!
# An actual infinite history law for softmax attention sampling

ArXiv:2508.09628v1, §5, `eq: softmax.process`. At each step,
each particle samples a previous particle according to its original
softmax row and moves to their original convex combination.
The joint choice weights below are products of those actual row weights.
They sum to one, and their marginals are exactly the original rows.

A history records the actual choices already made. Its recursively
computed configuration starts at the original input and applies every
recorded update. The history kernel prepends the next sampled choice.
It is an actual probability kernel on a countable discrete space.
Mathlib's Ionescu–Tulcea theorem then constructs an infinite history law.
Its finite-prefix marginals and its conditional next-step joint law
are proved below. The following module maps it to the actual token paths
and proves the complete original `IsSAProcess` transition identity.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators ENNReal
namespace Transformer.FrankWolfe

/-- The actual joint sampling weight for all particle choices.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def jointAttWeight {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d)
    (c : Idx n → Idx n) : ℝ≥0∞ :=
  ∏ i, ENNReal.ofReal (attWeight β X i (c i))

/-- The original softmax rows give total joint sampling mass one.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem sum_jointAttWeight {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d) :
    ∑ c : Idx n → Idx n, jointAttWeight β X c = 1 := by
  unfold jointAttWeight
  rw [← Fintype.prod_sum (fun i j : Idx n => ENNReal.ofReal (attWeight β X i j))]
  have hsum (i : Idx n) : ∑ j : Idx n, ENNReal.ofReal (attWeight β X i j) = 1 := by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun j _ => (attWeight_pos β X i j).le), sum_attWeight]
    norm_num
  simp only [hsum, Finset.prod_const_one]

/-- Every marginal of the actual joint sampling weights is the original row.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem sum_jointAttWeight_test {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d)
    (i : Idx n) (f : Idx n → ℝ≥0∞) :
    (∑ c : Idx n → Idx n, jointAttWeight β X c * f (c i)) =
      ∑ j : Idx n, ENNReal.ofReal (attWeight β X i j) * f j := by
  classical
  have hprod (c : Idx n → Idx n) :
      (∏ k : Idx n, ENNReal.ofReal (attWeight β X k (c k)) * (if k = i then f (c k) else 1)) =
        jointAttWeight β X c * f (c i) := by
    rw [Finset.prod_mul_distrib]
    simp [jointAttWeight]
  rw [← Finset.sum_congr rfl (fun c _ => hprod c),
    ← Fintype.prod_sum (fun k j : Idx n => ENNReal.ofReal (attWeight β X k j) *
      (if k = i then f j else 1))]
  have hother (k : Idx n) (hk : k ≠ i) :
      (∑ j : Idx n, ENNReal.ofReal (attWeight β X k j) * (if k = i then f j else 1)) = 1 := by
    simp only [hk, ite_false, mul_one]
    rw [← ENNReal.ofReal_sum_of_nonneg (fun j _ => (attWeight_pos β X k j).le), sum_attWeight]
    norm_num
  rw [Finset.prod_eq_single i]
  · simp
  · intro k _ hk
    exact hother k hk
  · simp

/-- The actual simultaneous sampled update of every original particle.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saUpdate {d n : ℕ} (γ : ℝ) (X : Idx n → EucSpace d)
    (c : Idx n → Idx n) : Idx n → EucSpace d :=
  fun i => (1 - γ) • X i + γ • X (c i)

/-- The countable type of actual finite particle-choice histories.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def SAHistory (n : ℕ) : Type := List (Idx n → Idx n)

instance (n : ℕ) : Countable (SAHistory n) := inferInstanceAs (Countable (List (Idx n → Idx n)))

instance (n : ℕ) : MeasurableSpace (SAHistory n) := ⊤

instance (n : ℕ) : MeasurableSingletonClass (SAHistory n) :=
  ⟨fun _ => MeasurableSpace.measurableSet_top⟩

/-- The empty history precedes every original sampled update.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def SAHistory.empty (n : ℕ) : SAHistory n := []

/-- The next actual sampling choice is prepended to the existing history.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def SAHistory.cons {n : ℕ} (c : Idx n → Idx n) (h : SAHistory n) : SAHistory n := c :: h

/-- The original configuration after the recorded actual sampled updates.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saHistoryConfig {d n : ℕ} (γ : ℝ) (X₀ : Idx n → EucSpace d) :
    SAHistory n → Idx n → EucSpace d
  | [] => X₀
  | c :: h => saUpdate γ (saHistoryConfig γ X₀ h) c

/-- The actual one-step probability kernel on choice histories.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saHistoryKernel {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    Kernel (SAHistory n) (SAHistory n) :=
  Kernel.ofFunOfCountable (α := SAHistory n) (β := SAHistory n) fun h =>
    ∑ c : Idx n → Idx n, jointAttWeight β (saHistoryConfig γ X₀ h) c • Measure.dirac (SAHistory.cons c h)

instance isMarkovKernel_saHistoryKernel {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    IsMarkovKernel (saHistoryKernel β γ X₀) := by
  constructor
  intro h
  apply isProbabilityMeasure_iff.mpr
  change (∑ c : Idx n → Idx n, jointAttWeight β (saHistoryConfig γ X₀ h) c • Measure.dirac (SAHistory.cons c h)) univ = 1
  simp only [Measure.finsetSum_apply, Measure.smul_apply, Measure.dirac_apply_of_mem (mem_univ _),
    smul_eq_mul, mul_one]
  exact sum_jointAttWeight β _

/-- The genuine history kernel has exactly the original particle transition marginal.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem saHistoryKernel_marginal {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (h : SAHistory n) (i : Idx n) (A : Set (EucSpace d)) :
    saHistoryKernel β γ X₀ h {h' | saHistoryConfig γ X₀ h' i ∈ A} =
      ∑ j : Idx n, ENNReal.ofReal (attWeight β (saHistoryConfig γ X₀ h) i j) *
        A.indicator (fun _ => (1 : ℝ≥0∞))
          ((1 - γ) • saHistoryConfig γ X₀ h i + γ • saHistoryConfig γ X₀ h j) := by
  classical
  change (∑ c : Idx n → Idx n, jointAttWeight β (saHistoryConfig γ X₀ h) c •
    Measure.dirac (SAHistory.cons c h)) _ = _
  simp only [Measure.finsetSum_apply, Measure.smul_apply,
    Measure.dirac_apply' _ ((Set.to_countable _).measurableSet), smul_eq_mul]
  have heq (c : Idx n → Idx n) :
      ({h' : SAHistory n | saHistoryConfig γ X₀ h' i ∈ A}).indicator 1 (SAHistory.cons c h) =
      A.indicator (fun _ => (1 : ℝ≥0∞))
        ((1 - γ) • saHistoryConfig γ X₀ h i + γ • saHistoryConfig γ X₀ h (c i)) := by
    rfl
  simp only [heq]
  exact sum_jointAttWeight_test β (saHistoryConfig γ X₀ h) i
    (fun j => A.indicator (fun _ => (1 : ℝ≥0∞))
      ((1 - γ) • saHistoryConfig γ X₀ h i + γ • saHistoryConfig γ X₀ h j))

/-- The actual next-history kernel reads the latest history in each prefix.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saHistoryKernels {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (t : ℕ) : Kernel ((Finset.Iic t) → SAHistory n) (SAHistory n) :=
  (saHistoryKernel β γ X₀).comap (fun h => h ⟨t, by simp⟩) (measurable_pi_apply _)

instance isMarkovKernel_saHistoryKernels {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (t : ℕ) : IsMarkovKernel (saHistoryKernels β γ X₀ t) := by
  unfold saHistoryKernels
  infer_instance

/-- The genuine infinite law obtained by iterating the actual history kernels.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saHistoryLaw {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    Measure (ℕ → SAHistory n) :=
  Kernel.traj (saHistoryKernels β γ X₀) 0 (fun _ => SAHistory.empty n)

instance isProbabilityMeasure_saHistoryLaw {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    IsProbabilityMeasure (saHistoryLaw β γ X₀) := by
  unfold saHistoryLaw
  infer_instance

/-- Every finite-prefix law is the corresponding actual iterated kernel.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem saHistoryLaw_prefix {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) (t : ℕ) :
    (saHistoryLaw β γ X₀).map (Preorder.frestrictLe t) =
      Kernel.partialTraj (X := fun _ => SAHistory n) (saHistoryKernels β γ X₀) 0 t
        (fun _ => SAHistory.empty n) :=
  Kernel.traj_map_frestrictLe_apply (X := fun _ => SAHistory n)
    (κ := saHistoryKernels β γ X₀) 0 t (fun _ => SAHistory.empty n)

/-- The actual next history has the stated conditional kernel given its prefix.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem saHistoryLaw_joint {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) (t : ℕ) :
    ((saHistoryLaw β γ X₀).map (Preorder.frestrictLe t)).compProd
      (saHistoryKernels β γ X₀ t) =
      (saHistoryLaw β γ X₀).map (fun h => (Preorder.frestrictLe t h, h (t + 1))) := by
  rw [saHistoryLaw_prefix]
  exact Kernel.partialTraj_compProd_eq_map_traj (X := fun _ => SAHistory n)
    (κ := saHistoryKernels β γ X₀) (x₀ := fun _ => SAHistory.empty n) (Nat.zero_le t)

/-- Two distinct original scalar tokens give a genuine probability law
on infinite histories at positive temperature and a convex step size. -/
example : IsProbabilityMeasure (saHistoryLaw 1 (1 / 2)
    (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)])) := inferInstance

end Transformer.FrankWolfe
