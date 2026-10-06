import Transformer.GPTMini.Sparsemax.CubeContentLearning
import Transformer.GPTMini.Sparsemax.PeriodicEnergyTopology

/-!
# A coefficient box enforces every learned content-attention budget

New implementation-oriented restriction after arXiv:1602.02068v2,
Eq. (1). With r>0 content bits and a self floor in (1/2,1], set each readout offset and gain to
`(1-floor)/(2*r)`. Coefficients in [-1,1] generate nonnegative routing
weights whose total is at most 1-floor. Therefore the complete joint
domain is exactly the coefficient box; no additional dictionary-sized
PSD projection or learned routing table is required.

The box is nonempty and compact. Projection of coefficients into this
box suffices to preserve the actual sparsemax and original-value inverse
guarantees. This does not establish convergence of an optimizer or
convexity of a later nonlinear FFN/language criterion. The r>0 and
floor<=1 hypotheses suffice for the elementary budget facts below;
the actual attention/value guarantee also needs its stated strict floor.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι κ : Type*} [Fintype ι]

/-- A size-normalized fixed readout budget for every generated content bit.
Source: bounded implementation chart following sparsemax Eq. (1). -/
def cubeContentNormalizedGain (ι : Type*) [Fintype ι] (floor : ℝ) : ℝ :=
  (1 - floor) / (2 * Fintype.card ι)

/-- Compatible floors and nonempty bit dimensions give a nonnegative fixed gain.
Source: normalized content budgets following sparsemax Eq. (1). -/
theorem cubeContentNormalizedGain_nonneg (floor : ℝ) (hf : floor ≤ 1) (hr : 0 < Fintype.card ι) :
    0 ≤ cubeContentNormalizedGain ι floor := by
  have hn : 0 < (Fintype.card ι : ℝ) := by exact_mod_cast hr
  unfold cubeContentNormalizedGain
  exact div_nonneg (by linarith) (by positivity)

/-- Two actual content bits give a positive ordinary implementation gain. -/
example : 0 ≤ cubeContentNormalizedGain (Fin 2) (3 / 4) :=
  cubeContentNormalizedGain_nonneg _ (by norm_num) (by norm_num)

/-- Maximum generated masses sum to the prescribed total outgoing budget exactly.
Source: the normalized affine implementation readout following sparsemax Eq. (1). -/
theorem cubeContentNormalizedGain_total (floor : ℝ) (hr : 0 < Fintype.card ι) :
    (Fintype.card ι : ℝ) * (2 * cubeContentNormalizedGain ι floor) = 1 - floor := by
  have hn : 0 < (Fintype.card ι : ℝ) := by exact_mod_cast hr
  unfold cubeContentNormalizedGain
  field_simp

/-- Two content coordinates give the exact quarter-mass outgoing budget. -/
example : (Fintype.card (Fin 2) : ℝ) * (2 * cubeContentNormalizedGain (Fin 2) (3 / 4)) = 1 / 4 := by
  rw [cubeContentNormalizedGain_total _ (by norm_num)]
  norm_num

/-- Coefficient caps alone enforce every genuine learned attention outgoing budget.
Source: affine normalized routing readouts following sparsemax Eq. (1). -/
theorem cubeContentNormalizedRoutes_mem {D : ℕ} (floor : ℝ) (pick : ι → κ) (channel : Fin D)
    (W : Matrix κ (Fin D) ℝ) (hf : floor ≤ 1) (hr : 0 < Fintype.card ι)
    (hW : ∀ k, -1 ≤ W k channel ∧ W k channel ≤ 1) :
    cubeContentLearnedRoutes (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel W ∈ cubeContentRouteDomain ι floor := by
  have hg := cubeContentNormalizedGain_nonneg floor hf hr
  have hl (i : ι) : 0 ≤ cubeContentLearnedRoutes (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel W i := by
    have h := mul_le_mul_of_nonneg_left (hW (pick i)).1 hg
    unfold cubeContentLearnedRoutes
    nlinarith
  have hu (i : ι) : cubeContentLearnedRoutes (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel W i ≤ 2 * cubeContentNormalizedGain ι floor := by
    have h := mul_le_mul_of_nonneg_left (hW (pick i)).2 hg
    unfold cubeContentLearnedRoutes
    nlinarith
  refine ⟨hl, ?_⟩
  have hs : (∑ i, cubeContentLearnedRoutes (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel W i) ≤
        ∑ i : ι, 2 * cubeContentNormalizedGain ι floor := Finset.sum_le_sum (fun i hi => hu i)
  rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ] at hs
  have he := cubeContentNormalizedGain_total floor hr
  rwa [he] at hs

/-- Opposite learned XOR coefficients generate identity and nonidentity routes in one box. -/
example (edge : Fin 2) : cubeContentLearnedRoutes
    (fun _ : Fin 2 => cubeContentNormalizedGain (Fin 2) (3 / 4))
    (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => (1 : Fin 2)) (0 : Fin 1)
      !![0; if edge = 0 then -1 else 1] ∈ cubeContentRouteDomain (Fin 2) (3 / 4) := by
  apply cubeContentNormalizedRoutes_mem _ _ _ _ (by norm_num) (by norm_num)
  intro k
  fin_cases edge <;> fin_cases k <;> norm_num

/-- The full joint implementation domain is exactly a small learned coefficient box.
Source: sufficient normalized affine budgets following sparsemax Eq. (1). -/
theorem cubeContentNormalizedDomain_iff {D : ℕ} (floor : ℝ) (pick : ι → κ) (channel : Fin D)
    (W : Matrix κ (Fin D) ℝ) (hf : floor ≤ 1) (hr : 0 < Fintype.card ι) :
    W ∈ cubeContentLearningDomain 1 floor (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel ↔ ∀ k d, -1 ≤ W k d ∧ W k d ≤ 1 := by
  constructor
  · exact fun h => h.1
  · intro h
    exact ⟨h, cubeContentNormalizedRoutes_mem floor pick channel W hf hr (fun k => h k channel)⟩

/-- A genuinely nonconstant coefficient table inhabits every box-equivalence premise. -/
example : (!![0; 1] : Matrix (Fin 2) (Fin 1) ℝ) ∈
    cubeContentLearningDomain 1 (3 / 4) (fun _ : Fin 2 => cubeContentNormalizedGain (Fin 2) (3 / 4))
      (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => (1 : Fin 2)) 0 ↔
      ∀ k : Fin 2, ∀ d : Fin 1, -1 ≤ (!![0; 1] : Matrix (Fin 2) (Fin 1) ℝ) k d ∧
        (!![0; 1] : Matrix (Fin 2) (Fin 1) ℝ) k d ≤ 1 :=
  cubeContentNormalizedDomain_iff _ _ _ _ (by norm_num) (by norm_num)

/-- The normalized joint content domain is compact, independently of the virtual-state dictionary size.
Source: exact coefficient-box topology after sparsemax Eq. (1). -/
theorem cubeContentNormalizedDomain_compact {D : ℕ} (floor : ℝ) (pick : ι → κ) (channel : Fin D)
    (hf : floor ≤ 1) (hr : 0 < Fintype.card ι) :
    IsCompact (cubeContentLearningDomain 1 floor (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel) := by
  have he : cubeContentLearningDomain 1 floor (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel =
        ((Set.Icc (-1 : ℝ) 1).matrix : Set (Matrix κ (Fin D) ℝ)) := by
    ext W
    exact cubeContentNormalizedDomain_iff floor pick channel W hf hr
  rw [he]
  exact (isCompact_Icc : IsCompact (Set.Icc (-1 : ℝ) 1)).matrix

/-- A nonlinear-content response chart with two learned rows is a real compact instance. -/
example : IsCompact (cubeContentLearningDomain 1 (3 / 4)
    (fun _ : Fin 2 => cubeContentNormalizedGain (Fin 2) (3 / 4))
    (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => (1 : Fin 2)) (0 : Fin 1)) :=
  cubeContentNormalizedDomain_compact _ _ _ (by norm_num) (by norm_num)

/-- A zero response is feasible without supplying an optimizer or a fitted endpoint.
Source: nonemptiness of the normalized joint content chart following sparsemax Eq. (1). -/
theorem zero_mem_cubeContentNormalizedDomain {D : ℕ} (floor : ℝ) (pick : ι → κ) (channel : Fin D)
    (hf : floor ≤ 1) (hr : 0 < Fintype.card ι) :
    (0 : Matrix κ (Fin D) ℝ) ∈
      cubeContentLearningDomain 1 floor (fun _ => cubeContentNormalizedGain ι floor)
        (fun _ => cubeContentNormalizedGain ι floor) pick channel := by
  rw [cubeContentNormalizedDomain_iff floor pick channel _ hf hr]
  intro k d
  norm_num

/-- Two actual content bits give a nonempty small coefficient-box witness. -/
example : (0 : Matrix (Fin 2) (Fin 1) ℝ) ∈ cubeContentLearningDomain 1 (3 / 4)
    (fun _ : Fin 2 => cubeContentNormalizedGain (Fin 2) (3 / 4))
    (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => (1 : Fin 2)) 0 :=
  zero_mem_cubeContentNormalizedDomain _ _ _ (by norm_num) (by norm_num)

/-- Every bounded finite answer function has full learned coefficients feasible in the genuine joint box.
Source: normalized budgets and bounded content kernel inversion after sparsemax Eq. (1). -/
theorem cubeContentFullLearning_mem [DecidableEq ι] {D : ℕ} (floor : ℝ)
    (pick : ι → Finset ι) (channel : Fin D) (Y : Matrix (CubeContentState ι) (Fin D) ℝ)
    (hf : floor ≤ 1) (hr : 0 < Fintype.card ι) (hY : ∀ x d, -1 ≤ Y x d ∧ Y x d ≤ 1) :
    cubeContentFullCoefficients Y ∈ cubeContentLearningDomain 1 floor
      (fun _ => cubeContentNormalizedGain ι floor) (fun _ => cubeContentNormalizedGain ι floor) pick channel := by
  rw [cubeContentNormalizedDomain_iff floor pick channel _ hf hr]
  exact cubeContentFullCoefficients_bounds Y hY

/-- Full nonlinear XOR answers inhabit every bounded-target joint feasibility premise. -/
example : cubeContentFullCoefficients (Matrix.of (fun x : CubeContentState (Fin 2) => fun _ : Fin 1 =>
    cubeContentCharacter ({0, 1} : Finset (Fin 2)) x)) ∈ cubeContentLearningDomain 1 (3 / 4)
      (fun _ : Fin 2 => cubeContentNormalizedGain (Fin 2) (3 / 4))
      (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => ({0, 1} : Finset (Fin 2))) 0 := by
  apply cubeContentFullLearning_mem _ _ _ _ (by norm_num) (by norm_num)
  intro x d
  change -1 ≤ cubeContentCharacter ({0, 1} : Finset (Fin 2)) x ∧ cubeContentCharacter {0, 1} x ≤ 1
  rw [cubeContentXor_eq]
  split_ifs <;> norm_num

/-- Actual jointly learned sparsemax/common values fit all bounded finite content answers on one box.
Source: the full coefficient kernel inverse and generated physical Eq. (1) forward; capacity is exponential. -/
theorem cubeContentFullLearnedForward_eq [DecidableEq ι] {R D : ℕ} (floor : ℝ)
    (code : Fin R → CubeContentState ι) (pick : ι → Finset ι) (channel : Fin D)
    (Y : Matrix (CubeContentState ι) (Fin D) ℝ) (hf : 1 / 2 < floor) (hf1 : floor ≤ 1)
    (hr : 0 < Fintype.card ι) (hY : ∀ x d, -1 ≤ Y x d ∧ Y x d ≤ 1) :
    cubeContentLearnedForward code id (fun _ => cubeContentNormalizedGain ι floor)
      (fun _ => cubeContentNormalizedGain ι floor) pick channel (cubeContentFullCoefficients Y) =
        Matrix.of (fun r d => Y (code r) d) := by
  rw [cubeContentLearnedForward_eq 1 floor _ _ _ _ _ _ _ hf
    (cubeContentFullLearning_mem floor pick channel Y hf1 hr hY), cubeContentFullOutput_eq]
  rfl

/-- A two-bit XOR query is fit by genuinely feasible jointly learned physical parameters. -/
example : cubeContentLearnedForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0)) id
    (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4))
    (fun _ => cubeContentNormalizedGain (Fin 2) (3 / 4)) (fun _ => ({0, 1} : Finset (Fin 2))) (0 : Fin 1)
    (cubeContentFullCoefficients (Matrix.of (fun x : CubeContentState (Fin 2) => fun _ : Fin 1 =>
      cubeContentCharacter ({0, 1} : Finset (Fin 2)) x))) = Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (-1 : ℝ)) := by
  rw [cubeContentFullLearnedForward_eq (3 / 4) _ _ _ _ (by norm_num) (by norm_num) (by norm_num)]
  · ext r d
    norm_num [cubeContentCharacter, cubeContentSign]
  · intro x d
    change -1 ≤ cubeContentCharacter ({0, 1} : Finset (Fin 2)) x ∧ cubeContentCharacter {0, 1} x ≤ 1
    rw [cubeContentXor_eq]
    split_ifs <;> norm_num

end Transformer.GPTMini.Sparsemax
