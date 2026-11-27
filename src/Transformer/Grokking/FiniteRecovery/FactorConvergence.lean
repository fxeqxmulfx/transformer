module

public import Transformer.Grokking.FiniteRecovery.Examples
public import Transformer.Grokking.FiniteRecovery.PhaseDecoding
public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun
public import Transformer.Grokking.FlatSpectrum.Training.Recovery
public import Transformer.Grokking.FlatSpectrum.Training.UnitBall
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitBall

/-! A continuation of actual squared-loss factor runs from the stacked
identity. At each positive decay the inner run converges. The outer
limit as decay vanishes recovers the observed entries, and the entire
phase table whenever nuclear completion is unique. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Matrix Filter Transformer.Grokking.FlatSpectrum
open Transformer.Grokking.FlatSpectrum.Training
open scoped Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- The decay used for the checked examples. -/
noncomputable def decaySchedule (j : ℕ) : ℝ := 1 / ((j : ℝ) + 1)

theorem decaySchedule_pos (j : ℕ) : 0 < decaySchedule j := by
  unfold decaySchedule
  positivity

theorem decaySchedule_tendsto : Tendsto decaySchedule atTop (𝓝 0) := by
  change Tendsto (fun j : ℕ => 1 / ((j : ℝ) + 1)) atTop (𝓝 0)
  simpa only [Nat.cast_add, Nat.cast_one] using
    (tendsto_add_atTop_iff_nat 1).2 (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ))

omit [DecidableEq n] in
/-- Zeroing the unobserved labels preserves the squared objective. -/
theorem fitLoss_mask_labels (Ω : Set (m × n)) (K : Matrix m n ℂ) :
    fitLoss Ω (mask Ω K) = fitLoss Ω K := by
  funext X
  have he : mask Ω (X - mask Ω K) = mask Ω (X - K) := by
    ext i j
    by_cases hij : (i, j) ∈ Ω <;> simp [mask, hij]
  simp only [fitLoss, he]

/-- At each decay, the displayed factors are limits of descent from the
stacked identity. The run only receives the mask and the observed labels. -/
def ObservedFactorRun [DecidableEq m] (Ω : Set (m × n)) (K : Matrix m n ℂ) (lams : ℕ → ℝ)
    (Us : ℕ → Matrix m (m ⊕ n) ℂ) (Vs : ℕ → Matrix n (m ⊕ n) ℂ) : Prop :=
  ∀ j, IsDecayedMin (fitLoss Ω K) (lams j) (Us j) (Vs j) ∧
    ∃ η > 0, Tendsto (decayRun (fun X => mask Ω (X - mask Ω K)) η (lams j)
      (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ))
      (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ))) atTop (𝓝 (Us j, Vs j))

theorem exists_observedFactorRun [DecidableEq m] (Ω : Set (m × n)) (K : Matrix m n ℂ)
    (lams : ℕ → ℝ) (hpos : ∀ j, 0 < lams j) :
    ∃ Us Vs, ObservedFactorRun Ω K lams Us Vs := by
  choose η hη Us Vs hlim hmin using fun j =>
    exists_tendsto_decayRun_fitLoss_fullStart Ω (mask Ω K) (hpos j)
  refine ⟨Us, Vs, fun j => ⟨?_, η j, hη j, hlim j⟩⟩
  simpa only [fitLoss_mask_labels] using hmin j

/-- Every observed phase converges along the continuation. -/
theorem ObservedFactorRun.tendsto_cell [DecidableEq m]
    {Ω : Set (m × n)} {K : Matrix m n ℂ} {lams : ℕ → ℝ}
    {Us : ℕ → Matrix m (m ⊕ n) ℂ} {Vs : ℕ → Matrix n (m ⊕ n) ℂ}
    (run : ObservedFactorRun Ω K lams Us Vs) (hpos : ∀ j, 0 < lams j)
    (hzero : Tendsto lams atTop (𝓝 0)) (i : m) (j : n) (hij : (i, j) ∈ Ω) :
    Tendsto (fun t => (Us t * (Vs t)ᴴ) i j) atTop (𝓝 (K i j)) := by
  let P := fun t => Us t * (Vs t)ᴴ
  have hupper : Tendsto (fun t => lams t * nuclearNorm K) atTop (𝓝 0) := by
    simpa only [zero_mul] using hzero.mul_const (nuclearNorm K)
  have hfit : Tendsto (fun t => fitLoss Ω K (P t)) atTop (𝓝 0) :=
    squeeze_zero (fun t => fitLoss_nonneg Ω K (P t))
      (fun t => (run t).1.fitLoss_le (hpos t).le (X' := K) (fun _ _ => rfl)) hupper
  have hsq : Tendsto (fun t => frobeniusSq (mask Ω (P t - K))) atTop (𝓝 0) := by
    have h := hfit.const_mul 2
    simp only [fitLoss, mul_zero, mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)] at h
    exact h
  have hm := Scope.tendsto_zero_of_tendsto_frobeniusSq hsq
  have hc := (tendsto_pi_nhds.mp ((tendsto_pi_nhds.mp hm) i)) j
  have hdiff : Tendsto (fun t => P t i j - K i j) atTop (𝓝 0) := by
    simpa [mask, hij] using hc
  simpa only [zero_add, sub_add_cancel] using hdiff.add_const (K i j)

/-- With unique nuclear completion, the complete phase prediction converges. -/
theorem exists_observedFactorRun_tendsto_kernel [DecidableEq m] {Ω : Set (m × n)}
    {K : Matrix m n ℂ} (hrec : NuclearRecovers Ω K) :
    ∃ Us Vs, ObservedFactorRun Ω K decaySchedule Us Vs ∧
      Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 K) := by
  obtain ⟨Us, Vs, hr⟩ := exists_observedFactorRun Ω K decaySchedule decaySchedule_pos
  exact ⟨Us, Vs, hr,
    tendsto_mul_conjTranspose_of_forall_isMinOn hrec decaySchedule_pos decaySchedule_tendsto
      (fun j => (hr j).1)⟩

/-- A kernel with a smaller-norm feasible competitor cannot be the limit
of the raw trained products at any sequence of positive decays. -/
theorem not_tendsto_decayedMin_of_not_isMinOn [Fintype k] {Ω : Set (m × n)}
    {K : Matrix m n ℂ} (h : ¬ IsMinOn nuclearNorm (completions Ω K) K)
    {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}
    (hmin : ∀ j, IsDecayedMin (fitLoss Ω K) (lams j) (Us j) (Vs j)) :
    ¬ Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 K) := by
  intro ht
  apply h
  apply isMinOn_iff.mpr
  intro X hX
  exact nuclearNorm_le_of_tendsto ht (Filter.Eventually.of_forall fun j =>
    (hmin j).nuclearNorm_le (hpos j) hX)

variable {p length size : ℕ} [NeZero p]

/-- Modular sums of any even length converge from the staircase mask. -/
theorem exists_modularSum_factorRun_tendsto (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    let K := targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e)
    ∃ Us Vs, ObservedFactorRun (staircase size) K decaySchedule Us Vs ∧
      Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 K) :=
  exists_observedFactorRun_tendsto_kernel (modularSum_nuclear_recovers e)

theorem exists_modularSum_factorRun_eventually_exact (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    let cut := enumeratedCut e
    let code := ZMod.stdAddChar (N := p)
    ∃ Us Vs, ObservedFactorRun (staircase size) (targetKernel cut code (modularSum e)) decaySchedule Us Vs ∧
      ∀ᶠ step in atTop, (fun word => phaseDecode code
        ((Us step * (Vs step)ᴴ) (cut.symm word).1 (cut.symm word).2)) = modularSum e := by
  obtain ⟨Us, Vs, hr, ht⟩ := exists_modularSum_factorRun_tendsto e
  exact ⟨Us, Vs, hr, phaseDecodeKernel_eventually _ _ ZMod.injective_stdAddChar _ _ ht⟩

example := exists_modularSum_factorRun_tendsto (size := 8) (wordEnumeration 3 2)
example := exists_modularSum_factorRun_tendsto (size := 15) (wordEnumeration 2 4)
example := exists_modularSum_factorRun_tendsto (size := 255) (wordEnumeration 2 8)
example := exists_modularSum_factorRun_tendsto (size := 65535) (wordEnumeration 2 16)
example := exists_modularSum_factorRun_eventually_exact (size := 65535) (wordEnumeration 2 16)

end Transformer.Grokking.FiniteRecovery
