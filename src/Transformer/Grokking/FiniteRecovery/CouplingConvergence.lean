module

public import Transformer.Grokking.FiniteRecovery.FactorConvergence
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatBest

/-! Convergent phase forecasts for the paired examples. The two axes of
the observation cross have only the additive side terms. A cross readout
combines their fitted phases with the coupling. If the coupling is unknown,
the additional unit-vector cells in PT4 give it by mixed differences. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Filter Matrix Transformer.Grokking.FlatSpectrum
open Transformer.Grokking.FlatSpectrum.TablePart
open scoped Topology

variable {p left right T : ℕ} [NeZero p]

/-- Complete from the fitted cross and insert the phase of the interaction.
This readout only uses entries in the zero row and zero column of the fit. -/
noncomputable def coupledCrossReadout (M : Matrix (Fin left) (Fin right) (ZMod p))
    (P : Matrix (Word (ZMod p) left) (Word (ZMod p) right) ℂ) :
    Matrix (Word (ZMod p) left) (Word (ZMod p) right) ℂ :=
  Matrix.of fun u v => P u 0 * P 0 v / P 0 0 * ZMod.stdAddChar (u ⬝ᵥ M *ᵥ v)

theorem coupledCrossReadout_target
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) :
    let K := targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)
    coupledCrossReadout M K = K := by
  rw [targetKernel_bilinearTask]
  ext u v
  have h₁ : ZMod.stdAddChar (g 0) ≠ 0 := norm_ne_zero_iff.mp (by simp)
  have h₂ : ZMod.stdAddChar (h 0) ≠ 0 := norm_ne_zero_iff.mp (by simp)
  simp only [coupledCrossReadout, Matrix.of_apply, bilinearKernel, mulVec_zero, dotProduct_zero,
    zero_dotProduct, AddChar.map_add_eq_mul, AddChar.map_zero_eq_one, mul_one]
  field_simp [h₁, h₂]

/-- Descent fits the observed arms of the cross. Their limit determines
the phase at every held-out point through the coupling-aware readout. -/
theorem exists_bilinear_factorRun_tendsto_cross
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p))
    (Ω : Set (Word (ZMod p) left × Word (ZMod p) right))
    (hrow : ∀ v, (0, v) ∈ Ω) (hcol : ∀ u, (u, 0) ∈ Ω) :
    let K := targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)
    ∃ Us Vs, ObservedFactorRun Ω K decaySchedule Us Vs ∧
      Tendsto (fun step => coupledCrossReadout M (Us step * (Vs step)ᴴ)) atTop (𝓝 K) := by
  let K := targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)
  obtain ⟨Us, Vs, hr⟩ := exists_observedFactorRun Ω K decaySchedule decaySchedule_pos
  have hz : K 0 0 ≠ 0 := by
    change (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)) 0 0 ≠ 0
    rw [targetKernel_bilinearTask]
    exact bilinearKernel_ne_zero p g h M 0 0
  refine ⟨Us, Vs, hr, ?_⟩
  have hlim : Tendsto (fun step => coupledCrossReadout M (Us step * (Vs step)ᴴ))
      atTop (𝓝 (coupledCrossReadout M K)) := by
    apply tendsto_pi_nhds.mpr
    intro u
    apply tendsto_pi_nhds.mpr
    intro v
    have ht (u v) (huv : (u, v) ∈ Ω) := hr.tendsto_cell decaySchedule_pos decaySchedule_tendsto u v huv
    exact (((ht u 0 (hcol u)).mul (ht 0 v (hrow v))).div (ht 0 0 (hrow 0)) hz).mul_const
      (ZMod.stdAddChar (u ⬝ᵥ M *ᵥ v))
  rwa [coupledCrossReadout_target cut g h M] at hlim

/-- Convergence of the coupled fit eventually gives the exact token at
every word, with the same observations as the phase readout. -/
theorem exists_bilinear_factorRun_eventually_exact
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p))
    (Ω : Set (Word (ZMod p) left × Word (ZMod p) right))
    (hrow : ∀ v, (0, v) ∈ Ω) (hcol : ∀ u, (u, 0) ∈ Ω) :
    ∃ Us Vs, ObservedFactorRun Ω (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M))
        decaySchedule Us Vs ∧
      ∀ᶠ step in atTop, (fun word => phaseDecode ZMod.stdAddChar
        (coupledCrossReadout M (Us step * (Vs step)ᴴ) (cut.symm word).1 (cut.symm word).2)) =
        bilinearTask cut g h M := by
  obtain ⟨Us, Vs, hr, ht⟩ := exists_bilinear_factorRun_tendsto_cross cut g h M Ω hrow hcol
  exact ⟨Us, Vs, hr, phaseDecodeKernel_eventually _ _ ZMod.injective_stdAddChar _ _ ht⟩

/-- For the flat examples, convergence of the raw factors to the true
phase table is impossible on any proper mask with positive decay. -/
theorem bilinear_not_tendsto_raw_fit {k : Type*} [Fintype k]
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    {M : Matrix (Fin left) (Fin right) (ZMod p)} (hM : Function.Injective M.vecMul)
    (inputs : Finset (Word (ZMod p) T)) (hmissing : ∃ word, word ∉ inputs)
    {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    {Us : ℕ → Matrix (Word (ZMod p) left) k ℂ} {Vs : ℕ → Matrix (Word (ZMod p) right) k ℂ}
    (hfit : ∀ j, FlatSpectrum.Training.IsDecayedMin
      (FlatSpectrum.Training.fitLoss (observedCells cut inputs)
        (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M))) (lams j) (Us j) (Vs j)) :
    ¬ Tendsto (fun j => Us j * (Vs j)ᴴ) atTop
      (𝓝 (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M))) :=
  not_tendsto_decayedMin_of_not_isMinOn (bilinearTask_not_minimum cut g h hM inputs hmissing) hpos hfit

variable [Fact p.Prime]

/-- Recover the coupling coefficients from the four corners for each
pair of coordinate vectors. -/
def couplingFromLabels
    (labels : Word (ZMod p) left × Word (ZMod p) right → ZMod p) :
    Matrix (Fin left) (Fin right) (ZMod p) :=
  Matrix.of fun i j =>
    labels (Pi.single i 1, Pi.single j 1) - labels (Pi.single i 1, 0) -
      labels (0, Pi.single j 1) + labels (0, 0)

theorem couplingFromLabels_eq
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p))
    (labels : Word (ZMod p) left × Word (ZMod p) right → ZMod p)
    (seen : ∀ cell ∈ crossPart (ZMod p) (Fin left) (Fin right),
      labels cell = bilinearTask cut g h M (cut cell)) :
    couplingFromLabels labels = M := by
  ext i j
  rw [couplingFromLabels, Matrix.of_apply,
    seen _ (Finset.mem_union_right _ (Finset.mem_image_of_mem _ (Finset.mem_univ (i, j)))),
    seen _ (Finset.mem_union_left _ (row_mem_cross (Pi.single i 1))),
    seen _ (Finset.mem_union_left _ (col_mem_cross (Pi.single j 1))),
    seen _ (Finset.mem_union_left _ (row_mem_cross 0))]
  simp [bilinearTask, dotProduct, Pi.single_apply]
  ring

/-- The cross with the unit-vector cells supports both learning the
coupling and convergent prediction of the whole phase table. -/
theorem exists_unknownCoupling_factorRun_tendsto
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p))
    (labels : Word (ZMod p) left × Word (ZMod p) right → ZMod p)
    (seen : ∀ cell ∈ crossPart (ZMod p) (Fin left) (Fin right),
      labels cell = bilinearTask cut g h M (cut cell)) :
    let K := targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)
    let Ω : Set (Word (ZMod p) left × Word (ZMod p) right) := crossPart (ZMod p) (Fin left) (Fin right)
    ∃ Us Vs, ObservedFactorRun Ω K decaySchedule Us Vs ∧
      Tendsto (fun j => coupledCrossReadout (couplingFromLabels labels) (Us j * (Vs j)ᴴ))
        atTop (𝓝 K) := by
  rw [couplingFromLabels_eq cut g h M labels seen]
  apply exists_bilinear_factorRun_tendsto_cross
  · intro v
    exact Finset.mem_union_left _ (col_mem_cross v)
  · intro u
    exact Finset.mem_union_left _ (row_mem_cross u)

example :=
  exists_bilinear_factorRun_eventually_exact (appendCut (ZMod 3) 2 2)
    (fun u => ∑ i, u i ^ 3) 0 1
    (observedCells (appendCut (ZMod 3) 2 2) fourTokenCross)
    (fun v => by simp [observedCells, fourTokenCross]) (fun u => by simp [observedCells, fourTokenCross])

example := exists_unknownCoupling_factorRun_tendsto (appendCut (ZMod 2) 4 4) 0 0 1 _ (fun _ _ => rfl)
example := exists_unknownCoupling_factorRun_tendsto (appendCut (ZMod 2) 8 8) 0 0 1 _ (fun _ _ => rfl)
example := exists_unknownCoupling_factorRun_tendsto (appendCut (ZMod 2) 16 16) 0 0 1 _ (fun _ _ => rfl)

end Transformer.Grokking.FiniteRecovery
