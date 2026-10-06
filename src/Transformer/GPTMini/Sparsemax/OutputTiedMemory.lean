import Transformer.GPTMini.Sparsemax.PeriodicMemoryDescent

/-!
# Affine sharing between learned answer coordinates and attention edges

New architectural restriction after arXiv:1602.02068v2, Eq. (1).
Each learned path weight equals a fixed offset plus a fixed gain times
the sum of adjacent learned output coordinates in one designated channel.
The coordinates are model parameters, not supplied attention labels. Q/K
and all common values are still decoded from the learned joint point.

The tie is affine, hence preserves the convex energy-coupled domain. With
every registered prototype observed, actual ordinary predictions determine
all intrinsic parameters. Independent edge/norm/frame gauges are removed
by this explicit parameter sharing and the canonical width-three chart.
Offsets, gain, channel and possible local connections are fixed choices;
jointly learning the readout is not claimed to remain convex.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Fixed affine parameter sharing from adjacent learned answer coordinates to path edges.
Source: the new convex restriction of the sparsemax chart following Eq. (1). -/
def outputTiedMemoryEdges {N D : ℕ} (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) : Fin N → ℝ :=
  fun e => offset e + gain * (Z e.castSucc channel + Z e.succ channel)

/-- Every output-affine combination obeys the exact edge-affine sharing rule.
Source: the new architectural tie before arXiv:1602.02068v2, Eq. (1). -/
theorem outputTiedMemoryEdges_affine {N D : ℕ} (offset : Fin N → ℝ) (gain : ℝ)
    (channel : Fin D) (Z W : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (a b : ℝ) (hab : a + b = 1) :
    outputTiedMemoryEdges offset gain channel (a • Z + b • W) =
      a • outputTiedMemoryEdges offset gain channel Z +
        b • outputTiedMemoryEdges offset gain channel W := by
  funext e
  change offset e + gain * ((a * Z e.castSucc channel + b * W e.castSucc channel) +
    (a * Z e.succ channel + b * W e.succ channel)) =
      a * (offset e + gain * (Z e.castSucc channel + Z e.succ channel)) +
        b * (offset e + gain * (W e.castSucc channel + W e.succ channel))
  have ha : a = 1 - b := by linarith
  rw [ha]
  ring

/-- Nonconstant answer tables and nonzero gain inhabit all affine-tie premises. -/
example : outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1)
    ((1 / 2 : ℝ) • taskEnergyTarget 0 + (1 / 2 : ℝ) • taskEnergyTarget 1) =
    (1 / 2 : ℝ) • outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0
      (taskEnergyTarget 0) +
    (1 / 2 : ℝ) • outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0
      (taskEnergyTarget 1) :=
  outputTiedMemoryEdges_affine _ _ _ _ _ _ _ (by norm_num)

/-- Zero learned outputs leave exactly the fixed offset, for every gain and memory size.
Source: the explicit affine sharing formula preceding sparsemax Eq. (1). -/
theorem outputTiedMemoryEdges_zero {N D : ℕ} (offset : Fin N → ℝ) (gain : ℝ)
    (channel : Fin D) :
    outputTiedMemoryEdges offset gain channel (0 : Matrix (Fin (N + 1)) (Fin D) ℝ) = offset := by
  funext e
  change offset e + gain * (0 + 0) = offset e
  ring

/-- The true joint domain intersected with the explicit affine parameter-sharing graph.
Source: the new restricted training block after arXiv:1602.02068v2, Eq. (1). -/
def outputTiedEnergyDomain (N D : ℕ) (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D) :
    Set ((Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) :=
  {x | x ∈ periodicEnergyDomain N D floor budget energy ∧
    x.1 = outputTiedMemoryEdges offset gain channel x.2}

/-- The tied domain stays jointly convex in all intrinsic learned parameters.
Source: the affine restriction of the energy-coupled sparsemax chart after Eq. (1). -/
theorem outputTiedEnergyDomain_convex (N D : ℕ) (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D) :
    Convex ℝ (outputTiedEnergyDomain N D floor budget energy offset gain channel) := by
  intro x hx y hy a b ha hb hab
  refine ⟨periodicEnergyDomain_convex N D floor budget energy hx.1 hy.1 ha hb hab, ?_⟩
  change a • x.1 + b • y.1 = outputTiedMemoryEdges offset gain channel (a • x.2 + b • y.2)
  rw [hx.2, hy.2, outputTiedMemoryEdges_affine offset gain channel x.2 y.2 a b hab]

/-- The tied constant-width domain is inhabited at every dictionary size with nonzero gain.
Source: zero-output feasibility and the explicit affine restriction after sparsemax Eq. (1). -/
theorem zero_mem_outputTiedEnergyDomain (N D : ℕ) (channel : Fin D) (floor : ℝ)
    (hf : floor ≤ 1) :
    ((0 : Fin N → ℝ), (0 : Matrix (Fin (N + 1)) (Fin D) ℝ)) ∈
      outputTiedEnergyDomain N D floor 0 1 0 1 channel := by
  refine ⟨zero_mem_periodicEnergyDomain N D floor hf, ?_⟩
  exact (outputTiedMemoryEdges_zero 0 1 channel).symm

/-- Eight prototypes inhabit all zero-domain premises without increasing physical width. -/
example : ((0 : Fin 7 → ℝ), (0 : Matrix (Fin 8) (Fin 2) ℝ)) ∈
    outputTiedEnergyDomain 7 2 (3 / 4) 0 1 0 1 0 :=
  zero_mem_outputTiedEnergyDomain _ _ _ _ (by norm_num)

/-- Equal learned output tables force equal complete intrinsic parameters on the tied domain.
Source: the actual affine sharing restriction after sparsemax Eq. (1). -/
theorem outputTiedEnergy_output_injective {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hZ : x.2 = y.2) : x = y := by
  apply Prod.ext
  · rw [hx.2, hy.2, hZ]
  · exact hZ

/-- Nonzero-gain feasible points inhabit every intrinsic-injectivity premise. -/
example : ((0 : Fin 7 → ℝ), (0 : Matrix (Fin 8) (Fin 2) ℝ)) = (0, 0) :=
  outputTiedEnergy_output_injective (3 / 4) 0 1 0 1 0 _ _
    (zero_mem_outputTiedEnergyDomain _ _ _ _ (by norm_num))
    (zero_mem_outputTiedEnergyDomain _ _ _ _ (by norm_num)) rfl

/-- Registered actual predictions determine all tied learned parameters, including geometry.
Source: true masked sparsemax/common values and the new affine tie after Eq. (1).
Every prototype is observed; no assumption that sparsemax itself is injective is made. -/
theorem outputTiedEnergyForward_injective {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (x y : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hx : x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hy : y ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel)
    (hp : periodicEnergyForward (fun j : Fin (N + 1) => j) x =
      periodicEnergyForward (fun j : Fin (N + 1) => j) y) : x = y := by
  have he := hp
  rw [periodicEnergyForward_eq floor budget energy _ x hf hx.1,
    periodicEnergyForward_eq floor budget energy _ y hf hy.1] at he
  exact outputTiedEnergy_output_injective floor budget energy offset gain channel x y hx hy he

/-- Genuine decoded width-three attention inhabits the registered-prediction premises. -/
example : ((0 : Fin 7 → ℝ), (0 : Matrix (Fin 8) (Fin 2) ℝ)) = (0, 0) :=
  outputTiedEnergyForward_injective (3 / 4) 0 1 0 1 0 _ _ (by norm_num)
    (zero_mem_outputTiedEnergyDomain _ _ _ _ (by norm_num))
    (zero_mem_outputTiedEnergyDomain _ _ _ _ (by norm_num)) rfl

/-- Any convex ordinary criterion remains jointly convex under affine parameter sharing.
Source: the genuine masked sparsemax chart and its new affine restriction after Eq. (1). -/
theorem outputTiedEnergyObjective_convex {R N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D) (code : Fin R → Fin (N + 1))
    (objective : Matrix (Fin R) (Fin D) ℝ → ℝ)
    (hf : 1 / 2 < floor) (hl : ConvexOn ℝ Set.univ objective) :
    ConvexOn ℝ (outputTiedEnergyDomain N D floor budget energy offset gain channel)
      (fun x => objective (periodicEnergyForward code x)) := by
  refine ⟨outputTiedEnergyDomain_convex N D floor budget energy offset gain channel, ?_⟩
  intro x hx y hy a b ha hb hab
  exact (periodicEnergyObjective_convex floor budget energy code objective hf hl).2
    hx.1 hy.1 ha hb hab

/-- A nonconstant ordinary answer criterion inhabits all tied joint-convexity premises. -/
example : ConvexOn ℝ (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) 0)
    (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0)) :=
  outputTiedEnergyObjective_convex _ _ _ _ _ _ _ _ (by norm_num) (matrixOutputError_convex _)

end Transformer.GPTMini.Sparsemax
