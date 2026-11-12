module

public import Transformer.Grokking.Protocol.RenameSymmetry
public import Transformer.Grokking.Protocol.LabelProbability

/-!
# A split that a renaming of the tokens carries to another split

Source: arXiv:2201.02177v1 (Power et al.), section 1, "distinct abstract symbols", and the
appendix *Additional experimental details*; openai/grok at 3d64b1d8;
`experiments/grok_division/PROTOCOL.md`, *The split* and *Randomness*.

`Exchange` and `ReadoutSymmetry` use renamings of the tokens that leave every training
example alone. A renaming can also move the training examples among themselves. A renaming
`τ` of the tokens and a reordering `ρ` of the equations carry a split `π` (`Carries`) when in
every training slot the example of the reordered split `ρ * π`, renamed, is the example of
`π`. Then every gradient of the run of `ρ * π` at the permuted parameters is the permuted
gradient of the run of `π` (`divisionGrad_carried`), so the run of `ρ * π` from the permuted
initial parameters is the permuted run of `π`, with the same shuffles of the epochs
(`trajectory_carried`), under Adam and under AdamW alike.

The law of a run does not see the change: composing the split with a fixed reordering
preserves the uniform law (`measurePreserving_mul_left_uniformPerm`), and a renaming preserves
the law of the initial parameters (`measurePreserving_comp_paramPerm`), so
`carrySample` preserves the law of a sample (`measurePreserving_carrySample`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- **A renaming `τ` of the tokens and a reordering `ρ` of the equations carry the split
`π`**: in every training slot the example of the split `ρ * π`, renamed, is that of `π`. -/
def Carries (τ : Equiv.Perm (Fin 239)) (ρ π : Equiv.Perm (Fin (97 * 96))) : Prop :=
  ∀ i : Fin 4656, (supervised (trainEq (ρ * π) i)).rename τ = supervised (trainEq π i)

variable {τ : Equiv.Perm (Fin 239)} {ρ π : Equiv.Perm (Fin (97 * 96))}

/-- **On a carried split the gradient of a batch of the reordered split at the permuted
parameters is the permuted gradient of the same slots of the split.** -/
theorem divisionGrad_carried (h : Carries τ ρ π) (slots : List (Fin 4656))
    (θ : Param paperSizes → ℝ) :
    divisionGrad (slots.map (trainEq (ρ * π))) (θ ∘ paramPerm τ) =
      divisionGrad (slots.map (trainEq π)) θ ∘ paramPerm τ := by
  funext p
  have hB : ((slots.map (trainEq (ρ * π))).map supervised).map (Example.rename τ) =
      (slots.map (trainEq π)).map supervised := by
    simp only [List.map_map]
    exact List.map_congr_left fun i _ => h i
  rw [divisionGrad, sourceGrad_comp_paramPerm_rename, hB]
  rfl

/-- **The run of a carried split from the permuted initial parameters is the permuted run**,
with the same shuffles of the epochs, at every weight decay. -/
theorem trajectory_carried (decay : ℝ) (h : Carries τ ρ π) (θ₀ : Param paperSizes → ℝ)
    (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (ρ * π, θ₀ ∘ paramPerm τ, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ paramPerm τ :=
  (adamRun_comp (paramPerm (S := paperSizes) τ)
    (fun s θ => divisionGrad_carried h (batchSlots 4656 512 σ s) θ) θ₀ t).1

/-- **Composing the split with a fixed reordering preserves the uniform law.** -/
theorem measurePreserving_mul_left_uniformPerm {n : ℕ} (ρ : Equiv.Perm (Fin n)) :
    MeasurePreserving (fun π : Equiv.Perm (Fin n) => ρ * π) (uniformPerm n) (uniformPerm n) := by
  have hm : Measurable fun π : Equiv.Perm (Fin n) => ρ * π :=
    fun _ _ => MeasurableSpace.measurableSet_top
  refine ⟨hm, Measure.ext_of_singleton fun π => ?_⟩
  have hpre : (fun π' : Equiv.Perm (Fin n) => ρ * π') ⁻¹' {π} = {ρ⁻¹ * π} := by
    ext π'
    simp [eq_inv_mul_iff_mul_eq]
  rw [Measure.map_apply hm MeasurableSpace.measurableSet_top, hpre, uniformPerm,
    Measure.uniformOfFinset_apply_singleton_of_mem (Finset.mem_univ _),
    Measure.uniformOfFinset_apply_singleton_of_mem (Finset.mem_univ _)]

variable (τ) in
/-- A renaming keeps the kind of a parameter, hence its initialization law. -/
lemma initLaw_paramPerm (p : Param paperSizes) :
    initLaw paperSizes (paramPerm τ p) = initLaw paperSizes p := by
  cases p <;> rfl

variable (τ) in
/-- **A renaming of the tokens preserves the law of the initial parameters.** -/
theorem measurePreserving_comp_paramPerm :
    MeasurePreserving (fun θ : Param paperSizes → ℝ => θ ∘ paramPerm τ) (initMeasure paperSizes)
      (initMeasure paperSizes) :=
  measurePreserving_comp_of_initLaw (paramPerm τ) (initLaw_paramPerm τ)

variable (τ ρ) in
/-- The sample with its split reordered and its initial parameters permuted. -/
def carrySample : Sample (97 * 96) 4656 (Param paperSizes) →
    Sample (97 * 96) 4656 (Param paperSizes) :=
  Prod.map (fun π => ρ * π) (Prod.map (fun θ => θ ∘ paramPerm τ) id)

variable (τ ρ) in
/-- **Reordering the split and permuting the initial parameters preserves the law of a
run.** -/
theorem measurePreserving_carrySample :
    MeasurePreserving (carrySample τ ρ) (law (97 * 96) 4656 (initMeasure paperSizes))
      (law (97 * 96) 4656 (initMeasure paperSizes)) :=
  (measurePreserving_mul_left_uniformPerm ρ).prod
    ((measurePreserving_comp_paramPerm τ).prod (MeasurePreserving.id _))

/-! The witness: the exchange of the two hidden tokens with no reordering carries every
uncovered split, because it leaves its training examples alone. -/

/-- The exchange of the hidden tokens carries every uncovered split to itself. -/
theorem carries_hiddenSwap_one {π : Equiv.Perm (Fin (97 * 96))} (hπ : π ∈ uncovered) :
    Carries hiddenSwap 1 π := fun i => by
  rw [one_mul]
  exact Example.rename_of_fixed (S := paperSizes) hiddenSwap (fixed_supervised (hπ i))

example (decay : ℝ) (θ₀ : Param paperSizes → ℝ) (σ : ℕ → Equiv.Perm (Fin 4656)) (t : ℕ) :
    ∃ π, trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (1 * π, θ₀ ∘ paramPerm hiddenSwap, σ) t =
      trajectory (Hyper.source decay) divisionGrad equations (by norm_num : 4656 ≤ 97 * 96) 512
        (π, θ₀, σ) t ∘ paramPerm hiddenSwap := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨π, trajectory_carried decay (carries_hiddenSwap_one hπ) θ₀ σ t⟩

example (slots : List (Fin 4656)) (θ : Param paperSizes → ℝ) :
    ∃ π, divisionGrad (slots.map (trainEq (1 * π))) (θ ∘ paramPerm hiddenSwap) =
      divisionGrad (slots.map (trainEq π)) θ ∘ paramPerm hiddenSwap := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact ⟨π, divisionGrad_carried (carries_hiddenSwap_one hπ) slots θ⟩

end Transformer.Grokking.Protocol
