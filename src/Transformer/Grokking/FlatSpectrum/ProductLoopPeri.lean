module

public import Transformer.Grokking.FlatSpectrum.ProductLoopCopies

/-!
# Phasor's Peri-RMS around a product layer

Source: `review/flat_spectrum.md`, *Loops*, item LP6. Every sublayer of Phasor reads the
RMS-normalized state and normalizes what its branch writes before the depth scale `c`
(`MHC.PeriRMS.step`, `Phasor.Router.step_apply`): Peri-RMS. Its feed-forward branch is a complex
product without a gate nonlinearity (`Phasor.FFN`), the branch `B z = ∑_u (g_u z)(w_u z) o_u` of
a product layer (`productBranch`). On `n` complex coordinates, `2n` real ones, the RMS is
`R z = z √(2n) / √(‖z‖² + 2n ε)` with `‖z‖² = ∑ⱼ |zⱼ|²` (`periRMS`, `stateEnergy`), as
`GPTMini.rmsNormEps` on a stream of width `2n`. A *Peri layer* is `F z = z + c R(B(R z))`
(`periLayer`), without Phasor's write gate and stream mixing.

Both RMS multiply their argument by a real number, and `B` is homogeneous of degree two, so
`F z = z + μ B z` with one real factor `μ` of `‖z‖²` and `‖B z‖²` (`periLayer_apply`): a gate on
the whole branch, whose logit reads two norms. Where they do not depend on the input, `μ` is a
number and the layer at most doubles the order, as a product layer does (`periLayer_order`).
LP2's prelude behind Peri-RMS is such a layer when its leaves are characters: the one-hot
features have energy `2k` (`stateEnergy_features`) and the leaves have modulus one
(`productBranch_leaf_features`). With the ideal RMS, `ε = 0`, the input RMS drops out:
`F z = z + c √(2n) B z / ‖B z‖` (`periLayer_eps_zero`). `ProductLoopPeriParity` lets `‖B z‖`
vary with the input.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {ι : Type*}

/-- The branch of a product layer, `B z = ∑_u (g_u z)(w_u z) o_u`. -/
def productBranch {η : Type*} [Fintype η] (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ)
    (z : ι → ℂ) : ι → ℂ :=
  ∑ u, (gate u z * up u z) • out u

/-- The branch is homogeneous of degree two. -/
lemma productBranch_smul {η : Type*} [Fintype η] (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ)
    (out : η → ι → ℂ) (c : ℂ) (z : ι → ℂ) :
    productBranch gate up out (c • z) = c ^ 2 • productBranch gate up out z := by
  simp only [productBranch, map_smul, smul_eq_mul, Finset.smul_sum, smul_smul]
  exact Finset.sum_congr rfl fun u _ => by ring_nf

/-- The factor of Phasor's RMS on `n` complex coordinates of energy `t`: `√(2n) / √(t + 2n ε)`. -/
noncomputable def rmsFactor (n : ℕ) (eps t : ℝ) : ℝ :=
  Real.sqrt (2 * n) / Real.sqrt (t + 2 * n * eps)

/-- The factor of a Peri layer on a state of energy `a` whose branch has energy `b`:
`c ρ(ρ(a)⁴ b) ρ(a)²`, with `ρ = rmsFactor n ε`. -/
noncomputable def periFactor (n : ℕ) (eps scale a b : ℝ) : ℝ :=
  scale * rmsFactor n eps (rmsFactor n eps a ^ 4 * b) * rmsFactor n eps a ^ 2

variable [Fintype ι]

/-- The energy `‖z‖² = ∑ⱼ |zⱼ|²` of a complex state. -/
noncomputable def stateEnergy (z : ι → ℂ) : ℝ := ∑ j, ‖z j‖ ^ 2

lemma stateEnergy_smul (c : ℂ) (z : ι → ℂ) : stateEnergy (c • z) = ‖c‖ ^ 2 * stateEnergy z := by
  simp [stateEnergy, mul_pow, Finset.mul_sum]

lemma eq_zero_of_stateEnergy_eq_zero {z : ι → ℂ} (h : stateEnergy z = 0) : z = 0 := by
  funext j
  simpa using (Finset.sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg ‖z j‖).1 h j (mem_univ j)

example := eq_zero_of_stateEnergy_eq_zero (z := (0 : Unit → ℂ)) (by simp [stateEnergy])

/-- **Phasor's RMS on a complex state**: `R z = z √(2n) / √(‖z‖² + 2n ε)`, `n = card ι`. -/
noncomputable def periRMS (eps : ℝ) (z : ι → ℂ) : ι → ℂ :=
  (rmsFactor (Fintype.card ι) eps (stateEnergy z) : ℂ) • z

/-- **A product layer between Phasor's two RMS**: the branch reads the normalized state, and its
normalized output is scaled by the depth scale `c` and added, `F z = z + c R(B(R z))`. -/
noncomputable def periLayer {η : Type*} [Fintype η] (eps scale : ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) : (ι → ℂ) → ι → ℂ :=
  fun z => z + (scale : ℂ) • periRMS eps (productBranch gate up out (periRMS eps z))

/-- **The two RMS are one real factor on the branch**: `F z = z + μ B z`, with `μ` the
`periFactor` of the energies of `z` and of `B z`. -/
theorem periLayer_apply {η : Type*} [Fintype η] (eps scale : ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) (z : ι → ℂ) :
    periLayer eps scale gate up out z = z + (periFactor (Fintype.card ι) eps scale
      (stateEnergy z) (stateEnergy (productBranch gate up out z)) : ℂ) •
        productBranch gate up out z := by
  have h4 (r : ℝ) : ‖(r : ℂ) ^ 2‖ ^ 2 = r ^ 4 := by
    rw [norm_pow, Complex.norm_real, Real.norm_eq_abs, sq_abs, ← pow_mul]
  simp only [periLayer, periRMS, productBranch_smul, stateEnergy_smul, h4, smul_smul, periFactor]
  congr 2
  push_cast
  ring

/-- **Constant norms keep the count**: if the coordinates of `Z` have order at most `r` and the
energies of `Z x y` and of its branch do not depend on the input, the Peri layer is a product
layer on these states, and its coordinates have order at most `2 r`. -/
theorem periLayer_order {κ α β η : Type} [Fintype η] {r : ℕ} (eps scale : ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) {a b : ℝ}
    (ha : ∀ x y, stateEnergy (Z x y) = a)
    (hb : ∀ x y, stateEnergy (productBranch gate up out (Z x y)) = b) (c : ι) :
    (fun x y => periLayer eps scale gate up out (Z x y) c) ∈ orderSpan (2 * r) := by
  set μ : ℂ := (periFactor (Fintype.card ι) eps scale a b : ℂ)
  have hG : IsProductLayer (Fintype.card η) fun z => z + μ • productBranch gate up out z := by
    refine ⟨η, inferInstance, le_rfl, gate, up, fun u => μ • out u, fun z => ?_⟩
    simp only [productBranch, Finset.smul_sum, smul_smul]
    congr 1
    exact Finset.sum_congr rfl fun u _ => by rw [mul_comm]
  have e : (fun x y => periLayer eps scale gate up out (Z x y) c) =
      fun x y => (fun z => z + μ • productBranch gate up out z) (Z x y) c := by
    funext x y
    rw [periLayer_apply, ha, hb]
  rw [e]
  exact hG.order hZ c

/-- **The ideal RMS**: with `ε = 0` the input RMS drops out, and the layer adds
`c √(2n) B z / ‖B z‖`. -/
theorem periLayer_eps_zero {η : Type*} [Fintype η] (scale : ℝ)
    (gate up : η → (ι → ℂ) →ₗ[ℂ] ℂ) (out : η → ι → ℂ) (z : ι → ℂ) :
    periLayer 0 scale gate up out z = z + ((scale * Real.sqrt (2 * Fintype.card ι) /
      Real.sqrt (stateEnergy (productBranch gate up out z)) : ℝ) : ℂ) •
        productBranch gate up out z := by
  rw [periLayer_apply]
  by_cases hB : stateEnergy (productBranch gate up out z) = 0
  · rw [eq_zero_of_stateEnergy_eq_zero hB, smul_zero, smul_zero]
  have hz : z ≠ 0 := by
    rintro rfl
    exact hB (by simp [productBranch, stateEnergy])
  obtain ⟨j, -⟩ := Function.ne_iff.1 hz
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j⟩
  have ha : 0 < stateEnergy z := lt_of_le_of_ne (Finset.sum_nonneg fun j _ => sq_nonneg _)
    fun h => hz (eq_zero_of_stateEnergy_eq_zero h.symm)
  have hb : 0 < stateEnergy (productBranch gate up out z) :=
    lt_of_le_of_ne (Finset.sum_nonneg fun j _ => sq_nonneg _) (Ne.symm hB)
  have h2 : 0 < Real.sqrt (2 * Fintype.card ι) := Real.sqrt_pos.2 (by positivity)
  have hsa : 0 < Real.sqrt (stateEnergy z) := Real.sqrt_pos.2 ha
  have hsb : 0 < Real.sqrt (stateEnergy (productBranch gate up out z)) := Real.sqrt_pos.2 hb
  have hroot : Real.sqrt ((Real.sqrt (2 * Fintype.card ι) / Real.sqrt (stateEnergy z)) ^ 4 *
      stateEnergy (productBranch gate up out z)) =
      (Real.sqrt (2 * Fintype.card ι) / Real.sqrt (stateEnergy z)) ^ 2 *
        Real.sqrt (stateEnergy (productBranch gate up out z)) := by
    rw [Real.sqrt_mul (by positivity), show (Real.sqrt (2 * Fintype.card ι) /
      Real.sqrt (stateEnergy z)) ^ 4 = ((Real.sqrt (2 * Fintype.card ι) /
        Real.sqrt (stateEnergy z)) ^ 2) ^ 2 by ring, Real.sqrt_sq (by positivity)]
  simp only [periFactor, rmsFactor, mul_zero, add_zero, hroot]
  congr 3
  field_simp

/-- The outputs of the leaf layer: unit `(t, b)` writes into scratch slot `t`. -/
def leafOut {κ α β T : Type} [DecidableEq κ] [DecidableEq α] [DecidableEq β] [DecidableEq T]
    (tb : T × β) : ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ :=
  Pi.single (Sum.inr tb.1) 1

variable {κ α β T : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Fintype T]

/-- The one-hot features of the pairs and a zero scratch have energy `2 card κ`. -/
lemma stateEnergy_features [Fintype κ] (x : κ → α) (y : κ → β) :
    stateEnergy (Sum.elim (pairFeatures x y) (0 : T → ℂ)) = 2 * Fintype.card κ := by
  rw [stateEnergy, Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type,
    Fintype.sum_prod_type]
  simp [pairFeatures, oneHot, apply_ite]
  ring

/-- On the features and a zero scratch, the branch of the leaf layer is `G_t(x_{π t}, y_{π t})`
on scratch slot `t` and zero on the features. -/
lemma productBranch_leaf_features [DecidableEq κ] [DecidableEq T] (π : T → κ)
    (G : T → α → β → ℂ) (x : κ → α) (y : κ → β) :
    productBranch (leafGate π G) (leafUp π) leafOut (Sum.elim (pairFeatures x y) 0) =
      Sum.elim (0 : (κ × α) ⊕ (κ × β) → ℂ) fun t => G t (x (π t)) (y (π t)) := by
  have e (z : ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) :
      productBranch (leafGate π G) (leafUp π) leafOut z = leafLayer π G z - z := by
    simp [leafLayer, productBranch, leafOut]
  funext c
  rw [e]
  rcases c with c | t
  · simp [leafLayer, Finset.sum_apply]
  · rw [Pi.sub_apply, leafLayer_features]
    simp

/-- **LP2's prelude behind Peri-RMS**: with characters as the leaf values, the energies are `2k`
and `k`, so the normalized leaf layer on the features has order at most `2`. -/
example {p k : ℕ} [NeZero p] (eps scale : ℝ) (π : Fin k → Fin k) (q : ZMod p → ZMod p → ZMod p)
    (c : ((Fin k × ZMod p) ⊕ (Fin k × ZMod p)) ⊕ Fin k) :
    (fun x y => periLayer eps scale (leafGate π fun _ a b => ZMod.stdAddChar (q a b)) (leafUp π)
      leafOut (Sum.elim (pairFeatures x y) 0) c) ∈ orderSpan (2 * 1) :=
  periLayer_order (a := 2 * k) (b := k) eps scale _ _ _ features_mem_orderSpan
    (fun x y => by simpa using stateEnergy_features (T := Fin k) x y)
    (fun x y => by
      rw [productBranch_leaf_features]
      simp [stateEnergy, Fintype.sum_sum_type]) c

end Transformer.Grokking.FlatSpectrum
