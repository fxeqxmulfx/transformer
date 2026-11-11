module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation
public import Transformer.Grokking.FlatSpectrum.FlatCapacity

/-!
# Flat spectrum as a capacity bound on actual channel outputs

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC6;
R3 and AR4 supply the rank/error obstruction. The scalar readout of the
actual forward map factors through the fixed categorical binding features
`(key,value)`. This is a necessary capacity bound, not a learning theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C X : Type*} [Fintype K] [Fintype V] [Fintype C]
  [DecidableEq V]

/-- Task-independent categorical binding features. Neither the target
nor any arithmetic operation appears in this map (CC6). -/
def categoricalFeatures (data : X → K → V) : Matrix (K × V) X ℂ :=
  fun kv x => if data x kv.1 = kv.2 then 1 else 0

/-- The generated output evaluated by a fixed scalar class readout (CC6).
The readout is for the spectral diagnostic, not an input feature. -/
noncomputable def outputKernel (φ : C → ℂ) (W : Channel Q K V C)
    (data : X → K → V) : Matrix Q X ℂ :=
  fun q x => ∑ c, (logits W q (data x) c : ℂ) * φ c

/-- Learned scalar coefficients on the categorical binding features (CC6). -/
noncomputable def scalarCoefficients (φ : C → ℂ) (W : Channel Q K V C) :
    Matrix Q (K × V) ℂ := fun q kv => ∑ c, (W q kv.1 kv.2 c : ℂ) * φ c

/-- The actual output has a factorization through `|K| |V|` categorical
features, for every parameter tensor and every input family (CC6). -/
theorem outputKernel_factorization (φ : C → ℂ) (W : Channel Q K V C)
    (data : X → K → V) :
    outputKernel φ W data = scalarCoefficients φ W * categoricalFeatures data := by
  ext q x
  simp only [outputKernel, logits, Complex.ofReal_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  simp [Matrix.mul_apply, Fintype.sum_prod_type, scalarCoefficients,
    categoricalFeatures, mul_ite, eq_comm]

variable [Fintype Q] [Fintype X] [DecidableEq Q] [DecidableEq X]

/-- Flat spectrum bounds the error of every channel in this family:
each missing categorical feature costs at least the squared singular
value. A nonpositive lower bound gives no positive error floor (CC6). -/
theorem flat_error_bound {T : Matrix Q X ℂ} {σ₂ : ℝ} (hσ : 0 < σ₂)
    (hT : IsFlat T σ₂) (φ : C → ℂ) (W : Channel Q K V C)
    (data : X → K → V) :
    σ₂ * ((min (Fintype.card Q) (Fintype.card X) : ℕ) - Fintype.card (K × V)) ≤
      frobeniusSq (T - outputKernel φ W data) := by
  rw [outputKernel_factorization]
  exact hT.mul_sub_card_le_frobeniusSq_sub hσ _ _

/-- Exact representation of a nonzero flat target requires enough
categorical width. Flatness alone supplies no sufficiency claim (CC6). -/
theorem flat_width_necessary {T : Matrix Q X ℂ} {σ₂ : ℝ} (hσ : σ₂ ≠ 0)
    (hT : IsFlat T σ₂) (φ : C → ℂ) (W : Channel Q K V C)
    (data : X → K → V) (hexact : T = outputKernel φ W data) :
    min (Fintype.card Q) (Fintype.card X) ≤ Fintype.card (K × V) := by
  rw [outputKernel_factorization] at hexact
  exact hT.card_le_of_eq_mul hσ hexact

/-- Fixed signed class readout used to exhibit a full-rank flat output.
It is not supplied as a feature to the learned model (CC6). -/
def boolSign (c : Bool) : ℂ := if c then -1 else 1

/-- A feasible channel realizes a full-rank flat `2 × 2` kernel from
ordinary categorical inputs, with width exactly two (CC6). -/
theorem flat_bool_table :
    IsFlat (outputKernel boolSign (table Bool.and) (fun v : Bool => fun _ : Unit => v)) 2 := by
  left
  ext q r
  cases q <;> cases r <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, outputKernel,
      logits, table, boolSign, Fintype.sum_bool, Matrix.one_apply]

example := flat_error_bound (K := Unit) (V := Bool)
  (by norm_num : (0 : ℝ) < 2) flat_bool_table boolSign (table Bool.and)
  (fun v : Bool => fun _ : Unit => v)

example := flat_width_necessary (K := Unit) (V := Bool)
  (by norm_num : (2 : ℝ) ≠ 0) flat_bool_table boolSign (table Bool.and)
  (fun v : Bool => fun _ : Unit => v) rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel
