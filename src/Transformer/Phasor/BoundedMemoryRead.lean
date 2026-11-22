module

public import Transformer.Phasor.EmbeddingInput
public import Transformer.RecurrentGradients.Section2_Mechanics
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# A bounded recurrent read of Phasor's embedding features

Source: the bounded recurrent readout in `experiments/phasor_value_transport/README.md`.
Three learned linear maps of each hidden vector produce a retention score,
a proposed memory value and a read gate. A fourth learned map projects the
read into the residual. Each sequence starts with zero memory. All positions
share the same four maps, including the first.

The state stays in the unit interval in absolute value at arbitrary lengths.
Perturbations of the initial state are attenuated by the product of the
learned retentions. The scalar adjoint below includes a cotangent from the
future state as well as the current output.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer

/-- Update one memory coordinate from the two projected activations. -/
noncomputable def gatedMemoryStep (forget write previous : ℝ) : ℝ :=
  Real.sigmoid forget * previous + (1 - Real.sigmoid forget) * Real.tanh write

/-- The state after the first `n` vectors, from an arbitrary initial value. -/
noncomputable def gatedMemoryScan (forget write : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => gatedMemoryStep (forget n) (write n) (gatedMemoryScan forget write initial n)

/-- Every sequence starts at zero and stays bounded independently of its
length, input scale and the magnitudes of both learned projections. -/
theorem gatedMemoryScan_bounded (a b : ℕ → ℝ) (n : ℕ) :
    |gatedMemoryScan a b 0 n| ≤ 1 := by
  induction n with
  | zero => norm_num [gatedMemoryScan]
  | succ n ih =>
    let g := Real.sigmoid (a n)
    have hg : 0 ≤ g := (Real.sigmoid_pos _).le
    have hw : 0 ≤ 1 - g := sub_nonneg.mpr (Real.sigmoid_lt_one _).le
    change |g * gatedMemoryScan a b 0 n + (1 - g) * Real.tanh (b n)| ≤ 1
    calc
      _ ≤ |g * gatedMemoryScan a b 0 n| + |(1-g) * Real.tanh (b n)| := abs_add_le _ _
      _ = g * |gatedMemoryScan a b 0 n| + (1-g) * |Real.tanh (b n)| := by
        simp only [abs_mul, abs_of_nonneg hg, abs_of_nonneg hw]
      _ ≤ g * 1 + (1-g) * 1 := add_le_add (mul_le_mul_of_nonneg_left ih hg)
        (mul_le_mul_of_nonneg_left (Real.abs_tanh_lt_one _).le hw)
      _ = 1 := by ring

/-- The dependence on the entering state is exactly the product of the
retentions, even when every vector gives a different gate. -/
theorem gatedMemoryScan_initial_difference (a b : ℕ → ℝ) (z z' : ℝ) (n : ℕ) :
    gatedMemoryScan a b z n - gatedMemoryScan a b z' n =
      (∏ i ∈ Finset.range n, Real.sigmoid (a i)) * (z-z') := by
  induction n with
  | zero => simp [gatedMemoryScan]
  | succ n ih =>
    rw [gatedMemoryScan, gatedMemoryScan, gatedMemoryStep, gatedMemoryStep,
      Finset.prod_range_succ]
    linear_combination Real.sigmoid (a n) * ih

/-- The derivative of the current read and next state paired with arbitrary
upstream gradients. Its coefficients are the four reverse-mode updates for
the previous state and the forget, write and read projections. -/
theorem gatedMemoryStep_adjoint (a b c z da db dc dz g future : ℝ) :
    HasDerivAt
      (fun t : ℝ => (future + g * Real.sigmoid (c+t*dc)) *
        gatedMemoryStep (a+t*da) (b+t*db) (z+t*dz))
      (((future + g * Real.sigmoid c) * Real.sigmoid a) * dz +
       ((future + g * Real.sigmoid c) * (z-Real.tanh b) *
         Real.sigmoid a * (1-Real.sigmoid a)) * da +
       ((future + g * Real.sigmoid c) * (1-Real.sigmoid a) * (1-Real.tanh b ^ 2)) * db +
       (g * gatedMemoryStep a b z * Real.sigmoid c * (1-Real.sigmoid c)) * dc) 0 := by
  have line (v dv : ℝ) : HasDerivAt (fun t : ℝ => v + t*dv) dv 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dv).const_add v
  have ha := (Real.hasDerivAt_sigmoid (a+0*da)).comp 0 (line a da)
  have hb := (RecurrentGradients.hasDerivAt_tanh (b+0*db)).comp 0 (line b db)
  have hc := (Real.hasDerivAt_sigmoid (c+0*dc)).comp 0 (line c dc)
  have update := (ha.mul (line z dz)).add (((hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub ha).mul hb)
  have objective := ((hc.const_mul g).const_add future).mul update
  convert! objective using 1
  simp [gatedMemoryStep]
  ring

/-- Four shared, trainable linear projections on dense hidden vectors.
The memory dimension is independent of the vocabulary size. -/
structure BoundedMemoryRead (D m : ℕ) where
  forget : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  write : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  read : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  output : (Fin m → ℝ) →ₗ[ℝ] EucSpace D

/-- Add the projected gated state to the current hidden vector. -/
noncomputable def BoundedMemoryRead.feature {D m : ℕ} (A : BoundedMemoryRead D m)
    (x : ℕ → EucSpace D) (i : ℕ) : EucSpace D :=
  x i + A.output (fun k => Real.sigmoid (A.read (x i) k) *
    gatedMemoryScan (fun j => A.forget (x j) k) (fun j => A.write (x j) k) 0 (i+1))

/-- A zero output map recovers the original hidden sequence for arbitrary
input projections and arbitrary context, as in the native initialization. -/
theorem BoundedMemoryRead.feature_zero_output {D m : ℕ} (A : BoundedMemoryRead D m)
    (x : ℕ → EucSpace D) (i : ℕ) :
    ({A with output := 0} : BoundedMemoryRead D m).feature x i = x i := by
  simp [BoundedMemoryRead.feature]

end Transformer.Phasor
