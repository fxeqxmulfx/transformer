module

public import Transformer.Sandwich.Section5_Locality
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Phasor v2: Peri-norm

The third element of Phasor v2 normalizes the input and the output of every branch. Source:
DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*:
the parameter-free `R_n(x) = x / √(mean(x²) + ε)`, `ε = 2^{−23}`, normalizes the embedding,
`h⁰_i = R_D(E[token_i])`, the input of every branch, `z_i = R_D(∑_s read_{i,s} h_{i,s})`, and
its output, `R_D(W_O a_i)` or `f_i`; "alpha=1/sqrt(2L) is applied AFTER each branch output
RMSNorm".

The model is complex: its `D` real coordinates pair into `c = D/2` complex channels. The source
pairs them inside each head block; another fixed pairing permutes the coordinates, and the
projections absorb the permutation. A vector is `CSpace c = ℂ^c`, its squared length that of its
`2c` real coordinates, and `R_D` is `rms` (`rms_eq_div`).

* `R_D` has length at most `√D`, RMS at most one, for every input (`norm_rms_le`).
* The Peri-norm branch `periBranch eps F z` is `R_D(F(R_D ∘ z))`: the raw branch `F` maps the
  normalized sequence to a sequence, attention reading every position up to `i`, a feed-forward
  branch only its own. Its output has length at most `√D` whatever `F` (`norm_periBranch_le`),
  and a causal `F` gives a causal branch (`isCausal_periBranch`).

`Transformer.PhasorV2.MHC` adds the output, scaled by the write gates and `α`, to the four
streams.
-/

@[expose] public section

open Transformer.Sandwich

namespace Transformer.PhasorV2

variable {c : ℕ}

/-- A vector of the model: `c` complex channels, `D = 2c` real coordinates. -/
abbrev CSpace (c : ℕ) := EuclideanSpace ℂ (Fin c)

/-- **`R_D`** (DiffGPT's main model, *Architecture*), parameter-free:
`R_D(x) = √D x / √(‖x‖² + D ε)` over the `D = 2c` real coordinates of `x`. -/
noncomputable def rms (eps : ℝ) (x : CSpace c) : CSpace c :=
  ((Real.sqrt (2 * c) / Real.sqrt (‖x‖ ^ 2 + 2 * c * eps) : ℝ) : ℂ) • x

/-- **The source's form** `x / √(mean(x²) + ε)`, the mean over the `D = 2c` real coordinates. -/
theorem rms_eq_div (hc : 0 < c) (eps : ℝ) (x : CSpace c) :
    rms eps x = ((1 / Real.sqrt (‖x‖ ^ 2 / (2 * c) + eps) : ℝ) : ℂ) • x := by
  have hD : (0 : ℝ) < 2 * c := mul_pos two_pos (Nat.cast_pos.2 hc)
  rw [rms, div_add' _ _ _ hD.ne', Real.sqrt_div' _ hD.le, one_div_div, mul_comm eps]

/-- The main model's width `D = 128`. -/
example (x : CSpace 64) := rms_eq_div (by norm_num) (1 / 2 ^ 23) x

/-- **`R_D` has length at most `√D`**: RMS at most one, for every input and every `ε ≥ 0`. -/
theorem norm_rms_le {eps : ℝ} (heps : 0 ≤ eps) (x : CSpace c) :
    ‖rms eps x‖ ≤ Real.sqrt (2 * c) := by
  have hε : 0 ≤ 2 * (c : ℝ) * eps := mul_nonneg (by positivity) heps
  have hx : ‖x‖ ≤ Real.sqrt (‖x‖ ^ 2 + 2 * c * eps) :=
    Real.le_sqrt_of_sq_le (le_add_of_nonneg_right hε)
  rw [rms, norm_smul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)), div_mul_eq_mul_div,
    mul_div_assoc]
  exact mul_le_of_le_one_right (Real.sqrt_nonneg _) (div_le_one_of_le₀ hx (Real.sqrt_nonneg _))

/-- The main model: `D = 128`, `ε = 2^{−23}`. -/
example (x : CSpace 64) := norm_rms_le (eps := 1 / 2 ^ 23) (by norm_num) x

/-- **A Peri-norm branch** (DiffGPT's main model, *Architecture*): at the position `i`, `R_D` of
the raw branch `F` applied to `R_D` of the input `z` at every position. -/
noncomputable def periBranch (eps : ℝ) (F : (ℕ → CSpace c) → ℕ → CSpace c) (z : ℕ → CSpace c)
    (i : ℕ) : CSpace c :=
  rms eps (F (fun j => rms eps (z j)) i)

/-- **A Peri-norm branch has output of length at most `√D`**, whatever the raw branch. -/
theorem norm_periBranch_le {eps : ℝ} (heps : 0 ≤ eps) (F : (ℕ → CSpace c) → ℕ → CSpace c)
    (z : ℕ → CSpace c) (i : ℕ) : ‖periBranch eps F z i‖ ≤ Real.sqrt (2 * c) :=
  norm_rms_le heps _

/-- Any raw branch of the main model's width. -/
example (F : (ℕ → CSpace 64) → ℕ → CSpace 64) (z : ℕ → CSpace 64) :=
  norm_periBranch_le (eps := 1 / 2 ^ 23) (by norm_num) F z 7

/-- **A causal raw branch gives a causal Peri-norm branch.** -/
theorem isCausal_periBranch (eps : ℝ) {F : (ℕ → CSpace c) → ℕ → CSpace c} (hF : IsCausal F) :
    IsCausal (periBranch eps F) := by
  intro x y i h
  simp only [periBranch]
  rw [hF _ _ i fun j hj => by rw [h j hj]]

/-- A feed-forward branch reads only its own position. -/
example (f : CSpace 4 → CSpace 4) :=
  isCausal_periBranch 1 (F := fun y i => f (y i)) fun _ _ i h => congrArg f (h i le_rfl)

end Transformer.PhasorV2
