module

public import Transformer.GPTMini.RMSNorm
public import Transformer.Sandwich.Section5_Locality

/-!
# Phasor v2: the Peri-norm residual

The third element of Phasor v2 is Phasor's Peri-RMS residual. Source: DiffGPT
(`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*: every
branch reads the stream through the parameter-free `R_D(x) = x / √(mean(x²) + ε)`
(`GPTMini.rmsNormEps`), its output is normalized by `R_D` again, and "alpha … is applied AFTER
each branch output RMSNorm". Phasor routes four streams through mHC, an element not yet chosen
for v2; on one stream, with the read and write gates and the skip map at one, a sublayer is

    h'_i = h_i + α R_D(F(R_D h)_i)

(`periStep`), where the branch `F` maps the normalized sequence to a sequence: attention reads
every position up to `i`, a feed-forward branch only its own. `periRun` stacks sublayers.

* Every update has length at most `|α| √D`, whatever the branch and the stream
  (`norm_periStep_sub_le`).
* After `n` sublayers the stream has grown by at most `n |α| √D` (`norm_periRun_le`); from a
  normalized embedding, `h⁰ = R_D(E[token])` in the source, its RMS `‖h‖/√D` is at most
  `1 + n |α|` (`rms_periRun_le`), for all weights.
* Causal branches give causal sublayers and stacks (`isCausal_periStep`, `isCausal_periRun`).
-/

@[expose] public section

open Transformer.GPTMini Transformer.Sandwich

namespace Transformer.PhasorV2

variable {D : ℕ}

/-- **A Peri-norm sublayer on one stream** (DiffGPT's main model, *Architecture*): at every
position `i` the stream `x` adds `α` times the RMS normalization of the branch's output, and the
branch reads the RMS normalization of the stream at every position. -/
noncomputable def periStep (eps α : ℝ) (F : (ℕ → EucSpace D) → ℕ → EucSpace D)
    (x : ℕ → EucSpace D) (i : ℕ) : EucSpace D :=
  x i + α • rmsNormEps eps (F (fun j => rmsNormEps eps (x j)) i)

/-- **`n` Peri-norm sublayers**, the sublayer `l` with the branch `F l`. -/
noncomputable def periRun (eps α : ℝ) (F : ℕ → (ℕ → EucSpace D) → ℕ → EucSpace D) :
    ℕ → (ℕ → EucSpace D) → ℕ → EucSpace D
  | 0, x => x
  | n + 1, x => periStep eps α (F n) (periRun eps α F n x)

/-- **Every update has length at most `|α| √D`**, whatever the branch and the stream. -/
theorem norm_periStep_sub_le {eps : ℝ} (heps : 0 < eps) (hD : 0 < D) (α : ℝ)
    (F : (ℕ → EucSpace D) → ℕ → EucSpace D) (x : ℕ → EucSpace D) (i : ℕ) :
    ‖periStep eps α F x i - x i‖ ≤ |α| * Real.sqrt D := by
  rw [periStep, add_sub_cancel_left, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_left (rmsNormEps_norm_le eps heps hD _) (abs_nonneg α)

/-- The main model's `ε = 2^{−23}` and width `128`, with any branch. -/
example (F : (ℕ → EucSpace 128) → ℕ → EucSpace 128) (x : ℕ → EucSpace 128) :=
  norm_periStep_sub_le (eps := 1 / 2 ^ 23) (by norm_num) (by norm_num) (1 / 4) F x 7

/-- **After `n` sublayers the stream has grown by at most `n |α| √D`**, whatever the branches. -/
theorem norm_periRun_le {eps : ℝ} (heps : 0 < eps) (hD : 0 < D) (α : ℝ)
    (F : ℕ → (ℕ → EucSpace D) → ℕ → EucSpace D) (n : ℕ) (x : ℕ → EucSpace D) (i : ℕ) :
    ‖periRun eps α F n x i‖ ≤ ‖x i‖ + n * (|α| * Real.sqrt D) := by
  induction n with
  | zero => simp [periRun]
  | succ n ih =>
    have h := norm_periStep_sub_le heps hD α (F n) (periRun eps α F n x) i
    have h' := norm_sub_norm_le (periStep eps α (F n) (periRun eps α F n x) i)
      (periRun eps α F n x i)
    rw [periRun]
    push_cast
    linarith

/-- Eight sublayers at `α = 1/√8`, four layers of attention and feed-forward branches. -/
example (F : ℕ → (ℕ → EucSpace 128) → ℕ → EucSpace 128) (x : ℕ → EucSpace 128) :=
  norm_periRun_le (eps := 1 / 2 ^ 23) (by norm_num) (by norm_num) (1 / Real.sqrt 8) F 8 x 7

/-- **From a normalized embedding the RMS after `n` sublayers is at most `1 + n |α|`**: the RMS
`‖h‖/√D` of the stream, for all branches and weights. -/
theorem rms_periRun_le {eps : ℝ} (heps : 0 < eps) (hD : 0 < D) (α : ℝ)
    (F : ℕ → (ℕ → EucSpace D) → ℕ → EucSpace D) (n : ℕ) (e : ℕ → EucSpace D) (i : ℕ) :
    ‖periRun eps α F n (fun j => rmsNormEps eps (e j)) i‖ / Real.sqrt D ≤ 1 + n * |α| := by
  have hs : 0 < Real.sqrt D := Real.sqrt_pos.2 (by exact_mod_cast hD)
  rw [div_le_iff₀ hs]
  have h0 := rmsNormEps_norm_le eps heps hD (e i)
  calc ‖periRun eps α F n (fun j => rmsNormEps eps (e j)) i‖
      ≤ ‖rmsNormEps eps (e i)‖ + n * (|α| * Real.sqrt D) := norm_periRun_le heps hD α F n _ i
    _ ≤ Real.sqrt D + n * (|α| * Real.sqrt D) := by linarith
    _ = (1 + n * |α|) * Real.sqrt D := by ring

/-- The same eight sublayers from a normalized embedding: RMS at most `1 + 8/√8 = 1 + 2√2`. -/
example (F : ℕ → (ℕ → EucSpace 128) → ℕ → EucSpace 128) (e : ℕ → EucSpace 128) :=
  rms_periRun_le (eps := 1 / 2 ^ 23) (by norm_num) (by norm_num) (1 / Real.sqrt 8) F 8 e 7

/-- **A causal branch gives a causal sublayer.** -/
theorem isCausal_periStep (eps α : ℝ) {F : (ℕ → EucSpace D) → ℕ → EucSpace D}
    (hF : IsCausal F) : IsCausal (periStep eps α F) := by
  intro x y i h
  simp only [periStep]
  rw [h i le_rfl, hF _ _ i fun j hj => by rw [h j hj]]

/-- A feed-forward branch reads only its own position. -/
example (f : EucSpace 4 → EucSpace 4) :=
  isCausal_periStep (D := 4) 1 1 (F := fun y i => f (y i)) fun _ _ i h => congrArg f (h i le_rfl)

/-- **Causal branches give a causal stack.** -/
theorem isCausal_periRun (eps α : ℝ) {F : ℕ → (ℕ → EucSpace D) → ℕ → EucSpace D}
    (hF : ∀ l, IsCausal (F l)) (n : ℕ) : IsCausal (periRun eps α F n) := by
  induction n with
  | zero => exact fun x y i h => h i le_rfl
  | succ n ih => exact (isCausal_periStep eps α (hF n)).comp ih

/-- A stack of feed-forward branches. -/
example (f : ℕ → EucSpace 4 → EucSpace 4) :=
  isCausal_periRun (D := 4) 1 1 (F := fun l y i => f l (y i))
    (fun l _ _ i h => congrArg (f l) (h i le_rfl)) 3

end Transformer.PhasorV2
