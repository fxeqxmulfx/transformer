module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Basic

/-!
# Projection inequalities are the actual raw complex activation constraints

Source: `review/flat_spectrum.md`, CM79. A finite row describes which learned
bias and position/object coordinates the original raw input reads. It is
used only to certify that update constraints match that computation; it
does not replace the model by a readout on supplied task features.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection

open scoped BigOperators
open PatternReduction

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- The exact finite row of the original raw lookup, including its bias,
CM79. Missing positions contribute no coordinate. -/
def rawRow (tokens : Prefix O T) : Coordinate O T → ℝ
  | none => 1
  | some (i, o) => if tokens.val[i.val]? = some o then 1 else 0

/-- Applying the row to all learned coordinates equals the original real
raw input, CM79. No training target occurs on either side. -/
theorem rawRow_dot (tokens : Prefix O T) (w : Coordinate O T → ℝ) :
    dot (rawRow tokens) w = realInput tokens w := by
  rw [realInput_apply]
  unfold dot
  simp only [Fintype.sum_option, rawRow, one_mul, Fintype.sum_prod_type]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  have choose (value : Option O) :
      (∑ o, (if value = some o then (1 : ℝ) else 0) * w (some (i, o))) =
        match value with | none => 0 | some o => w (some (i, o)) := by
    cases value <;> simp
  exact choose _

/-- Orientation of a complete raw row, CM79. Zero coefficients remain. -/
def signedRow (active : Bool) (tokens : Prefix O T) (d : Coordinate O T) : ℝ :=
  if active then rawRow tokens d else -(rawRow tokens d)

/-- Signed row evaluation is exactly the model's original sign condition,
CM79, for both orientations. -/
theorem signedRow_dot (active : Bool) (tokens : Prefix O T) (w : Coordinate O T → ℝ) :
    dot (signedRow active tokens) w = orient active (realInput tokens w) := by
  cases active with
  | true => exact rawRow_dot tokens w
  | false =>
    change (∑ d, -(rawRow tokens d) * w d) = -(realInput tokens w)
    rw [← rawRow_dot]
    simp [dot]

omit [DecidableEq O] in
/-- The raw complex component uses the same actual real coordinate read,
CM79. Both real and imaginary cases are retained. -/
theorem raw_components (tokens : Prefix O T) (k : Bool) (w : ObjectWeights O T) :
    components rawInput (tokens, k) w = realInput tokens (project k w) := by
  cases k with
  | false =>
    change (rawInput tokens w).im = realInput tokens (project false w)
    rw [rawInput_complexInput]
    rfl
  | true =>
    change (rawInput tokens w).re = realInput tokens (project true w)
    rw [rawInput_complexInput]
    rfl

/-- Every componentwise certificate enforces the original complex model's
whole input domain, CM79. No class, slot, sign or component is removed. -/
theorem certificates_mem {J : Type*} {p q : Parameters J O (ObjectWeights O T)}
    (mask : J → Pattern (Prefix O T))
    (dual : J → O → Bool → Bool → Prefix O T → ℝ)
    (h : ∀ j c s k, Certificate (fun x => signedRow (mask j (x, k)) x)
      (project k (p j c s)) (project k (q j c s)) (dual j c s k)) :
    q ∈ complexDomain rawInput Set.univ mask := by
  intro j c s x hx
  have hf := (h j c s x.2).2.1 x.1
  rw [signedRow_dot] at hf
  change 0 ≤ orient (mask j x) (components rawInput x (q j c s))
  rcases x with ⟨tokens, k⟩
  rw [raw_components]
  exact hf

example := certificates_mem (O := Bool) (T := 1) (J := Unit) (p := 0) (q := 0)
  (fun _ _ => true) (fun _ _ _ _ _ => 0) (by
    intro j c s k
    apply identity_certificate
    intro tokens
    simp [dot, project])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
