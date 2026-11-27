module

public import Transformer.PhasorV2.FFN
public import Mathlib.Algebra.Group.EvenFunction

/-!
# A single residual stream with a learned phase

The ordinary residual in Xie et al., arXiv:2512.24880v2, Eq. (1), adds a branch write to
the input. Here the proposed sublayer is `exp(i θ ‖x‖) • x + write x`. A single learned
real coefficient controls the rotation at every token and channel of the sublayer.
The incoming norm is already computed by the input RMS. At zero initialization the
update is ordinary addition. The skip preserves the norm and has an explicit inverse.

The example branch follows `experiments/argand/ARGAND.md`, §§ Embedding,
complex_ffn, and Forward pass and loss. Both projected factors of the FFN change sign
when the input is negated. Their product, hence the normalized and scaled write, is even.
The two occurrences of RMS use the runner's denominator `‖x‖ + ε`.

On the pair `x, -x`, both inputs rotate by the same angle. Against a skip with a fixed
complex coefficient, the parallelogram law splits the paired mean squared output error
into the skip error and the write error. At unit norm a quarter turn gives a mean squared
distance of at least 2 to every ordinary FFN. Norms one and two require different phases,
so even a freely chosen constant coefficient cannot reproduce this sublayer on all inputs.
-/

@[expose] public section

namespace Transformer.PhasorV2

variable {c m n : ℕ}

/-- Rotate all channels by the same angle, proportional to the incoming token's norm. -/
noncomputable def turnByNorm (θ : ℝ) (x : CSpace c) : CSpace c :=
  Complex.exp ((θ * ‖x‖ : ℝ) * Complex.I) • x

/-- The proposed one-stream extension of arXiv:2512.24880v2, Eq. (1).
`write` includes the branch normalization and residual scale. -/
noncomputable def phaseResidual (θ : ℝ) (write : CSpace c → CSpace c) (x : CSpace c) :
    CSpace c :=
  turnByNorm θ x + write x

/-- Initializing the angle to zero recovers the ordinary residual for every branch. -/
@[simp] theorem phaseResidual_zero (write : CSpace c → CSpace c) (x : CSpace c) :
    phaseResidual 0 write x = x + write x := by
  simp [phaseResidual, turnByNorm]

/-- The incoming norm can be read from the output of the skip. -/
theorem norm_turnByNorm (θ : ℝ) (x : CSpace c) : ‖turnByNorm θ x‖ = ‖x‖ := by
  rw [turnByNorm, norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]

/-- Using the opposite coefficient undoes the skip for every input, including zero. -/
theorem turnByNorm_inverse (θ : ℝ) (x : CSpace c) : turnByNorm (-θ) (turnByNorm θ x) = x := by
  rw [turnByNorm, norm_turnByNorm, turnByNorm, smul_smul, ← Complex.exp_add]
  simp [neg_mul]

/-- Argand's complex RMS from § Embedding of `ARGAND.md` multiplies by the square root
of the complex width and divides by the Euclidean norm plus the offset. -/
noncomputable def argandRMS (eps : ℝ) (x : CSpace c) : CSpace c :=
  ((Real.sqrt c / (‖x‖ + eps) : ℝ) : ℂ) • x

/-- Reversing the sign preserves the denominator and reverses the normalized vector. -/
theorem argandRMS_neg (eps : ℝ) (x : CSpace c) : argandRMS eps (-x) = -argandRMS eps x := by
  simp only [argandRMS, norm_neg, smul_neg]

/-- The single-stream FFN write from `ARGAND.md`, §§ complex_ffn and Forward pass and loss.
The three complex projections are bias-free. The scale is applied after the output RMS. -/
noncomputable def FFN.argandWrite (F : FFN c m) (eps scale : ℝ) (x : CSpace c) : CSpace c :=
  let z := argandRMS eps x
  (scale : ℂ) • argandRMS eps (F.WF (F.Wg z * F.Wu z))

/-- The Argand FFN writes the same vector at both inputs of an antipodal pair. -/
theorem FFN.argandWrite_even (F : FFN c m) (eps scale : ℝ) :
    Function.Even (F.argandWrite eps scale) := by
  intro x
  simp only [FFN.argandWrite, argandRMS_neg, map_neg, neg_mul_neg]

/-- Mean squared distance between two sublayer outputs on the two inputs `x` and `-x`. -/
noncomputable def antipodalMSE (f g : CSpace c → CSpace c) (x : CSpace c) : ℝ :=
  (‖f x - g x‖ ^ 2 + ‖f (-x) - g (-x)‖ ^ 2) / 2

/-- Exact paired error against any fixed skip coefficient `q` and arbitrary even writes.
The ordinary residual has `q = 1`. -/
theorem phaseResidual_antipodalMSE (θ : ℝ) (q : ℂ) {f g : CSpace c → CSpace c}
    (hf : Function.Even f) (hg : Function.Even g) (x : CSpace c) :
    antipodalMSE (phaseResidual θ f) (fun z => q • z + g z) x =
      ‖Complex.exp ((θ * ‖x‖ : ℝ) * Complex.I) - q‖ ^ 2 * ‖x‖ ^ 2 +
        ‖f x - g x‖ ^ 2 := by
  let p (r : ℝ) := Complex.exp ((θ * r : ℝ) * Complex.I)
  have diff (z : CSpace c) : phaseResidual θ f z - (q • z + g z) =
      (f z - g z) + (p ‖z‖ - q) • z := by
    rw [phaseResidual, turnByNorm, add_sub_add_comm, sub_smul]
    exact add_comm _ _
  rw [antipodalMSE, diff x, diff (-x), hf x, hg x, norm_neg, smul_neg, ← sub_eq_add_neg,
    parallelogram_law_with_norm ℂ, norm_smul, mul_pow]
  dsimp only [p]
  ring

/-- The FFN in the width-64 example of `ARGAND.md` satisfies both evenness hypotheses
with independently chosen weights and the runner's RMS offset. -/
example (F G : FFN 64 256) (scale θ : ℝ) (x : CSpace 64) :=
  phaseResidual_antipodalMSE θ 1 (F.argandWrite_even 0.000001 scale)
    (G.argandWrite_even 0.000001 scale) x

/-- At unit norm, a quarter turn incurs a sharp error floor against every ordinary
Argand FFN, even with a different hidden width. Matching the writes attains the floor. -/
theorem FFN.quarterTurn_antipodalMSE (F : FFN c m) (G : FFN c n) (eps scale : ℝ)
    {x : CSpace c} (hx : ‖x‖ = 1) :
    antipodalMSE (phaseResidual (Real.pi / 2) (F.argandWrite eps scale))
        (fun z => z + G.argandWrite eps scale z) x =
      2 + ‖F.argandWrite eps scale x - G.argandWrite eps scale x‖ ^ 2 := by
  have h := phaseResidual_antipodalMSE (Real.pi / 2) 1 (F.argandWrite_even eps scale)
    (G.argandWrite_even eps scale) x
  norm_num [hx, Complex.sq_norm, Complex.normSq_apply] at h
  exact h

/-- A unit coordinate in the example width from `ARGAND.md`. -/
example (F G : FFN 64 256) (scale : ℝ) :=
  F.quarterTurn_antipodalMSE G 0.000001 scale (x := EuclideanSpace.single 0 1)
    (by simp [EuclideanSpace.single])

/-- At coefficient `π/2`, inputs of norms one and two demand the distinct phases `i` and
`-1`. No fixed complex skip and even write can represent this Argand FFN sublayer. -/
theorem FFN.quarterTurn_ne_fixed (F : FFN (c + 1) m) (eps scale : ℝ) (q : ℂ)
    {g : CSpace (c + 1) → CSpace (c + 1)} (hg : Function.Even g) :
    phaseResidual (Real.pi / 2) (F.argandWrite eps scale) ≠
      (fun z => q • z + g z) := by
  intro heq
  have atRadius (r : ℝ) (hr : 0 < r) :
      Complex.exp ((Real.pi / 2 * r : ℝ) * Complex.I) = q := by
    let x : CSpace (c + 1) := EuclideanSpace.single 0 (r : ℂ)
    have hx : ‖x‖ = r := by simp [x, EuclideanSpace.single, abs_of_pos hr]
    have h := phaseResidual_antipodalMSE (Real.pi / 2) q (F.argandWrite_even eps scale) hg x
    rw [heq, hx] at h
    simp only [antipodalMSE, sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0),
      zero_add, zero_div] at h
    by_contra hne
    have hd := norm_pos_iff.mpr (sub_ne_zero.mpr hne)
    have hp := mul_pos (sq_pos_of_pos hd) (sq_pos_of_pos hr)
    linarith [sq_nonneg ‖F.argandWrite eps scale x - g x‖]
  have hi : Complex.I = q := by simpa using atRadius 1 zero_lt_one
  have hneg : (-1 : ℂ) = q := by simpa using atRadius 2 zero_lt_two
  have him := congrArg Complex.im (hi.trans hneg.symm)
  norm_num at him

/-- Every ordinary product FFN supplies an admissible competing write. -/
example (F G : FFN 64 256) (scale : ℝ) :=
  F.quarterTurn_ne_fixed 0.000001 scale 1 (G.argandWrite_even 0.000001 scale)

end Transformer.PhasorV2
