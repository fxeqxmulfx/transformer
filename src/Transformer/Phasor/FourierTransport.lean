module

public import Transformer.Phasor.TiedToeplitz
public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Mathlib.RingTheory.IntegralDomain

/-!
# A learned cyclic read of Fourier coordinates

Source: “Learned Fourier transport” in `experiments/phasor_value_transport/README.md`.
The prefix controls a distribution over cyclic channel shifts and a constant
route, followed by a unit complex phase for each output. The spectrum and its
adjoint decoder share a learned real matrix. The exact reference is the affine
action on additive characters. Residue values occur only in the reference.
-/

@[expose] public section

namespace Transformer.Phasor.FourierTransport

variable {m p D d H F : ℕ}

/-- The affine action before decoding. The constant source has complex value one. -/
noncomputable def act (c : ℝ) (r : Fin m → ℝ) (phase z : Fin m → ℂ) (j : Fin m) : ℂ :=
  phase j * ((c : ℂ) + ∑ s, (r s : ℂ) * z (j+s))

/-- A vertex of the shift simplex selects a common cyclic permutation. -/
theorem act_shift (s : Fin m) (phase z : Fin m → ℂ) (j : Fin m) :
    act 0 (fun t => if t = s then 1 else 0) phase z j = phase j * z (j+s) := by
  classical
  simp [act, apply_ite, ite_mul]

/-- The other simplex vertex drops all dependence on the current input. -/
theorem act_constant (phase z : Fin m → ℂ) (j : Fin m) :
    act 1 (fun _ => 0) phase z j = phase j := by simp [act]

/-- Additive characters in the frequency-times-input convention. -/
noncomputable def character [NeZero p] (k x : ZMod p) : ℂ := ZMod.stdAddChar (k*x)

/-- The phase of any affine map factors from its permutation of characters. -/
theorem character_affine [NeZero p] (k t a b : ZMod p) :
    character k (t + a*b) = character k t * character (k*a) b := by
  simp only [character, mul_add, mul_assoc, AddChar.map_add_eq_mul]

/-- A nonzero scalar permutes frequencies over the prime field. -/
theorem frequency_bijective [Fact p.Prime] {a : ZMod p} (ha : a ≠ 0) :
    Function.Bijective (fun k : ZMod p => k*a) :=
  mul_left_bijective_of_finite₀ ha

example : Function.Bijective (fun k : ZMod 97 => k*5) := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  exact frequency_bijective (by decide)

/-- If the chart identifies cyclic shift with frequency multiplication, the
single selected route evaluates every NV11 output character exactly. -/
theorem cubic_shift [NeZero p] (freq : Fin m → ZMod p) (s j : Fin m) (a b : ZMod p)
    (chart : ∀ i, freq (i+s) = freq i * a) :
    act 0 (fun t => if t = s then 1 else 0) (fun i => character (freq i) (a^3))
      (fun i => character (freq i) b) j = character (freq j) (a^3+a*b) := by
  rw [act_shift, chart, character_affine]

example (j : Fin 96) (b : ZMod 97) :
    act 0 (fun t : Fin 96 => if t = 0 then 1 else 0) (fun i => character (i.val : ZMod 97) (1^3))
      (fun i => character (i.val : ZMod 97) b) j = character (j.val : ZMod 97) (1^3+1*b) :=
  cubic_shift (fun i : Fin 96 => (i.val : ZMod 97)) 0 j 1 b (by simp)

/-- On NV11's zero prefix the constant route yields the constant character at
every output frequency for every possible current embedding spectrum. -/
theorem cubic_zero [NeZero p] (k b : ZMod p) (z : Fin m → ℂ) (j : Fin m) :
    act 1 (fun _ => 0) (fun _ => 1) z j = character k (0^3+0*b) := by
  simp [act_constant, character]

/-- Generator order for the exact 97-class reference check. -/
def chart97 (j : Fin 96) : ZMod 97 := 5 ^ j.val

/-- Wrapping a frequency index leaves its field value unchanged after one
period. Adding a shift therefore multiplies every frequency by one scalar. -/
theorem chart97_add (i s : Fin 96) : chart97 (i+s) = chart97 i * chart97 s := by
  simp only [chart97, Fin.val_add, ← pow_add]
  generalize (i.val + s.val) = n
  conv_rhs => rw [← Nat.mod_add_div n 96]
  rw [pow_add, pow_mul, show (5 : ZMod 97)^96 = 1 by decide, one_pow, mul_one]

set_option maxRecDepth 20000 in
private theorem chart97_covers : ∀ a : ZMod 97, a ≠ 0 → ∃ s : Fin 96, chart97 s = a := by
  decide

/-- Every nonzero NV11 row has an exact single-shift realization in the
concrete chart. The same shift works for every frequency and every operand. -/
theorem cubic97_nonzero (a : ZMod 97) (ha : a ≠ 0) :
    ∃ s : Fin 96, ∀ (j : Fin 96) (b : ZMod 97),
      act 0 (fun t => if t = s then 1 else 0) (fun i => character (chart97 i) (a^3))
        (fun i => character (chart97 i) b) j = character (chart97 j) (a^3+a*b) := by
  obtain ⟨s, hs⟩ := chart97_covers a ha
  exact ⟨s, fun j b => cubic_shift chart97 s j a b (fun i => by rw [chart97_add, hs])⟩

example := cubic97_nonzero 17 (by decide)

/-- Learned frequency coordinates and affine prefix control projections. -/
structure Read (D m : ℕ) where
  spectrum : EucSpace D →ₗ[ℝ] ToeplitzChannels m
  route : EucSpace D →ₗ[ℝ] (Option (Fin m) → ℝ)
  routeBias : Option (Fin m) → ℝ
  angle : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  angleBias : Fin m → ℝ

/-- A single softmax normalizes the shift and constant choices together. -/
noncomputable def Read.weight (P : Read D m) (h : EucSpace D) (s : Option (Fin m)) : ℝ :=
  Real.exp (P.route h s + P.routeBias s) / ∑ t, Real.exp (P.route h t + P.routeBias t)

/-- With an empty prefix the current spectrum is read unchanged. -/
noncomputable def Read.channels (P : Read D m) (prior : Option (EucSpace D))
    (x : EucSpace D) : ToeplitzChannels m :=
  match prior with
  | none => P.spectrum x
  | some h => WithLp.toLp 2 (act (P.weight h none) (fun s => P.weight h (some s))
      (fun j => Complex.exp ((P.angle h j + P.angleBias j : ℝ) * Complex.I)) (P.spectrum x).ofLp)

/-- The hidden feature is decoded with the transpose of the spectrum matrix. -/
noncomputable def Read.feature (P : Read D m) (prior : Option (EucSpace D))
    (x : EucSpace D) : EucSpace D := LinearMap.adjoint P.spectrum (P.channels prior x)

/-- Class scoring compares learned spectral coordinates to the transported
coordinates. Their construction sees only the current vector and strict prefix. -/
theorem Read.score (P : Read D m) (q x : EucSpace D) (prior : Option (EucSpace D)) :
    inner ℝ q (P.feature prior x) = inner ℝ (P.spectrum q) (P.channels prior x) :=
  LinearMap.adjoint_inner_right _ _ _

/-- Apply the read to every position of an embedding sequence. -/
noncomputable def sequence (A : EmbeddingBackbone D d H F) (P : Read D m)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :=
  let prior := match i with
    | 0 => none
    | t+1 => some (A.feature θ eps scale x l t)
  P.feature prior (GPTMini.rmsNormEps eps (x i))

/-- Neither future vectors nor their identities can affect the Fourier read. -/
theorem sequence_congr_prefix (A : EmbeddingBackbone D d H F) (P : Read D m)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ)
    (h : ∀ j ≤ i, x j = y j) : sequence A P θ eps scale x l i = sequence A P θ eps scale y l i := by
  cases i with
  | zero => simp [sequence, h 0 (by omega)]
  | succ i =>
    simp only [sequence, EmbeddingBackbone.feature, h (i+1) (by omega),
      A.states_congr_prefix eps scale x y l i fun j hj => h j (by omega)]

example (A : EmbeddingBackbone 128 32 4 171) (P : Read 128 96) (θ : StreamReadWeights 128)
    (x fill : ℕ → EucSpace 128) (l : ℕ) :
    sequence A P θ 1 1 x l 9 = sequence A P θ 1 1 (fun i => if i ≤ 9 then x i else fill i) l 9 :=
  sequence_congr_prefix _ _ _ _ _ _ _ _ _ (fun i hi => by simp [hi])

end Transformer.Phasor.FourierTransport
