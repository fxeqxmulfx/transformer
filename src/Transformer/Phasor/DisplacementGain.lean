module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# Gain interpolation for the tied successor read

Source: the residual-gain comparison in `experiments/phasor_value_transport/README.md`.
The direction read from the successor is fixed while the learned scalar gain
varies. These are exact score and margin results for the model at any position.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ} [NeZero p]

/-- At any token and depth, interpolating two gains interpolates the entire
score vector coordinate by coordinate, with all other weights held fixed.
Real interpolation also covers extrapolation beyond the two endpoints. -/
theorem Model.displacedSuccessorLogit_interpolate (M : Model D d H F p) (T : PairTables D p)
    (strength g h t eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    M.displacedSuccessorLogit T strength ((1-t)*g+t*h) eps scale xs L i c =
      (1-t) * M.displacedSuccessorLogit T strength g eps scale xs L i c +
        t * M.displacedSuccessorLogit T strength h eps scale xs L i c := by
  simp only [displacedSuccessorLogit, inner_add_right, real_inner_smul_right]
  ring

/-- Gain changes have a uniform bound on the margin of two classes. The
constant uses the distance between their tied embeddings, for arbitrary
contexts and arbitrary strengths of the causal read. -/
theorem Model.displacedSuccessor_margin_gain_bound (M : Model D d H F p) (T : PairTables D p)
    (strength g h : ℝ) {eps : ℝ} (heps : 0 < eps) (hD : 0 < D)
    (scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (u v : Fin p) :
    let gap := fun gain => M.displacedSuccessorLogit T strength gain eps scale xs L i u -
      M.displacedSuccessorLogit T strength gain eps scale xs L i v
    |gap g - gap h| ≤ |g-h| * (‖M.embedding u - M.embedding v‖ * Real.sqrt D) := by
  let a := xs (i-1)
  let b := xs i
  let w := Perspective.softmaxWeight (M.logits eps scale xs L (i-1))
  let raw := (transportedValue M.embedding w (a*b) + (T.withSuccessor xs i).prefixRead .xor (0,0) xs i) +
    strength • (M.embedding (successorAddress xs i) - M.embedding b)
  let r := GPTMini.rmsNormEps eps raw
  let x := M.feature eps scale xs L i
  change |(⟪M.embedding u, x + g • r⟫ - ⟪M.embedding v, x + g • r⟫) -
    (⟪M.embedding u, x + h • r⟫ - ⟪M.embedding v, x + h • r⟫)| ≤ _
  rw [← inner_sub_left (M.embedding u) (M.embedding v) (x + g • r),
    ← inner_sub_left (M.embedding u) (M.embedding v) (x + h • r)]
  simp only [inner_add_right, real_inner_smul_right, add_sub_add_left_eq_sub,
    ← sub_mul, abs_mul]
  exact mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hD raw) (norm_nonneg _)))
    (abs_nonneg _)

example (M : Model 64 16 4 86 68) (T : PairTables 64 68)
    (xs : ℕ → Fin 68) (L i : ℕ) (u v : Fin 68) :
    let gap := fun g => M.displacedSuccessorLogit T 1 g 1 1 xs L i u -
      M.displacedSuccessorLogit T 1 g 1 1 xs L i v
    |gap (3/4) - gap (1/2)| ≤ |(3/4 : ℝ)-1/2| * (‖M.embedding u - M.embedding v‖ * Real.sqrt 64) :=
  M.displacedSuccessor_margin_gain_bound T _ _ _ (by norm_num) (by decide) _ xs L i u v


/-- A gain multiplier can depend on whether the causal read returns a
different token. It uses only completed transitions and the query input. -/
def retrievalGain {α : Type*} [DecidableEq α] (gain factor : ℝ) (xs : ℕ → α) (i : ℕ) : ℝ :=
  if successorAddress xs i = xs i then gain else factor * gain

/-- Injectively renaming all input tokens preserves the gain at every position
even if the target vocabulary is larger. -/
theorem retrievalGain_map {α β : Type*} [DecidableEq α] [DecidableEq β]
    (f : α → β) (hf : Function.Injective f) (gain factor : ℝ) (xs : ℕ → α) (i : ℕ) :
    retrievalGain gain factor (fun j => f (xs j)) i = retrievalGain gain factor xs i := by
  simp only [retrievalGain, successorAddress, latestSuccessor_map f hf, Option.getD_map, hf.eq_iff]
  split <;> rename_i h <;> simp [h]

/-- Place the input-dependent gain on the normalized hidden residual. -/
noncomputable def Model.retrievalGainLogit (M : Model D d H F p) (T : PairTables D p)
    (strength gain factor eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  M.displacedSuccessorLogit T strength (retrievalGain gain factor xs i) eps scale xs L i c

/-- On a self-read the full score agrees for all parameters and all
multipliers. This includes a missing transition, a self-loop, and the first
two positions of any prompt. -/
theorem Model.retrievalGainLogit_eq_of_self (M : Model D d H F p) (T : PairTables D p)
    (strength gain factor eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ)
    (h : successorAddress xs i = xs i) (c : Fin p) :
    M.retrievalGainLogit T strength gain factor eps scale xs L i c =
      M.displacedSuccessorLogit T strength gain eps scale xs L i c := by
  simp [retrievalGainLogit, retrievalGain, h]

/-- Interpolating the gain multipliers interpolates the entire tied logit
vector for every input, including prefixes whose successor is a self-read.
Source: the retrieval-gain trials in `experiments/phasor_value_transport/README.md`. -/
theorem Model.retrievalGainLogit_factor_interpolate (M : Model D d H F p) (T : PairTables D p)
    (strength gain lo hi t eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    M.retrievalGainLogit T strength gain ((1-t)*lo+t*hi) eps scale xs L i c =
      (1-t) * M.retrievalGainLogit T strength gain lo eps scale xs L i c +
        t * M.retrievalGainLogit T strength gain hi eps scale xs L i c := by
  by_cases h : successorAddress xs i = xs i
  · simp only [retrievalGainLogit, retrievalGain, ite_eq_left h]
    ring
  · simpa only [retrievalGainLogit, retrievalGain, ite_eq_right h, add_mul, mul_assoc] using
      M.displacedSuccessorLogit_interpolate T strength (lo*gain) (hi*gain) t eps scale xs L i c

/-- At multiplier 1.75, every score is a quarter of the ordinary score
plus three quarters of the doubled-multiplier score, with shared weights. -/
example (M : Model 64 16 4 86 68) (T : PairTables 64 68)
    (xs : ℕ → Fin 68) (L i : ℕ) (c : Fin 68) :
    M.retrievalGainLogit T 1 (1/2) (7/4) 1 1 xs L i c =
      (1/4) * M.retrievalGainLogit T 1 (1/2) 1 1 1 xs L i c +
        (3/4) * M.retrievalGainLogit T 1 (1/2) 2 1 1 xs L i c := by
  convert M.retrievalGainLogit_factor_interpolate T 1 (1/2) 1 2 (3/4) 1 1 xs L i c using 1 <;>
    norm_num

/-- The conditional gain can change a two-class margin by at most the
absolute change in gain on a hit times the class distance and square root
of the width. The estimate holds for arbitrary embeddings and prefixes. -/
theorem Model.retrievalGain_margin_bound (M : Model D d H F p) (T : PairTables D p)
    (strength gain factor : ℝ) {eps : ℝ} (heps : 0 < eps) (hD : 0 < D)
    (scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (u v : Fin p) :
    let read := M.retrievalGainLogit T strength gain factor eps scale xs L i
    let base := M.displacedSuccessorLogit T strength gain eps scale xs L i
    |(read u - read v) - (base u - base v)| ≤
      (|factor-1| * |gain|) * (‖M.embedding u - M.embedding v‖ * Real.sqrt D) := by
  by_cases h : successorAddress xs i = xs i
  · simp only [M.retrievalGainLogit_eq_of_self T _ _ _ _ _ xs L i h, sub_self, abs_zero]
    positivity
  · have gap := M.displacedSuccessor_margin_gain_bound T strength (factor * gain) gain
      heps hD scale xs L i u v
    have diff : factor * gain - gain = (factor-1) * gain := by ring
    simpa [retrievalGainLogit, retrievalGain, h, diff, abs_mul] using gap

example (xs : ℕ → ℕ) (i : ℕ) :
    retrievalGain (1/2) 2 (fun j => xs j + 17) i = retrievalGain (1/2) 2 xs i :=
  retrievalGain_map (fun a : ℕ => a+17) (by intro a b h; dsimp at h; omega) _ _ xs i

example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.retrievalGainLogit T 1 (1/2) 2 1 1 xs L 1 c =
      M.displacedSuccessorLogit T 1 (1/2) 1 1 xs L 1 c :=
  M.retrievalGainLogit_eq_of_self T _ _ _ _ _ xs L 1 (successorAddress_start xs 1 (by decide)) c

example (M : Model 64 16 4 86 68) (T : PairTables 64 68)
    (xs : ℕ → Fin 68) (L i : ℕ) (u v : Fin 68) :
    let read := M.retrievalGainLogit T 1 (1/2) 2 1 1 xs L i
    let base := M.displacedSuccessorLogit T 1 (1/2) 1 1 xs L i
    |(read u - read v) - (base u - base v)| ≤
      (|(2:ℝ)-1| * |(1/2:ℝ)|) * (‖M.embedding u - M.embedding v‖ * Real.sqrt 64) :=
  M.retrievalGain_margin_bound T _ _ _ (by norm_num) (by decide) _ xs L i u v

/-- An independent learned increment on changed successor reads; both
coefficients are shared by all positions and sequences. The bonus is added
to the amplitude of the normalized hidden residual before the tied readout. -/
def successorBonus {α : Type*} [DecidableEq α] (base bonus : ℝ) (xs : ℕ → α) (i : ℕ) : ℝ :=
  if successorAddress xs i = xs i then base else base + bonus

/-- Any two prescribed amplitudes, for a self-read and a changed successor,
determine exactly one base gain and bonus. Source: the independent-gain
experiment in `experiments/phasor_value_transport/README.md`. -/
theorem existsUnique_successorBonus {α : Type*} [DecidableEq α] (xs : ℕ → α) (miss hit : ℕ)
    (hm : successorAddress xs miss = xs miss) (hh : successorAddress xs hit ≠ xs hit)
    (a b : ℝ) : ∃! θ : ℝ × ℝ,
      successorBonus θ.1 θ.2 xs miss = a ∧ successorBonus θ.1 θ.2 xs hit = b := by
  refine ⟨(a, b-a), ?_, ?_⟩
  · simp [successorBonus, hm, hh]
  · rintro ⟨u,v⟩ huv
    simp only [successorBonus, ite_eq_left hm, ite_eq_right hh] at huv
    apply Prod.ext
    · exact huv.1
    · change v = b-a
      linarith [huv.1, huv.2]

/-- Initializing the bonus to the excess multiplier times the base
recovers every score of the fixed-multiplier architecture. -/
theorem Model.successorBonusLogit_eq (M : Model D d H F p) (T : PairTables D p)
    (strength gain factor eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    M.displacedSuccessorLogit T strength (successorBonus gain ((factor-1)*gain) xs i) eps scale xs L i c =
      M.retrievalGainLogit T strength gain factor eps scale xs L i c := by
  by_cases h : successorAddress xs i = xs i
  · simp [successorBonus, retrievalGainLogit, retrievalGain, h]
  · have eq : gain + (factor-1)*gain = factor*gain := by ring
    simp [successorBonus, retrievalGainLogit, retrievalGain, h, eq]

example (a b : ℝ) :
    let xs := fun i => (![0,1,0] : Fin 3 → Fin 2) (Fin.ofNat 3 i)
    ∃! θ : ℝ × ℝ, successorBonus θ.1 θ.2 xs 0 = a ∧ successorBonus θ.1 θ.2 xs 2 = b :=
  existsUnique_successorBonus _ 0 2 (by decide) (by decide) a b

end Transformer.Phasor
