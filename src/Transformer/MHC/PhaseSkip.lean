module

public import Transformer.Residual.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

/-!
# The phase-conjugated skip map

The main model's four-stream mHC replaces the skip matrix `P` (the finite Sinkhorn map of
arXiv:2512.24880v2, §4.2) by `D P D*`, with `D = diag(r_s)` one unit phase per stream, `r_0 = 1`
and `r_s = e^{i arctan b_s}` for one learned `b_s` per stream and sublayer
(`research/main_model/README.txt`, *Architecture*). The streams are vectors and a phase turns a
whole stream: `(D P D* x)_i = r_i ∑_j P_ij conj(r_j) x_j` (`phaseMix`). A `PhaseAction` is all of
`e^{iφ}` this needs: isometries `rot φ` composing by adding angles; `complexPhase` on `ℂ` is one,
and `Transformer.SequenceModel.RotateHead` gives another on the half-split heads.

* `phaseMix_norm_le`: a nonnegative row-normalized `P` keeps every common norm bound, with no
  column normalization and no Sinkhorn limit.
* `phaseMix_zero`: zero phases are the ordinary mixer. `phaseMix_aligned`: the aligned constant
  mode `r_i z` is conserved, but the ordinary constant mode is not, even for a positive doubly
  stochastic `P` (`phaseMix_complex_constant_not_preserved`).
* `phaseMix_energy_bound_iff`: `‖D P D* x‖² ≤ C ‖x‖²` for all `x` iff it holds for `P`, the phases
  being isometries; the implemented finite Sinkhorn matrix need not be doubly stochastic.
* `unitPhase b = e^{i arctan b}` is the implemented `r_s`, of norm one.

Coefficients are frozen at one token: no Jacobian of the router, no write-back, no floating point.
Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/proofs/
ComplexPhaseRouting.lean` (`MainModelComplexPhaseRouting`), there for one complex scalar per stream.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.MHC

open Transformer.Residual

/-- Isometries `rot φ` of a feature space, one per angle `φ`, composing by adding angles: the
action of the phases `e^{iφ}`. -/
structure PhaseAction (E : Type*) [NormedAddCommGroup E] where
  /-- The rotation by an angle. -/
  rot : ℝ → E → E
  /-- Angle `0` is the identity. -/
  rot_zero : ∀ x, rot 0 x = x
  /-- Rotations compose by adding angles. -/
  rot_add : ∀ a b x, rot a (rot b x) = rot (a + b) x
  /-- A rotation preserves the norm. -/
  norm_rot : ∀ a x, ‖rot a x‖ = ‖x‖

section Group

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] (P : PhaseAction E)

/-- Turning back by the angle just turned undoes it. -/
theorem PhaseAction.rot_neg_rot (a : ℝ) (x : E) : P.rot (-a) (P.rot a x) = x := by
  rw [P.rot_add, neg_add_cancel, P.rot_zero]

/-- The squared length `∑ᵢ ‖xᵢ‖²` of a tuple of streams. -/
noncomputable def streamEnergy (x : ι → E) : ℝ := ∑ i, ‖x i‖ ^ 2

/-- Turning every stream by its own angle preserves the energy. -/
theorem streamEnergy_action (φ : ι → ℝ) (x : ι → E) :
    streamEnergy (fun j => P.rot (φ j) (x j)) = streamEnergy x := by
  unfold streamEnergy
  exact Finset.sum_congr rfl fun j _ => by rw [P.norm_rot]

end Group

section Mix

variable {ι : Type*} [Fintype ι] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  (P : PhaseAction E)

/-- The phase-conjugated skip `D H D*` on streams: turn stream `j` back by `φ_j`, mix with `H`,
turn row `i` forward by `φ_i`. -/
noncomputable def phaseMix (H : Matrix ι ι ℝ) (φ : ι → ℝ) (x : ι → E) : ι → E :=
  fun i => P.rot (φ i) (weightedSum (H i) fun j => P.rot (-φ j) (x j))

/-- Nonnegative row-normalized coefficients keep the maximum stream norm after phase conjugation.
No column normalization and no ideal Sinkhorn limit is assumed. -/
theorem phaseMix_norm_le (H : Matrix ι ι ℝ) (φ : ι → ℝ) (x : ι → E) (R : ℝ)
    (hp : ∀ i j, 0 ≤ H i j) (hs : ∀ i, ∑ j, H i j = 1) (hx : ∀ i, ‖x i‖ ≤ R) (i : ι) :
    ‖phaseMix P H φ x i‖ ≤ R := by
  change ‖P.rot (φ i) (weightedSum (H i) fun j => P.rot (-φ j) (x j))‖ ≤ R
  rw [P.norm_rot]
  exact weightedSum_convex_norm_le _ _ _ (hp i) (hs i) fun j => (P.norm_rot _ _).trans_le (hx j)

/-- The zero-initialized phases give the original mixer for every input. -/
theorem phaseMix_zero (H : Matrix ι ι ℝ) (x : ι → E) :
    phaseMix P H (fun _ => 0) x = mix H x := by
  funext i
  simp only [phaseMix, mix, neg_zero, P.rot_zero]

/-- A phase-aligned constant mode replaces ordinary identical-stream conservation for this
conjugated skip map. Write-back is outside this claim. -/
theorem phaseMix_aligned (H : Matrix ι ι ℝ) (φ : ι → ℝ) (z : E) (hs : ∀ i, ∑ j, H i j = 1) :
    phaseMix P H φ (fun j => P.rot (φ j) z) = fun j => P.rot (φ j) z := by
  funext i
  simp only [phaseMix, P.rot_neg_rot]
  rw [weightedSum_const, hs, one_smul]

/-- Exact equivalence of every squared-energy upper bound of the frozen map. The phases are
isometries, so the conjugated and the ordinary skip are bounded by the same constants; this does
not assume that the implemented finite Sinkhorn matrix is doubly stochastic. -/
theorem phaseMix_energy_bound_iff (H : Matrix ι ι ℝ) (φ : ι → ℝ) (C : ℝ) :
    (∀ x : ι → E, streamEnergy (phaseMix P H φ x) ≤ C * streamEnergy x) ↔
      (∀ x : ι → E, streamEnergy (mix H x) ≤ C * streamEnergy x) := by
  constructor
  · intro h x
    have hx := h fun j => P.rot (φ j) (x j)
    have e : phaseMix P H φ (fun j => P.rot (φ j) (x j)) = fun i => P.rot (φ i) (mix H x i) := by
      funext i
      simp only [phaseMix, mix, P.rot_neg_rot]
    rwa [e, streamEnergy_action, streamEnergy_action] at hx
  · intro h x
    have e : phaseMix P H φ x =
        fun i => P.rot (φ i) (mix H (fun j => P.rot (-φ j) (x j)) i) := rfl
    have hu : streamEnergy (fun j => P.rot (-φ j) (x j)) = streamEnergy x :=
      streamEnergy_action P (fun j => -φ j) x
    rw [e, streamEnergy_action, ← hu]
    exact h _

end Mix

section Complex

/-- Multiplication by `e^{iφ}` on `ℂ`: the action of the model's scalar phases. -/
noncomputable def complexPhase : PhaseAction ℂ where
  rot a z := Complex.exp (a * Complex.I) * z
  rot_zero z := by simp
  rot_add a b z := by
    rw [← mul_assoc, ← Complex.exp_add]
    congr 2
    push_cast
    ring
  norm_rot a z := by rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

/-- The implemented stream phase `r = (1 + i b) / √(1 + b²)` of the learned `b`. -/
noncomputable def unitPhase (b : ℝ) : ℂ :=
  ⟨1 / Real.sqrt (1 + b ^ 2), b / Real.sqrt (1 + b ^ 2)⟩

/-- `r = e^{i arctan b}`: the angle of the phase is `arctan b`. -/
theorem unitPhase_eq_exp_arctan (b : ℝ) :
    unitPhase b = Complex.exp ((Real.arctan b : ℝ) * Complex.I) := by
  apply Complex.ext
  · simp [unitPhase, Real.cos_arctan]
  · simp [unitPhase, Real.sin_arctan]

/-- The implemented phase has modulus one. -/
theorem unitPhase_norm (b : ℝ) : ‖unitPhase b‖ = 1 := by
  rw [unitPhase_eq_exp_arctan]
  exact Complex.norm_exp_ofReal_mul_I _

/-- A strictly positive doubly stochastic two-stream mixer need not preserve the ordinary constant
streams after a nontrivial finite phase conjugation: stream `0` of the constant streams `1`
leaves `ℝ`. -/
theorem phaseMix_complex_constant_not_preserved :
    phaseMix complexPhase (fun _ _ : Fin 2 => (1 / 2 : ℝ)) ![0, Real.arctan 1]
      (fun _ => (1 : ℂ)) 0 ≠ 1 := by
  intro h
  have hi := congrArg Complex.im h
  have hn : 0 < 1 / Real.sqrt (1 + 1 ^ 2) := by positivity
  have e : phaseMix complexPhase (fun _ _ : Fin 2 => (1 / 2 : ℝ)) ![0, Real.arctan 1]
      (fun _ => (1 : ℂ)) 0 = (1 / 2 : ℝ) • (1 : ℂ) +
        (1 / 2 : ℝ) • Complex.exp (((-Real.arctan 1 : ℝ) : ℂ) * Complex.I) := by
    simp [phaseMix, weightedSum, complexPhase, Fin.sum_univ_two]
  rw [e] at hi
  simp only [Complex.add_im, Complex.smul_im, Complex.one_im, Complex.exp_ofReal_mul_I_im,
    Real.sin_neg, Real.sin_arctan, smul_eq_mul] at hi
  linarith

/-- Four streams, uniform `P`, four phases: the bound `1` on every stream holds for the
conjugated skip, as for the plain one. -/
example : ∀ i : Fin 4,
    ‖phaseMix complexPhase (fun _ _ : Fin 4 => (1 / 4 : ℝ)) (fun j => (j : ℝ)) (fun _ => (1 : ℂ))
      i‖ ≤ 1 :=
  phaseMix_norm_le complexPhase _ _ _ 1 (fun _ _ => by norm_num)
    (fun _ => by norm_num [Fin.sum_univ_four]) (fun _ => by simp)

/-- The aligned constant mode `r_i z` is fixed. -/
example :
    phaseMix complexPhase (fun _ _ : Fin 4 => (1 / 4 : ℝ)) (fun j => (j : ℝ))
        (fun j : Fin 4 => complexPhase.rot (j : ℝ) (2 + Complex.I)) =
      fun j : Fin 4 => complexPhase.rot (j : ℝ) (2 + Complex.I) :=
  phaseMix_aligned complexPhase _ _ _ (fun _ => by norm_num [Fin.sum_univ_four])

/-- The identity skip is bounded by `1` in energy, so is its conjugation. -/
example (φ : Fin 4 → ℝ) (x : Fin 4 → ℂ) :
    streamEnergy (phaseMix complexPhase (1 : Matrix (Fin 4) (Fin 4) ℝ) φ x) ≤
      1 * streamEnergy x :=
  (phaseMix_energy_bound_iff complexPhase 1 φ 1).mpr
    (fun x => by simp [mix_one]) x

end Complex

end Transformer.MHC
