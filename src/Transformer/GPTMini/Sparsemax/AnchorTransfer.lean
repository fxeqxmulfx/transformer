import Transformer.GPTMini.Sparsemax.AnchoredScores

/-!
# Lifting active score transfers into trainable anchor parameters

Derived from arXiv:1602.02068v2, §2.5, `sparsemax_gradient`, using
the independent bounded score architecture rather than `QKNormScores`
at `73f8a0b`. Invert the bounded coordinates along an exact active-pair
score transfer. The resulting parameter curve starts at the original
parameters, is differentiable there, and realizes the same raw scores
throughout a neighborhood of zero.

This closes the distinction between a useful raw score direction and
a direction accessible to this architecture's learned anchor parameters.
No inverse property is asserted for steps leaving the open score cap.
Ordinary scores and values are frozen along this particular curve.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall
open Filter
open scoped Topology

/-- Lift an anchor-pair score transfer through the coordinate inverse.
Source context: the derived score architecture for §2.5 of
arXiv:1602.02068v2; this changes parameters rather than their backward rule. -/
def boundedTransferParameters {A : ℕ} (cap : ℝ) (parameters : Fin A → ℝ)
    (j k : Fin A) (t : ℝ) : Fin A → ℝ :=
  fun a => inverseCoordinate cap
    (transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a)

/-- The lifted curve starts at the actual trainable parameters.
Source context: the derived local inverse for §2.5 of
arXiv:1602.02068v2; no surrogate or detached coordinate is used. -/
theorem boundedTransferParameters_zero {A : ℕ} (cap : ℝ) (parameters : Fin A → ℝ)
    (j k : Fin A) (hc : 0 < cap) :
    boundedTransferParameters cap parameters j k 0 = parameters := by
  funext a
  simp only [boundedTransferParameters, transferScores, zero_mul, add_zero]
  exact inverseCoordinate_bounded cap (parameters a) hc

/-- The starting-point hypothesis is inhabited with two finite anchors.
Source context: arXiv:1602.02068v2, §2.5, derived parameter lift. -/
example : boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 0 1 0 = fun _ => 0 :=
  boundedTransferParameters_zero _ _ 0 1 (by norm_num)

/-- The lifted parameter curve has an ordinary derivative at its start.
Source context: §2.5 of arXiv:1602.02068v2, the derived inverse-score
curve; strict coordinate bounds make all logarithms differentiable. -/
theorem boundedTransferParameters_differentiableAt {A : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (j k : Fin A) (hc : 0 < cap) :
    DifferentiableAt ℝ (boundedTransferParameters cap parameters j k) 0 := by
  apply differentiableAt_pi.mpr
  intro a
  have hr := boundedCoordinate_bounds cap (parameters a) hc
  have hi := inverseCoordinate_differentiableAt cap (boundedCoordinate cap (parameters a))
    hr.1 hr.2
  have hd : DifferentiableAt ℝ
      (fun t : ℝ => transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a) 0 := by
    have h := (hasDerivAt_const (0 : ℝ) (boundedCoordinate cap (parameters a))).add
      ((hasDerivAt_id (0 : ℝ)).mul_const (basis j a - basis k a))
    simpa only [transferScores, Pi.add_def, id_eq] using h.differentiableAt
  have hi' : DifferentiableAt ℝ (inverseCoordinate cap)
      (transferScores (fun b => boundedCoordinate cap (parameters b)) j k 0 a) := by
    simpa only [transferScores, zero_mul, add_zero] using hi
  simpa only [boundedTransferParameters, Function.comp_def] using hi'.comp (0 : ℝ) hd

/-- The parameter differentiability premise has the same sparse instance.
Source context: arXiv:1602.02068v2, §2.5, derived inverse curve. -/
example : DifferentiableAt ℝ
    (boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 0 1) 0 :=
  boundedTransferParameters_differentiableAt _ _ 0 1 (by norm_num)

/-- The local lift realizes exactly the raw anchor-pair transfer,
including unchanged ordinary scores. Source context: the score direction
from §2.5 of arXiv:1602.02068v2, through the new independent chart.
This identity holds on a neighborhood, not merely to first order. -/
theorem boundedTransferParameters_realizes_transfer {A N : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (j k : Fin A) (hc : 0 < cap) :
    ∀ᶠ t : ℝ in 𝓝 0,
      anchoredScores cap (boundedTransferParameters cap parameters j k t) ordinary =
        transferScores (anchoredScores cap parameters ordinary) (Fin.castAdd N j)
          (Fin.castAdd N k) t := by
  classical
  have hinside : ∀ᶠ t : ℝ in 𝓝 0, ∀ a : Fin A,
      -cap < transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a ∧
      transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a < cap := by
    apply Filter.eventually_all.mpr
    intro a
    have hb := boundedCoordinate_bounds cap (parameters a) hc
    have hd := (hasDerivAt_const (0 : ℝ) (boundedCoordinate cap (parameters a))).add
      ((hasDerivAt_id (0 : ℝ)).mul_const (basis j a - basis k a))
    have hs : ContinuousAt
        (fun t => transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a) 0 := by
      simpa only [transferScores, Pi.add_def, id_eq] using hd.continuousAt
    have hl : -cap < transferScores (fun b => boundedCoordinate cap (parameters b)) j k 0 a := by
      simpa only [transferScores, zero_mul, add_zero] using hb.1
    have hu : transferScores (fun b => boundedCoordinate cap (parameters b)) j k 0 a < cap := by
      simpa only [transferScores, zero_mul, add_zero] using hb.2
    exact (continuousAt_const.eventually_lt hs hl).and
      (hs.eventually_lt continuousAt_const hu)
  apply hinside.mono
  intro t ht
  funext n
  refine Fin.addCases ?_ ?_ n
  · intro a
    have hb (b : Fin A) : basis (Fin.castAdd N b) (Fin.castAdd N a) = basis b a := by
      by_cases he : a = b
      · have hh : Fin.castAdd N a = Fin.castAdd N b := congrArg (Fin.castAdd N) he
        simp only [basis, ite_eq_left he, ite_eq_left hh]
      · have hh : Fin.castAdd N a ≠ Fin.castAdd N b :=
          fun ht => he (Fin.castAdd_injective A N ht)
        simp only [basis, ite_eq_right he, ite_eq_right hh]
    have hi := boundedCoordinate_inverse cap
      (transferScores (fun b => boundedCoordinate cap (parameters b)) j k t a)
      (ht a).1 (ht a).2
    simpa only [anchoredScores, Fin.addCases_left, boundedTransferParameters,
      transferScores, hb j, hb k] using hi
  · intro a
    have hj : Fin.natAdd A a ≠ Fin.castAdd N j := by
      intro he
      have hv := congrArg Fin.val he
      change A + a.val = j.val at hv
      omega
    have hk : Fin.natAdd A a ≠ Fin.castAdd N k := by
      intro he
      have hv := congrArg Fin.val he
      change A + a.val = k.val at hv
      omega
    simp [anchoredScores, transferScores, basis, hj, hk]

/-- The exact local-lift premise is inhabited while an ordinary slot
has zero sparsemax weight. Source context: arXiv:1602.02068v2, §2.5. -/
example : ∀ᶠ t : ℝ in 𝓝 0,
    anchoredScores (1 / 8) (boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 0 1 t)
      (fun _ : Fin 1 => 0) =
    transferScores (anchoredScores (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0))
      0 1 t :=
  boundedTransferParameters_realizes_transfer _ _ _ 0 1 (by norm_num)

/-- Small score transfers lift to nearby trainable parameters.
Source context: the derived inverse chart for §2.5 of
arXiv:1602.02068v2; this permits transporting local-minimum claims. -/
theorem boundedTransferParameters_continuousAt {A : ℕ} (cap : ℝ)
    (parameters : Fin A → ℝ) (j k : Fin A) (hc : 0 < cap) :
    ContinuousAt (boundedTransferParameters cap parameters j k) 0 :=
  (boundedTransferParameters_differentiableAt cap parameters j k hc).continuousAt

/-- The local continuity premise holds for finite sparse-row parameters.
Source context: arXiv:1602.02068v2, §2.5, derived parameter lift. -/
example : ContinuousAt (boundedTransferParameters (1 / 8) (fun _ : Fin 2 => 0) 0 1) 0 :=
  boundedTransferParameters_continuousAt _ _ 0 1 (by norm_num)

end Transformer.GPTMini.Sparsemax
