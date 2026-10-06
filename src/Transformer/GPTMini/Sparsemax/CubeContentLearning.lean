import Transformer.GPTMini.Sparsemax.CubeContentExpressivity
import Transformer.GPTMini.Sparsemax.MatrixOutputError

/-!
# Joint convex learning of content embeddings, attention and generated values

New restricted block after arXiv:1602.02068v2, Eq. (1). A single K*D
coefficient table learns selected causal content interactions. Fixed
affine readouts of its coefficients generate all flip weights and hence
the actual physical Q/K and inverse-adjusted common values. Coefficient
caps and the outgoing budget are jointly convex. Genuine sparsemax/value
predictions are affine in the entire learned state, including support
changes, for arbitrary fixed input codes and selected interaction sets.

The feature encoder, structural virtual-memory mask and scalar readouts
are fixed architecture. Free Q/K matrices, trainable normalization,
stacked layers and a downstream FFN are not included in this chart.
The later causal block supplies a conditional convex-output-criterion
theorem; no particular language loss or successful Basis training is assumed.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ]

/-- Fixed affine coefficient readouts generate all selected learned content flip masses.
Source: parameter sharing in the new sparsemax Eq. (1) architecture. -/
def cubeContentLearnedRoutes {D : ℕ} (offset gain : ι → ℝ) (pick : ι → κ) (channel : Fin D)
    (W : Matrix κ (Fin D) ℝ) : ι → ℝ := fun i => offset i + gain i * W (pick i) channel

/-- All learned content coefficients, physical embeddings and common values share one convex chart.
Source: bounded affine routing and compact value coordinates after sparsemax Eq. (1). -/
def cubeContentLearningDomain {D : ℕ} (cap floor : ℝ) (offset gain : ι → ℝ) (pick : ι → κ)
    (channel : Fin D) : Set (Matrix κ (Fin D) ℝ) :=
  {W | (∀ k d, -cap ≤ W k d ∧ W k d ≤ cap) ∧
    cubeContentLearnedRoutes offset gain pick channel W ∈ cubeContentRouteDomain ι floor}

omit [DecidableEq ι] [Fintype ι] [Fintype κ] in
/-- Actual generated flip masses preserve every convex coefficient interpolation.
Source: the fixed affine readouts before sparsemax Eq. (1). -/
theorem cubeContentLearnedRoutes_affine {D : ℕ} (offset gain : ι → ℝ) (pick : ι → κ) (channel : Fin D)
    (W V : Matrix κ (Fin D) ℝ) (a b : ℝ) (hab : a + b = 1) :
    cubeContentLearnedRoutes offset gain pick channel (a • W + b • V) =
      a • cubeContentLearnedRoutes offset gain pick channel W +
        b • cubeContentLearnedRoutes offset gain pick channel V := by
  ext i
  change offset i + gain i * (a * W (pick i) channel + b * V (pick i) channel) =
    a * (offset i + gain i * W (pick i) channel) + b * (offset i + gain i * V (pick i) channel)
  have ha : a = 1 - b := by linarith
  rw [ha]
  ring

/-- Two different nonlinear-content coefficient tables inhabit the affine-routing premise. -/
example : cubeContentLearnedRoutes (fun _ : Fin 2 => (1 / 16 : ℝ)) (fun _ => (1 / 16 : ℝ))
    (fun _ => (1 : Fin 2)) (0 : Fin 1) ((1 / 2 : ℝ) • !![0; -1] + (1 / 2 : ℝ) • !![0; 1]) =
    (1 / 2 : ℝ) • cubeContentLearnedRoutes (fun _ : Fin 2 => (1 / 16 : ℝ)) (fun _ => (1 / 16 : ℝ))
      (fun _ => (1 : Fin 2)) (0 : Fin 1) !![0; -1] +
    (1 / 2 : ℝ) • cubeContentLearnedRoutes (fun _ : Fin 2 => (1 / 16 : ℝ)) (fun _ => (1 / 16 : ℝ))
      (fun _ => (1 : Fin 2)) (0 : Fin 1) !![0; 1] :=
  cubeContentLearnedRoutes_affine _ _ _ _ _ _ _ _ (by norm_num)

omit [DecidableEq ι] [Fintype κ] in
/-- The entire tied content-learning domain is convex without frozen Q/K, values or positive supports.
Source: bounded affine readout and routing budgets following sparsemax Eq. (1). -/
theorem cubeContentLearningDomain_convex {D : ℕ} (cap floor : ℝ) (offset gain : ι → ℝ)
    (pick : ι → κ) (channel : Fin D) :
    Convex ℝ (cubeContentLearningDomain cap floor offset gain pick channel) := by
  intro W hW V hV a b ha hb hab
  constructor
  · intro k d
    exact (convex_Icc (-cap) cap) (hW.1 k d) (hV.1 k d) ha hb hab
  · rw [cubeContentLearnedRoutes_affine offset gain pick channel W V a b hab]
    exact cubeContentRouteDomain_convex floor hW.2 hV.2 ha hb hab

/-- Genuine sparsemax, actual physical content embeddings and decoded common values are jointly learned.
Source: the fixed-readout content chart following sparsemax Eq. (1). -/
def cubeContentLearnedForward {R D : ℕ} (code : Fin R → CubeContentState ι) (features : κ → Finset ι)
    (offset gain : ι → ℝ) (pick : ι → κ) (channel : Fin D) (W : Matrix κ (Fin D) ℝ) :
    Matrix (Fin R) (Fin D) ℝ :=
  cubeContentForward code features (cubeContentLearnedRoutes offset gain pick channel W) W

/-- All jointly learned physical parameters give the intended content-interaction response exactly.
Source: original variational sparsemax and the proved value decoder following Eq. (1). -/
theorem cubeContentLearnedForward_eq {R D : ℕ} (cap floor : ℝ) (code : Fin R → CubeContentState ι)
    (features : κ → Finset ι) (offset gain : ι → ℝ) (pick : ι → κ) (channel : Fin D)
    (W : Matrix κ (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ cubeContentLearningDomain cap floor offset gain pick channel) :
    cubeContentLearnedForward code features offset gain pick channel W =
      fun r d => cubeContentOutput features W (code r) d :=
  cubeContentForward_eq floor code features _ W hf hW.2

/-- Arbitrary chosen content interactions retain affine actual forward interpolation in all learned parameters.
Source: the exact joint content chart following sparsemax Eq. (1), including support changes. -/
theorem cubeContentLearnedForward_affine {R D : ℕ} (cap floor : ℝ) (code : Fin R → CubeContentState ι)
    (features : κ → Finset ι) (offset gain : ι → ℝ) (pick : ι → κ) (channel : Fin D)
    (W V : Matrix κ (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ cubeContentLearningDomain cap floor offset gain pick channel)
    (hV : V ∈ cubeContentLearningDomain cap floor offset gain pick channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    cubeContentLearnedForward code features offset gain pick channel (a • W + b • V) =
      a • cubeContentLearnedForward code features offset gain pick channel W +
        b • cubeContentLearnedForward code features offset gain pick channel V := by
  have hm := cubeContentLearningDomain_convex cap floor offset gain pick channel hW hV ha hb hab
  rw [cubeContentLearnedForward_eq cap floor code features offset gain pick channel _ hf hm,
    cubeContentLearnedForward_eq cap floor code features offset gain pick channel W hf hW,
    cubeContentLearnedForward_eq cap floor code features offset gain pick channel V hf hV]
  have he := cubeContentOutput_linear features W V a b
  ext r d
  exact congrArg (fun f => f (code r) d) he

/-- Opposite XOR response coefficients share one bounded feasible domain with genuinely different routes.
Source: a nontrivial joint-content witness following sparsemax Eq. (1). -/
theorem cubeContentXor_mem (edge : Fin 2) :
    (!![0; if edge = 0 then -1 else 1] : Matrix (Fin 2) (Fin 1) ℝ) ∈
      cubeContentLearningDomain 1 (3 / 4) (fun _ : Fin 2 => (1 / 16 : ℝ))
        (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) 0 := by
  constructor
  · intro k d
    fin_cases edge <;> fin_cases k <;> fin_cases d <;> norm_num
  · constructor
    · intro i
      fin_cases edge <;> norm_num [cubeContentLearnedRoutes]
    · fin_cases edge <;> norm_num [cubeContentLearnedRoutes, Fin.sum_univ_two]

/-- Both genuine opposite nonlinear responses satisfy every exact-forward premise. -/
example (edge : Fin 2) : cubeContentLearnedForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0))
    (fun k : Fin 2 => if k = 0 then ∅ else {0, 1}) (fun _ => (1 / 16 : ℝ))
    (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) 0 !![0; if edge = 0 then -1 else 1] =
      fun _ d => cubeContentOutput (fun k : Fin 2 => if k = 0 then (∅ : Finset (Fin 2)) else {0, 1})
        !![0; if edge = 0 then -1 else 1] (fun i : Fin 2 => decide (i = 0)) d :=
  cubeContentLearnedForward_eq 1 (3 / 4) _ _ _ _ _ _ _ (by norm_num) (cubeContentXor_mem edge)

/-- Two opposite learned XOR responses and their midpoint satisfy every joint-affinity premise. -/
example : cubeContentLearnedForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0))
    (fun k : Fin 2 => if k = 0 then ∅ else {0, 1}) (fun _ => (1 / 16 : ℝ))
    (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) 0
      ((1 / 2 : ℝ) • !![0; -1] + (1 / 2 : ℝ) • !![0; 1]) =
    (1 / 2 : ℝ) • cubeContentLearnedForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0))
      (fun k : Fin 2 => if k = 0 then ∅ else {0, 1}) (fun _ => (1 / 16 : ℝ))
      (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) 0 !![0; -1] +
    (1 / 2 : ℝ) • cubeContentLearnedForward (fun _ : Fin 1 => fun i : Fin 2 => decide (i = 0))
      (fun k : Fin 2 => if k = 0 then ∅ else {0, 1}) (fun _ => (1 / 16 : ℝ))
      (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) 0 !![0; 1] :=
  cubeContentLearnedForward_affine 1 (3 / 4) _ _ _ _ _ _ _ _ (by norm_num)
    (cubeContentXor_mem 0) (cubeContentXor_mem 1) _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Learning opposite content interactions genuinely changes the actual sparsemax support.
Source: tied coefficient readouts and original variational Eq. (1), with both endpoints feasible. -/
theorem cubeContentXor_support_changes :
    cubeContentAttention (cubeContentLearnedRoutes (fun _ : Fin 2 => (1 / 16 : ℝ))
      (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) (0 : Fin 1) !![0; 1])
        (fun _ => false) (cubeContentFlip 0 (fun _ => false)) = 1 / 8 ∧
    cubeContentAttention (cubeContentLearnedRoutes (fun _ : Fin 2 => (1 / 16 : ℝ))
      (fun _ => (1 / 16 : ℝ)) (fun _ => (1 : Fin 2)) (0 : Fin 1) !![0; -1])
        (fun _ => false) (cubeContentFlip 0 (fun _ => false)) = 0 := by
  constructor
  · have h := (cubeContentXor_mem 1).2
    norm_num at h
    rw [cubeContentAttention_eq (3 / 4) _ (by norm_num) h, cubeContentRouting_flip]
    norm_num [cubeContentLearnedRoutes]
  · have h := (cubeContentXor_mem 0).2
    norm_num at h
    rw [cubeContentAttention_eq (3 / 4) _ (by norm_num) h, cubeContentRouting_flip]
    norm_num [cubeContentLearnedRoutes]

end Transformer.GPTMini.Sparsemax
