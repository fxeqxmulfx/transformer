module

public import Transformer.Rotate90.Training.Defs

/-!
# Differentiability of the actual finite-depth training objective

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, derivative
justification. All hidden states here use the current Peri-RMSNorm loop. The
branch matrices are fixed only while taking the embedding partial derivative;
the result holds for every matrix choice and every finite number of blocks.
The auxiliary untied loss and its tied restriction are analytic everywhere
for positive epsilon. No differentiability hypothesis is left unproved when
the symmetry theorem is applied to this network.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural
open scoped BigOperators

variable {P V K H : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]
  [Fintype V] [Fintype K] [Fintype H]

/-- Both local normalizers and the true SiLU gate preserve analyticity in the
incoming state. This is one iteration of the source loop, without simplification. -/
theorem analyticAt_block {eps : ℝ} (heps : 0 < eps)
    (B : Power.BlockWeights K H) {v : P → Channels K → ℝ} {x : P}
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (i : Channels K) :
    AnalyticAt ℝ (fun y => Power.block eps B (v y) i) x := by
  have hr := fun j => analyticAt_rmsNorm heps hv j
  have hg := fun j => analyticAt_dense (A := fun _ : P => B.1)
    (fun _ _ => analyticAt_const) hr j
  have hu := fun j => analyticAt_dense (A := fun _ : P => B.2.1)
    (fun _ _ => analyticAt_const) hr j
  have hout : ∀ j, AnalyticAt ℝ
      (fun y => dense B.2.2 (fun h => swiglu (dense B.1 (rmsNorm eps (v y)) h)
        (dense B.2.1 (rmsNorm eps (v y)) h)) j) x := by
    intro j
    exact analyticAt_dense (A := fun _ : P => B.2.2) (fun _ _ => analyticAt_const)
      (fun h => analyticAt_swiglu_comp (hg h) (hu h)) j
  exact (hv i).fun_add (analyticAt_rmsNorm heps hout i)

/-- Induction over the actual residual loop, for an arbitrary finite depth. -/
theorem analyticAt_blocks {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) {v : P → Channels K → ℝ} {x : P}
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (i : Channels K) :
    AnalyticAt ℝ
      (fun y => blocks.foldl (fun z B => Power.block eps B z) (v y) i) x := by
  induction blocks generalizing v with
  | nil => exact hv i
  | cons B blocks ih =>
    exact ih (fun j => analyticAt_block heps B hv j)

/-- The two outer normalizers are included as well as every block normalizer. -/
theorem analyticAt_hidden {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) {a b : P → Channels K → ℝ} {x : P}
    (ha : ∀ i, AnalyticAt ℝ (fun y => a y i) x)
    (hb : ∀ i, AnalyticAt ℝ (fun y => b y i) x) (i : Channels K) :
    AnalyticAt ℝ (fun y => hidden eps blocks (a y) (b y) i) x := by
  have hraw := fun j => (ha j).fun_add (analyticAt_rotate90 hb j)
  exact analyticAt_rmsNorm heps
    (fun j => analyticAt_blocks heps blocks (fun c => analyticAt_rmsNorm heps hraw c) j) i

/-- Every untied logit is analytic jointly in all three table occurrences. -/
theorem analyticAt_untiedLogit {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (x : Tables V (Channels K)) (a b q : V) :
    AnalyticAt ℝ (fun y => untiedLogit eps blocks y a b q) x := by
  have hA : ∀ r i, AnalyticAt ℝ (fun y : Tables V (Channels K) => entry y.1 r i) x := by
    intro r i
    exact ((PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × Channels K => ℝ) (r, i)).comp
      (ContinuousLinearMap.fst ℝ _ _)).analyticAt x
  have hB : ∀ r i, AnalyticAt ℝ (fun y : Tables V (Channels K) => entry y.2.1 r i) x := by
    intro r i
    exact ((PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × Channels K => ℝ) (r, i)).comp
      ((ContinuousLinearMap.fst ℝ _ _).comp (ContinuousLinearMap.snd ℝ _ _))).analyticAt x
  have hC : ∀ r i, AnalyticAt ℝ (fun y : Tables V (Channels K) => entry y.2.2 r i) x := by
    intro r i
    exact ((PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × Channels K => ℝ) (r, i)).comp
      ((ContinuousLinearMap.snd ℝ _ _).comp (ContinuousLinearMap.snd ℝ _ _))).analyticAt x
  exact analyticAt_tiedReadout hC (fun i => analyticAt_hidden heps blocks (hA a) (hB b) i) q

variable {p : ℕ} [Fact p.Prime]

/-- The full softmax and all valid operand pairs preserve analyticity. -/
theorem analyticAt_fullLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (x : Tables (ZMod p) (Channels K)) :
    AnalyticAt ℝ (fullLoss eps blocks k) x := by
  unfold fullLoss
  apply AnalyticAt.div_const
  apply Finset.analyticAt_fun_sum
  intro a ha
  apply Finset.analyticAt_fun_sum
  intro b hb
  exact analyticAt_crossEntropy (fun q => analyticAt_untiedLogit heps blocks x a b q)
    ((a / (b : ZMod p)) ^ k)

/-- Reusing the same parameters in all three roles preserves analyticity. -/
theorem analyticAt_sharedLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (E : Table (ZMod p) (Channels K)) :
    AnalyticAt ℝ (sharedLoss eps blocks k) E := by
  exact (analyticAt_fullLoss heps blocks k (diagonal E)).comp (diagonal.analyticAt E)

/-- The derivative used in the alias theorem exists at every parameter value,
including repeated embedding rows and zero hidden activations. -/
theorem differentiableAt_fullLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (x : Tables (ZMod p) (Channels K)) :
    DifferentiableAt ℝ (fullLoss eps blocks k) x :=
  (analyticAt_fullLoss heps blocks k x).differentiableAt

/-- The actual embedding gradient therefore differentiates the full tied loss. -/
theorem differentiableAt_sharedLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ)
    (E : Table (ZMod p) (Channels K)) :
    DifferentiableAt ℝ (sharedLoss eps blocks k) E :=
  (analyticAt_sharedLoss heps blocks k E).differentiableAt

/-- The auxiliary loss has no nonsmooth exceptional parameter set. -/
theorem analyticOnNhd_fullLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ) :
    AnalyticOnNhd ℝ (fullLoss (p := p) eps blocks k) Set.univ := by
  intro x hx
  exact analyticAt_fullLoss heps blocks k x

/-- The same global statement holds after tying the embedding occurrences. -/
theorem analyticOnNhd_sharedLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (k : ℕ) :
    AnalyticOnNhd ℝ (sharedLoss (p := p) eps blocks k) Set.univ := by
  intro x hx
  exact analyticAt_sharedLoss heps blocks k x

example := analyticAt_block (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) Power.zeroBlock
  (v := fun t : ℝ => fun i => t + i.2.val) (x := 0) (by intro i; fun_prop) (0, 1)
example := analyticAt_blocks (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock, Power.zeroBlock]
  (v := fun t : ℝ => fun i => t + i.2.val) (x := 0) (by intro i; fun_prop) (0, 1)
example := analyticAt_hidden (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock]
  (a := fun t : ℝ => fun i => t + i.2.val) (b := fun t : ℝ => fun _ => t ^ 2)
  (x := 0) (by intro i; fun_prop) (by intro i; fun_prop) (0, 1)
example := analyticAt_untiedLogit (V := Fin 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (0, 0, 0) 0 1 2
example := analyticAt_fullLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2 (0, 0, 0)
example := analyticAt_sharedLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2 0
example := differentiableAt_fullLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2 (0, 0, 0)
example := differentiableAt_sharedLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2 0
example := analyticOnNhd_fullLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2
example := analyticOnNhd_sharedLoss (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 2

end Transformer.Rotate90.Training
