module

public import Transformer.Landscape.Neural.Loss
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Shared embedding, pairwise rotation and dense maps

Source: `grokking_abmod/web/own-channel/README.md`, Architecture and
initialization, revision f49969393a7482490d45483d8a4fd3046b1adede; the
matching equations are in `landscape_grokking_plan.md`, Model and protocol.

Channels are indexed by a pair number and a coordinate in `Fin 2`, so the
quarter-turn is exactly the lab's adjacent-coordinate operation. The same
embedding argument is used for input lookup and for the final readout.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Landscape.Neural

/-- An even channel width, indexed by adjacent coordinate pairs. -/
abbrev Channels (K : Type*) := K × Fin 2

variable {P V K C H I : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]
  [Fintype C] [Fintype I]

/-- Row-vector multiplication by a dense trainable matrix, as in `z = r A`. -/
noncomputable def dense (A : C → H → ℝ) (v : C → ℝ) (j : H) : ℝ :=
  ∑ i, v i * A i j

/-- The browser's fixed `J(u,v) = (-v,u)` on every adjacent pair. -/
def rotate90 (v : Channels K → ℝ) (i : Channels K) : ℝ :=
  if i.2 = 0 then -v (i.1, 1) else v (i.1, 0)

/-- Both operands use the same embedding, with a fixed rotation on the
second operand (Model and protocol, `x0`). -/
def tiedInput (E : V → Channels K → ℝ) (a b : V) (i : Channels K) : ℝ :=
  E a i + rotate90 (E b) i

/-- The transposed readout of that same embedding (Model and protocol,
`logit[q]`). It is not a separately trainable matrix. -/
noncomputable def tiedReadout (E : V → C → ℝ) (v : C → ℝ) (q : V) : ℝ :=
  ∑ i, v i * E q i

/-- Two quarter-turns negate every coordinate, verifying the sign and
pair indexing in the lab's definition of `J`. -/
theorem rotate90_rotate90 (v : Channels K → ℝ) (i : Channels K) :
    rotate90 (rotate90 v) i = -v i := by
  obtain ⟨k, c⟩ := i
  fin_cases c
  · simp [rotate90]
  · simp [rotate90]

/-- The fixed J commutes with addition (Model and protocol, pairwise rotation). -/
theorem rotate90_add (v w : Channels K → ℝ) (i : Channels K) :
    rotate90 (fun j => v j + w j) i = rotate90 v i + rotate90 w i := by
  by_cases h : i.2 = 0
  · simp [rotate90, h, add_comm]
  · simp [rotate90, h]

/-- The fixed J commutes with real scaling, so it introduces no trainable
nonlinearity into either operand path (Model and protocol). -/
theorem rotate90_mul (t : ℝ) (v : Channels K → ℝ) (i : Channels K) :
    rotate90 (fun j => t * v j) i = t * rotate90 v i := by
  by_cases h : i.2 = 0
  · simp [rotate90, h]
  · simp [rotate90, h]

/-- A coordinate of a Euclidean parameter vector is analytic. This gives
actual trainable weights, rather than an assumed analytic model. -/
theorem analyticAt_parameter (i : I) (x : EuclideanSpace ℝ I) :
    AnalyticAt ℝ (fun y : EuclideanSpace ℝ I => y.ofLp i) x := by
  exact (PiLp.proj (𝕜 := ℝ) 2 (fun _ : I => ℝ) i).analyticAt x

/-- Dense multiplication is analytic jointly in the varying input and all
matrix entries (Model and protocol, `z = r A`). -/
theorem analyticAt_dense {A : P → C → H → ℝ} {v : P → C → ℝ} {x : P}
    (hA : ∀ i j, AnalyticAt ℝ (fun y => A y i j) x)
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (j : H) :
    AnalyticAt ℝ (fun y => dense (A y) (v y) j) x := by
  exact Finset.analyticAt_fun_sum _ fun i _ => (hv i).fun_mul (hA i j)

/-- A fixed quarter-turn preserves coordinatewise analyticity; it does not
apply a parameter-dependent rotation (Model and protocol, `J`). -/
theorem analyticAt_rotate90 {v : P → Channels K → ℝ} {x : P}
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (i : Channels K) :
    AnalyticAt ℝ (fun y => rotate90 (v y) i) x := by
  by_cases h : i.2 = 0
  · simpa only [rotate90, h, ↓reduceIte] using (hv (i.1, 1)).fun_neg
  · simpa only [rotate90, h, ↓reduceIte] using hv (i.1, 0)

/-- Both input occurrences of E may vary with the same parameters
(Model and protocol, `x0 = E[a] + J(E[b])`). -/
theorem analyticAt_tiedInput {E : P → V → Channels K → ℝ} {x : P}
    (hE : ∀ q i, AnalyticAt ℝ (fun y => E y q i) x) (a b : V) (i : Channels K) :
    AnalyticAt ℝ (fun y => tiedInput (E y) a b i) x := by
  exact (hE a i).fun_add (analyticAt_rotate90 (hE b) i)

/-- The readout remains analytic when both activations and the reused E
depend on the same parameters (Model and protocol, final logits). -/
theorem analyticAt_tiedReadout {E : P → V → C → ℝ} {v : P → C → ℝ} {x : P}
    (hE : ∀ q i, AnalyticAt ℝ (fun y => E y q i) x)
    (hv : ∀ i, AnalyticAt ℝ (fun y => v y i) x) (q : V) :
    AnalyticAt ℝ (fun y => tiedReadout (E y) (v y) q) x := by
  exact Finset.analyticAt_fun_sum _ fun i _ => (hv i).fun_mul (hE q i)

/- Nonconstant witnesses for the joint analyticity hypotheses. The matrices,
embeddings and hidden states vary, and all finite index types are nonempty. -/

example : (∀ i j : Fin 2,
      AnalyticAt ℝ (fun t : ℝ => t + i.val + j.val) 0) ∧
    (∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (i.val + 1)) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => dense
      (fun i j : Fin 2 => t + i.val + j.val)
      (fun i : Fin 2 => t * (i.val + 1)) 0) 0 := by
  have hA : ∀ i j : Fin 2, AnalyticAt ℝ (fun t : ℝ => t + i.val + j.val) 0 := by
    intro i j
    fun_prop
  have hv : ∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (i.val + 1)) 0 := by
    intro i
    fun_prop
  exact ⟨hA, hv, analyticAt_dense hA hv 0⟩

example : (∀ i : Channels (Fin 1), AnalyticAt ℝ (fun t : ℝ => t + i.2.val) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => rotate90 (fun i : Channels (Fin 1) => t + i.2.val) (0, 0)) 0 := by
  have h : ∀ i : Channels (Fin 1), AnalyticAt ℝ (fun t : ℝ => t + i.2.val) 0 := by
    intro i
    fun_prop
  exact ⟨h, analyticAt_rotate90 h (0, 0)⟩

example : (∀ q : Fin 3, ∀ i : Channels (Fin 1),
      AnalyticAt ℝ (fun t : ℝ => t + q.val + i.2.val) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => tiedInput
      (fun (q : Fin 3) (i : Channels (Fin 1)) => t + q.val + i.2.val) 0 1 (0, 0)) 0 := by
  have h : ∀ q : Fin 3, ∀ i : Channels (Fin 1),
      AnalyticAt ℝ (fun t : ℝ => t + q.val + i.2.val) 0 := by
    intro q i
    fun_prop
  exact ⟨h, analyticAt_tiedInput h 0 1 (0, 0)⟩

example : (∀ q i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t + q.val + i.val) 0) ∧
    (∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (i.val + 1)) 0) ∧
    AnalyticAt ℝ (fun t : ℝ => tiedReadout
      (fun q i : Fin 2 => t + q.val + i.val)
      (fun i : Fin 2 => t * (i.val + 1)) 0) 0 := by
  have hE : ∀ q i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t + q.val + i.val) 0 := by
    intro q i
    fun_prop
  have hv : ∀ i : Fin 2, AnalyticAt ℝ (fun t : ℝ => t * (i.val + 1)) 0 := by
    intro i
    fun_prop
  exact ⟨hE, hv, analyticAt_tiedReadout hE hv 0⟩

end Transformer.Landscape.Neural
