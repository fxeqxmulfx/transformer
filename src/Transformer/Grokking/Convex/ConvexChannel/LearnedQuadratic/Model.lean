module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Basic

/-!
# Signed learned quadratic channels and ordinary answer cross-entropy

Source: `review/convex/moments.md`, CM85. Every class and both readout signs
retain a complete complex PSD matrix. Actual inference recovers the full
factor, learns its balanced representation and readout, and squares the
magnitudes of its complex responses. The whole-domain loss is jointly convex.
The fixed phase and the two readout signs are architecture, not learned
independent factors. There are no internal-state targets or label constraints.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open Landscape.Neural (crossEntropy)

variable {D C : Type*}

/-- Both complete complex matrices of every original output object, CM85. -/
abbrev Parameters (D C : Type*) := C → Bool → Matrix D D ℂ

/-- Joint, label-independent PSD domain; no rank or activation constraint, CM85. -/
def domain : Set (Parameters D C) := {p | ∀ c s, (p c s).PosSemidef}

/-- All learned coordinates vary in the same convex domain, CM85. -/
theorem domain_convex : Convex ℝ (domain : Set (Parameters D C)) := by
  intro p hp q hq a b ha hb hab c s
  exact ((hp c s).smul ha).add ((hq c s).smul hb)

/-- Zero retains every class, sign and coordinate and is feasible, CM85. -/
theorem zero_mem : (0 : Parameters D C) ∈ domain := by
  intro c s
  exact Matrix.PosSemidef.zero

variable [Fintype D] [DecidableEq D]

/-- The full learned complex product in one output branch, CM85.
The readout magnitude changes with the recovered factor; its phase is fixed. -/
noncomputable def branch (M : Matrix D D ℂ) (x : D → ℂ) : ℂ :=
  (LearnedHinge.amplitude (factor M) ^ 2 : ℝ) * (1 + Complex.I) *
    (energy (LearnedHinge.representation (factor M)) x : ℝ)

/-- Expanding both learned stages gives the same complex quadratic output,
CM85. Imaginary representation coordinates participate in every norm square. -/
theorem branch_eq {M : Matrix D D ℂ} (hM : M.PosSemidef) (x : D → ℂ) :
    branch M x = (1 + Complex.I) * (quadratic M x : ℝ) := by
  rw [← network_factor hM x]
  simp only [branch, network, Complex.ofReal_mul]
  ring

/-- Prediction reads the real part after both complete complex products,
CM85. The competing output is an original object. -/
noncomputable def logits (p : Parameters D C) (x : D → ℂ) : C → ℝ :=
  fun c => (branch (p c true) x - branch (p c false) x).re

/-- Actual logits equal signed joint readings on the whole domain, CM85. -/
theorem logits_eq {p : Parameters D C} (hp : p ∈ domain) (x : D → ℂ) (c : C) :
    logits p x c = quadratic (p c true) x - quadratic (p c false) x := by
  simp [logits, branch_eq (hp c true), branch_eq (hp c false)]

/-- Both recovered layers are recomputed from the parameter combination;
their actual logits preserve that combination, CM85. -/
theorem logits_combination {p q : Parameters D C} (hp : p ∈ domain) (hq : q ∈ domain)
    (x : D → ℂ) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    logits (a • p + b • q) x = a • logits p x + b • logits q x := by
  have hm := domain_convex hp hq ha hb hab
  funext c
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, logits_eq hm,
    logits_eq hp, logits_eq hq, quadratic_combination]
  ring

variable [Fintype C]

/-- Only ordinary original-object CE receives a target, CM85. -/
noncomputable def loss (x : D → ℂ) (target : C) (p : Parameters D C) : ℝ :=
  crossEntropy (logits p x) target

/-- Ordinary answer CE is convex in every joint matrix coordinate, CM85. -/
theorem loss_convex (x : D → ℂ) (target : C) :
    ConvexOn ℝ (domain : Set (Parameters D C)) (loss x target) := by
  refine ⟨domain_convex, ?_⟩
  intro p hp q hq a b ha hb hab
  simp only [loss, logits_combination hp hq x a b ha hb hab]
  exact (ConvexRecall.convex_crossEntropy target).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- The actual original-object probabilities are strictly positive, CM85. -/
theorem probability_pos (p : Parameters D C) (x : D → ℂ) (c : C) :
    0 < Geometry.probability (logits p x) c := Geometry.probability_pos _ _

/-- Every original output object remains in the normalized softmax, CM85. -/
theorem probability_sum (p : Parameters D C) (x : D → ℂ) (witness : C) :
    ∑ c, Geometry.probability (logits p x) c = 1 := Geometry.sum_probability _ witness

/-- An indexed mean minibatch, including repetitions and a short tail, CM85.
The zero-record sum has value zero and is not a prescribed training update. -/
noncomputable def batchLoss {N : ℕ} (data : Fin N → D → ℂ) (target : Fin N → C)
    (p : Parameters D C) : ℝ := (N : ℝ)⁻¹ * ∑ i, loss (data i) (target i) p

/-- The complete indexed minibatch objective is jointly convex, CM85.
This does not assert convergence of a native gradient or forward-only loop. -/
theorem batchLoss_convex {N : ℕ} (data : Fin N → D → ℂ) (target : Fin N → C) :
    ConvexOn ℝ (domain : Set (Parameters D C)) (batchLoss data target) := by
  refine ⟨domain_convex, ?_⟩
  intro p hp q hq a b ha hb hab
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) =>
    (loss_convex (data i) (target i)).2 hp hq ha hb hab)
  have hs := mul_le_mul_of_nonneg_left h (inv_nonneg.mpr (Nat.cast_nonneg N : (0 : ℝ) ≤ N))
  simpa only [batchLoss, smul_eq_mul, Finset.sum_add_distrib, ← Finset.mul_sum,
    mul_add, mul_left_comm] using hs

example := branch_eq (D := Bool) (M := 0) Matrix.PosSemidef.zero (fun _ => Complex.I)
example := logits_eq (D := Bool) (C := Bool) zero_mem (fun _ => Complex.I) true
example := logits_combination (D := Bool) (C := Bool) zero_mem zero_mem
  (fun _ => Complex.I) (1 / 2) (1 / 2) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic
