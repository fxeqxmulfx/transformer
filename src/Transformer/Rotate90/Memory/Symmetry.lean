module

public import Transformer.Rotate90.Learning.Basic

/-!
# Parameter negation and nonconvex prediction averaging

Source: `review/flat_spectrum.md`, *Optimizer memory*, M3--M4. Negating all
parameters preserves the actual finite-stack logits: input and hidden states
change sign, the gate/value products do not, and the tied readout changes sign
again. This is not oddness of SiLU. The zero parameter midpoint has uniform
logits. Parameter averaging therefore cannot inherit a convex prediction or
loss guarantee merely because its endpoints predict correctly.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Memory

open Landscape.Neural Learning
open scoped BigOperators

variable {V K H D : Type*} [Fintype K] [Fintype H]

/-- Positive-epsilon RMSNorm is odd, including the zero input. The identity
also holds under total real division without a positivity hypothesis. -/
theorem rms_neg {C : Type*} [Fintype C] (eps : ℝ) (v : C → ℝ) :
    rmsNorm eps (fun i => -v i) = fun i => -rmsNorm eps v i := by
  funext i
  simp [rmsNorm, meanSquare, neg_div]

/-- Simultaneously negating input and matrix leaves their dense product fixed. -/
theorem dense_neg_both {C J : Type*} [Fintype C]
    (A : C → J → ℝ) (v : C → ℝ) :
    dense (fun i j => -A i j) (fun i => -v i) = dense A v := by
  funext j
  simp [dense]

/-- Every layer retains both local normalizers. The SiLU arguments are equal,
not sign-reversed, and the output matrix supplies the branch's minus sign. -/
theorem block_neg (eps : ℝ) (w : Weights V K H D) (d : D)
    (v : Channels K → ℝ) :
    Power.block eps (matrices (-w) d) (fun i => -v i) =
      fun i => -Power.block eps (matrices w d) v i := by
  have hmat : matrices (-w) d =
      (fun i j => -(matrices w d).1 i j,
       fun i j => -(matrices w d).2.1 i j,
       fun i j => -(matrices w d).2.2 i j) := by rfl
  rw [hmat]
  unfold Power.block
  rw [rms_neg]
  simp only [dense_neg_both]
  have hout (u : H → ℝ) : dense (fun i j => -(matrices w d).2.2 i j) u =
      fun j => -dense (matrices w d).2.2 u j := by
    funext j
    simp [dense, Finset.sum_neg_distrib]
  rw [hout, rms_neg]
  funext i
  ring

/-- Sign propagation through an arbitrary finite ordered residual loop. -/
theorem stack_neg (eps : ℝ) (order : List D) (w : Weights V K H D)
    (v : Channels K → ℝ) :
    (order.map (matrices (-w))).foldl (fun z B => Power.block eps B z) (fun i => -v i) =
      fun i => -((order.map (matrices w)).foldl (fun z B => Power.block eps B z) v i) := by
  induction order generalizing v with
  | nil => rfl
  | cons d order ih =>
    simp only [List.map_cons, List.foldl_cons]
    rw [block_neg, ih]

/-- M3: all actual finite-stack logits are even in the complete parameter
vector. This includes every tied occurrence and every layer's matrices. -/
theorem score_neg (eps : ℝ) (order : List D) (w : Weights V K H D) (a b q : V) :
    score eps order (-w) a b q = score eps order w a b q := by
  have hinput : tiedInput (Learning.embedding (-w)) a b =
      fun i => -tiedInput (Learning.embedding w) a b i := by
    funext i
    simp [tiedInput, rotate90, Learning.embedding]
    split_ifs <;> ring
  unfold score Power.stackLogit Power.stackState
  rw [hinput, rms_neg, stack_neg, rms_neg]
  simp [tiedReadout, Learning.embedding]

/-- The corresponding full-softmax losses agree on every fixed split. -/
theorem loss_neg [Fintype V] (eps : ℝ) (order : List D) (w : Weights V K H D)
    (rows : Finset (V × V)) (target : V × V → V) :
    loss eps order rows target (-w) = loss eps order rows target w := by
  unfold loss meanCrossEntropy crossEntropy
  simp only [score_neg]

/-- M4: the midpoint of antipodal weights has identically zero output scores. -/
theorem score_zero (eps : ℝ) (order : List D) (a b q : V) :
    score (K := K) (H := H) eps order 0 a b q = 0 := by
  simp [score, Power.stackLogit, tiedReadout, Learning.embedding]

/-- The full-softmax loss at the midpoint is log vocabulary size on every row. -/
theorem row_loss_zero [Fintype V] (eps : ℝ) (order : List D) (a b target : V) :
    crossEntropy (fun q => score (K := K) (H := H) eps order 0 a b q) target =
      Real.log (Fintype.card V) := by
  simp [crossEntropy, score_zero]

example := score_neg (V := Fin 3) (K := Fin 8) (H := Fin 43)
  (1/8388608) (List.finRange 3) (WithLp.toLp 2 (fun _ => 1)) 0 1 2
example := loss_neg (K := Fin 8) (H := Fin 43) (1/8388608) (List.finRange 3)
  (WithLp.toLp 2 (fun _ => 1)) (Finset.univ : Finset (ZMod 3 × ZMod 3))
  (fun r => (r.1 / r.2) ^ 2)

end Transformer.Rotate90.Memory
