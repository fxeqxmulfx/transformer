module

public import Transformer.Rotate90.Training.Sampling.Optimizer

/-!
# Actual sampled Rotate90 updates break exact aliases

Source: `review/flat_spectrum.md`, *Sampled updates can break aliases*, equations
(S3), (S6) and the nonzero witness. The gradient is that of the real normalized
tied model, and the step is the existing source Schedule-Free recurrence. A
single valid equation a=b=1 works for every natural exponent. Arbitrary finite
stacks are covered when their common hidden coordinate is positive; replicated
zero branches and nonzero embeddings explicitly satisfy this hypothesis.
The full-table comparison is specifically K=1, where division permutes all
residues and equation (S4) cancels the embedding gradient exactly.
-/

@[expose] public section

namespace Transformer.Rotate90.Training.Sampling

open Landscape.Neural ScheduleFree
open scoped BigOperators Classical

variable {V K H : Type*} [Fintype V] [Fintype K] [Fintype H]

variable {p : ℕ} [Fact p.Prime]

/-- The sampled objective on the complete indexed domain equals the existing
full-table tied objective, including zero numerators and all unit denominators. -/
theorem batchLoss_full_eq (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ) :
    batchLoss eps blocks (fun x : ZMod p × (ZMod p)ˣ => x.1)
      (fun x => x.2) (fun x => (x.1 / (x.2 : ZMod p)) ^ k) = sharedLoss eps blocks k := by
  funext E
  unfold batchLoss sharedLoss fullLoss
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul]
  congr 1

/-- For each denominator, division permutes all residues. Thus the complete
division table has frequency 1/p for every answer, rather than just nonzero ones. -/
theorem frequency_full_division (q : ZMod p) :
    frequency (fun x : ZMod p × (ZMod p)ˣ => x.1 / (x.2 : ZMod p)) q = (p : ℝ)⁻¹ := by
  classical
  have hc (b : (ZMod p)ˣ) :
      (∑ a : ZMod p, if a / (b : ZMod p) = q then (1 : ℝ) else 0) = 1 := by
    calc
      _ = ∑ a : ZMod p, if a = q then (1 : ℝ) else 0 := by
        simpa only [Units.mulRight_apply, Units.val_inv_eq_inv_val, div_eq_mul_inv] using
          Equiv.sum_comp (Units.mulRight b⁻¹) (fun a => if a = q then (1 : ℝ) else 0)
      _ = 1 := by simp
  unfold frequency
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  change (∑ b : (ZMod p)ˣ, ∑ a : ZMod p,
    if a / (b : ZMod p) = q then (1 : ℝ) else 0) /
      Fintype.card (ZMod p × (ZMod p)ˣ) = (p : ℝ)⁻¹
  simp_rw [hc]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one,
    Fintype.card_prod, Nat.cast_mul, ZMod.card]
  have hp : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  have hu : (Fintype.card (ZMod p)ˣ : ℝ) ≠ 0 := by
    exact_mod_cast ne_of_gt (Fintype.card_pos (α := (ZMod p)ˣ))
  field_simp

/-- **Full division cancels the actual embedding gradient at every collapsed
table.** Arbitrary finite branch weights and all normalizers remain present. -/
theorem gradient_sharedLoss_division_constant {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ) :
    gradient (sharedLoss (p := p) eps blocks 1) (constantTable v) = 0 := by
  rw [← batchLoss_full_eq]
  simp only [pow_one]
  apply gradient_batch_zero_of_uniform heps
  intro q
  simpa only [ZMod.card] using frequency_full_division q

omit [Fintype K] in
/-- Constant embeddings are invariant under every multiplicative subgroup. -/
theorem constantTable_aliased (S : Subgroup (ZMod p)ˣ) (v : Channels K → ℝ) :
    Aliased S (constantTable v) := by
  intro h hh q i
  rfl

/-- **The full-table invariant cannot be extended to arbitrary sampled steps.**
The legal pair (1,1) has target 1 for every K, including zero. A subgroup element
different from 1 supplies a competing row, and one source step breaks its alias. -/
theorem rowStep_not_aliased (params : AdamW.Parameters)
    (hrate : 0 < AdamW.rate params 0) (hbeta : params.adam.betaTwo < 1)
    (hopt : 0 < params.adam.epsilon) {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ) (k : ℕ)
    (S : Subgroup (ZMod p)ˣ) (r : (ZMod p)ˣ) (hr : r ∈ S) (hne : (r : ZMod p) ≠ 1)
    (i : Channels K) (hh : 0 < hidden eps blocks v v i) :
    ¬ Aliased S (WithLp.toLp 2
      (rowStep params eps blocks 1 1 ((1 / 1 : ZMod p) ^ k) (constantTable v)).point) := by
  simp only [div_one, one_pow]
  intro hE
  have hsep := rowStep_separates params hrate hbeta hopt heps blocks v
    (1 : ZMod p) 1 1 r hne.symm i hh
  have heq := hE r hr 1 i
  simp only [entry, mul_one] at heq
  exact (ne_of_lt hsep) heq

/-- **One explicit model exhibits both behaviors.** At width 16, hidden size 43
and every positive finite depth, a nonzero embedding has zero full-table division
gradient and exact aliases. A valid singleton update breaks the aliases in both
training and evaluation buffers, with ordinary positive-decay Schedule-Free
parameters. This is an escape witness, not full-task learning in one step. -/
theorem full_zero_sample_escapes {eps : ℝ} (heps : 0 < eps) (d : ℕ) :
    let blocks := List.replicate (d + 1) (Power.zeroBlock (K := Fin 8) (H := Fin 43))
    let E : Table (ZMod 3) (Channels (Fin 8)) := constantTable fun _ => 1
    gradient (sharedLoss eps blocks 1) E = 0 ∧ Aliased ⊤ E ∧
      ¬ Aliased ⊤ (WithLp.toLp 2 (rowStep witnessParameters eps blocks 1 1 1 E).point) ∧
      ¬ Aliased ⊤ (WithLp.toLp 2
        (AdamW.evaluation witnessParameters (rowStep witnessParameters eps blocks 1 1 1 E))) := by
  let blocks := List.replicate (d + 1) (Power.zeroBlock (K := Fin 8) (H := Fin 43))
  let E : Table (ZMod 3) (Channels (Fin 8)) := constantTable fun _ => 1
  have hr : 0 < AdamW.rate witnessParameters 0 := by
    norm_num [witnessParameters, AdamW.rate, warmupFactor]
  have hb : witnessParameters.adam.betaTwo < 1 := by norm_num [witnessParameters]
  have ho : 0 < witnessParameters.adam.epsilon := by norm_num [witnessParameters]
  have he : ¬ Aliased ⊤
      (WithLp.toLp 2 (rowStep witnessParameters eps blocks 1 1 1 E).point) := by
    simpa only [div_one, one_pow] using rowStep_not_aliased witnessParameters hr hb ho
      heps blocks (fun _ => 1) 1 ⊤ (Units.mk0 (2 : ZMod 3) (by decide)) (by simp)
      (by decide) (0, 1) (hidden_ones_pos heps (d + 1) 0)
  have hx : AdamW.evaluation witnessParameters (rowStep witnessParameters eps blocks 1 1 1 E) =
      (rowStep witnessParameters eps blocks 1 1 1 E).point :=
    initial_step_evaluation witnessParameters hr hb _ _
  refine ⟨gradient_sharedLoss_division_constant heps blocks _, constantTable_aliased ⊤ _, he, ?_⟩
  change ¬ Aliased ⊤ (WithLp.toLp 2
    (AdamW.evaluation witnessParameters (rowStep witnessParameters eps blocks 1 1 1 E)))
  rw [hx]
  exact he

example := batchLoss_full_eq (p := 3) (K := Fin 8) (H := Fin 43)
  (1 / 8388608) [Power.zeroBlock] 2
example := frequency_full_division (p := 3) 0
example := frequency_full_division (p := 3) 1
example := gradient_sharedLoss_division_constant (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)

example := constantTable_aliased (p := 3) (K := Fin 8) ⊤ (fun _ => 1)
example := rowStep_not_aliased (p := 3) (K := Fin 8) (H := Fin 43) witnessParameters
  (by norm_num [witnessParameters, AdamW.rate, warmupFactor])
  (by norm_num [witnessParameters]) (by norm_num [witnessParameters])
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1) 2
  ⊤ (Units.mk0 (2 : ZMod 3) (by decide)) (by simp) (by decide)
  (0, 1) (hidden_ones_pos (by norm_num) 1 0)
example := full_zero_sample_escapes (eps := 1 / 8388608) (by norm_num) 0
example := full_zero_sample_escapes (eps := 1 / 8388608) (by norm_num) 3

end Transformer.Rotate90.Training.Sampling
