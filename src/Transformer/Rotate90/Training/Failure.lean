module

public import Transformer.Rotate90.Training.Optimizer
public import Transformer.Rotate90.Training.Readout

/-!
# A genuine training obstruction despite sufficient actual-model capacity

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, failure and
limit theorems. Exact alias symmetry prevents a unique correct answer, even
though `Power.exists_weights_width16_any_depth` supplies successful weights.
This disproves unconditional convergence from every initialization. It does
not assign a probability to such initializations or prove a stable basin.

The result covers the source Schedule-Free recurrence and arbitrary external
parameter schedules satisfying that same update. It is independent of the
branch trajectory: simultaneous learning of every branch is allowed. Infinite
training time does not remove an exact row equality if a pointwise limit exists.
This is not a theorem about partial tables, stochastic batches, or roundoff.
In particular, the measured half-table failures are not asserted to remain in
this exact subspace: their nonsymmetric gradients are reported separately.
The result supplies a mechanism and rules out a universal convergence claim;
it does not turn an empirical plateau into a proof of permanent failure.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural ScheduleFree
open scoped Topology

variable {p : ℕ} [Fact p.Prime] {K H C : Type*} [Fintype K] [Fintype H] [Fintype C]

omit [Fintype C] in
/-- Pointwise limits cannot split rows that agree throughout a trajectory.
No limit of the hidden matrices is required for this embedding statement. -/
theorem aliased_limit {S : Subgroup (ZMod p)ˣ}
    {E : ℕ → Table (ZMod p) C} {limit : Table (ZMod p) C}
    (hE : ∀ t, Aliased S (E t))
    (hlim : ∀ q i, Filter.Tendsto (fun t => entry (E t) q i)
      Filter.atTop (𝓝 (entry limit q i))) : Aliased S limit := by
  intro h hh q i
  exact tendsto_nhds_unique_of_forall (hlim ((h : ZMod p) * q) i) (hlim q i)
    (fun t => hE t h hh q i)

/-- A convergent sequence of exactly aliased embeddings still cannot give a
strictly correct complete table at its limit, for any final branch matrices. -/
theorem limit_not_solves {S : Subgroup (ZMod p)ˣ}
    {E : ℕ → Table (ZMod p) (Channels K)} {limit : Table (ZMod p) (Channels K)}
    (hE : ∀ t, Aliased S (E t))
    (hlim : ∀ q i, Filter.Tendsto (fun t => entry (E t) q i)
      Filter.atTop (𝓝 (entry limit q i)))
    {h : (ZMod p)ˣ} (hh : h ∈ S) (hne : h ≠ 1)
    (eps : ℝ) (blocks : List (Power.BlockWeights K H)) (k : ℕ) :
    ¬ (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
      Power.stackLogit eps (entry limit) blocks a b q <
        Power.stackLogit eps (entry limit) blocks a b ((a / b) ^ k)) :=
  not_solves_of_aliased (aliased_limit hE hlim) hh hne eps blocks k

/-- **Actual-model training can fail forever despite representability.** At
any finite update count, a nontrivial exact alias prevents solving the table.
The embedding is the optimizer's reported average, not just its training query.
Evaluation branch matrices are arbitrary and may be the optimizer's averages. -/
theorem trainingRun_not_solves {eps : ℝ} (heps : 0 < eps)
    (params : AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : (ZMod p × Channels K) → ℝ} (hE : Aliased S (WithLp.toLp 2 E))
    {h : (ZMod p)ˣ} (hh : h ∈ S) (hne : h ≠ 1)
    (t : ℕ) (evalBlocks : List (Power.BlockWeights K H)) :
    let evalE := WithLp.toLp 2 (AdamW.evaluation params
      (AdamW.trainingRun params (embeddingOracle eps blocks k) E t))
    ¬ (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
      Power.stackLogit eps (entry evalE) evalBlocks a b q <
        Power.stackLogit eps (entry evalE) evalBlocks a b ((a / b) ^ k)) := by
  exact not_solves_of_aliased
    (evaluation_trainingRun_aliased heps params blocks k hk hE t) hh hne eps evalBlocks k

/-- The same trajectory retains the quantitative cross-entropy obstruction
on each nonzero target. Other classes in softmax can only increase this bound. -/
theorem trainingRun_loss_floor {eps : ℝ} (heps : 0 < eps)
    (params : AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : (ZMod p × Channels K) → ℝ} (hE : Aliased S (WithLp.toLp 2 E))
    (t : ℕ) (evalBlocks : List (Power.BlockWeights K H)) (a b : ZMod p)
    {q : ZMod p} (hq : q ≠ 0) :
    let evalE := WithLp.toLp 2 (AdamW.evaluation params
      (AdamW.trainingRun params (embeddingOracle eps blocks k) E t))
    Real.log (Nat.card S : ℝ) ≤
      crossEntropy (Power.stackLogit eps (entry evalE) evalBlocks a b) q := by
  exact crossEntropy_ge_log_subgroup
    (evaluation_trainingRun_aliased heps params blocks k hk hE t) eps evalBlocks a b hq

/-- External schedules and arbitrary valid initial optimizer buffers cannot
repair exact full-table alias symmetry. The update equation is the independently
defined source recurrence, with the actual model gradient on its right side. -/
theorem trajectory_not_solves {eps : ℝ} (heps : 0 < eps)
    (params : ℕ → AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    (s : ℕ → AdamW.TrainingState (ZMod p × Channels K)) (hzero : StateAliased S (s 0))
    (hstep : ∀ t, s (t + 1) = AdamW.trainingStep (params t) t (s t)
      (embeddingOracle eps blocks k t (s t).point))
    {h : (ZMod p)ˣ} (hh : h ∈ S) (hne : h ≠ 1)
    (t : ℕ) (evalParams : AdamW.Parameters) (evalBlocks : List (Power.BlockWeights K H)) :
    let evalE := WithLp.toLp 2 (AdamW.evaluation evalParams (s t))
    ¬ (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
      Power.stackLogit eps (entry evalE) evalBlocks a b q <
        Power.stackLogit eps (entry evalE) evalBlocks a b ((a / b) ^ k)) := by
  exact not_solves_of_aliased
    (evaluation_aliased evalParams (trajectory_aliased heps params blocks k hk s hzero hstep t))
    hh hne eps evalBlocks k

private noncomputable def sampleParameters : AdamW.Parameters :=
  { adam := { learningRate := 3 / 1000, decay := 1, warmup := 10,
              betaOne := 9 / 10, betaTwo := 98 / 100, epsilon := 1 / 100000000 } }

-- These witnesses use positive rate, positive epsilon and a nonzero embedding.
-- Its zero row is different from its nonzero rows, so it is not a zero network.
-- Changing the branch matrices does not affect any symmetry hypothesis.
example := aliased_limit (p := 3) (C := Fin 16) (S := ⊤)
  (E := fun _ => WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (limit := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro t h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (fun _ _ => tendsto_const_nhds)
example := limit_not_solves (p := 3) (K := Fin 8) (H := Fin 43) (S := ⊤)
  (E := fun _ => WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (limit := WithLp.toLp 2 fun j => if j.1 = 0 then 2 else 1)
  (by intro t h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (fun _ _ => tendsto_const_nhds)
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) (by decide)
  (1 / 8388608) [Power.zeroBlock] 1
example := trainingRun_not_solves (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) sampleParameters (fun _ => [Power.zeroBlock]) 1
  (S := ⊤) (by simp) (E := fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) (by decide) 20000 [Power.zeroBlock]
example := trainingRun_loss_floor (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) sampleParameters (fun _ => [Power.zeroBlock]) 1
  (S := ⊤) (by simp) (E := fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
  20000 [Power.zeroBlock] 1 2 (q := 2) (by decide)
example := trajectory_not_solves (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) (fun _ => sampleParameters)
  (fun _ => [Power.zeroBlock]) 1 (S := ⊤) (by simp)
  (AdamW.trainingRun sampleParameters
    (embeddingOracle (1 / 8388608) (fun _ => [Power.zeroBlock]) 1) (fun _ => 1))
  (by exact ⟨fun h hh q i => rfl, fun h hh q i => rfl, fun h hh q i => rfl⟩)
  (fun _ => rfl) (h := Units.mk0 (2 : ZMod 3) (by decide)) (by simp) (by decide)
  20000 sampleParameters [Power.zeroBlock]

end Transformer.Rotate90.Training
