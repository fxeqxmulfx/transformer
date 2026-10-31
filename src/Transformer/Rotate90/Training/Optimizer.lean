module

public import Transformer.Rotate90.Training.Gradient
public import Transformer.ScheduleFree.AdamW.Defs

/-!
# Schedule-Free AdamW preserves the actual full-table alias symmetry

Source: `review/flat_spectrum.md`, *Alias symmetry and training*, optimizer theorem;
the revised source recurrence formalized in `ScheduleFree.AdamW.Defs`. Symmetry
must hold in the query, base and second-moment rows, not only the evaluated E.
Initialization supplies all three conditions. Coordinatewise normalization, decay
and averaging preserve them. The oracle below is the actual tied-model loss
gradient, with arbitrary branch matrices at each step. This is a full-batch
real-arithmetic result; stochastic partial batches need not preserve the symmetry.
-/

@[expose] public section

namespace Transformer.Rotate90.Training

open Landscape.Neural

open ScheduleFree

/-- Equal coordinates in all optimizer buffers and the gradient remain equal
after the source's complete update, including bias correction and query decay. -/
theorem trainingStep_equal_coordinates {ι : Type*} (params : AdamW.Parameters) (t : ℕ)
    (s : AdamW.TrainingState ι) (g : ι → ℝ) (i j : ι)
    (hy : s.point i = s.point j) (hz : s.base i = s.base j)
    (hv : s.history.second i = s.history.second j) (hg : g i = g j) :
    (AdamW.trainingStep params t s g).point i = (AdamW.trainingStep params t s g).point j ∧
      (AdamW.trainingStep params t s g).base i = (AdamW.trainingStep params t s g).base j ∧
      (AdamW.trainingStep params t s g).history.second i =
        (AdamW.trainingStep params t s g).history.second j := by
  simp only [AdamW.trainingStep, AdamW.direction, AdamW.denominator, AdamW.nextSecond, AdamW.nextHistory]
  rw [hy, hz, hv, hg]
  exact ⟨rfl, rfl, rfl⟩

variable {p : ℕ} [Fact p.Prime] {K H : Type*} [Fintype K] [Fintype H]

/-- All embedding-coordinate optimizer buffers share the alias symmetry. -/
def StateAliased (S : Subgroup (ZMod p)ˣ)
    (s : AdamW.TrainingState (ZMod p × Channels K)) : Prop :=
  Aliased S (WithLp.toLp 2 s.point) ∧ Aliased S (WithLp.toLp 2 s.base) ∧
    Aliased S (WithLp.toLp 2 s.history.second)

omit [Fintype K] in
/-- The complete Schedule-Free update preserves the three buffer invariants. -/
theorem stateAliased_step (params : AdamW.Parameters) (t : ℕ)
    {S : Subgroup (ZMod p)ˣ} {s : AdamW.TrainingState (ZMod p × Channels K)}
    (hs : StateAliased S s) {g : (ZMod p × Channels K) → ℝ}
    (hg : Aliased S (WithLp.toLp 2 g)) : StateAliased S (AdamW.trainingStep params t s g) := by
  obtain ⟨hy, hz, hv⟩ := hs
  have heq (h : (ZMod p)ˣ) (hh : h ∈ S) (q : ZMod p) (i : Channels K) :=
    trainingStep_equal_coordinates params t s g ((h * q, i)) (q, i)
      (hy h hh q i) (hz h hh q i) (hv h hh q i) (hg h hh q i)
  exact ⟨fun h hh q i => (heq h hh q i).1,
    fun h hh q i => (heq h hh q i).2.1, fun h hh q i => (heq h hh q i).2.2⟩

/-- The oracle is the actual embedding gradient at the current query. Branch
matrices may follow any trajectory, including simultaneous optimizer updates. -/
noncomputable def embeddingOracle (eps : ℝ) (blocks : ℕ → List (Power.BlockWeights K H))
    (k t : ℕ) (E : (ZMod p × Channels K) → ℝ) : (ZMod p × Channels K) → ℝ :=
  (gradient (sharedLoss eps (blocks t) k) (WithLp.toLp 2 E)).ofLp

/-- The gradient theorem supplies the invariant gradient required by the update. -/
theorem embeddingOracle_aliased {eps : ℝ} (heps : 0 < eps)
    (blocks : ℕ → List (Power.BlockWeights K H)) (k t : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : (ZMod p × Channels K) → ℝ} (hE : Aliased S (WithLp.toLp 2 E)) :
    Aliased S (WithLp.toLp 2 (embeddingOracle eps blocks k t E)) := by
  exact gradient_sharedLoss_aliased heps (blocks t) k hk hE

/-- **Full-batch Schedule-Free cannot leave the exact alias subspace.** This is
induction on the independently defined source recurrence, using the actual loss
gradient. It covers arbitrary finite width/depth and arbitrary branch trajectories. -/
theorem trainingRun_aliased {eps : ℝ} (heps : 0 < eps)
    (params : AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : (ZMod p × Channels K) → ℝ} (hE : Aliased S (WithLp.toLp 2 E)) (t : ℕ) :
    StateAliased S (AdamW.trainingRun params (embeddingOracle eps blocks k) E t) := by
  induction t with
  | zero => exact ⟨hE, hE, fun h hh q i => rfl⟩
  | succ t ih =>
    exact stateAliased_step params t ih (embeddingOracle_aliased heps blocks k t hk ih.1)

/-- External schedules may change any scalar optimizer parameter at each step.
Every trajectory satisfying the actual source update retains the invariant;
branch matrices can also change arbitrarily. This covers the lab's external
warmup without identifying it with the optimizer's internal warmup convention. -/
theorem trajectory_aliased {eps : ℝ} (heps : 0 < eps)
    (params : ℕ → AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H))
    (k : ℕ) {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    (s : ℕ → AdamW.TrainingState (ZMod p × Channels K)) (hzero : StateAliased S (s 0))
    (hstep : ∀ t, s (t + 1) = AdamW.trainingStep (params t) t (s t)
      (embeddingOracle eps blocks k t (s t).point)) (t : ℕ) : StateAliased S (s t) := by
  induction t with
  | zero => exact hzero
  | succ t ih =>
    rw [hstep]
    exact stateAliased_step (params t) t ih (embeddingOracle_aliased heps blocks k t hk ih.1)

omit [Fintype K] in
/-- Evaluation uses the source's averaged weights; they retain the same symmetry. -/
theorem evaluation_aliased (params : AdamW.Parameters)
    {S : Subgroup (ZMod p)ˣ} {s : AdamW.TrainingState (ZMod p × Channels K)}
    (hs : StateAliased S s) : Aliased S (WithLp.toLp 2 (AdamW.evaluation params s)) := by
  intro h hh q i
  change s.point (h * q, i) + _ * (s.base (h * q, i) - s.point (h * q, i)) =
    s.point (q, i) + _ * (s.base (q, i) - s.point (q, i))
  have hy := hs.1 h hh q i
  have hz := hs.2.1 h hh q i
  change s.point (h * q, i) = s.point (q, i) at hy
  change s.base (h * q, i) = s.base (q, i) at hz
  rw [hy, hz]

/-- The full trajectory's reported embedding, not merely its training query,
remains aliased at every finite update count. -/
theorem evaluation_trainingRun_aliased {eps : ℝ} (heps : 0 < eps)
    (params : AdamW.Parameters) (blocks : ℕ → List (Power.BlockWeights K H)) (k : ℕ)
    {S : Subgroup (ZMod p)ˣ} (hk : (Nat.card S).Coprime k)
    {E : (ZMod p × Channels K) → ℝ} (hE : Aliased S (WithLp.toLp 2 E)) (t : ℕ) :
    Aliased S (WithLp.toLp 2 (AdamW.evaluation params
      (AdamW.trainingRun params (embeddingOracle eps blocks k) E t))) :=
  evaluation_aliased params (trainingRun_aliased heps params blocks k hk hE t)

private noncomputable def sampleParameters : AdamW.Parameters :=
  { adam := { learningRate := 3 / 1000, decay := 1, warmup := 0,
              betaOne := 9 / 10, betaTwo := 98 / 100, epsilon := 1 / 100000000 } }

example := trainingStep_equal_coordinates sampleParameters 4
  (⟨fun _ : Fin 2 => 2, fun _ => 3, ⟨fun _ => 4, 5, 3 / 1000⟩⟩)
  (fun _ => 7) 0 1 rfl rfl rfl rfl
example := stateAliased_step (p := 3) (K := Fin 1) sampleParameters 4
  (S := ⊤) (s := ⟨fun _ => 2, fun _ => 3, ⟨fun _ => 4, 5, 3 / 1000⟩⟩)
  (by exact ⟨fun h hh q i => rfl, fun h hh q i => rfl, fun h hh q i => rfl⟩)
  (g := fun _ => 7) (by intro h hh q i; rfl)
example := embeddingOracle_aliased (p := 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) (fun _ => [Power.zeroBlock]) 1 4 (S := ⊤) (by simp)
  (E := fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero])
example := trainingRun_aliased (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) sampleParameters (fun _ => [Power.zeroBlock]) 1
  (S := ⊤) (by simp) (E := fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero]) 5000
example := evaluation_aliased (p := 3) (K := Fin 1) sampleParameters
  (S := ⊤) (s := ⟨fun _ => 2, fun _ => 3, ⟨fun _ => 4, 5, 3 / 1000⟩⟩)
  (by exact ⟨fun h hh q i => rfl, fun h hh q i => rfl, fun h hh q i => rfl⟩)
example := evaluation_trainingRun_aliased (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) sampleParameters (fun _ => [Power.zeroBlock]) 1
  (S := ⊤) (by simp) (E := fun j => if j.1 = 0 then 2 else 1)
  (by intro h hh q i; simp [entry, mul_eq_zero, h.ne_zero]) 5000

example := trajectory_aliased (p := 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) (fun _ => sampleParameters)
  (fun _ => [Power.zeroBlock]) 1 (S := ⊤) (by simp)
  (AdamW.trainingRun sampleParameters
    (embeddingOracle (1 / 8388608) (fun _ => [Power.zeroBlock]) 1) (fun _ => 1))
  (by exact ⟨fun h hh q i => rfl, fun h hh q i => rfl, fun h hh q i => rfl⟩)
  (fun _ => rfl) 5000

end Transformer.Rotate90.Training
