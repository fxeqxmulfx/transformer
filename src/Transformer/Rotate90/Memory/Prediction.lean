module

public import Transformer.Rotate90.Memory.Geometry

/-!
# Predictions inherited by a stopped evaluation trajectory

Source: `review/flat_spectrum.md`, *Optimizer memory*, M6. At positive epsilon,
continuity of the actual all-parameter network transfers the stopped optimizer
limit to logits and full-softmax loss. Every strict comparison at the saved
base eventually holds along the evaluation path, including a wrong winner.
Base ties need not resolve consistently, and no useful numerical time bound
follows from continuity alone. This is a conditional continuation theorem,
not a claim about the endpoint reached by positive-rate random-start training.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Memory

open Learning ScheduleFree

variable {V K H D : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype D]

/-- M6: all actual finite-stack logits converge to those of the fixed base. -/
theorem stopped_score_tendsto {eps : ℝ} (heps : 0 < eps) (order : List D)
    (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (Index V K H D → ℝ) → (Index V K H D → ℝ))
    (s : AdamW.TrainingState (Index V K H D))
    (hW : 0 < s.history.total) (hM : 0 < s.history.maximumRate) (a b q : V) :
    Filter.Tendsto
      (fun n => score eps order (evaluated (halt p) (stoppedRun p offset oracle s n)) a b q)
      Filter.atTop (nhds (score eps order (WithLp.toLp 2 s.base) a b q)) :=
  (analyticAt_score heps order (WithLp.toLp 2 s.base) a b q).continuousAt.tendsto.comp
    (stopped_tendsto p offset oracle s hW hM)

/-- Any strict base comparison eventually persists, whether correct or wrong. -/
theorem stopped_comparison {eps : ℝ} (heps : 0 < eps) (order : List D)
    (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (Index V K H D → ℝ) → (Index V K H D → ℝ))
    (s : AdamW.TrainingState (Index V K H D))
    (hW : 0 < s.history.total) (hM : 0 < s.history.maximumRate)
    (a b q r : V)
    (hqr : score eps order (WithLp.toLp 2 s.base) a b q <
      score eps order (WithLp.toLp 2 s.base) a b r) :
    ∀ᶠ n in Filter.atTop,
      score eps order (evaluated (halt p) (stoppedRun p offset oracle s n)) a b q <
        score eps order (evaluated (halt p) (stoppedRun p offset oracle s n)) a b r :=
  (stopped_score_tendsto heps order p offset oracle s hW hM a b q).eventually_lt
    (stopped_score_tendsto heps order p offset oracle s hW hM a b r) hqr

/-- Finite vocabulary makes one eventual time work for every strict comparison
on the entire table. Equality at the base is deliberately not included. -/
theorem stopped_all_comparisons {eps : ℝ} (heps : 0 < eps) (order : List D)
    (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (Index V K H D → ℝ) → (Index V K H D → ℝ))
    (s : AdamW.TrainingState (Index V K H D))
    (hW : 0 < s.history.total) (hM : 0 < s.history.maximumRate) :
    ∀ᶠ n in Filter.atTop, ∀ a b q r : V,
      score eps order (WithLp.toLp 2 s.base) a b q <
        score eps order (WithLp.toLp 2 s.base) a b r →
      score eps order (evaluated (halt p) (stoppedRun p offset oracle s n)) a b q <
        score eps order (evaluated (halt p) (stoppedRun p offset oracle s n)) a b r := by
  apply Filter.eventually_all.mpr
  intro a
  apply Filter.eventually_all.mpr
  intro b
  apply Filter.eventually_all.mpr
  intro q
  apply Filter.eventually_all.mpr
  intro r
  by_cases hqr : score eps order (WithLp.toLp 2 s.base) a b q <
      score eps order (WithLp.toLp 2 s.base) a b r
  · exact (stopped_comparison heps order p offset oracle s hW hM a b q r hqr).mono
      (fun _ h _ => h)
  · exact Filter.Eventually.of_forall (fun _ h => (hqr h).elim)

/-- The fixed split's actual full-softmax loss converges to base loss too. -/
theorem stopped_loss_tendsto {eps : ℝ} (heps : 0 < eps) (order : List D)
    (p : AdamParameters) (offset : ℕ)
    (oracle : ℕ → (Index V K H D → ℝ) → (Index V K H D → ℝ))
    (s : AdamW.TrainingState (Index V K H D))
    (hW : 0 < s.history.total) (hM : 0 < s.history.maximumRate)
    (rows : Finset (V × V)) (target : V × V → V) :
    Filter.Tendsto
      (fun n => loss eps order rows target (evaluated (halt p) (stoppedRun p offset oracle s n)))
      Filter.atTop (nhds (loss eps order rows target (WithLp.toLp 2 s.base))) :=
  ((analyticOnNhd_loss heps order rows target) (WithLp.toLp 2 s.base)
    (Set.mem_univ _)).continuousAt.tendsto.comp (stopped_tendsto p offset oracle s hW hM)

example := stopped_score_tendsto (V := Fin 3) (K := Fin 8) (H := Fin 43)
  (eps := 1/8388608) (by norm_num) (List.finRange 3)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun _ => 0, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num) 0 1 2
example := stopped_all_comparisons (V := Fin 3) (K := Fin 8) (H := Fin 43)
  (eps := 1/8388608) (by norm_num) (List.finRange 3)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun _ => 0, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num)
example := stopped_loss_tendsto (V := ZMod 3) (K := Fin 8) (H := Fin 43)
  (eps := 1/8388608) (by norm_num) (List.finRange 3)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750 (fun _ y => y)
  ⟨fun _ => 1, fun _ => 0, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (by norm_num) (by norm_num) Finset.univ (fun r => (r.1 / r.2) ^ 2)

/-- A nonvacuous strict-comparison witness: a genuine powered-division solver
is the base, and the starting query is displaced from it. -/
example : ∃ w : Weights (ZMod 3) (Power.PairIndex (Fin 1))
      ((Fin 10 × Fin 2) ⊕ Fin 23) (Fin 1),
    ∀ᶠ n in Filter.atTop,
      score (1/8388608) [0]
        (evaluated (halt ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩)
          (stoppedRun ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750
            (fun _ y => y) ⟨fun i => w.ofLp i + 1, w.ofLp, ⟨fun _ => 1, 1, 1⟩⟩ n)) 1 1 0 <
      score (1/8388608) [0]
        (evaluated (halt ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩)
          (stoppedRun ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ 750
            (fun _ y => y) ⟨fun i => w.ofLp i + 1, w.ofLp, ⟨fun _ => 1, 1, 1⟩⟩ n)) 1 1 1 := by
  obtain ⟨w, hw, _, _⟩ := antipodal_solvers (p := 3) (K := Fin 1) (Z := Fin 23)
    (eps := 1/8388608) (by norm_num) 2
  refine ⟨w, stopped_comparison (by norm_num) [0] _ 750 (fun _ y => y)
    ⟨fun i => w.ofLp i + 1, w.ofLp, ⟨fun _ => 1, 1, 1⟩⟩ (by norm_num) (by norm_num)
    1 1 0 1 ?_⟩
  simpa using hw 1 1 (by decide) 0 (by norm_num)

end Transformer.Rotate90.Memory
