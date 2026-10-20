import Transformer.ScheduleFree.Section2_Muon
import Transformer.ScheduleFree.Section4_AdamW

/-!
Kernel-checked wrapper invariants and rational checkpoints. The same values
are written independently in optimizers/tests/test_schedule_free_muon.py.
General wrapper sources: arXiv:2405.15682v4, §§2–2.1; native Muon source:
arXiv:2502.16982, §§2.1–2.2. This script asserts no convergence of the composition.
-/

open Transformer.ScheduleFree

noncomputable section

example :
    let w : Matrix (Fin 1) (Fin 1) ℝ := Matrix.of (fun _ _ => 1)
    let s := muonWrapperRun (1 / 2) 0 (1 / 2) (fun _ => 1 / 4) (fun _ _ => 0) w 1
    s.base 0 0 = 7 / 8 ∧ s.evaluation 0 0 = 15 / 16 ∧
      wrapperQuery (1 / 2) s 0 0 = 29 / 32 := muonWrapper_first_decay_fixture

example :
    let w : Matrix (Fin 1) (Fin 1) ℝ := Matrix.of (fun _ _ => 1)
    let s := muonWrapperRun (1 / 2) 0 (1 / 2) (fun _ => 1 / 4) (fun _ _ => 0) w 2
    s.base 0 0 = 49 / 64 ∧ s.evaluation 0 0 = 169 / 192 ∧
      wrapperQuery (1 / 2) s 0 0 = 79 / 96 := muonWrapper_second_decay_fixture

#print axioms wrapperCoefficient_bounds
#print axioms wrapperQuery_endpoints
#print axioms wrapperQuery_displacement
#print axioms wrapperStep_balance
#print axioms wrapperRun_updates
#print axioms wrapperRun_average_balance
#print axioms wrapperRun_evaluation
#print axioms wrapper_initial_query
#print axioms muonWrapperStep_base
#print axioms muonWrapperStep_momentum
#print axioms muonWrapperRun_evaluation
#print axioms muonWrapperStep_zero_gradient
#print axioms muonWrapperStep_zero_rate
#print axioms muonWrapper_first_decay_fixture
#print axioms muonWrapper_second_decay_fixture
#print axioms adam_total_monotone
#print axioms adamRun_total_nonneg
#print axioms adamRun_second_nonneg
#print axioms adam_averaging_bounds
#print axioms adam_weighted_balance
