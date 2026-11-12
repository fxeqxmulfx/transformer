use crate::*;
use propagation::{Observation, Problem};
use search::{AttemptResult, Limits, Stop, discover, replay};

fn obs(row: usize, column: usize, label: usize) -> Observation {
    Observation { row, column, label }
}
fn limits() -> Limits {
    Limits {
        anchors: 16,
        sweeps: 4,
        producer_checks: 10_000_000,
        branches: 16384,
        memory_bytes: RUN_MEMORY_BYTES,
    }
}
fn triangle() -> Problem {
    Problem {
        side: 3,
        states: 3,
        vocab: 2,
        decoders: vec![vec![0, 1, 1]],
        observations: (0..3)
            .flat_map(|i| {
                (0..3).filter_map(move |j| {
                    ((i, j) != (0, 0)).then_some(obs(i, j, usize::from(i != j)))
                })
            })
            .collect(),
    }
}

#[test]
fn extra_observed_anchors_resolve_the_lean_fixed_point_stall() {
    let p = triangle();
    let mut cap = limits();
    cap.anchors = 1;
    let first = discover(&p, cap).unwrap();
    assert_eq!(first.record.attempts[0].config.anchor, obs(1, 1, 0));
    assert_eq!(first.verified.unwrap().answer(0, 0), None);
    let found = discover(&p, limits()).unwrap();
    assert_eq!(found.record.stop, Stop::FullCoverage);
    assert!(found.record.attempts.len() > 1);
    assert_eq!(found.verified.as_ref().unwrap().answer(0, 0), Some(0));
    assert_eq!(found.record.producer_checks, found.record.checker_checks);
    replay(&p, &found.record).unwrap();
}

#[test]
fn larger_fibers_and_every_declared_bank_member_reach_the_solver() {
    let d = [0, 0, 1, 1, 0];
    let p = Problem {
        side: 4,
        states: 5,
        vocab: 3,
        decoders: decoder_bank(3).unwrap(),
        observations: (0..4)
            .flat_map(|i| {
                (0..4).filter_map(move |j| {
                    ((i, j) != (0, 0)).then_some(obs(i, j, d[(i + 5 - j) % 5]))
                })
            })
            .collect(),
    };
    let found = discover(&p, limits()).unwrap();
    let verified = found.verified.unwrap();
    assert_eq!(verified.answer(0, 0), Some(0));
    assert_eq!(verified.certificate().problem.decoders.len(), 4);
    assert!(
        verified
            .certificate()
            .branches
            .iter()
            .all(|b| !b.live || b.decoder == 1)
    );
    for mode in [
        general::Mode::Full,
        general::Mode::Mini,
        general::Mode::Forward,
    ] {
        let mut solver = general::Solver::new(&verified, 1, mode).unwrap();
        solver.step(&[0]).unwrap();
        solver
            .snapshot(12)
            .unwrap()
            .model
            .certified(&verified)
            .unwrap();
    }
    let p = Problem {
        side: 2,
        states: 4,
        vocab: 4,
        decoders: vec![vec![0, 1, 2, 3], vec![0, 1, 3, 2]],
        observations: vec![obs(0, 0, 0), obs(0, 1, 1), obs(1, 0, 2)],
    };
    let found = discover(&p, limits()).unwrap();
    assert_eq!(found.verified.as_ref().unwrap().answer(1, 1), None);
    assert_eq!(
        found
            .record
            .selected
            .as_ref()
            .unwrap()
            .problem
            .decoders
            .len(),
        2
    );
    assert!(found.record.attempts.iter().all(|a| matches!(&a.result,
        AttemptResult::Complete(c) if c.coverage == 3)));
}

#[test]
fn budget_refusal_never_certifies_a_partial_union() {
    let p = triangle();
    let complete = discover(&p, limits()).unwrap();
    let AttemptResult::Complete(first) = &complete.record.attempts[0].result else {
        panic!()
    };
    let first_cost = first.costs.support_checks + first.costs.output_checks;
    let mut cap = limits();
    cap.producer_checks = first_cost + 1;
    let found = discover(&p, cap).unwrap();
    assert_eq!(found.record.stop, Stop::WorkBudget);
    assert_eq!(found.record.selected_attempt, Some(0));
    assert_eq!(found.record.attempts.len(), 2);
    assert!(matches!(
        &found.record.attempts[1].result,
        AttemptResult::Refused(_)
    ));
    assert_eq!(found.verified.as_ref().unwrap().answer(0, 0), None);
    assert_eq!(found.record.producer_checks, cap.producer_checks);
    assert_eq!(found.record.checker_checks, first_cost);
    replay(&p, &found.record).unwrap();
    cap.producer_checks = 0;
    let none = discover(&p, cap).unwrap();
    assert!(none.record.selected.is_none() && none.verified.is_none());
    cap = limits();
    cap.branches = 0;
    assert_eq!(
        discover(&p, cap).unwrap().record.stop,
        Stop::ResourceRefusal
    );
    // The second attempt must account for the saved first certificate too.
    let base = propagation::preflight(&p, propagation::Config::new(obs(1, 1, 0), 4))
        .unwrap()
        .memory_bound;
    cap = limits();
    cap.memory_bytes = 2 * base + 64 * 1024 * 1024;
    let memory = discover(&p, cap).unwrap();
    assert_eq!(memory.record.selected_attempt, Some(0));
    assert_eq!(memory.record.stop, Stop::ResourceRefusal);
    assert!(matches!(
        &memory.record.attempts[1].result,
        AttemptResult::Refused(propagation::Refusal {
            reason: propagation::RefusalReason::Memory { .. },
            ..
        })
    ));
    replay(&p, &memory.record).unwrap();
}

#[test]
fn arbitrary_label_renaming_and_input_arrival_order_preserve_discovery() {
    let p = triangle();
    let original = discover(&p, limits()).unwrap();
    let mut reversed = p.clone();
    reversed.observations.reverse();
    assert_eq!(
        original.record,
        discover(&reversed, limits()).unwrap().record
    );
    for batch in 1..=p.observations.len() {
        let mut assembled = p.clone();
        assembled.observations.clear();
        for chunk in p.observations.chunks(batch) {
            assembled.observations.extend_from_slice(chunk);
        }
        assert_eq!(
            original.record,
            discover(&assembled, limits()).unwrap().record
        );
    }
    let mut renamed = p.clone();
    renamed.vocab = 5;
    for d in &mut renamed.decoders {
        for y in d {
            *y = [4, 2][*y];
        }
    }
    for t in &mut renamed.observations {
        t.label = [4, 2][t.label];
    }
    let found = discover(&renamed, limits()).unwrap();
    assert_eq!(found.verified.unwrap().answer(0, 0), Some(4));
    assert_eq!(original.record.attempts.len(), found.record.attempts.len());
    let mut repeated = p.clone();
    repeated.observations.extend_from_slice(&p.observations);
    let repeats = discover(&repeated, limits()).unwrap();
    assert_eq!(repeats.verified.unwrap().answer(0, 0), Some(0));
    assert_eq!(
        repeats.record.selected.unwrap().problem.observations.len(),
        16
    );
}

#[test]
fn contradiction_and_tampered_discovery_have_explicit_outcomes() {
    let mut p = triangle();
    p.observations.push(obs(1, 1, 1));
    let no_fit = discover(&p, limits()).unwrap();
    assert_eq!(no_fit.record.stop, Stop::Inconsistent);
    assert!(no_fit.verified.unwrap().inconsistent());
    let p = triangle();
    let found = discover(&p, limits()).unwrap();
    let bytes = serde_json::to_vec(&found.record).unwrap();
    let saved = serde_json::from_slice(&bytes).unwrap();
    replay(&p, &saved).unwrap();
    for kind in 0..4 {
        let mut bad = found.record.clone();
        match kind {
            0 => {
                bad.attempts.remove(0);
            }
            1 => bad.selected_attempt = Some(0),
            2 => bad.producer_checks += 1,
            _ => bad
                .selected
                .as_mut()
                .unwrap()
                .branches
                .pop()
                .map(|_| ())
                .unwrap(),
        }
        assert!(replay(&p, &bad).is_err());
    }
    assert!(decoder_bank(usize::MAX).is_err());
    let bank = decoder_bank(5).unwrap();
    assert_eq!(bank[0], vec![0, 1, 2, 3, 4, 4, 3, 2, 1]);
    assert_eq!(bank[1], vec![0, 0, 1, 1, 2, 2, 1, 1, 0]);
    assert_eq!(bank.len(), 4);
    let mut invalid = limits();
    invalid.memory_bytes = 1;
    assert!(discover(&p, invalid).is_err());
}
