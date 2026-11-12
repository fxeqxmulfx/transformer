//! Independent enumeration of complete phase assignments, not of pruning paths.
use super::*;

fn validate_all_completions(p: &Problem, c: Config, saved: &Certificate) {
    let checked = verify(p, c, saved).unwrap();
    let q = p.states;
    let mut completions = 0;
    for (k, d) in p.decoders.iter().enumerate() {
        // Fix row 0 to zero: every assignment has an equivalent representative.
        for r1 in 0..q {
            for s0 in 0..q {
                for s1 in 0..q {
                    let (r, s) = ([0, r1], [s0, s1]);
                    if !p
                        .observations
                        .iter()
                        .all(|t| d[(r[t.row] + s[t.column]) % q] == t.label)
                    {
                        continue;
                    }
                    completions += 1;
                    let shift = r[c.anchor.row];
                    let z = (shift + s[c.anchor.column]) % q;
                    let branch = saved
                        .branches
                        .iter()
                        .find(|b| b.decoder == k && b.anchor_state == z)
                        .unwrap();
                    assert!(branch.live);
                    for i in 0..2 {
                        assert!(branch.domains.rows[i].contains(&((r[i] + q - shift) % q)));
                        assert!(branch.domains.columns[i].contains(&((s[i] + shift) % q)));
                        for j in 0..2 {
                            let y = d[(r[i] + s[j]) % q];
                            assert!(saved.answers[i * 2 + j].contains(&y));
                            if let Some(answer) = checked.answer(i, j) {
                                assert_eq!(answer, y);
                            }
                        }
                    }
                }
            }
        }
    }
    if checked.inconsistent() {
        assert_eq!(completions, 0);
    }
}

fn cases(decoders: Vec<Vec<usize>>, budgets: &[usize]) -> usize {
    let q = decoders[0].len();
    let mut count = 0;
    // Each cell is unobserved, observed as 0, or observed as 1. Include every
    // nonempty pattern, even when no compatible global completion exists.
    for encoded in 1..81 {
        let (mut pattern, mut observations) = (encoded, Vec::new());
        for cell in 0..4 {
            let digit = pattern % 3;
            pattern /= 3;
            if digit != 0 {
                observations.push(obs(cell / 2, cell % 2, digit - 1));
            }
        }
        let p = Problem {
            side: 2,
            states: q,
            vocab: 2,
            decoders: decoders.clone(),
            observations,
        };
        let mut previous: Option<Vec<Vec<usize>>> = None;
        for &steps in budgets {
            let c = Config::new(p.observations[0], steps);
            let saved = complete(&p, c);
            validate_all_completions(&p, c, &saved);
            if let Some(old) = previous {
                for (before, after) in old.iter().zip(&saved.answers) {
                    assert!(after.iter().all(|y| before.contains(y)));
                }
            }
            previous = Some(saved.answers);
            count += 1;
        }
    }
    count
}

#[test]
fn every_binary_decoder_on_four_states_every_observation_pattern() {
    let mut total = 0;
    for encoded in 0..16 {
        let decoder = (0..4).map(|z| (encoded >> z) & 1).collect();
        total += cases(vec![decoder], &[0, 1, 2, 3]);
    }
    assert_eq!(total, 5120);
}

#[test]
fn every_distinct_pair_of_binary_decoders_on_three_states() {
    let mut total = 0;
    for a in 0..8 {
        for b in a + 1..8 {
            let decoders = [a, b]
                .map(|encoded| (0..3).map(|z| (encoded >> z) & 1).collect())
                .to_vec();
            total += cases(decoders, &[0, 1, 3]);
        }
    }
    assert_eq!(total, 6720);
}

#[test]
fn no_singleton_fiber_four_by_four_witness_and_all_phase_completions() {
    // Every decoder fiber has at least two states, so an observed singleton
    // fiber cannot force an equality of two columns as in the ZMod 4 witness.
    let d = vec![0, 0, 1, 1, 0];
    let mut p = Problem {
        side: 4,
        states: 5,
        vocab: 2,
        decoders: vec![d.clone()],
        observations: vec![],
    };
    for i in 0..4 {
        for j in 0..4 {
            if (i, j) != (0, 0) {
                p.observations.push(obs(i, j, d[(i + 5 - j) % 5]));
            }
        }
    }
    let c = Config::new(obs(0, 2, 1), 4);
    let saved = complete(&p, c);
    let checked = verify(&p, c, &saved).unwrap();
    assert_eq!(checked.answer(0, 0), Some(0));
    let mut completions = 0;
    for encoded in 0..5_usize.pow(7) {
        let (mut value, mut r, mut s) = (encoded, [0; 4], [0; 4]);
        for phase in r[1..].iter_mut().chain(s.iter_mut()) {
            *phase = value % 5;
            value /= 5;
        }
        if p.observations
            .iter()
            .all(|t| d[(r[t.row] + s[t.column]) % 5] == t.label)
        {
            completions += 1;
            assert_eq!(d[(r[0] + s[0]) % 5], 0);
        }
    }
    assert!(completions > 1);
}
