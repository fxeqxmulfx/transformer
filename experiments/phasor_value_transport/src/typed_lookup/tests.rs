use super::*;

fn nearest(row: &[f32]) -> Option<usize> {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
}
fn make(keys: &[usize], d: usize) -> Vec<f32> {
    keys.iter()
        .flat_map(|&k| (0..d).map(move |j| if j == k { 1. } else { 0. }))
        .collect()
}
fn choose(w: &mut [f32], cue: usize, main: usize, tail: usize) {
    let code = paths::code(cue, main, tail);
    w[code as usize % BINS] = if code & BINS as u32 == 0 { 1. } else { -1. };
}

#[test]
fn sharing_budget_prefixes_keeps_every_floating_point_vote() {
    // Repeated keys, missing addresses, and ties in the requested depths.
    let keys: Vec<_> = (0..123).map(|i| (i % 23 != 0).then_some(i * (i + 3) % 17)).collect();
    let weights: Vec<_> = (0..BINS).map(|i| (i as f32 * 0.283).sin()).collect();
    let budgets = [1, 2, 10, 10, 20, 39, 40, 41, 80];
    for ordered in [false, true] {
        let together = final_vote_budgets(&keys, &budgets, &weights, ordered, None);
        for (out, budget) in together.iter().zip(budgets) {
            let separate = final_vote_budgets(&keys, &[budget], &weights, ordered, None);
            assert!(out.iter().zip(&separate[0]).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

#[test]
fn repeated_reads_use_the_latest_link_then_a_different_final_relation() {
    let d = 12;
    // Both occurrences of the same directed relation at address 0 are live.
    // Only the later write participates. The terminal is read in reverse.
    let keys = [0, 1, 9, 11, 0, 1, 2, 11, 2, 1, 3, 10, 4, 5, 3, 11, 7, 0];
    let x = make(&keys, d);
    let mut w = vec![0.; parameters(d)];
    choose(&mut w, 7, 2, 11);
    for (mode, exists) in [(1, false), (2, true), (5, true)] {
        let tape = Tape::new(
            &x,
            &x,
            &w,
            (d, keys.len()),
            PathRead::linear(mode),
            true,
            nearest,
        );
        let mut out = vec![0.; x.len()];
        tape.add_output(&mut out);
        let last = row(&out, keys.len() - 1, d);
        assert_eq!(last[4], f32::from(exists));
        assert_eq!(last.iter().copied().sum::<f32>(), f32::from(exists));
        for n in 1..keys.len() {
            let mut prefix = vec![0.; n * d];
            Tape::new(
                &x[..n * d],
                &x[..n * d],
                &w,
                (d, n),
                PathRead::linear(mode),
                true,
                nearest,
            )
            .add_output(&mut prefix);
            assert_eq!(prefix, out[..n * d]);
        }
    }
}

#[test]
fn signed_path_sum_and_gate_differentiate_through_dense_values() {
    for (gain_power, normalize_vote) in [(0, false), (1, false), (0, true), (1, true), (4, true)] {
        for ordered in [false, true] {
            check_gradient(PathRead {
                passes: 2,
                gain_power,
                normalize_vote,
                ordered,
                soft_address: false,
                agreement: false,
            });
        }
    }
}

#[test]
fn paired_addresses_differentiate_through_inputs_hidden_gate_and_both_banks() {
    for ordered in [false, true] {
        check_gradient(PathRead {
            passes: 2,
            gain_power: 4,
            ordered,
            agreement: true,
            ..PathRead::NORMALIZED
        });
    }
}

fn check_gradient(read: PathRead) {
    let d = 6;
    let x = make(&[0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 4, 3, 2, 5, 4, 3, 1, 2], d);
    let length = 9;
    let h: Vec<_> = (0..x.len())
        .map(|i| ((i * 23 % 67) as f32 - 32.) / 83.)
        .collect();
    let y: Vec<_> = (0..x.len())
        .map(|i| ((i * 17 % 83) as f32 - 40.) / 71.)
        .collect();
    let mut w: Vec<_> = (0..parameters(d))
        .map(|i| ((i * 17 % 113) as f32 - 50.) / 299.)
        .collect();
    w[3] = 0.29;
    w[5] = -0.225;
    if read.agreement {
        // Activate both coordinates of a path present in the input. With
        // uncorrelated random coordinates the contextual gate can be tiny.
        let mut store = Store::new(read.ordered);
        for (i, v) in x[..length * d].chunks_exact(d).enumerate() {
            store.observe(nearest(v), i);
        }
        let mut first = None;
        store.visit(read.passes, |path| {
            first.get_or_insert(path);
        });
        let [(a, s), (b, t)] = first.expect("fixture must contain a path").addresses();
        w[a] = (0.24 * s) as f32;
        w[b] = (0.22 * t) as f32;
    }
    let tape = Tape::new(&x, &h, &w, (d, length), read, true, nearest);
    let mut dx = vec![0.; x.len()];
    let mut dh = vec![0.; x.len()];
    let mut dw = vec![0.; w.len()];
    tape.backward(&h, &w, &y, &mut dx, &mut dh, &mut dw);
    let objective = |a: &[f32], b: &[f32], c: &[f32]| {
        let mut out = vec![0.; a.len()];
        Tape::new(a, b, c, (d, length), read, true, nearest).add_output(&mut out);
        out.iter()
            .zip(&y)
            .map(|(&v, &k)| f64::from(v) * f64::from(k))
            .sum::<f64>()
    };
    for (id, (base, pullback)) in [(&x, &dx), (&h, &dh), (&w, &dw)].into_iter().enumerate() {
        let direction: Vec<_> = (0..base.len())
            .map(|i| ((i * 97 % 193) as f32 - 96.) / 107.)
            .collect();
        let at = |eps: f32| {
            let z: Vec<_> = base
                .iter()
                .zip(&direction)
                .map(|(a, b)| a + eps * b)
                .collect();
            match id {
                0 => objective(&z, &h, &w),
                1 => objective(&x, &z, &w),
                _ => objective(&x, &h, &z),
            }
        };
        let analytic = dot(pullback, &direction);
        let numeric = (at(0.0007) - at(-0.0007)) / 0.0014;
        assert!(
            (numeric - f64::from(analytic)).abs() < 0.002 + 0.002 * f64::from(analytic).abs(),
            "input {id}, analytic {analytic}, numeric {numeric}"
        );
    }
    assert!(dw[..BINS].iter().any(|v| v.abs() > 0.01));
    if read.agreement {
        assert!(dw[..BINS / 2].iter().any(|v| v.abs() > 0.01));
        assert!(dw[BINS / 2..BINS].iter().any(|v| v.abs() > 0.01));
    }
    assert!(dh.iter().any(|v| v.abs() > 0.0001));
}

#[test]
fn frozen_path_votes_match_every_training_row_without_retaining_adjoint_entries() {
    let d = 16;
    let mut state = 3581u32;
    let w: Vec<_> = (0..parameters(d))
        .map(|i| ((i * 17 % 113) as f32 - 50.) / 299.)
        .collect();
    for length in [2, 17, 96] {
        let keys: Vec<_> = (0..2 * length)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as usize % d
            })
            .collect();
        let mut x = make(&keys, d);
        for row in x.chunks_exact_mut(d).step_by(13) {
            row.fill(0.);
        }
        let address = |v: &[f32]| nearest(v).filter(|&i| v[i] > 0.);
        for passes in [1, 5, 10] {
            let train = Tape::new(
                &x,
                &x,
                &w,
                (d, length),
                PathRead::linear(passes),
                true,
                address,
            );
            let score = Tape::new(
                &x,
                &x,
                &w,
                (d, length),
                PathRead::linear(passes),
                false,
                address,
            );
            let mut expected = x.clone();
            let mut actual = x.clone();
            train.add_output(&mut expected);
            score.add_output(&mut actual);
            assert!(
                expected
                    .iter()
                    .zip(&actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(score.input.is_empty());
            assert!(
                score
                    .frames
                    .iter()
                    .all(|r| r.witnesses.capacity() == 0 && r.terminals.capacity() == 0)
            );
            if length > 2 {
                assert!(
                    train
                        .frames
                        .iter()
                        .any(|r| !r.witnesses.is_empty() && !r.terminals.is_empty())
                );
            }
        }
    }
}

/// Run the test executable once per mode so peak resident memory is measured
/// in a fresh process. Every row goes through the same five-pass read.
#[test]
#[ignore = "benchmarks host path storage at length 512"]
fn measure_path_storage() {
    let retain = match std::env::var("PHASOR_PATH_STORAGE").as_deref() {
        Ok("adjoint") => true,
        Ok("frozen") => false,
        _ => panic!("set PHASOR_PATH_STORAGE to adjoint or frozen"),
    };
    let (d, length, batch) = (64, 512, 4);
    let mut state = 0x17b4d67au32;
    let mut x = vec![0.; batch * length * d];
    for (i, row) in x.chunks_exact_mut(d).enumerate() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = if i % 3 == 1 {
            (state >> 28) % 4
        } else {
            4 + (state >> 26) % 48
        };
        row[key as usize] = 1.;
    }
    let w: Vec<_> = (0..parameters(d))
        .map(|i| ((i * 17 % 113) as f32 - 50.) / 299.)
        .collect();
    let address = |v: &[f32]| v.iter().position(|&x| x == 1.);
    let mut times = Vec::new();
    let mut entries = 0;
    let mut out = x.clone();
    for _ in 0..6 {
        out.copy_from_slice(&x);
        let now = std::time::Instant::now();
        let tape = Tape::new(
            &x,
            &x,
            &w,
            (d, length),
            PathRead::linear(5),
            retain,
            address,
        );
        tape.add_output(&mut out);
        times.push(now.elapsed().as_secs_f64());
        entries = tape.frames.iter().map(|r| r.witnesses.len()).sum::<usize>();
    }
    let digest = crate::hash(&out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    println!("witnesses={entries} output={digest} seconds={times:?}");
    assert_eq!(entries > 0, retain);
}

#[test]
fn shared_prefixes_match_independent_rows_with_their_own_hidden_gates() {
    let d = 12;
    let base = [0, 1, 9, 11, 0, 1, 2, 11, 2, 1, 3, 10, 4, 5, 3, 11, 7, 0];
    for repeats in [1, 9] {
        let keys = base.repeat(repeats);
        let mut changed = keys.clone();
        changed[keys.len() / 2] = 8;
        let mut x = make(&keys, d);
        for seq in [&changed, &changed, &keys] {
            x.extend(make(seq, d));
        }
        let h: Vec<_> = (0..x.len())
            .map(|i| ((i * 37 % 137) as f32 - 68.) / 42.)
            .collect();
        let w: Vec<_> = (0..parameters(d))
            .map(|i| ((i * 17 % 113) as f32 - 50.) / 40.)
            .collect();
        let stride = keys.len() * d;
        for read in [
            PathRead::linear(1),
            PathRead::NORMALIZED,
            PathRead {
                passes: 40,
                ordered: true,
                gain_power: 4,
                ..PathRead::NORMALIZED
            },
        ] {
            let mut actual = vec![0.; x.len()];
            Tape::new(&x, &h, &w, (d, keys.len()), read, false, nearest).add_output(&mut actual);
            for positions in [[0, keys.len() / 2, keys.len() - 1, 4], [2, 1, 0, 3]] {
                let mut requested = vec![0.; x.len()];
                Tape::with_outputs(&x, &h, &w, (d, keys.len()), read, Outputs::Positions(&positions), nearest)
                    .add_output(&mut requested);
                for (b, t) in positions.into_iter().enumerate() {
                    let i = b * stride + t * d;
                    assert!(actual[i..i+d].iter().zip(&requested[i..i+d])
                        .all(|(full, selected)| full.to_bits() == selected.to_bits()));
                }
            }
            for (batch, (input, hidden)) in x
                .chunks_exact(stride)
                .zip(h.chunks_exact(stride))
                .enumerate()
            {
                let mut independent = vec![0.; stride];
                Tape::new(input, hidden, &w, (d, keys.len()), read, true, nearest)
                    .add_output(&mut independent);
                assert!(
                    independent
                        .iter()
                        .zip(&actual[batch * stride..])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            assert_ne!(&actual[stride..2 * stride], &actual[2 * stride..3 * stride]);
        }
    }
}
