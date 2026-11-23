//! Check both split order and the actual DiffGPT temporal codec and decoder.
//! Reconstruct each orbit backwards using the inverse transition. Decode a
//! center-token continuation using only prefix observations, as in the source.
#[path = "../src/stress/temporal.rs"]
mod temporal;
use arithmetic::{hash, reference::SplitMix64};
use serde_json::Value;
use source_codec::{Difference, Domain, Matrix, Tokenizer, decode, encode};
use std::{fs, path::Path};

type State = (i64, i64);

fn inverse_orbits(fibonacci: bool) -> [Vec<State>; 2] {
    let mut visited = [[false; 97]; 97];
    let mut orbits = Vec::new();
    for a in 0..97i64 {
        for b in 0..97i64 {
            if visited[a as usize][b as usize]
                || (if fibonacci {
                    a == 0 && b == 0
                } else {
                    a == 0 || b == 0
                })
            {
                continue;
            }
            let mut orbit = Vec::new();
            let mut state = (a, b);
            while !visited[state.0 as usize][state.1 as usize] {
                visited[state.0 as usize][state.1 as usize] = true;
                orbit.push(state);
                let (u, v) = state;
                state = (
                    if fibonacci {
                        (v - u).rem_euclid(97)
                    } else {
                        v * u % 97
                    },
                    u,
                );
            }
            assert_eq!(state, (a, b));
            orbit[1..].reverse();
            orbits.push(orbit);
        }
    }
    let mut order: Vec<_> = (0..orbits.len()).collect();
    SplitMix64(1 ^ 0x6f72626974730001).shuffle(&mut order);
    let (mut train, mut held) = (vec![], vec![]);
    for i in order {
        let dest = if train.len() <= held.len() {
            &mut train
        } else {
            &mut held
        };
        dest.extend_from_slice(&orbits[i]);
    }
    SplitMix64(1 ^ 0x747261696e000002).shuffle(&mut train);
    SplitMix64(1 ^ 0x7465737400000003).shuffle(&mut held);
    [train, held]
}

fn trajectories(seeds: &[State], n: usize, fibonacci: bool) -> Matrix<f64> {
    let mut raw = Vec::new();
    let mut states = seeds.to_vec();
    for _ in 0..n {
        raw.extend(states.iter().map(|&(a, _)| a as f64));
        for (a, b) in &mut states {
            let future = if fibonacci {
                (*a + *b) % 97
            } else {
                (0..97).find(|y| y * *b % 97 == *a).unwrap()
            };
            (*a, *b) = (*b, future);
        }
    }
    Matrix::new(seeds.len(), raw).unwrap()
}

#[test]
fn complete_orbit_splits_match_the_public_tokens_masks_and_decoded_futures() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.lake/reproduction/gpt");
    for (name, digest) in [
        (
            "research/current_grokking/adapters.py",
            "4c36782edbb785432a7dfd71cf291acde2a1bf2c9337f27b5df854ec826cdbc2",
        ),
        (
            "src/diff_gpt/application/grokking/recurrence.py",
            "ceba42d1d2eb915d6e9748e3984997f183ae7f7bdf8ef28b6445da18da37964c",
        ),
        (
            "src/diff_gpt/application/grokking/recurrence_field_sampler.py",
            "8a9c620eb432184d08ca217b29f96a157d589c72a528ab2e24cc9ebfc53143cf",
        ),
    ] {
        assert_eq!(hash(&fs::read(source.join(name)).unwrap()), digest);
    }
    let catalog: Value = serde_json::from_slice(
        &fs::read(source.join("research/current_grokking/plan.json")).unwrap(),
    )
    .unwrap();
    let canonical: Value = serde_json::from_str(arithmetic::ORIGINAL35).unwrap();
    let canonical = canonical["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "recurrence-fibonacci-k1-c2-lgt")
        .unwrap();
    let mut cases = 0;
    for case in catalog["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "recurrence" && r["data"]["tokenizer"] == "lgt")
    {
        let (task, hashes) = temporal::build(case).unwrap();
        let order = case["data"]["order"].as_u64().unwrap() as usize;
        let context = case["data"]["context"].as_u64().unwrap() as usize;
        let fib = case["data"]["task"] == "fibonacci";
        let pair = inverse_orbits(fib);
        let cut = pair[0].len();
        assert_eq!(
            [cut, pair[1].len()],
            if fib { [4704, 4704] } else { [4603, 4613] }
        );
        let states = pair.concat();
        let n = states.len();
        let raw = trajectories(&states, context + order + 1, fib);
        let mut digest = format!("torch.int64:({n}, {}):", raw.rows()).into_bytes();
        for i in 0..n {
            for t in 0..raw.rows() {
                digest.extend((raw.values()[t * n + i] as i64).to_le_bytes());
            }
        }
        assert_eq!(hash(&digest), hashes["raw"]);
        if fib && order == 1 {
            assert_eq!(hashes, canonical["data_fingerprints"]);
        }
        let vocab = task.vocab as i64;
        let ops = vec![Difference::order(order).unwrap(); n];
        let domain = vec![Domain::symmetric(((vocab - (1 << order)) / 2) as f64); n];
        let full = encode(&raw, vocab, &domain, &ops, Tokenizer::Lgt).unwrap();
        assert_eq!(full.clipped_tokens, 0);
        let past = Matrix::new(n, raw.values()[..raw.values().len() - n].to_vec()).unwrap();
        let history = encode(&past, vocab, &domain, &ops, Tokenizer::Lgt).unwrap();
        assert_eq!(full.prefix, history.prefix);
        assert_eq!(
            history.tokens.values(),
            &full.tokens.values()[..context * n]
        );
        assert!(history.grid.scale.iter().all(|&s| s == 1.));
        let mut center_tokens = history.tokens.values().to_vec();
        center_tokens.extend(std::iter::repeat_n(vocab / 2, n));
        let baseline = decode(
            &history.prefix,
            &history.grid,
            &Matrix::new(n, center_tokens).unwrap(),
            vocab,
            &ops,
            Tokenizer::Lgt,
        )
        .unwrap();
        assert_eq!(
            decode(
                &full.prefix,
                &full.grid,
                &full.tokens,
                vocab,
                &ops,
                Tokenizer::Lgt
            )
            .unwrap(),
            raw
        );
        for (i, e) in task.splits["train"]
            .iter()
            .chain(&task.splits["validation"])
            .enumerate()
        {
            let past = (0..context)
                .map(|t| history.tokens.values()[t * n + i])
                .collect::<Vec<_>>();
            assert_eq!(e.tokens, past);
            let future = raw.values()[(raw.rows() - 1) * n + i] as i64;
            let base = baseline.values()[(baseline.rows() - 1) * n + i] as i64;
            let target = full.tokens.values()[context * n + i];
            assert_eq!(e.targets[..context - 1], vec![-100; context - 1]);
            assert_eq!(e.targets[context - 1], target);
            assert_eq!(target - vocab / 2 + base, future);
            assert_eq!(e.field, fib.then_some((base, future)));
            assert_eq!(e.row_index, Some(i as i64));
            assert!(e.changes.iter().all(|&v| !v));
        }
        eprintln!(
            "{} ordered examples={n} train={cut} raw={}",
            case["name"], hashes["raw"]
        );
        cases += 1;
    }
    assert_eq!(cases, 5);
}
