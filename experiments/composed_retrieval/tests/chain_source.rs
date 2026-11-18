//! Independent literal construction of the DiffGPT dense tensor. The source
//! generator's seed 11 archive and this seed 1 check cover different records.
#![cfg(feature = "chain-data")]
use baseline::hash;
use composed_retrieval::{
    Result,
    chain_data::{self, PRIMES, STEPS},
};
use std::collections::BTreeSet;

fn reference_draws(seed: u64) -> impl Iterator<Item = u64> {
    (1u64..).map(move |i| {
        let z = seed.wrapping_add(i.wrapping_mul(0x9e3779b97f4a7c15));
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    })
}

fn choice(draws: &mut impl Iterator<Item = u64>, size: usize) -> usize {
    let cutoff = (1u128 << 64) % size as u128;
    let x = draws.find(|&x| u128::from(x) >= cutoff).unwrap();
    (x % size as u64) as usize
}

fn reference_inputs(p: usize, width: usize, count: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut rng = reference_draws(seed);
    // p17 at one short step is the sole enumerated domain in the 12 cases.
    if (p, width, count) == (17, 3, 3072) {
        let mut codes: Vec<_> = (0..4913).collect();
        for last in (1..4913).rev() {
            codes.swap(choice(&mut rng, last + 1), last);
        }
        return codes[..count]
            .iter()
            .map(|&code| vec![code % 17, code / 17 % 17, code / 289])
            .collect();
    }
    let mut seen = BTreeSet::new();
    std::iter::from_fn(|| {
        Some(
            (0..width)
                .map(|_| choice(&mut rng, p) as i64)
                .collect::<Vec<_>>(),
        )
    })
    .filter(|row| seen.insert(row.clone()))
    .take(count)
    .collect()
}

fn dense_hash(rows: &[Vec<i64>], p: usize) -> String {
    let width = rows[0].len();
    let mut bytes = format!("torch.int64:({}, 3, {}):", rows.len(), width).into_bytes();
    for x in rows {
        let mut value = x[0];
        for i in 0..(width - 1) / 2 {
            value = (value * ((value * value + x[2 * i + 1]) % p as i64) + x[2 * i + 2]) % p as i64;
        }
        let tensor = x
            .iter()
            .copied()
            .chain(std::iter::repeat_n(-100, width - 1))
            .chain([value])
            .chain(std::iter::repeat_n(0, width));
        bytes.extend(tensor.flat_map(i64::to_le_bytes));
    }
    hash(&bytes)
}

#[test]
fn entire_original_seed_one_order_labels_masks_and_longer_rng() -> Result<()> {
    for p in PRIMES {
        for l in STEPS {
            let short = reference_inputs(p, 1 + 2 * l, 3072, 0x73706c6974730003);
            let long = reference_inputs(p, 1 + 4 * l, 1024, 0x6c6f6e6765720000);
            let actual = chain_data::build(p, l, 1)?;
            assert_eq!(actual.selection, "validation");
            assert_eq!(actual.vocab, p);
            assert_eq!(actual.splits.len(), 3);
            for (name, rows) in [
                ("train", &short[..2048]),
                ("validation", &short[2048..]),
                ("test/length-double", long.as_slice()),
            ] {
                let split = &actual.splits[name];
                assert_eq!(split.len(), rows.len());
                for (original, x) in split.iter().zip(rows) {
                    assert_eq!(&original.tokens, x);
                }
                assert_eq!(
                    chain_data::fingerprints(split, p)?.dense_tensor_sha256,
                    dense_hash(rows, p),
                    "p={p} steps={l} split={name}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_changed_final_only_supervision_and_nonoriginal_cases() -> Result<()> {
    assert!(chain_data::build(19, 1, 1).is_err());
    assert!(chain_data::build(17, 3, 1).is_err());
    let mut task = chain_data::build(17, 1, 1)?;
    let row = &task.splits.remove("train").unwrap()[0];
    for column in [0, 1, 2] {
        let mut r = row.clone();
        match column {
            0 => r.targets[0] = 0,
            1 => r.changes[1] = true,
            _ => {
                r.targets.pop();
            }
        }
        assert!(chain_data::fingerprints(&[r], 17).is_err());
    }
    Ok(())
}
