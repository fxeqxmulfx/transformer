//! Fixed-label generators from grokking 308912f2, training/domain/tasks.py.
//! Their only consumer is the external data adapter. Each reconstructed
//! label and its position are authenticated by the original tensor hashes.
use crate::{Result, reference::SplitMix64};
use serde_json::Value;

fn draw(rng: &mut SplitMix64, bound: u32) -> u32 {
    let bound = u64::from(bound);
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = rng.next();
        if word >= threshold {
            return (word % bound) as u32;
        }
    }
}

pub(crate) fn table(spec: &Value) -> Result<Vec<[u32; 3]>> {
    if spec["prime"] != 97 {
        return Err("fixed registry tables expect the catalog modulus 97".into());
    }
    let p = 97;
    let op = spec["operation"].as_str().ok_or("missing operation")?;
    let pairs = (0..p).flat_map(|a| (0..p).map(move |b| (a, b)));
    if op == "random" {
        let mut rng = SplitMix64(0x72616e646f6d3031);
        return Ok(pairs.map(|(a, b)| [a, b, draw(&mut rng, p)]).collect());
    }
    let half = (p - 1) / 2;
    let residues: Vec<_> = if op == "sum-random-half" {
        let mut values: Vec<_> = (0..p).collect();
        SplitMix64(0x68616c6673657421).shuffle(&mut values);
        values.truncate(half as usize);
        values
    } else {
        let swaps = op
            .strip_prefix("sum-interval-")
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|n| matches!(n, 2 | 4 | 8 | 12))
            .ok_or("unknown fixed registry set")?;
        let mut inside: Vec<_> = (0..half).collect();
        let mut outside: Vec<_> = (half..p).collect();
        let mut rng = SplitMix64(0x696e74657276616c);
        rng.shuffle(&mut inside);
        rng.shuffle(&mut outside);
        inside
            .into_iter()
            .skip(swaps)
            .chain(outside.into_iter().take(swaps))
            .collect()
    };
    let mut member = [false; 97];
    for residue in residues {
        member[residue as usize] = true;
    }
    Ok(pairs
        .map(|(a, b)| [a, b, u32::from(member[((a + b) % p) as usize])])
        .collect())
}
