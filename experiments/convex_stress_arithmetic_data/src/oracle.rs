//! Port of the exact definitions retained in DiffGPT5a37c515's
//! research/grokking_task_inventory.json and stress_tests_for_gpt.md.
//! Missing fixed random tables/sets are refused, never reconstructed by guess.
use crate::{Result, reference::SplitMix64};
use serde_json::Value;

pub const UNAVAILABLE: [&str; 6] = [
    "random",
    "sum-interval-2",
    "sum-interval-4",
    "sum-interval-8",
    "sum-interval-12",
    "sum-random-half",
];

fn power(a: u32, n: u32, p: u32) -> u32 {
    (0..n).fold(1, |z, _| z * a % p)
}
fn inverse(a: u32, p: u32) -> Result<u32> {
    (1..p)
        .find(|&b| a * b % p == 1)
        .ok_or_else(|| "noninvertible denominator".into())
}
pub fn table(spec: &Value) -> Result<Option<Vec<[u32; 3]>>> {
    let p = spec["prime"].as_u64().ok_or("missing prime")?;
    if ![17, 97, 193].contains(&p) {
        return Err("unsupported source modulus".into());
    }
    let p = p as u32;
    let op = spec["operation"].as_str().ok_or("missing operation")?;
    if UNAVAILABLE.contains(&op) {
        return Ok(None);
    }
    let mut permutation: Vec<_> = (0..p).collect();
    SplitMix64(0x7065726D75746531).shuffle(&mut permutation);
    let first = u32::from(matches!(op, "div" | "power-division"));
    let mut rows = vec![];
    for a in 0..p {
        for b in first..p {
            let sum = (a + b) % p;
            let cubic = (a * a * a + a * b) % p;
            let parity = (2 * (cubic / 2) + (a + b) % 2) % p;
            let y = match op {
                "add" => sum,
                "sub" => (a + p - b) % p,
                "mul" => a * b % p,
                "div" => a * inverse(b, p)? % p,
                "power-division" => {
                    let n = spec["power"].as_u64().ok_or("missing exponent")?;
                    if ![0, 2, 3, 5].contains(&n) {
                        return Err("unsupported exponent".into());
                    }
                    power(a * inverse(b, p)? % p, n as u32, p)
                }
                "mixed" => {
                    if b % 2 == 1 {
                        a * inverse(b, p)? % p
                    } else {
                        (a + p - b) % p
                    }
                }
                "square-sum" => (a * a + b * b) % p,
                "quadratic" => (a * a + a * b + b * b) % p,
                "quadratic-linear" => (a * a + a * b + b * b + a) % p,
                "cubic-product" => cubic,
                "cubic-mixed" => (a * a * a + a * b * b + b) % p,
                "half-sum" => sum * inverse(2, p)? % p,
                "perm-add" => permutation[sum as usize],
                "cube-plus" => (a * a * a + b) % p,
                "square-times" => a * a * b % p,
                "affine-product" => (a + a * b) % p,
                "abs-diff" => a.abs_diff(b),
                "abs-diff/blind" => a.abs_diff(b) / 2,
                "floor-product" => a * b / p,
                "max" => a.max(b),
                "sum-square" => power(sum, 2, p),
                "sum-cube" => power(sum, 3, p),
                "square-product" => (a * a + a * b) % p,
                "parity-cubic" => parity,
                "parity-cubic/blind" => parity / 2,
                "sum-interval" => u32::from(sum <= (p - 3) / 2),
                "sum-squares" => u32::from((1..p).any(|x| x * x % p == sum)),
                _ => {
                    let n = op
                        .strip_prefix("sum-pow")
                        .and_then(|x| x.parse::<u32>().ok())
                        .filter(|x| [4, 6, 8, 12, 16, 24, 32, 48].contains(x))
                        .ok_or("unsupported operation")?;
                    power(sum, n, p)
                }
            };
            if y >= p {
                return Err("out of vocabulary source label".into());
            }
            rows.push([a, b, y]);
        }
    }
    Ok(Some(rows))
}
