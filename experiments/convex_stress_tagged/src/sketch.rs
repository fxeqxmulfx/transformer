//! Categorical centered sketches, ported from decode/search.rs at4510ee7d.
//! Unlike its permutation clock, angular quantization allows repeated phases
//! and an independently declared q. Every power state and measured norm stays
//! in the record. No correct phase, input-token formula or hidden label enters.
use crate::{C, Result, Sample, random::Random};
use serde::{Deserialize, Serialize};

fn pairs(x: &[C]) -> Vec<[f64; 2]> {
    x.iter().map(|z| [z.re, z.im]).collect()
}
fn normalized(x: &mut [C]) -> Option<f64> {
    let norm = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return None;
    }
    for z in x {
        *z /= norm;
    }
    Some(norm)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerState {
    pub left: Vec<[f64; 2]>,
    pub right: Vec<[f64; 2]>,
    pub left_norm: Option<f64>,
    pub right_norm: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    pub codes: Vec<[f64; 2]>,
    pub mean: [f64; 2],
    pub matrix: Vec<[f64; 2]>,
    pub states: Vec<PowerState>,
    pub left: Vec<[f64; 2]>,
    pub conjugate_right: Vec<[f64; 2]>,
    pub observed_rows: Vec<bool>,
    pub observed_columns: Vec<bool>,
    pub matvecs: usize,
    pub products: usize,
    pub reason: String,
}
pub fn build(
    n: usize,
    vocab: usize,
    samples: &[Sample],
    seed: u32,
    steps: usize,
) -> Result<Sketch> {
    crate::validate_observations(n, vocab, samples)?;
    if !(1..=128).contains(&steps) {
        return Err("invalid sketch power budget".into());
    }
    let mut classes = vec![];
    let (mut observed_rows, mut observed_columns) = (vec![false; n], vec![false; n]);
    for o in samples {
        if !classes.contains(&o.label) {
            classes.push(o.label);
        }
        observed_rows[o.cell % n] = true;
        observed_columns[o.cell / n] = true;
    }
    let mut rng = Random::new(seed);
    let mut codes = vec![C::default(); vocab];
    for &c in &classes {
        codes[c] = C::new(rng.gauss(0., 1.), rng.gauss(0., 1.));
    }
    let mean = samples.iter().map(|s| codes[s.label]).sum::<C>() / samples.len() as f64;
    let mut matrix = vec![C::default(); n * n];
    for o in samples {
        matrix[o.cell] = codes[o.label] - mean;
    }
    let mut right: Vec<_> = (0..n)
        .map(|_| C::new(rng.gauss(0., 1.), rng.gauss(0., 1.)))
        .collect();
    let initial_norm = normalized(&mut right);
    let mut left = vec![C::default(); n];
    let mut out = Sketch {
        codes: pairs(&codes),
        mean: [mean.re, mean.im],
        matrix: pairs(&matrix),
        states: vec![PowerState {
            left: pairs(&left),
            right: pairs(&right),
            left_norm: None,
            right_norm: initial_norm,
        }],
        left: vec![],
        conjugate_right: vec![],
        observed_rows,
        observed_columns,
        matvecs: 0,
        products: 0,
        reason: "initial_normalization_failed".into(),
    };
    if initial_norm.is_none() {
        return Ok(out);
    }
    for _ in 0..steps {
        left.fill(C::default());
        for j in 0..n {
            for i in 0..n {
                left[i] += matrix[i + n * j] * right[j];
            }
        }
        out.matvecs += 1;
        out.products += n * n;
        let left_norm = normalized(&mut left);
        if left_norm.is_none() {
            out.states.push(PowerState {
                left: pairs(&left),
                right: pairs(&right),
                left_norm,
                right_norm: None,
            });
            out.reason = "left_normalization_failed".into();
            return Ok(out);
        }
        right.fill(C::default());
        for j in 0..n {
            for i in 0..n {
                right[j] += matrix[i + n * j].conj() * left[i];
            }
        }
        out.matvecs += 1;
        out.products += n * n;
        let right_norm = normalized(&mut right);
        out.states.push(PowerState {
            left: pairs(&left),
            right: pairs(&right),
            left_norm,
            right_norm,
        });
        if right_norm.is_none() {
            out.reason = "right_normalization_failed".into();
            return Ok(out);
        }
    }
    left.fill(C::default());
    for j in 0..n {
        for i in 0..n {
            left[i] += matrix[i + n * j] * right[j];
        }
    }
    out.matvecs += 1;
    out.products += n * n;
    out.left = pairs(&left);
    out.conjugate_right = pairs(&right.iter().map(|z| z.conj()).collect::<Vec<_>>());
    out.reason = "power_complete".into();
    Ok(out)
}
pub fn quantize(values: &[[f64; 2]], observed: &[bool], q: usize) -> Result<Vec<usize>> {
    if values.len() != observed.len() || !(2..=257).contains(&q) {
        return Err("invalid angular phase clock".into());
    }
    let anchor = observed
        .iter()
        .position(|&o| o)
        .ok_or("no observed phase anchor")?;
    let z: Vec<_> = values.iter().map(|&[re, im]| C::new(re, im)).collect();
    if z.iter()
        .zip(observed)
        .any(|(z, &o)| o && (!z.re.is_finite() || !z.im.is_finite() || z.norm() < 1e-12))
    {
        return Err("observed angular phase is zero or nonfinite".into());
    }
    let origin = z[anchor].arg();
    Ok(z.iter()
        .zip(observed)
        .map(|(z, &o)| {
            if o {
                (((z.arg() - origin) * q as f64 / std::f64::consts::TAU)
                    .round()
                    .rem_euclid(q as f64)) as usize
            } else {
                0
            }
        })
        .collect())
}
