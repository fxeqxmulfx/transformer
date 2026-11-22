//! Post-training comparison against additive characters. This module never constructs model inputs or weights.
use crate::Result;
use serde_json::{Value, json};
use std::f64::consts::TAU;

pub type Complex = [f64; 2];

pub fn character(k: usize, x: usize, p: usize) -> Complex {
    let (s, c) = (TAU * ((k * x) % p) as f64 / p as f64).sin_cos();
    [c, s]
}

pub fn multiply([a, b]: Complex, [c, d]: Complex) -> Complex {
    [a * c - b * d, a * d + b * c]
}
pub fn conjugate([a, b]: Complex) -> Complex {
    [a, -b]
}
fn norm2(z: Complex) -> f64 {
    z[0] * z[0] + z[1] * z[1]
}
fn inner(z: Complex, w: Complex) -> f64 {
    z[0] * w[0] + z[1] * w[1]
}
fn unit(z: Complex) -> Option<Complex> {
    let r = norm2(z).sqrt();
    (r > 0.).then(|| [z[0] / r, z[1] / r])
}

/// Hungarian assignment. The row potentials and augmenting paths minimize
/// minus the squared correlations, assigning each nonzero frequency once.
fn match_channels(weight: &[Vec<f64>]) -> Vec<usize> {
    let n = weight.len();
    let mut row_price = vec![0.; n + 1];
    let mut col_price = vec![0.; n + 1];
    let mut owner = vec![0; n + 1];
    let mut previous = vec![0; n + 1];
    for root in 1..=n {
        owner[0] = root;
        let mut column = 0;
        let mut distance = vec![f64::INFINITY; n + 1];
        let mut seen = vec![false; n + 1];
        loop {
            seen[column] = true;
            let row = owner[column];
            let mut next = 0;
            let mut delta = f64::INFINITY;
            for j in 1..=n {
                if seen[j] {
                    continue;
                }
                let cost = -weight[row - 1][j - 1] - row_price[row] - col_price[j];
                if cost < distance[j] {
                    distance[j] = cost;
                    previous[j] = column;
                }
                if distance[j] < delta {
                    delta = distance[j];
                    next = j;
                }
            }
            for j in 0..=n {
                if seen[j] {
                    row_price[owner[j]] += delta;
                    col_price[j] -= delta;
                } else {
                    distance[j] -= delta;
                }
            }
            column = next;
            if owner[column] == 0 {
                break;
            }
        }
        while column != 0 {
            let parent = previous[column];
            owner[column] = owner[parent];
            column = parent;
        }
    }
    let mut frequency = vec![0; n];
    for j in 1..=n {
        frequency[owner[j] - 1] = j;
    }
    frequency
}

pub struct Measurements {
    pub p: usize,
    pub m: usize,
    /// Row b of the learned coordinates, row a of the rotations and route probabilities.
    pub coordinates: Vec<Complex>,
    pub phases: Vec<Complex>,
    pub routes: Vec<f64>,
}

fn average(total: f64, count: usize) -> Option<f64> {
    (count > 0).then(|| total / count as f64)
}

pub fn analyze(v: &Measurements) -> Result<Value> {
    let (p, m) = (v.p, v.m);
    if m + 1 != p
        || m == 0
        || v.coordinates.len() != p * m
        || v.phases.len() != p * m
        || v.routes.len() != p * (m + 1)
        || !v
            .coordinates
            .iter()
            .chain(&v.phases)
            .flatten()
            .chain(&v.routes)
            .all(|a| a.is_finite())
    {
        return Err(
            "Fourier reference needs one finite learned channel per nonzero frequency".into(),
        );
    }
    let mut coefficient = vec![vec![[0.; 2]; m]; m];
    let mut quality = vec![vec![0.; m]; m];
    for j in 0..m {
        let power = (0..p).map(|b| norm2(v.coordinates[b * m + j])).sum::<f64>() / p as f64;
        for k in 1..p {
            let c = &mut coefficient[j][k - 1];
            for b in 0..p {
                let z = multiply(v.coordinates[b * m + j], conjugate(character(k, b, p)));
                c[0] += z[0] / p as f64;
                c[1] += z[1] / p as f64;
            }
            quality[j][k - 1] = if power > 0. { norm2(*c) / power } else { 0. };
        }
    }
    let frequency = match_channels(&quality);
    let mut slot = vec![0; p];
    let mut gains = Vec::with_capacity(m);
    let mut matched = Vec::with_capacity(m);
    for (j, &k) in frequency.iter().enumerate() {
        slot[k] = j;
        gains.push(coefficient[j][k - 1]);
        matched.push(quality[j][k - 1]);
    }
    let mut correct = 0;
    let mut mass = 0.;
    let mut collapse = 0.;
    let mut phase_dot = [0.; 2];
    let mut phase_count = [0; 2];
    let mut action_dot = 0.;
    let mut predicted_energy = 0.;
    let mut reference_energy = 0.;
    for a in 0..p {
        let route = &v.routes[a * (m + 1)..(a + 1) * (m + 1)];
        if (route.iter().sum::<f64>() - 1.).abs() > 2e-5 || route.iter().any(|&v| v < 0.) {
            return Err("invalid learned routing distribution".into());
        }
        let strongest = route
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        let intercept = (a * a % p) * a % p;
        if a > 0 {
            collapse += route[m];
        }
        for j in 0..m {
            let k = frequency[j];
            let target = slot[k * a % p];
            if a > 0 {
                let ideal = (target + m - j) % m;
                mass += route[ideal];
                correct += usize::from(strongest == ideal);
            }
            let source_origin = if a == 0 {
                Some([1., 0.])
            } else {
                unit(gains[target])
            };
            if let (Some(dst), Some(src)) = (unit(gains[j]), source_origin) {
                let ideal = multiply(multiply(dst, conjugate(src)), character(k, intercept, p));
                let group = usize::from(a != 0);
                phase_dot[group] += inner(v.phases[a * m + j], ideal);
                phase_count[group] += 1;
            }
            for b in 0..p {
                let mut mixed = [route[m], 0.];
                for (s, &prob) in route[..m].iter().enumerate() {
                    let z = v.coordinates[b * m + (j + s) % m];
                    mixed[0] += prob * z[0];
                    mixed[1] += prob * z[1];
                }
                let predicted = multiply(v.phases[a * m + j], mixed);
                let reference = multiply(gains[j], character(k, (intercept + a * b) % p, p));
                action_dot += inner(predicted, reference);
                predicted_energy += norm2(predicted);
                reference_energy += norm2(reference);
            }
        }
    }
    let energy = (predicted_energy * reference_energy).sqrt();
    let count = (p - 1) * m;
    Ok(json!({
        "frequency_by_learned_slot":frequency,
        "channel_mean_squared_correlation":matched.iter().sum::<f64>()/m as f64,
        "channel_mean_correlation":matched.iter().map(|q|q.sqrt()).sum::<f64>()/m as f64,
        "alignment_gain_norm_min":gains.iter().map(|&z|norm2(z).sqrt()).fold(f64::INFINITY,f64::min),
        "routing":{"nonzero_top1_correct":correct,"nonzero_channel_count":count,
            "nonzero_mean_correct_mass":mass/count as f64,"nonzero_mean_collapse_mass":collapse/(p-1) as f64,
            "zero_collapse_mass":v.routes[m]},
        "phase":{"zero_mean_cosine":average(phase_dot[0],phase_count[0]),"zero_count":phase_count[0],
            "nonzero_mean_cosine":average(phase_dot[1],phase_count[1]),"nonzero_count":phase_count[1]},
        "transported_character_cosine":(energy>0.).then(||action_dot/energy)
    }))
}
