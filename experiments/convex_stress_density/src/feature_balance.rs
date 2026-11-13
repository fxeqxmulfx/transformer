//! CM24 / Lean 75ca9276. Observable radial balancing and explicit missing-axis
//! completion. The uniform density is feasible, not necessarily pure/optimal.
use crate::{C, Result, assignment_certificate, matrix, memory_bound};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Balanced {
    pub n: usize,
    pub rank: usize,
    pub original: Vec<C>,
    pub observed_rows: Vec<bool>,
    pub energies: Vec<f64>,
    /// None means an explicitly completed near-zero, unobserved row.
    pub scales: Vec<Option<f64>>,
    pub features: Vec<C>,
    pub uniform: Vec<C>,
}
pub fn build(b: &[C], n: usize, rank: usize, observed: &[bool]) -> Result<Balanced> {
    memory_bound(n, rank)?;
    if b.len() != n * rank
        || observed.len() != n
        || b.iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() > 1e4)
    {
        return Err("invalid bounded balancing input".into());
    }
    let energies: Vec<f64> = (0..n)
        .map(|i| (0..rank).map(|j| b[i + n * j].norm_sqr()).sum())
        .collect();
    let mut features = b.to_vec();
    let mut scales = vec![];
    for i in 0..n {
        let scale = if energies[i] <= 1e-24 {
            if observed[i] {
                return Err(format!("near_zero_observed_feature_row:{i}").into());
            }
            None
        } else {
            Some((rank as f64 / energies[i]).sqrt())
        };
        for j in 0..rank {
            features[i + n * j] = scale.map_or(C::new(1., 0.), |s| b[i + n * j] * s);
        }
        scales.push(scale);
    }
    let mut uniform = vec![C::default(); rank * rank];
    for i in 0..rank {
        uniform[i + rank * i] = C::new(1. / rank as f64, 0.);
    }
    Ok(Balanced {
        n,
        rank,
        original: b.to_vec(),
        observed_rows: observed.to_vec(),
        energies,
        scales,
        features,
        uniform,
    })
}

/// Rebuild the normalization and audit actual matrix multiplication, not a
/// presumed Fourier mode or objective optimum. Numerical f64 evidence only.
pub fn audit(out: &Balanced) -> Result<Value> {
    if build(&out.original, out.n, out.rank, &out.observed_rows)? != *out {
        return Err("feature balancing replay mismatch".into());
    }
    let (n, r) = (out.n, out.rank);
    let values = matrix::eigen(&out.uniform, r)?.values;
    let trace: C = (0..r).map(|i| out.uniform[i + r * i]).sum();
    let mut diagonal_error: f64 = 0.;
    let mut energy_error: f64 = 0.;
    for x in 0..n {
        let mut raw = C::default();
        let mut energy = 0.;
        for i in 0..r {
            let a = out.features[x + n * i];
            energy += a.re * a.re + a.im * a.im;
            for j in 0..r {
                raw += a * out.uniform[i + r * j] * out.features[x + n * j].conj();
            }
        }
        diagonal_error = diagonal_error.max((raw - 1.).norm());
        energy_error = energy_error.max((energy - r as f64).abs());
    }
    if values[0] < -1e-10
        || (trace - 1.).norm() > 1e-10
        || diagonal_error > 1e-10
        || energy_error > 1e-10
    {
        return Err("raw uniform density feasibility failed".into());
    }
    Ok(
        json!({"diagonal_error":diagonal_error,"row_energy_error":energy_error,
        "trace_error":(trace-1.).norm(),"eigenvalues":values,
        "completed_rows":out.scales.iter().enumerate().filter_map(|(i,s)|s.is_none().then_some(i)).collect::<Vec<_>>(),
        "raw_reading_products":n*r*r,"energy_terms":n*r,
        "objective_optimality_claim":false,"exact_real_arithmetic_certificate":false}),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clock {
    pub phases: Vec<usize>,
    pub missing: Vec<usize>,
}
/// Order measured phases only; fill the remaining ranks in index order.
/// This is a declared empirical convention for unseen entries, not inference.
pub fn clock(x: &[C], observed: &[bool]) -> Result<Clock> {
    if x.len() != observed.len() || !(2..=193).contains(&x.len()) {
        return Err("invalid partial clock dimensions".into());
    }
    let mut order: Vec<_> = (0..x.len()).filter(|&i| observed[i]).collect();
    if order.is_empty()
        || order
            .iter()
            .any(|&i| !x[i].re.is_finite() || !x[i].im.is_finite() || x[i].norm() < 1e-12)
    {
        return Err("invalid_observed_clock_coordinate".into());
    }
    if order.iter().enumerate().any(|(k, &i)| {
        order[..k]
            .iter()
            .any(|&j| (x[i] / x[i].norm() - x[j] / x[j].norm()).norm() < 1e-7)
    }) {
        return Err("observed_clock_collision".into());
    }
    order.sort_by(|&i, &j| x[i].arg().total_cmp(&x[j].arg()).then(i.cmp(&j)));
    let missing: Vec<_> = (0..x.len()).filter(|&i| !observed[i]).collect();
    order.extend(&missing);
    let mut phases = vec![0; x.len()];
    for (rank, &i) in order.iter().enumerate() {
        phases[i] = rank;
    }
    if !assignment_certificate::permutation(&phases, x.len()) {
        return Err("partial clock completion is not a permutation".into());
    }
    Ok(Clock { phases, missing })
}
