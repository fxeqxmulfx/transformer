//! TF1–TF3: freeze training-derived endpoints before evaluating the updated model.
use crate::{
    algebra::linear,
    domain::{
        Result,
        model::{E, W},
    },
    forecast_jet::norm,
    groups::Geometry,
    inputs::{hash, require},
    intervals::Frozen,
    reconstruction::RANKS,
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
pub const METHODS: [&str; 3] = ["old_geometry", "combined_bridge", "directional_bridge"];
pub struct Bridge {
    pub labels: Vec<Option<usize>>,
    pub upper: Vec<Vec<f64>>,
    pub geometry: Vec<Value>,
    pub old_geometry_violation: f64,
}
pub fn signed_change(old: f64, next_at_old: f64, tangent: f64) -> f64 {
    next_at_old - old + tangent
}
pub fn endpoint_bridge(old_bound: f64, old_score: f64, endpoint: f64) -> f64 {
    old_bound + (endpoint - old_score)
}
impl Bridge {
    pub fn new(g: Geometry, f: &Frozen, prior: &[f64], next: &[f64]) -> Result<Self> {
        let n = g.labels.len();
        let old = linear(&f.network.output.base.a, n, W, &prior[..E], 97);
        let moved = linear(&f.network.output.base.a, n, W, &next[..E], 97);
        let tangent = linear(&f.network.output.base.v, n, W, &next[..E], 97);
        let mut upper = vec![vec![f64::INFINITY; n * 97]; RANKS.len() * METHODS.len()];
        let mut violation = 0f64;
        for (q, label) in g.labels.iter().enumerate() {
            let Some(t) = *label else {
                continue;
            };
            for c in 0..97 {
                if c == t {
                    for m in &mut upper {
                        m[q * 97 + c] = 0.;
                    }
                    continue;
                }
                let gap = old[q * 97 + c] - old[q * 97 + t];
                let delta = signed_change(
                    gap,
                    moved[q * 97 + c] - moved[q * 97 + t],
                    tangent[q * 97 + c] - tangent[q * 97 + t],
                );
                let contrast = norm(
                    &(0..W)
                        .map(|k| next[c * W + k] - next[t * W + k])
                        .collect::<Vec<_>>(),
                );
                let old_norm = norm(
                    &(0..W)
                        .map(|k| prior[c * W + k] - prior[t * W + k])
                        .collect::<Vec<_>>(),
                );
                let remainder = contrast * (f.network.output.radii[3][q] + 1e-3) + 2e-3;
                for (r, geometry) in g.upper.iter().enumerate() {
                    let baseline = geometry[q * 97 + c];
                    violation = violation.max(gap - baseline);
                    upper[r * 3][q * 97 + c] = baseline + old_norm * 1e-3 + 2e-3;
                    upper[r * 3 + 1][q * 97 + c] = baseline + delta + remainder;
                    upper[r * 3 + 2][q * 97 + c] =
                        endpoint_bridge(baseline, gap, f.gaps[4][q * 97 + c][1]);
                    require(
                        baseline + crate::query_step::tolerance(&[baseline, gap]) >= gap,
                        "old geometry failed",
                    )?;
                }
            }
        }
        require(
            upper.iter().all(|m| {
                g.labels.iter().enumerate().all(|(q, t)| {
                    t.is_none() || m[q * 97..(q + 1) * 97].iter().all(|v| v.is_finite())
                })
            }),
            "nonfinite constructed bound",
        )?;
        Ok(Self {
            labels: g.labels,
            upper,
            geometry: g.groups,
            old_geometry_violation: violation,
        })
    }
    pub fn persist(&self, path: &Path) -> Result<Value> {
        let mut out = BufWriter::new(File::create_new(path)?);
        for x in [1u64, self.labels.len() as u64, 97, self.upper.len() as u64] {
            out.write_all(&x.to_le_bytes())?;
        }
        for t in &self.labels {
            out.write_all(&(t.map_or(u64::MAX, |v| v as u64)).to_le_bytes())?;
        }
        for track in &self.upper {
            for x in track {
                out.write_all(&x.to_le_bytes())?;
            }
        }
        out.flush()?;
        out.get_ref().sync_all()?;
        let bytes = fs::read(path)?;
        Ok(
            json!({"sha256":hash(&bytes),"bytes":bytes.len(),"frozen_before_updated_forward":true,
            "ranks":RANKS,"methods":METHODS,"format":"u64 little endian version,rows,classes,tracks; rows u64 labels (MAX abstain); rank-major method-major f64 upper endpoints"}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn change_retains_readout_tangent_product_and_finite_remainder() {
        // w=2,x=2,w'=3,v=-1,e=1/2: old gap=4, predicted=3, future=9/2.
        assert_eq!(signed_change(4., 6., -3.), -1.);
        assert_eq!(4. + signed_change(4., 6., -3.) + 3. * 0.5, 4.5);
    }
    #[test]
    fn endpoint_transport_retains_training_slack() {
        assert_eq!(endpoint_bridge(-4., -5., -6.), -5.);
        assert!(endpoint_bridge(-4., -5., -6.) >= -6.);
    }
}
