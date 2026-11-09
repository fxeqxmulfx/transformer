//! FT3: signed contrast intervals before final RMS, with independent allowances.
use crate::{
    algebra::linear,
    domain::{
        Result,
        model::{E, W},
    },
    forecast_jet::{EPS, norm},
    inputs::{hash, require},
    network::Network,
    query_step::{self, Interval},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
pub const TRACKS: usize = 6;
pub const NAMES: [&str; TRACKS] = [
    "legacy",
    "gram",
    "local-rms",
    "combined",
    "directional",
    "directional-raw",
];
pub struct Frozen {
    pub network: Network,
    pub gaps: [Vec<[f64; 2]>; TRACKS],
    pub outcomes: [Vec<Interval>; TRACKS],
}
pub fn quotient(lo: f64, hi: f64, den_lo: f64, den_hi: f64) -> [f64; 2] {
    assert!(den_lo > 0. && den_hi >= den_lo && lo <= hi);
    [
        lo / if lo >= 0. { den_hi } else { den_lo },
        hi / if hi >= 0. { den_lo } else { den_hi },
    ]
}
impl Frozen {
    /// Retained f64 buffer lengths and capacities, excluding allocator metadata,
    /// weights, temporary preparation buffers and the later independent audit.
    pub fn numeric_storage(&self) -> (usize, usize) {
        let mut vectors = Vec::new();
        for t in [&self.network.output, &self.network.pre] {
            vectors.extend([&t.base.a, &t.base.v, &t.base.r, &t.defects]);
            vectors.extend(t.radii.iter());
            vectors.extend(t.gains.iter());
        }
        for stage in &self.network.stages {
            vectors.extend([&stage.center, &stage.defects]);
            vectors.extend(stage.radii.iter());
            vectors.extend(stage.gains.iter());
        }
        let used = vectors.iter().map(|v| v.len()).sum::<usize>()
            + self.gaps.iter().map(|v| 2 * v.len()).sum::<usize>()
            + self.outcomes.iter().map(|v| 4 * v.len()).sum::<usize>();
        let capacity = vectors.iter().map(|v| v.capacity()).sum::<usize>()
            + self.gaps.iter().map(|v| 2 * v.capacity()).sum::<usize>()
            + self
                .outcomes
                .iter()
                .map(|v| 4 * v.capacity())
                .sum::<usize>();
        (8 * used, 8 * capacity)
    }
    pub fn new(network: Network, next: &[f64], rows: &[u32]) -> Result<Self> {
        let n = rows.len() / 3;
        let center: Vec<_> = network
            .output
            .base
            .a
            .iter()
            .zip(&network.output.base.v)
            .map(|(a, v)| a + v)
            .collect();
        let pre: Vec<_> = network
            .pre
            .base
            .a
            .iter()
            .zip(&network.pre.base.v)
            .map(|(a, v)| a + v)
            .collect();
        let logits = linear(&center, n, W, &next[..E], 97);
        let numerator = linear(&pre, n, W, &next[..E], 97);
        let contrasts: Vec<_> = (0..97)
            .flat_map(|t| {
                (0..97).map(move |c| {
                    (0..W)
                        .map(|k| (next[c * W + k] - next[t * W + k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
            })
            .collect();
        let mut gaps: [Vec<[f64; 2]>; TRACKS] = std::array::from_fn(|_| Vec::with_capacity(n * 97));
        let mut outcomes: [Vec<Interval>; TRACKS] = std::array::from_fn(|_| Vec::with_capacity(n));
        for (q, row) in rows.as_chunks::<3>().0.iter().enumerate() {
            let target = row[2] as usize;
            let length = norm(&pre[q * W..(q + 1) * W]);
            let r = network.pre.radii[3][q];
            let den_lo = ((length - r).max(0.).powi(2) / W as f64 + EPS).sqrt();
            let den_hi = ((length + r).powi(2) / W as f64 + EPS).sqrt();
            for c in 0..97 {
                let contrast = contrasts[target * 97 + c];
                let predicted = logits[q * 97 + c] - logits[q * 97 + target];
                let allowance = contrast * 1e-3 + 2e-3;
                for (m, track) in gaps.iter_mut().enumerate().take(4) {
                    let radius = contrast * (network.output.radii[m][q] + 1e-3) + 2e-3;
                    track.push(if c == target {
                        [0., 0.]
                    } else {
                        [predicted - radius, predicted + radius]
                    });
                }
                let num = numerator[q * 97 + c] - numerator[q * 97 + target];
                let raw = quotient(num - contrast * r, num + contrast * r, den_lo, den_hi);
                let raw = if c == target {
                    [0., 0.]
                } else {
                    [raw[0] - allowance, raw[1] + allowance]
                };
                let combined = gaps[3][q * 97 + c];
                gaps[4].push([raw[0].max(combined[0]), raw[1].min(combined[1])]);
                gaps[5].push(raw);
            }
            for m in 0..TRACKS {
                let lower: Vec<_> = gaps[m][q * 97..(q + 1) * 97].iter().map(|b| b[0]).collect();
                let upper: Vec<_> = gaps[m][q * 97..(q + 1) * 97].iter().map(|b| b[1]).collect();
                outcomes[m].push(query_step::interval(&lower, &upper, target));
            }
        }
        require(
            gaps.iter()
                .flatten()
                .all(|b| b[0].is_finite() && b[1].is_finite() && b[0] <= b[1]),
            "invalid frozen interval",
        )?;
        Ok(Self {
            network,
            gaps,
            outcomes,
        })
    }
    pub fn persist(&self, path: &Path) -> Result<Value> {
        let file = File::create_new(path)?;
        let mut file = BufWriter::new(file);
        let n = self.network.output.base.rows();
        for word in [1u64, n as u64, W as u64, TRACKS as u64] {
            file.write_all(&word.to_le_bytes())?;
        }
        for t in [&self.network.output, &self.network.pre] {
            for (a, v) in t.base.a.iter().zip(&t.base.v) {
                file.write_all(&(a + v).to_le_bytes())?;
            }
            for r in &t.radii {
                for x in r {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
        }
        for track in &self.gaps {
            for b in track {
                for x in b {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
        }
        file.flush()?;
        file.get_ref().sync_all()?;
        let bytes = fs::read(path)?;
        let stages: Vec<_> = self
            .network
            .stages
            .iter()
            .map(|s| {
                let bytes: Vec<_> = s
                    .center
                    .iter()
                    .chain(s.radii.iter().flatten())
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                json!({"name":s.name,"width":s.width,"center_and_radii_sha256":hash(&bytes)})
            })
            .collect();
        Ok(
            json!({"sha256":hash(&bytes),"bytes":bytes.len(),"rows":n,"tracks":NAMES,
            "format":"little endian u64 version,rows,width,tracks; final then pre-RMS f64 center and four radius vectors; each track row-major 97 pairs of f64 lower/upper gaps",
            "stage_hashes":stages,"frozen_before_updated_forward":true}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_quotients_cover_positive_negative_zero_and_crossing_intervals() {
        for (lo, hi) in [(-3., -1.), (0., 0.), (-2., 3.), (1., 4.)] {
            let out = quotient(lo, hi, 0.5, 2.);
            for i in 0..21 {
                for j in 0..21 {
                    let value = (lo + (hi - lo) * i as f64 / 20.) / (0.5 + 1.5 * j as f64 / 20.);
                    assert!(out[0] <= value && value <= out[1]);
                }
            }
        }
    }
}
