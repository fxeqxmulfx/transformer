//! Frozen cheap displacement rules and the FT1/FT2 tightening extension.
use crate::{
    cache_cpu::Node,
    domain::{
        Result,
        model::{Arm, E, W},
    },
    forecast_jet::norm,
    inputs::{hash, require},
    matrix,
};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
pub struct Bound {
    pub from: String,
    pub to: String,
    pub radii: Vec<f64>,
    pub contrasts: Vec<f64>,
    pub seconds: f64,
    pub gram_seconds: f64,
    pub factors: [f64; 2],
}
pub fn normalized(radius: f64, s: f64, tight: bool) -> f64 {
    let scale = (W as f64).sqrt();
    let length = scale * s;
    let legacy = 2. / s;
    let gain = if tight && radius < length {
        legacy.min(scale / (length * (length - radius)).sqrt())
    } else {
        legacy
    };
    (gain * radius).min(2. * scale)
}
impl Bound {
    pub fn prepare(
        arm: Arm,
        old: &[f32],
        next: &[f32],
        nodes: &[Node],
        from: &str,
        to: &str,
        tight: bool,
    ) -> Self {
        let start = Instant::now();
        let prior: Vec<_> = old.iter().map(|&x| x as f64).collect();
        let next: Vec<_> = next.iter().map(|&x| x as f64).collect();
        let delta: Vec<_> = next.iter().zip(&prior).map(|(n, o)| n - o).collect();
        let h = arm.hidden();
        let off = E + 2 * h * W;
        let grams = Instant::now();
        let factors = if tight {
            [
                matrix::bound(&next[E..off], 2 * h, W).factor,
                matrix::bound(&next[off..], W, h).factor,
            ]
        } else {
            [norm(&next[E..off]), norm(&next[off..])]
        };
        let gram_seconds = if tight {
            grams.elapsed().as_secs_f64()
        } else {
            0.
        };
        let d = [norm(&delta[E..off]), norm(&delta[off..])];
        let contrasts = (0..97)
            .flat_map(|a| {
                let p = &next;
                (0..97).map(move |b| {
                    (0..W)
                        .map(|k| (p[a * W + k] - p[b * W + k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
            })
            .collect();
        let radii = nodes
            .iter()
            .enumerate()
            .map(|(q, s)| {
                let a = q / 97;
                let b = q % 97;
                let embed = (0..W)
                    .map(|k| {
                        let x = delta[a * W + k]
                            + if k % 2 == 0 {
                                -delta[b * W + k + 1]
                            } else {
                                delta[b * W + k - 1]
                            };
                        x * x
                    })
                    .sum::<f64>()
                    .sqrt();
                let x0 = normalized(embed, s.denominators[0], tight);
                let n1 = normalized(x0, s.denominators[1], tight);
                let gate = factors[0] * n1 + d[0] * s.first_input;
                let gated = if arm == Arm::Plain {
                    gate
                } else {
                    (1. + s.gate_max / 4.) * gate
                };
                let product = (s.gated_max * gate + s.value_max * gated + gated * gate)
                    * if arm == Arm::Plain || arm == Arm::Product {
                        1. / 2f64.sqrt()
                    } else {
                        1.
                    };
                let branch = factors[1] * product + d[1] * s.hidden_norm;
                normalized(
                    x0 + normalized(branch, s.denominators[2], tight),
                    s.denominators[3],
                    tight,
                )
            })
            .collect();
        Self {
            from: from.into(),
            to: to.into(),
            radii,
            contrasts,
            seconds: start.elapsed().as_secs_f64(),
            gram_seconds,
            factors,
        }
    }
    pub fn bytes(&self) -> usize {
        8 * (self.radii.len() + self.contrasts.len())
    }
    pub fn capacity(&self) -> usize {
        8 * (self.radii.capacity() + self.contrasts.capacity())
    }
    pub fn persist(&self, path: &Path) -> Result<Value> {
        require(
            self.radii.len() == 9409 && self.contrasts.len() == 97 * 97,
            "invalid bound dimensions",
        )?;
        require(
            self.radii
                .iter()
                .chain(&self.contrasts)
                .all(|x| x.is_finite() && *x >= 0.),
            "nonfinite bound",
        )?;
        let bytes: Vec<_> = self
            .radii
            .iter()
            .chain(&self.contrasts)
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let mut file = BufWriter::new(File::create_new(path)?);
        file.write_all(&bytes)?;
        file.flush()?;
        file.get_ref().sync_all()?;
        Ok(
            json!({"from":self.from,"to":self.to,"sha256":hash(&bytes),"bytes":bytes.len(),
            "format":"little endian f64: 9409 radii, then 97x97 row-major contrast norms",
            "before_updated_forward":true,"seconds":self.seconds,"gram_seconds":self.gram_seconds,"factors":self.factors}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cache_cpu, domain::model};
    #[test]
    fn nonzero_steps_and_local_normalization_fallback_are_enclosed() {
        let scale = (W as f64).sqrt();
        assert_eq!(normalized(scale, 1., true), 2. * scale);
        assert_eq!(normalized(0., 1., true), 0.);
        for arm in Arm::ALL {
            let old = model::initialize(arm, 7);
            let next: Vec<_> = old
                .iter()
                .enumerate()
                .map(|(i, x)| x + 1e-5 * ((i % 7) as f32 - 3.))
                .collect();
            let pairs: Vec<_> = (0..9409)
                .map(|q| [(q / 97) as u32, (q % 97) as u32])
                .collect();
            let p: Vec<_> = old.iter().map(|&x| x as f64).collect();
            let pn: Vec<_> = next.iter().map(|&x| x as f64).collect();
            let prior = cache_cpu::graph(arm, &p, &pairs);
            let future = cache_cpu::graph(arm, &pn, &pairs);
            for tight in [false, true] {
                let b = Bound::prepare(arm, &old, &next, &prior.nodes, "old", "new", tight);
                assert!(b.radii.iter().all(|r| *r > 0.));
                for q in [0, 1, 512, 9408] {
                    let error = (0..W)
                        .map(|k| (future.features[q * W + k] - prior.features[q * W + k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    assert!(error <= b.radii[q] + 1e-8);
                }
            }
        }
    }
}
