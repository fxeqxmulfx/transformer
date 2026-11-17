//! Both modes use ordinary CE over all original classes. The finite-query
//! path calls only forward evaluation on feasible points.
use crate::{
    Result,
    model::{DIM, Model, dot, softmax},
};
use baseline::channel::cross_entropy;

pub struct Sample {
    pub prefix: Vec<usize>,
    pub target: usize,
}
pub struct Batch<'a> {
    pub records: Vec<&'a Sample>,
    lengths: Vec<usize>,
}
impl<'a> Batch<'a> {
    pub fn new(records: Vec<&'a Sample>) -> Result<Self> {
        if records.is_empty() {
            return Err("empty supervised minibatch".into());
        }
        let mut lengths: Vec<_> = records
            .iter()
            .map(|s| s.prefix.len().saturating_sub(1))
            .collect();
        lengths.sort_unstable();
        lengths.dedup();
        Ok(Self { records, lengths })
    }
    fn links(&self, model: &Model) -> Vec<Vec<f64>> {
        let mut cached = vec![vec![]; self.lengths.last().unwrap() + 1];
        for &n in &self.lengths {
            cached[n] = model.links(n);
        }
        cached
    }
    pub fn loss(&self, model: &Model) -> f64 {
        let links = self.links(model);
        self.records
            .iter()
            .map(|s| {
                let n = s.prefix.len().saturating_sub(1);
                cross_entropy(&model.pass(&s.prefix, &links[n]).logits, s.target)
            })
            .sum::<f64>()
            / self.records.len() as f64
    }
    pub fn finite(&self, model: &Model, coordinates: &[usize], h: f64) -> Vec<f64> {
        coordinates
            .iter()
            .map(|&i| {
                let plus = self.loss(&model.perturb(i, h));
                let minus = self.loss(&model.perturb(i, -h));
                (plus - minus) / (2. * h)
            })
            .collect()
    }
    pub fn analytic(&self, model: &Model) -> Vec<f64> {
        let links = self.links(model);
        let mut gradient = vec![0.; model.weights.len()];
        for sample in &self.records {
            let Some((&query, past)) = sample.prefix.split_last() else {
                continue;
            };
            let n = past.len();
            if n == 0 {
                continue;
            }
            let relation = &links[n];
            let pass = model.pass(&sample.prefix, relation);
            let mut residual = pass.logits;
            softmax(&mut residual);
            residual[sample.target] -= 1.;
            let mut dh = [0.; DIM];
            for (c, &res) in residual.iter().enumerate() {
                let coeff = model.capacity * res;
                let start = model.table_len() + c * DIM;
                add_scaled(&mut gradient[start..start + DIM], coeff, &pass.state);
                add_scaled(&mut dh, coeff, model.value(c));
            }
            let mut value_signal = Vec::with_capacity(n);
            for (&o, &mass) in past.iter().zip(&pass.mass) {
                let start = model.table_len() + o * DIM;
                add_scaled(&mut gradient[start..start + DIM], mass, &dh);
                value_signal.push(dot(&dh, model.value(o)));
            }
            let key_signal: Vec<_> = relation
                .chunks_exact(n)
                .map(|row| dot(row, &value_signal))
                .collect();
            let mean = dot(&pass.attention, &key_signal);
            for (j, &o) in past.iter().enumerate() {
                let a = pass.attention[j];
                let ds = a * (key_signal[j] - mean);
                gradient[model.scale_index()] += ds * dot(model.key(query), model.key(o));
                add_scaled(
                    &mut gradient[query * DIM..(query + 1) * DIM],
                    ds * model.scale(),
                    model.key(o),
                );
                add_scaled(
                    &mut gradient[o * DIM..(o + 1) * DIM],
                    ds * model.scale(),
                    model.key(query),
                );
                for (k, &signal) in value_signal.iter().enumerate() {
                    gradient[model.relative_index(j, k)] +=
                        a * relation[j * n + k] * (signal - key_signal[j]);
                }
            }
        }
        for g in &mut gradient {
            *g /= self.records.len() as f64;
        }
        gradient
    }
}

fn add_scaled(dst: &mut [f64], coeff: f64, source: &[f64]) {
    for (x, y) in dst.iter_mut().zip(source) {
        *x += coeff * y;
    }
}
