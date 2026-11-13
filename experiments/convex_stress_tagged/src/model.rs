//! CM25 / Lean d8efeb36: actual tagged phase coordinates. Unlike the earlier
//! permutation constructor, n and q may differ and phases may repeat. The
//! decoder/prediction layout ports convex_stress_decode at4510ee7d; dimensions
//! now allow n<=257, q<=771 and vocabulary<=1024 with streaming Fourier audits.
use crate::{C, Result, Sample, linalg, loss, metrics};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub fn memory_bound(n: usize, q: usize, vocab: usize) -> Result<u64> {
    if !(2..=257).contains(&n) || !(2..=771).contains(&q) || !(2..=1024).contains(&vocab) {
        return Err("phase dimensions exceed n257, q771, vocabulary1024 limits".into());
    }
    // Dense matrices are n by n, one frequency at a time. No n*n*q*vocab tensor.
    // Includes conservative SVD buffers and bounded integer refinement tables.
    Ok(512 * 1024 * 1024 + 1024 * (n as u64).pow(2) + 64 * (q * vocab + n * q) as u64)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub n: usize,
    pub q: usize,
    pub vocab: usize,
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub decoder: Vec<usize>,
    pub latent_labels: Vec<usize>,
    pub output_labels: Vec<usize>,
}
impl Model {
    pub fn construct(
        vocab: usize,
        rows: Vec<usize>,
        columns: Vec<usize>,
        decoder: Vec<usize>,
    ) -> Result<Self> {
        let (n, q) = (rows.len(), decoder.len());
        memory_bound(n, q, vocab)?;
        if columns.len() != n
            || rows.iter().chain(&columns).any(|&r| r >= q)
            || decoder.iter().any(|&c| c >= vocab)
        {
            return Err("invalid phase or decoder coordinates".into());
        }
        let mut latent_labels = Vec::with_capacity(n * n);
        let mut output_labels = Vec::with_capacity(n * n);
        for &s in &columns {
            for &r in &rows {
                let z = (r + s) % q;
                latent_labels.push(z);
                output_labels.push(decoder[z]);
            }
        }
        Ok(Self {
            n,
            q,
            vocab,
            rows,
            columns,
            decoder,
            latent_labels,
            output_labels,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if Self::construct(
            self.vocab,
            self.rows.clone(),
            self.columns.clone(),
            self.decoder.clone(),
        )? != *self
        {
            return Err("saved phase table differs from actual coordinates".into());
        }
        Ok(())
    }
    /// Read only the stored one-hot latent table, then independently measure all
    /// Fourier matrices with LAPACK. Original classes stay in the CE denominator.
    pub fn audit(&self, samples: &[Sample]) -> Result<Value> {
        self.validate()?;
        crate::validate_observations(self.n, self.vocab, samples)?;
        let mut norms = vec![];
        for f in 0..self.q {
            let a: Vec<_> = self
                .latent_labels
                .iter()
                .map(|&z| C::from_polar(1., std::f64::consts::TAU * (f * z) as f64 / self.q as f64))
                .collect();
            let sv = linalg::svd(&a, self.n)?.values;
            let nuclear = sv.iter().sum::<f64>();
            if sv[0] > self.n as f64 + 1e-7 || nuclear > self.n as f64 + 1e-7 {
                return Err("raw actual latent moment exceeds its radius-N budget".into());
            }
            norms.push(json!({"frequency":f,"operator":sv[0],"nuclear":nuclear}));
        }
        let (mut ce, mut errors) = (0., 0);
        for s in samples {
            let mut logits = vec![0.; self.vocab];
            logits[self.output_labels[s.cell]] = 1.;
            ce += loss(&logits, s.label);
            errors += usize::from(self.output_labels[s.cell] != s.label);
        }
        let ideal = (std::f64::consts::E + self.vocab as f64 - 1.).ln() - 1.;
        if (ce / samples.len() as f64 - ideal - errors as f64 / samples.len() as f64).abs() > 1e-10
        {
            return Err("raw ordinary CE/error identity failed".into());
        }
        Ok(json!({"latent_moment_norms":norms,"training_errors":errors,
            "training_observations":samples.len(),"train_ce":ce/samples.len() as f64,
            "ideal_ce":ideal,"exact_observed_fit":errors==0,"vocabulary":self.vocab,
            "n":self.n,"q":self.q,"raw_frequency_audits":self.q,
            "buffer_bound_bytes":memory_bound(self.n,self.q,self.vocab)?,
            "spectral_bounds_apply_to":"latent_moments","exact_real_arithmetic_certificate":false}))
    }
    /// Map original categories to an interior category, or mark a constant
    /// strip with None. The full output and latent tables are materialized.
    pub fn lift(
        &self,
        constant: usize,
        rows: &[Option<usize>],
        columns: &[Option<usize>],
    ) -> Result<Self> {
        self.validate()?;
        if self.q > 257
            || constant >= self.vocab
            || rows.len() != columns.len()
            || rows.iter().chain(columns).flatten().any(|&i| i >= self.n)
        {
            return Err("invalid bounded categorical tag lift".into());
        }
        memory_bound(rows.len(), 3 * self.q, self.vocab)?;
        let r = rows
            .iter()
            .map(|i| i.map_or(1, |i| 3 * self.rows[i]))
            .collect();
        let s = columns
            .iter()
            .map(|j| j.map_or(1, |j| 3 * self.columns[j]))
            .collect();
        let d = (0..3 * self.q)
            .map(|z| {
                if z % 3 == 0 {
                    self.decoder[z / 3]
                } else {
                    constant
                }
            })
            .collect();
        let out = Self::construct(self.vocab, r, s, d)?;
        for (j, col) in columns.iter().enumerate() {
            for (i, row) in rows.iter().enumerate() {
                let expected = match (row, col) {
                    (Some(r), Some(c)) => self.output_labels[r + self.n * c],
                    _ => constant,
                };
                if out.output_labels[i + out.n * j] != expected {
                    return Err("actual tag decoding identity failed".into());
                }
            }
        }
        Ok(out)
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.len() >= 2);
        let (r, c) = (prefix[prefix.len() - 1], prefix[prefix.len() - 2]);
        assert!(r >= 0 && c >= 0 && (r as usize) < self.n && (c as usize) < self.n);
        let mut out = vec![0.; self.vocab];
        out[self.output_labels[r as usize + self.n * c as usize]] = 1.;
        out
    }
}
