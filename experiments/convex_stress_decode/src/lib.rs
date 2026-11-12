//! CM15 after Lean b73b3e3f: fixed decoding of bounded latent phase candidates.
//! The saved endpoint is a compact one-hot vertex of the proved convex family.
//! Proposal search is heuristic; exact observed fit alone certifies ordinary CE
//! optimality, never an unseen answer. Inference reads the saved output table.
pub use convex_stress_relabel::{C, Result, Sample, linalg, loss, metrics, random};
use serde::{Deserialize, Serialize};
pub mod assignment;
pub mod cohort;
pub mod search;

pub fn memory_bound(n: usize, q: usize, vocab: usize) -> Result<u64> {
    if [n, q, vocab].iter().any(|&x| !(2..=193).contains(&x)) {
        return Err("decoded dimensions must lie between two and 193".into());
    }
    // Retain the conservative established dense-solver bound even though this
    // constructor and its one-frequency-at-a-time auditor store only O(N^2).
    let bytes = convex_stress_relabel::memory_bound(n.max(q).max(vocab))?;
    if bytes > 2 * 1024 * 1024 * 1024 {
        return Err("decoded buffer bound exceeds two GiB".into());
    }
    Ok(bytes)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub n: usize,
    pub q: usize,
    pub vocab: usize,
    pub row_phases: Vec<usize>,
    pub column_phases: Vec<usize>,
    pub decoder: Vec<usize>,
    /// Column-major one-hot latent probability table, stored by its nonzero indices.
    pub latent_labels: Vec<usize>,
    /// Materialized original-class table; the only learned values read at inference.
    pub output_labels: Vec<usize>,
}

impl Model {
    pub fn construct(
        vocab: usize,
        row_phases: Vec<usize>,
        column_phases: Vec<usize>,
        decoder: Vec<usize>,
    ) -> Result<Self> {
        let (n, q) = (row_phases.len(), decoder.len());
        memory_bound(n, q, vocab)?;
        if column_phases.len() != n
            || row_phases.iter().chain(&column_phases).any(|&x| x >= q)
            || decoder.iter().any(|&x| x >= vocab)
        {
            return Err("invalid latent phases or decoder".into());
        }
        let mut latent_labels = Vec::with_capacity(n * n);
        let mut output_labels = Vec::with_capacity(n * n);
        for &column in &column_phases {
            for &row in &row_phases {
                let latent = (row + column) % q;
                latent_labels.push(latent);
                output_labels.push(decoder[latent]);
            }
        }
        Ok(Self {
            n,
            q,
            vocab,
            row_phases,
            column_phases,
            decoder,
            latent_labels,
            output_labels,
        })
    }

    pub fn validate_shape(&self) -> Result<()> {
        memory_bound(self.n, self.q, self.vocab)?;
        if self.row_phases.len() != self.n
            || self.column_phases.len() != self.n
            || self.decoder.len() != self.q
            || self.latent_labels.len() != self.n * self.n
            || self.output_labels.len() != self.n * self.n
            || self
                .row_phases
                .iter()
                .chain(&self.column_phases)
                .any(|&x| x >= self.q)
            || self.decoder.iter().any(|&x| x >= self.vocab)
            || self.latent_labels.iter().any(|&x| x >= self.q)
            || self.output_labels.iter().any(|&x| x >= self.vocab)
        {
            return Err("invalid decoded model shape or coordinate".into());
        }
        for cell in 0..self.n * self.n {
            if self.latent_labels[cell]
                != (self.row_phases[cell % self.n] + self.column_phases[cell / self.n]) % self.q
                || self.output_labels[cell] != self.decoder[self.latent_labels[cell]]
            {
                return Err("saved probability tables disagree with their construction".into());
            }
        }
        Ok(())
    }

    /// Reconstruct each actual latent Fourier matrix from stored table coordinates,
    /// without using the row/column factor formula, and inspect its LAPACK spectrum.
    pub fn audit(&self, observations: &[Sample]) -> Result<serde_json::Value> {
        self.validate_shape()?;
        validate_observations(self.n, self.vocab, observations)?;
        let mut norms = vec![];
        for f in 0..self.q {
            let matrix: Vec<_> = self
                .latent_labels
                .iter()
                .map(|&l| C::from_polar(1., std::f64::consts::TAU * (f * l) as f64 / self.q as f64))
                .collect();
            let sv = linalg::svd(&matrix, self.n)?.values;
            let nuclear = sv.iter().sum::<f64>();
            if sv[0] > self.n as f64 + 1e-8 || nuclear > self.n as f64 + 1e-8 {
                return Err("raw latent table violates the operator/nuclear budgets".into());
            }
            norms.push(serde_json::json!({"frequency":f,"operator":sv[0],"nuclear":nuclear}));
        }
        let mut ce = 0.;
        let mut errors = 0;
        for s in observations {
            let mut logits = vec![0.; self.vocab];
            logits[self.output_labels[s.cell]] = 1.;
            ce += loss(&logits, s.label);
            errors += usize::from(self.output_labels[s.cell] != s.label);
        }
        Ok(
            serde_json::json!({"latent_moment_norms":norms,"train_ce":ce/observations.len() as f64,
            "training_errors":errors,"exact_observed_fit":errors==0,"training_observations":observations.len(),
            "original_vocabulary":self.vocab,"spectral_bounds_apply_to":"latent_moments",
            "exact_real_arithmetic_certificate":false}),
        )
    }
}

impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(
            prefix.len() >= 2,
            "decoded pair predictor needs two visible tokens"
        );
        let (row, column) = (prefix[prefix.len() - 1], prefix[prefix.len() - 2]);
        assert!(row >= 0 && column >= 0 && (row as usize) < self.n && (column as usize) < self.n);
        let mut out = vec![0.; self.vocab];
        out[self.output_labels[row as usize + self.n * column as usize]] = 1.;
        out
    }
}

pub fn validate_observations(n: usize, vocab: usize, observations: &[Sample]) -> Result<()> {
    memory_bound(n, n, vocab)?;
    if observations.is_empty() || observations.len() > n * n {
        return Err("invalid observed-cell count".into());
    }
    let mut seen = vec![false; n * n];
    for s in observations {
        if s.cell >= n * n || s.label >= vocab || seen[s.cell] {
            return Err("invalid or repeated observed cell".into());
        }
        seen[s.cell] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
