//! Independent checker for CM18 `Accepts` and the least reach relation `Pinned`.
//! Neither proposed coverage flags nor proposed answers are trusted or loaded.
//! Checking uses exact integer residues; spectral feasibility is a separate check.
use crate::{Model, Result, Training};
use serde::{Deserialize, Serialize};

/// Data of Lean `DecodedRecovery.Certificate`; it contains no held-out labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Certificate {
    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub top: usize,
    pub bot: usize,
    pub left: usize,
    pub right: usize,
}

/// Only `check` constructs this type. Deserialization must return to the checker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Verified {
    certificate: Certificate,
    rows: Vec<bool>,
    columns: Vec<bool>,
    labels: Vec<Option<usize>>,
    full_training_fit: bool,
}

impl Verified {
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }

    pub fn labels(&self) -> &[Option<usize>] {
        &self.labels
    }

    pub fn covered_cells(&self) -> usize {
        self.labels.iter().filter(|label| label.is_some()).count()
    }

    pub fn full_training_fit(&self) -> bool {
        self.full_training_fit
    }

    /// A spectrally feasible anchor by CM16 `pairedPhase_budget`, with exactly
    /// the certified output rows. The phase values outside coverage are arbitrary;
    /// this anchor need not fit the other observations. Native SVD checks remain
    /// numerical, not interval certificates of the exact-real theorem.
    pub fn anchor(&self, training: &Training) -> Result<Model> {
        // Bind the certificate to these observations and this decoder again.
        if check(training, &self.certificate)? != *self {
            return Err("verified certificate belongs to different training data".into());
        }
        let shape = training.shape();
        let q = shape.latent_states();
        let mut half = vec![0.; shape.cells() * shape.orbits];
        for j in 0..shape.side {
            for i in 0..shape.side {
                let z = (self.certificate.row[i] + self.certificate.col[j]) % q;
                let orbit = z.min(q - z);
                half[(i + shape.side * j) * shape.orbits + orbit] =
                    1. / convex_stress_paired::weight(orbit);
            }
        }
        Model::new(
            shape,
            training.decoder().clone(),
            shape.side as f64,
            shape.side as f64,
            half,
        )
    }
}

fn two_distinct(mut phases: impl Iterator<Item = usize>) -> bool {
    phases
        .next()
        .is_some_and(|first| phases.any(|phase| phase != first))
}

/// Exact integer checker, following `Certificate.lean` and `Rigidity.lean`.
pub fn check(training: &Training, c: &Certificate) -> Result<Verified> {
    let n = training.shape().side;
    let q = training.shape().latent_states();
    if c.row.len() != n
        || c.col.len() != n
        || [c.top, c.bot, c.left, c.right].iter().any(|&i| i >= n)
        || c.row.iter().chain(&c.col).any(|&z| z >= q)
    {
        return Err("certificate has invalid dimensions, indices or phase residues".into());
    }
    if [c.top, c.bot].iter().any(|&i| {
        [c.left, c.right]
            .iter()
            .any(|&j| training.at(i, j).is_none())
    }) || c.row[c.top] == c.row[c.bot]
        || c.col[c.left] == c.col[c.right]
        || (c.row[c.top] + c.row[c.bot] + c.col[c.left] + c.col[c.right]).is_multiple_of(q)
    {
        return Err("certificate fails the observed rectangle conditions".into());
    }
    let mut rows = vec![false; n];
    let mut columns = vec![false; n];
    rows[c.top] = true;
    rows[c.bot] = true;
    for (j, reached) in columns.iter_mut().enumerate() {
        *reached = training.at(c.top, j).is_some() && training.at(c.bot, j).is_some();
    }
    loop {
        let mut changed = false;
        for (i, reached) in rows.iter_mut().enumerate() {
            if !*reached
                && two_distinct((0..n).filter_map(|j| {
                    (columns[j] && training.at(i, j).is_some()).then_some(c.col[j])
                }))
            {
                *reached = true;
                changed = true;
            }
        }
        for (j, reached) in columns.iter_mut().enumerate() {
            if !*reached
                && two_distinct(
                    (0..n).filter_map(|i| {
                        (rows[i] && training.at(i, j).is_some()).then_some(c.row[i])
                    }),
                )
            {
                *reached = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut labels = vec![None; n * n];
    let mut full_training_fit = true;
    for j in 0..n {
        for i in 0..n {
            let label = training.decoder().class_of_state((c.row[i] + c.col[j]) % q);
            let fits = training.at(i, j).is_none_or(|observed| observed == label);
            full_training_fit &= fits;
            if rows[i] && columns[j] {
                if !fits {
                    return Err("certificate disagrees with a reached training observation".into());
                }
                labels[i + n * j] = Some(label);
            }
        }
    }
    Ok(Verified {
        certificate: c.clone(),
        rows,
        columns,
        labels,
        full_training_fit,
    })
}
