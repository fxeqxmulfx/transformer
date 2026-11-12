//! CM16 after Lean 3447d663: equal probabilities on opposite latent states of Z/q.
//! The learned coordinates are a latent probability table that is even in its
//! latent state; a fixed even decoder sums it into the original vocabulary.
//! Lean proves convexity of this domain, feasibility of a symmetric phase
//! witness and the observed-CE certificate. It proves no convergence of the
//! iterations below and no recovery of unseen labels.
//!
//! Three sizes are separate everywhere: the categorical input side `side`, the
//! latent orbits `orbits` (latent states q = 2 * orbits - 1) and the original
//! output classes `vocab`. An even table is stored by its half: one value per
//! orbit {c, -c}, c = 0..orbits-1, which is the probability of each state of it.
pub use convex_stress_relabel::{C, Mode, Result, Sample, linalg, metrics};
use serde::{Deserialize, Serialize};
pub mod cohort;
pub mod projection;
pub mod solver;
pub mod transform;

/// The frozen cap on each of the three sizes, as in the preceding cohorts.
pub const MAXIMUM_DIMENSION: usize = 193;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub side: usize,
    pub orbits: usize,
    pub vocab: usize,
}
impl Shape {
    pub fn latent_states(&self) -> usize {
        2 * self.orbits - 1
    }
    pub fn cells(&self) -> usize {
        self.side * self.side
    }
    pub fn validate(&self) -> Result<()> {
        if self.orbits < 2
            || [self.side, self.latent_states(), self.vocab]
                .iter()
                .any(|&x| !(2..=MAXIMUM_DIMENSION).contains(&x))
        {
            return Err("input side, latent states and vocabulary must lie in 2..=193".into());
        }
        Ok(())
    }
}

/// Number of latent states in the orbit `c`: the state 0 alone, or {c, -c}.
pub fn weight(orbit: usize) -> f64 {
    if orbit == 0 { 1. } else { 2. }
}

/// An upper bound on the explicit numerical buffers of one solver, replay or
/// audit process, computed from the sizes alone before any buffer exists:
/// twelve latent halves and four output tables of f64, eight complex side x side
/// matrices, two orbit x orbit cosine matrices, the JSON text of one model and
/// one splitting state at 32 bytes per number, and 64 MiB of overhead.
pub fn memory_bound(shape: Shape) -> Result<u64> {
    shape.validate()?;
    let (cells, orbits, vocab) = (
        shape.cells() as u64,
        shape.orbits as u64,
        shape.vocab as u64,
    );
    Ok(12 * 8 * cells * orbits
        + 4 * 8 * cells * vocab
        + 8 * 16 * cells
        + 2 * 8 * orbits * orbits
        + 32 * (2 * cells * orbits + cells * vocab)
        + 64 * 1024 * 1024)
}

/// Refuse a shape whose explicit buffer bound exceeds `limit` bytes.
pub fn preflight(shape: Shape, limit: u64) -> Result<u64> {
    let bytes = memory_bound(shape)?;
    if bytes > limit {
        return Err("explicit buffer bound exceeds the frozen memory limit".into());
    }
    Ok(bytes)
}

/// A fixed even decoder d(-z) = d(z) of Z/(2 * orbits - 1), given by the class
/// of each orbit. It is not learned and reads no input token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decoder {
    pub orbit_class: Vec<usize>,
}
impl Decoder {
    /// Lean `foldDecoder`: the orbit {c, -c} decodes to the class c. It depends
    /// on the size of the vocabulary and the given order of its classes only.
    pub fn fold(vocab: usize) -> Self {
        Self {
            orbit_class: (0..vocab).collect(),
        }
    }
    pub fn validate(&self, shape: Shape) -> Result<()> {
        shape.validate()?;
        if self.orbit_class.len() != shape.orbits
            || self.orbit_class.iter().any(|&c| c >= shape.vocab)
        {
            return Err("decoder does not map the latent orbits into the vocabulary".into());
        }
        Ok(())
    }
    /// The class of the latent state `z` in 0..q, by its orbit.
    pub fn class_of_state(&self, z: usize) -> usize {
        let q = 2 * self.orbit_class.len() - 1;
        self.orbit_class[z.min(q - z)]
    }
    /// The orbits of each class; a class may have none.
    pub fn fibers(&self, vocab: usize) -> Vec<Vec<usize>> {
        let mut out = vec![vec![]; vocab];
        for (orbit, &class) in self.orbit_class.iter().enumerate() {
            out[class].push(orbit);
        }
        out
    }
    /// Sum the even latent table over the fibers: the original-vocabulary table.
    pub fn decode(&self, shape: Shape, latent_half: &[f64]) -> Vec<f64> {
        let mut out = vec![0.; shape.cells() * shape.vocab];
        for (row, half) in out
            .chunks_exact_mut(shape.vocab)
            .zip(latent_half.chunks_exact(shape.orbits))
        {
            for (orbit, &p) in half.iter().enumerate() {
                row[self.orbit_class[orbit]] += weight(orbit) * p;
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub shape: Shape,
    pub decoder: Decoder,
    pub radius: f64,
    pub nuclear_budget: f64,
    /// Cell-major halves of the even latent probability table: the entry
    /// `cell * orbits + c` is the probability of the state c and of the state -c.
    pub latent_half: Vec<f64>,
    /// The decoded table in original class order; the only values read at inference.
    pub output: Vec<f64>,
}
impl Model {
    pub fn new(
        shape: Shape,
        decoder: Decoder,
        radius: f64,
        nuclear_budget: f64,
        latent_half: Vec<f64>,
    ) -> Result<Self> {
        decoder.validate(shape)?;
        if latent_half.len() != shape.cells() * shape.orbits {
            return Err("latent table has the wrong size".into());
        }
        let output = decoder.decode(shape, &latent_half);
        let model = Self {
            shape,
            decoder,
            radius,
            nuclear_budget,
            latent_half,
            output,
        };
        model.validate_shape()?;
        Ok(model)
    }
    /// Every latent state has the same probability; all nonconstant moments vanish.
    pub fn uniform(
        shape: Shape,
        decoder: Decoder,
        radius: f64,
        nuclear_budget: f64,
    ) -> Result<Self> {
        shape.validate()?;
        let value = 1. / shape.latent_states() as f64;
        Self::new(
            shape,
            decoder,
            radius,
            nuclear_budget,
            vec![value; shape.cells() * shape.orbits],
        )
    }
    /// The last token is the row and the previous token the column, as in the
    /// preceding pair models. Tokens are indices; no value of them is computed with.
    pub fn cell(&self, prefix: &[i64]) -> usize {
        assert!(
            prefix.len() >= 2,
            "paired predictor needs two visible tokens"
        );
        let (row, column) = (prefix[prefix.len() - 1], prefix[prefix.len() - 2]);
        assert!(row >= 0 && column >= 0);
        let (row, column) = (row as usize, column as usize);
        assert!(row < self.shape.side && column < self.shape.side);
        row + self.shape.side * column
    }
    pub fn validate_shape(&self) -> Result<()> {
        self.decoder.validate(self.shape)?;
        let (orbits, vocab, cells) = (self.shape.orbits, self.shape.vocab, self.shape.cells());
        if !self.radius.is_finite()
            || self.radius <= 0.
            || !self.nuclear_budget.is_finite()
            || self.nuclear_budget <= 0.
            || self.latent_half.len() != cells * orbits
            || self.output.len() != cells * vocab
            || self
                .latent_half
                .iter()
                .any(|p| !p.is_finite() || *p < -1e-10 || *p > 1. + 1e-10)
            || self.latent_half.chunks_exact(orbits).any(|half| {
                let mass: f64 = half.iter().enumerate().map(|(c, p)| weight(c) * p).sum();
                (mass - 1.).abs() > 1e-10
            })
            || self.output != self.decoder.decode(self.shape, &self.latent_half)
        {
            return Err("invalid saved latent or decoded probability table".into());
        }
        Ok(())
    }
    pub fn validate(&self, projector: &projection::Projector) -> Result<Vec<projection::Norms>> {
        self.validate_shape()?;
        let norms = projector.norms(&self.latent_half)?;
        if norms.iter().any(|s| {
            s.operator > self.radius * (1. + 1e-9) || s.nuclear > self.nuclear_budget * (1. + 1e-9)
        }) {
            return Err("saved latent table violates an operator or nuclear bound".into());
        }
        Ok(norms)
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.shape.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let start = self.cell(prefix) * self.shape.vocab;
        self.output[start..start + self.shape.vocab].to_vec()
    }
}

/// Observed cells are distinct, inside the table, and labelled in the vocabulary.
pub fn validate_observations(shape: Shape, observations: &[Sample]) -> Result<()> {
    shape.validate()?;
    if observations.is_empty() || observations.len() > shape.cells() {
        return Err("invalid observed-cell count".into());
    }
    let mut seen = vec![false; shape.cells()];
    for s in observations {
        if s.cell >= shape.cells() || s.label >= shape.vocab || seen[s.cell] {
            return Err("invalid or repeated observed cell".into());
        }
        seen[s.cell] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
