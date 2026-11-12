//! General-fiber variant of the CM18 solver at 6f2efed1. Full and mini minimize
//! ordinary CE in original output coordinates. Mini refreshes one persistent
//! gradient per observed sample; repetitions retain their empirical weight.
//! Forward uses observed-label sub-simplexes and evaluates zero derivatives.
//! These iterations have no proved convergence or finite-step guarantee.
use crate::*;
use propagation::{Certificate, Verified};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Full,
    Mini,
    Forward,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub certificate: Certificate,
    pub decoder_index: usize,
    pub mode: Mode,
    pub updates: u64,
    pub costs: Costs,
    pub residual: f64,
    pub z: Vec<f64>,
    pub gradients: Vec<f64>,
    pub gradient_sum: Vec<f64>,
    pub shadow: Vec<f64>,
}

pub struct Solver {
    endpoints: endpoint::Builder,
    decoder_index: usize,
    mode: Mode,
    confined: Vec<Option<usize>>,
    fibers: Vec<Vec<usize>>,
    step_size: f64,
    z: Vec<f64>,
    gradients: Vec<f64>,
    gradient_sum: Vec<f64>,
    shadow: Vec<f64>,
    updates: u64,
    costs: Costs,
    residual: f64,
}

/// Chain rule through the fixed linear decoder, over the entire output vocabulary.
pub fn latent_gradient(
    decoder: &[usize],
    vocab: usize,
    latent: &[f64],
    label: usize,
) -> Result<Vec<f64>> {
    if vocab < 2
        || label >= vocab
        || decoder.len() != latent.len()
        || decoder.is_empty()
        || decoder.iter().any(|&y| y >= vocab)
        || latent.iter().any(|x| !x.is_finite())
    {
        return Err("invalid decoded gradient input".into());
    }
    let mut logits = vec![0.; vocab];
    for (&y, &p) in decoder.iter().zip(latent) {
        logits[y] += p;
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("decoded gradient overflow".into());
    }
    let mut g = vec![0.; vocab];
    gradient(&logits, label, &mut g);
    Ok(decoder.iter().map(|&y| g[y]).collect())
}

impl Solver {
    pub fn new(verified: &Verified, decoder_index: usize, mode: Mode) -> Result<Self> {
        let p = &verified.certificate().problem;
        let shape = Shape {
            side: p.side,
            states: p.states,
            vocab: p.vocab,
        };
        memory_bound(
            shape,
            p.observations.len(),
            verified.certificate().memory_bound,
        )?;
        let decoder = p
            .decoders
            .get(decoder_index)
            .ok_or("invalid decoder index")?;
        let mut fibers = vec![vec![]; p.vocab];
        for (z, &y) in decoder.iter().enumerate() {
            fibers[y].push(z);
        }
        let mut labels = vec![None; shape.cells()];
        let mut multiplicities = vec![0_usize; shape.cells()];
        for t in &p.observations {
            let cell = shape.cell(t.row, t.column);
            if labels[cell].is_some_and(|y| y != t.label) || fibers[t.label].is_empty() {
                return Err(
                    "conflicting observations or a label outside this decoder's image".into(),
                );
            }
            labels[cell] = Some(t.label);
            multiplicities[cell] += 1;
        }
        let mut confined = vec![None; shape.cells()];
        for i in 0..p.side {
            for j in 0..p.side {
                confined[shape.cell(i, j)] = verified.answer(i, j);
            }
        }
        if mode == Mode::Forward {
            for (cell, label) in labels.iter().enumerate() {
                if let Some(y) = label {
                    if confined[cell].is_some_and(|x| x != *y) {
                        return Err("certificate/observation conflict".into());
                    }
                    confined[cell] = Some(*y);
                }
            }
        }
        let largest = fibers.iter().map(Vec::len).max().unwrap();
        let copies = *multiplicities.iter().max().unwrap();
        // Fixed scale for the largest decoder fiber and repeated-cell weight.
        // This is a numerical recipe; no convergence claim is imported from it.
        let step_size = 1. / (largest * copies) as f64;
        let endpoints = endpoint::Builder::new(verified, decoder_index)?;
        let entries = shape.entries();
        let mut result = Self {
            endpoints,
            decoder_index,
            mode,
            confined,
            fibers,
            step_size,
            z: vec![1. / p.states as f64; entries],
            gradients: vec![0.; p.observations.len() * p.states],
            gradient_sum: vec![0.; entries],
            shadow: vec![],
            updates: 0,
            costs: Costs::default(),
            residual: 0.,
        };
        result.shadow = result.rows(&result.z);
        Ok(result)
    }
    fn rows(&self, point: &[f64]) -> Vec<f64> {
        let q = self.endpoints.anchor.shape.states;
        let mut result = vec![0.; point.len()];
        for (cell, (input, output)) in point
            .chunks_exact(q)
            .zip(result.chunks_exact_mut(q))
            .enumerate()
        {
            projection::row(
                input,
                self.confined[cell].map(|y| self.fibers[y].as_slice()),
                output,
            );
        }
        result
    }
    /// Sample indices refer to the canonical multiset saved in the certificate.
    /// All changes, including the cost ledger, are committed only on success.
    pub fn step(&mut self, selected: &[usize]) -> Result<()> {
        let p = &self.endpoints.verified.certificate().problem;
        if selected.iter().any(|&i| i >= p.observations.len()) {
            return Err("invalid minibatch index".into());
        }
        let shape = self.endpoints.anchor.shape;
        let x = self.rows(&self.z);
        let mut memory = self.gradients.clone();
        let mut g = self.gradient_sum.clone();
        let mut evaluations = 0;
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..p.observations.len()).collect();
            let selected = if self.mode == Mode::Full {
                all.as_slice()
            } else {
                selected
            };
            for &index in selected {
                let t = p.observations[index];
                let cell = shape.cell(t.row, t.column);
                let first = cell * shape.states;
                let saved = index * shape.states;
                let next = latent_gradient(
                    &self.endpoints.anchor.decoder,
                    shape.vocab,
                    &x[first..first + shape.states],
                    t.label,
                )?;
                for ((sum, old), &new) in g[first..first + shape.states]
                    .iter_mut()
                    .zip(&mut memory[saved..saved + shape.states])
                    .zip(&next)
                {
                    *sum += new - *old;
                    *old = new;
                }
            }
            evaluations = selected.len() as u64;
        }
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(&g)
            .map(|((&x, &z), &g)| 2. * x - z - self.step_size * g)
            .collect();
        let (y, decompositions) = self.endpoints.projector.ball(&argument)?;
        let z: Vec<_> = self
            .z
            .iter()
            .zip(x.iter().zip(&y))
            .map(|(&z, (&x, &y))| z + y - x)
            .collect();
        if z.iter()
            .chain(&g)
            .any(|x| !x.is_finite() || x.abs() > 1e100)
        {
            return Err("invalid numerical splitting state".into());
        }
        let costs = Costs {
            gradients: self
                .costs
                .gradients
                .checked_add(evaluations)
                .ok_or("cost overflow")?,
            row_projections: self
                .costs
                .row_projections
                .checked_add(2)
                .ok_or("cost overflow")?,
            spectral_projections: self
                .costs
                .spectral_projections
                .checked_add(1)
                .ok_or("cost overflow")?,
            decompositions: self
                .costs
                .decompositions
                .checked_add(decompositions)
                .ok_or("cost overflow")?,
        };
        let updates = self.updates.checked_add(1).ok_or("update overflow")?;
        let residual = x
            .iter()
            .zip(&y)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        self.shadow = self.rows(&z);
        self.z = z;
        self.gradients = memory;
        self.gradient_sum = g;
        self.costs = costs;
        self.updates = updates;
        self.residual = residual;
        Ok(())
    }
    pub fn snapshot(&mut self, rounds: usize) -> Result<endpoint::Endpoint> {
        self.endpoints.build(&self.shadow, rounds)
    }
    pub fn state(&self) -> State {
        State {
            certificate: self.endpoints.verified.certificate().clone(),
            decoder_index: self.decoder_index,
            mode: self.mode,
            updates: self.updates,
            costs: self.costs,
            residual: self.residual,
            z: self.z.clone(),
            gradients: self.gradients.clone(),
            gradient_sum: self.gradient_sum.clone(),
            shadow: self.shadow.clone(),
        }
    }
    pub fn restore(
        verified: &Verified,
        decoder_index: usize,
        mode: Mode,
        state: &State,
    ) -> Result<Self> {
        let p = &verified.certificate().problem;
        let shape = Shape {
            side: p.side,
            states: p.states,
            vocab: p.vocab,
        };
        memory_bound(
            shape,
            p.observations.len(),
            verified.certificate().memory_bound,
        )?;
        if state.certificate != *verified.certificate()
            || state.decoder_index != decoder_index
            || state.mode != mode
            || state.z.len() != shape.entries()
            || state.shadow.len() != shape.entries()
            || state.gradient_sum.len() != shape.entries()
            || state.gradients.len() != p.observations.len() * p.states
            || !state.residual.is_finite()
            || state.residual < 0.
            || state
                .z
                .iter()
                .chain(&state.shadow)
                .chain(&state.gradient_sum)
                .chain(&state.gradients)
                .any(|x| !x.is_finite() || x.abs() > 1e100)
            || state.gradients.iter().any(|g| g.abs() > 1. + 1e-12)
            || state.updates.checked_mul(2) != Some(state.costs.row_projections)
            || state.updates != state.costs.spectral_projections
            || state
                .updates
                .checked_mul((p.states / 2) as u64)
                .is_none_or(|max| state.costs.decompositions > max)
            || (mode == Mode::Forward
                && (state.costs.gradients != 0 || state.gradients.iter().any(|&g| g != 0.)))
            || (mode == Mode::Full
                && state.updates.checked_mul(p.observations.len() as u64)
                    != Some(state.costs.gradients))
        {
            return Err("state dimensions, instance, numbers or ledger changed".into());
        }
        let mut expected = vec![0.; shape.entries()];
        for (index, t) in p.observations.iter().enumerate() {
            let first = shape.cell(t.row, t.column) * p.states;
            for z in 0..p.states {
                expected[first + z] += state.gradients[index * p.states + z];
            }
        }
        if expected
            .iter()
            .zip(&state.gradient_sum)
            .any(|(a, b)| (a - b).abs() > 1e-10 * (1. + a.abs()))
        {
            return Err("persistent gradient sum disagrees with sample memories".into());
        }
        let mut result = Self::new(verified, decoder_index, mode)?;
        if result.rows(&state.z) != state.shadow {
            return Err("saved shadow is not the row projection".into());
        }
        result.z = state.z.clone();
        result.shadow = state.shadow.clone();
        result.gradients = state.gradients.clone();
        result.gradient_sum = state.gradient_sum.clone();
        result.updates = state.updates;
        result.costs = state.costs;
        result.residual = state.residual;
        Ok(result)
    }
}
