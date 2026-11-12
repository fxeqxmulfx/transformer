//! CM18 restriction of the CM16 splitting solver (parent a56cf4f0).
//! Every row projection now enforces the checked certified equalities in all
//! three modes. The full original-vocabulary CE and latent metric are unchanged.
//! Minibatches refresh a persistent gradient table, not an unbiased SGD estimate;
//! forward-only uses no derivatives and also confines original observed rows.
//! The proof is convexity and conditional identification of the domain, not
//! convergence of these iterations. Snapshots use a certificate-preserving anchor.
use crate::endpoint::{Endpoint, EndpointBuilder};
use crate::{Certificate, Decoder, Mode, Observation, Result, Shape, Training};
use convex_stress_paired::{
    projection::{distance, simplex_even},
    solver::latent_gradient,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Costs {
    pub row_projections: u64,
    pub spectral_projections: u64,
    pub decompositions: u64,
    pub gradient_evaluations: u64,
}

/// The mathematical instance is included so restore cannot silently switch the
/// observations, decoder, certificate or training mode. This is numerical state,
/// not proof of its provenance; endpoint replay must check the recorded updates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub shape: Shape,
    pub decoder: Decoder,
    pub observations: Vec<Observation>,
    pub certificate: Certificate,
    pub mode: String,
    pub updates: u64,
    pub costs: Costs,
    pub residual: f64,
    pub z: Vec<f64>,
    pub gradients: Vec<f64>,
}

pub struct Solver {
    endpoints: EndpointBuilder,
    observations: Vec<Observation>,
    mode: Mode,
    confined: Vec<Option<usize>>,
    fibers: Vec<Vec<usize>>,
    z: Vec<f64>,
    gradients: Vec<f64>,
    shadow: Vec<f64>,
    updates: u64,
    costs: Costs,
    residual: f64,
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Full => "full_batch",
        Mode::Mini => "minibatch",
        Mode::Forward => "forward_only",
    }
}

impl Solver {
    /// Starts with a uniform splitting point. The first row projection already
    /// enforces the certificate. Discovery costs are shared preprocessing costs
    /// and must be recorded for each mode, not hidden as a free initialization.
    pub fn new(training: &Training, certificate: &Certificate, mode: Mode) -> Result<Self> {
        let endpoints = EndpointBuilder::new(training, certificate)?;
        let shape = training.shape();
        let observations = training.unique_observations();
        let fibers = training.decoder().fibers(shape.vocab);
        let mut confined = endpoints.verified().labels().to_vec();
        if mode == Mode::Forward {
            for s in &observations {
                if confined[s.cell].is_some_and(|label| label != s.label) {
                    return Err("certificate conflicts with an observed forward row".into());
                }
                confined[s.cell] = Some(s.label);
            }
        }
        let entries = shape.cells() * shape.orbits;
        let z = vec![1. / shape.latent_states() as f64; entries];
        let mut solver = Self {
            endpoints,
            observations,
            mode,
            confined,
            fibers,
            z,
            gradients: vec![0.; entries],
            shadow: vec![],
            updates: 0,
            costs: Costs::default(),
            residual: 0.,
        };
        solver.shadow = solver.rows(&solver.z);
        Ok(solver)
    }

    fn rows(&self, point: &[f64]) -> Vec<f64> {
        let orbits = self.endpoints.anchor().shape.orbits;
        let mut out = vec![0.; point.len()];
        for (cell, (input, row)) in point
            .chunks_exact(orbits)
            .zip(out.chunks_exact_mut(orbits))
            .enumerate()
        {
            simplex_even(
                input,
                self.confined[cell].map(|y| self.fibers[y].as_slice()),
                row,
            );
        }
        out
    }

    fn refresh(&self, x: &[f64], selected: &[usize], gradients: &mut [f64]) {
        let anchor = self.endpoints.anchor();
        for &index in selected {
            let s = self.observations[index];
            let row = s.cell * anchor.shape.orbits..(s.cell + 1) * anchor.shape.orbits;
            latent_gradient(
                &anchor.decoder,
                anchor.shape.vocab,
                &x[row.clone()],
                s.label,
                &mut gradients[row],
            );
        }
    }

    /// Indices address canonical unique observations. A runner preserving an
    /// original minibatch order must map its cell IDs with `sample_index`.
    pub fn step(&mut self, selected: &[usize]) -> Result<()> {
        if selected.iter().any(|&i| i >= self.observations.len()) {
            return Err("batch index outside the canonical training observations".into());
        }
        let x = self.rows(&self.z);
        let mut gradients = self.gradients.clone();
        let mut evaluations = 0;
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..self.observations.len()).collect();
            let selected = if self.mode == Mode::Full {
                &all
            } else {
                selected
            };
            self.refresh(&x, selected, &mut gradients);
            evaluations = selected.len() as u64;
        }
        // Exactly one orbit per class: the largest nonempty fiber has two
        // latent states, hence the inherited unit step is unchanged.
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(&gradients)
            .map(|((&a, &z), &g)| 2. * a - z - g)
            .collect();
        let anchor = self.endpoints.anchor();
        let (y, decompositions) =
            self.endpoints
                .projector()
                .ball(&argument, anchor.radius, anchor.nuclear_budget)?;
        let z: Vec<_> = self
            .z
            .iter()
            .zip(x.iter().zip(&y))
            .map(|(&z, (&a, &b))| z + b - a)
            .collect();
        if z.iter().chain(&gradients).any(|v| !v.is_finite()) {
            return Err("nonfinite numerical splitting state".into());
        }
        self.residual = distance(anchor.shape, &x, &y);
        self.shadow = self.rows(&z);
        self.z = z;
        self.gradients = gradients;
        self.updates += 1;
        self.costs.row_projections += 2;
        self.costs.spectral_projections += 1;
        self.costs.decompositions += decompositions;
        self.costs.gradient_evaluations += evaluations;
        Ok(())
    }

    pub fn sample_index(&self, cell: usize) -> Option<usize> {
        self.observations
            .binary_search_by_key(&cell, |s| s.cell)
            .ok()
    }

    pub fn snapshot(&self, repair_rounds: usize) -> Result<Endpoint> {
        self.endpoints.build(&self.shadow, repair_rounds)
    }

    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Fresh full-gradient fixed-point diagnostic, with its own work ledger.
    /// In forward-only mode it computes no gradient. It does not alter state.
    pub fn full_residual(&self) -> Result<(f64, Costs)> {
        let x = self.rows(&self.z);
        let mut g = vec![0.; x.len()];
        let mut evaluations = 0;
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..self.observations.len()).collect();
            self.refresh(&x, &all, &mut g);
            evaluations = all.len() as u64;
        }
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(&g)
            .map(|((&a, &z), &g)| 2. * a - z - g)
            .collect();
        let anchor = self.endpoints.anchor();
        let (y, decompositions) =
            self.endpoints
                .projector()
                .ball(&argument, anchor.radius, anchor.nuclear_budget)?;
        Ok((
            distance(anchor.shape, &x, &y),
            Costs {
                row_projections: 1,
                spectral_projections: 1,
                decompositions,
                gradient_evaluations: evaluations,
            },
        ))
    }

    pub fn state(&self) -> State {
        let anchor = self.endpoints.anchor();
        State {
            shape: anchor.shape,
            decoder: anchor.decoder.clone(),
            observations: self.observations.clone(),
            certificate: self.endpoints.verified().certificate().clone(),
            mode: mode_name(self.mode).into(),
            updates: self.updates,
            costs: self.costs,
            residual: self.residual,
            z: self.z.clone(),
            gradients: self.gradients.clone(),
        }
    }

    pub fn restore(&mut self, state: State) -> Result<()> {
        let anchor = self.endpoints.anchor();
        let lengths = anchor.shape.cells() * anchor.shape.orbits;
        if state.shape != anchor.shape
            || state.decoder != anchor.decoder
            || state.observations != self.observations
            || state.certificate != *self.endpoints.verified().certificate()
            || state.mode != mode_name(self.mode)
            || state.z.len() != lengths
            || state.gradients.len() != lengths
            || !state.residual.is_finite()
            || state.residual < 0.
            || state
                .z
                .iter()
                .chain(&state.gradients)
                .any(|v| !v.is_finite())
            || state.updates.checked_mul(2) != Some(state.costs.row_projections)
            || state.costs.spectral_projections != state.updates
            || (self.mode == Mode::Forward
                && (state.costs.gradient_evaluations != 0
                    || state.gradients.iter().any(|&g| g != 0.)))
            || (self.mode == Mode::Full
                && state.updates.checked_mul(self.observations.len() as u64)
                    != Some(state.costs.gradient_evaluations))
        {
            return Err("saved state does not match this certified solver".into());
        }
        self.shadow = self.rows(&state.z);
        self.z = state.z;
        self.gradients = state.gradients;
        self.updates = state.updates;
        self.costs = state.costs;
        self.residual = state.residual;
        Ok(())
    }
}
