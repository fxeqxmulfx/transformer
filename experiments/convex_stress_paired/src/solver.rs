//! One spectral projection per update, retaining the splitting point z.
//! With x = P_A(z): z += P_B(2x - z - step * g) - x, as in the CM5 solver.
//! Full: g is the gradient at x of the summed ordinary CE of the decoded rows.
//! Mini: only the selected observations refresh their part of a retained g
//! (incremental aggregated gradient, not unbiased SGD).
//! Forward: A also confines each observed row to the fiber of its label and
//! g = 0, so no derivative is evaluated.
//! The gradient with respect to the latent table is the class gradient read
//! through the fixed decoder: g(z) = (softmax(Q) - e_y)(d(z)), with Q the
//! decoded row over the whole original vocabulary.
//! No convergence and no rate are claimed for these iterations.
use crate::{
    Decoder, Mode, Model, Result, Sample,
    projection::{Norms, Projector, distance, simplex_even},
    weight,
};
use convex_stress_relabel::gradient;
use serde::{Deserialize, Serialize};

/// Work actually done by `Solver::step`, excluding snapshots and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Costs {
    pub row_projections: u64,
    pub spectral_projections: u64,
    pub decompositions: u64,
    pub gradient_evaluations: u64,
}

/// The ordinary-CE gradient of one observed cell with respect to its latent
/// row, one value per orbit: the class gradient at the class of the orbit.
pub fn latent_gradient(
    decoder: &Decoder,
    vocab: usize,
    half: &[f64],
    label: usize,
    out: &mut [f64],
) {
    let mut decoded = vec![0.; vocab];
    for (orbit, &p) in half.iter().enumerate() {
        decoded[decoder.orbit_class[orbit]] += weight(orbit) * p;
    }
    let mut class = vec![0.; vocab];
    gradient(&decoded, label, &mut class);
    for (g, &k) in out.iter_mut().zip(&decoder.orbit_class) {
        *g = class[k];
    }
}

pub struct Solver {
    pub projector: Projector,
    pub decoder: Decoder,
    pub shadow: Vec<f64>,
    pub residual: f64,
    pub radius: f64,
    pub nuclear_budget: f64,
    pub mode: Mode,
    /// The CE Hessian in the logits is at most 1/2 and decoding has squared norm
    /// equal to the largest fiber, so `2 / fiber` is below twice the inverse of
    /// the gradient's Lipschitz constant; it is the unit step of the CM5 solver
    /// whenever no class has more than two latent states.
    pub step: f64,
    pub costs: Costs,
    z: Vec<f64>,
    memory: Vec<f64>,
    samples: Vec<Sample>,
    /// Forward only: the orbits to which the row of an observed cell is confined.
    confined: Vec<Option<usize>>,
    fibers: Vec<Vec<usize>>,
}
impl Solver {
    pub fn new(model: &Model, samples: Vec<Sample>, mode: Mode) -> Result<Self> {
        model.validate_shape()?;
        crate::validate_observations(model.shape, &samples)?;
        let fibers = model.decoder.fibers(model.shape.vocab);
        let largest = fibers
            .iter()
            .map(|fiber| fiber.iter().map(|&c| weight(c)).sum::<f64>())
            .fold(0., f64::max);
        let mut confined = vec![None; model.shape.cells()];
        if mode == Mode::Forward {
            for s in &samples {
                if fibers[s.label].is_empty() {
                    return Err("forward-only needs a latent state for every observed label".into());
                }
                confined[s.cell] = Some(s.label);
            }
        }
        Ok(Self {
            projector: Projector::new(model.shape)?,
            decoder: model.decoder.clone(),
            shadow: model.latent_half.clone(),
            residual: 0.,
            radius: model.radius,
            nuclear_budget: model.nuclear_budget,
            mode,
            step: (2. / largest).min(1.),
            costs: Costs::default(),
            z: model.latent_half.clone(),
            memory: vec![0.; model.latent_half.len()],
            samples,
            confined,
            fibers,
        })
    }
    fn a(&self, point: &[f64]) -> Vec<f64> {
        let orbits = self.projector.shape.orbits;
        let mut x = vec![0.; point.len()];
        for ((input, out), label) in point
            .chunks_exact(orbits)
            .zip(x.chunks_exact_mut(orbits))
            .zip(&self.confined)
        {
            simplex_even(input, label.map(|y| self.fibers[y].as_slice()), out);
        }
        x
    }
    fn gradients(&self, x: &[f64], selected: &[usize], memory: &mut [f64]) -> Result<()> {
        let shape = self.projector.shape;
        for &index in selected {
            let s = self.samples.get(index).ok_or("invalid batch index")?;
            let row = s.cell * shape.orbits..(s.cell + 1) * shape.orbits;
            latent_gradient(
                &self.decoder,
                shape.vocab,
                &x[row.clone()],
                s.label,
                &mut memory[row],
            );
        }
        Ok(())
    }
    fn argument(&self, x: &[f64], g: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.z)
            .zip(g)
            .map(|((&a, &z), &g)| 2. * a - z - self.step * g)
            .collect()
    }
    pub fn step(&mut self, selected: &[usize]) -> Result<()> {
        let x = self.a(&self.z);
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..self.samples.len()).collect();
            let selected = if self.mode == Mode::Full {
                &all
            } else {
                selected
            };
            let mut memory = std::mem::take(&mut self.memory);
            let refreshed = self.gradients(&x, selected, &mut memory);
            self.memory = memory;
            refreshed?;
            self.costs.gradient_evaluations += selected.len() as u64;
        }
        let (y, decompositions) = self.projector.ball(
            &self.argument(&x, &self.memory),
            self.radius,
            self.nuclear_budget,
        )?;
        self.residual = distance(self.projector.shape, &x, &y);
        for (z, (&a, &b)) in self.z.iter_mut().zip(x.iter().zip(&y)) {
            *z += b - a;
        }
        self.shadow = self.a(&self.z);
        self.costs.row_projections += 2;
        self.costs.spectral_projections += 1;
        self.costs.decompositions += decompositions;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<(Model, f64, Vec<Norms>)> {
        self.projector.feasible(
            &self.decoder,
            &self.shadow,
            self.radius,
            self.nuclear_budget,
        )
    }
    /// Largest deviation of a decoded observed row from the one-hot row of its label.
    pub fn fit_defect(&self, model: &Model) -> f64 {
        let vocab = model.shape.vocab;
        self.samples
            .iter()
            .flat_map(|s| {
                model.output[s.cell * vocab..(s.cell + 1) * vocab]
                    .iter()
                    .enumerate()
                    .map(move |(c, &p)| (p - f64::from(c == s.label)).abs())
            })
            .fold(0., f64::max)
    }
    /// The fixed-point residual with the gradient of every observation at x.
    pub fn full_residual(&self) -> Result<f64> {
        let x = self.a(&self.z);
        let mut g = vec![0.; x.len()];
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..self.samples.len()).collect();
            self.gradients(&x, &all, &mut g)?;
        }
        let (y, _) =
            self.projector
                .ball(&self.argument(&x, &g), self.radius, self.nuclear_budget)?;
        Ok(distance(self.projector.shape, &x, &y))
    }
    pub fn diagnostic_state(&self) -> &[f64] {
        &self.z
    }
    /// Restore only the splitting point for endpoint diagnostics, not for resume.
    pub fn restore_diagnostic_state(&mut self, z: Vec<f64>) -> Result<()> {
        if z.len() != self.z.len() || z.iter().any(|v| !v.is_finite()) {
            return Err("invalid diagnostic splitting point".into());
        }
        self.z = z;
        self.shadow = self.a(&self.z);
        Ok(())
    }
}
