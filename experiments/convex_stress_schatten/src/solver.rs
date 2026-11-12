//! One simplex and one moment projection per iteration, retaining splitting state.
//! Full: x=P_A(z), y=P_B(2x-z-grad CE(x)), z+=y-x.
//! Mini: replace selected per-observation gradients in a retained aggregate; this
//! is incremental aggregated gradient, not unbiased SGD. No rate is claimed.
//! Forward: A fixes observed one-hot rows; use y=P_B(2x-z), without derivatives.
//! Port of coupled-moments 2de90819, with both spectral budgets passed to B.
use crate::{
    Model, Result, Sample, gradient,
    projection::{Projector, distance, simplex},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Full,
    Mini,
    Forward,
}
impl Mode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "full_batch" => Ok(Self::Full),
            "minibatch" => Ok(Self::Mini),
            "forward_only" => Ok(Self::Forward),
            _ => Err("unknown training arm".into()),
        }
    }
}
pub struct Solver {
    pub projector: Projector,
    pub shadow: Vec<f64>,
    pub residual: f64,
    pub radius: f64,
    pub nuclear_budget: f64,
    pub mode: Mode,
    z: Vec<f64>,
    memory: Vec<f64>,
    samples: Vec<Sample>,
}
impl Solver {
    pub fn new(model: &Model, samples: Vec<Sample>, mode: Mode, workers: usize) -> Result<Self> {
        model.validate_shape()?;
        let mut seen = std::collections::BTreeSet::new();
        if samples.is_empty()
            || samples.iter().any(|s| {
                s.cell >= model.vocab * model.vocab
                    || s.label >= model.vocab
                    || !seen.insert(s.cell)
            })
        {
            return Err("invalid or duplicate training cells".into());
        }
        Ok(Self {
            projector: Projector::new(model.vocab, workers)?,
            shadow: model.probabilities.clone(),
            residual: 0.,
            radius: model.radius,
            nuclear_budget: model.nuclear_budget,
            mode,
            z: model.probabilities.clone(),
            memory: vec![0.; model.probabilities.len()],
            samples,
        })
    }
    fn a(&self, point: &[f64]) -> Vec<f64> {
        let n = self.projector.n;
        let mut x = vec![0.; point.len()];
        for (input, out) in point.chunks_exact(n).zip(x.chunks_exact_mut(n)) {
            simplex(input, out);
        }
        if self.mode == Mode::Forward {
            for s in &self.samples {
                x[s.cell * n..(s.cell + 1) * n].fill(0.);
                x[s.cell * n + s.label] = 1.;
            }
        }
        x
    }
    pub fn step(&mut self, selected: &[usize]) -> Result<()> {
        let n = self.projector.n;
        let x = self.a(&self.z);
        if self.mode != Mode::Forward {
            let all: Vec<_> = (0..self.samples.len()).collect();
            let selected = if self.mode == Mode::Full {
                &all
            } else {
                selected
            };
            for &index in selected {
                let s = self.samples.get(index).ok_or("invalid batch index")?;
                let start = s.cell * n;
                gradient(
                    &x[start..start + n],
                    s.label,
                    &mut self.memory[start..start + n],
                );
            }
        }
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(&self.memory)
            .map(|((&a, &z), &g)| 2. * a - z - g)
            .collect();
        let y = self
            .projector
            .ball(&argument, self.radius, self.nuclear_budget)?;
        self.residual = distance(&x, &y);
        for (z, (&a, &b)) in self.z.iter_mut().zip(x.iter().zip(&y)) {
            *z += b - a;
        }
        self.shadow = self.a(&self.z);
        Ok(())
    }
    pub fn snapshot(&mut self) -> Result<(Model, f64, Vec<crate::projection::Norms>)> {
        self.projector
            .feasible(&self.shadow, self.radius, self.nuclear_budget)
    }
    pub fn fit_defect(&self, model: &Model) -> f64 {
        self.samples
            .iter()
            .flat_map(|s| {
                model.probabilities[s.cell * model.vocab..(s.cell + 1) * model.vocab]
                    .iter()
                    .enumerate()
                    .map(move |(c, &p)| (p - f64::from(c == s.label)).abs())
            })
            .fold(0., f64::max)
    }
    pub fn full_residual(&mut self) -> Result<f64> {
        let n = self.projector.n;
        let x = self.a(&self.z);
        let mut g = vec![0.; x.len()];
        if self.mode != Mode::Forward {
            for s in &self.samples {
                gradient(
                    &x[s.cell * n..(s.cell + 1) * n],
                    s.label,
                    &mut g[s.cell * n..(s.cell + 1) * n],
                );
            }
        }
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(g)
            .map(|((&a, &z), g)| 2. * a - z - g)
            .collect();
        let y = self
            .projector
            .ball(&argument, self.radius, self.nuclear_budget)?;
        Ok(distance(&x, &y))
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
