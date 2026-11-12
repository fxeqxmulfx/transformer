//! Splitting with fixed grouped cells and the original sum-CE objective.
//! Ported from deef98cb's solver; original observations are never deduplicated.
//! The step is 1/max(cell multiplicity). Mini refreshes per-observation memory
//! and its summed cell gradients. No convergence rate is asserted.
use crate::projection::{Norms, Projector, distance, simplex};
use crate::{Groups, Mode, Model, Result, Sample, Table, gradient};

pub struct Solver {
    pub projector: Projector,
    pub residual: f64,
    pub step_size: f64,
    pub maximum_multiplicity: usize,
    groups: Groups,
    radius: f64,
    nuclear_budget: f64,
    mode: Mode,
    samples: Vec<Sample>,
    multiplicities: Vec<usize>,
    z: Vec<f64>,
    shadow: Vec<f64>,
    memory: Vec<f64>,
    aggregate: Vec<f64>,
}

impl Solver {
    pub fn new(model: &Model, samples: Vec<Sample>, mode: Mode, workers: usize) -> Result<Self> {
        model.validate_shape()?;
        let n = model.table.vocab;
        let mut labels = vec![None; n * n];
        let mut multiplicities = vec![0; n * n];
        if samples.is_empty() {
            return Err("empty training set".into());
        }
        for s in &samples {
            if s.cell >= n * n
                || s.label >= n
                || s.cell % n >= model.groups.row_count
                || s.cell / n >= model.groups.column_count
                || labels[s.cell].is_some_and(|label| label != s.label)
            {
                return Err("invalid or inconsistent grouped training cells".into());
            }
            labels[s.cell] = Some(s.label);
            multiplicities[s.cell] += 1;
        }
        let maximum_multiplicity = *multiplicities.iter().max().unwrap();
        Ok(Self {
            projector: Projector::new(n, workers)?,
            residual: 0.,
            step_size: 1. / maximum_multiplicity as f64,
            maximum_multiplicity,
            groups: model.groups.clone(),
            radius: model.table.radius,
            nuclear_budget: model.table.nuclear_budget,
            mode,
            memory: vec![0.; samples.len() * n],
            samples,
            multiplicities,
            z: model.table.probabilities.clone(),
            shadow: model.table.probabilities.clone(),
            aggregate: vec![0.; n * n * n],
        })
    }

    fn a(&self, point: &[f64]) -> Vec<f64> {
        let n = self.projector.n;
        let mut x = vec![0.; point.len()];
        for (cell, (input, out)) in point.chunks_exact(n).zip(x.chunks_exact_mut(n)).enumerate() {
            if cell % n < self.groups.row_count && cell / n < self.groups.column_count {
                simplex(input, out);
            } else {
                // Inactive padding has zero nonconstant Fourier moments.
                out.fill(1. / n as f64);
            }
        }
        if self.mode == Mode::Forward {
            for s in &self.samples {
                x[s.cell * n..(s.cell + 1) * n].fill(0.);
                x[s.cell * n + s.label] = 1.;
            }
        }
        x
    }

    pub fn full_gradient(&self, x: &[f64]) -> Vec<f64> {
        let n = self.projector.n;
        let mut g = vec![0.; x.len()];
        let mut current = vec![0.; n];
        for s in &self.samples {
            let start = s.cell * n;
            gradient(&x[start..start + n], s.label, &mut current);
            for (sum, &value) in g[start..start + n].iter_mut().zip(&current) {
                *sum += value;
            }
        }
        g
    }

    pub fn step(&mut self, selected: &[usize]) -> Result<()> {
        let n = self.projector.n;
        let x = self.a(&self.z);
        if self.mode == Mode::Full {
            self.aggregate = self.full_gradient(&x);
        } else if self.mode == Mode::Mini {
            let mut current = vec![0.; n];
            for &index in selected {
                let s = self.samples.get(index).ok_or("invalid batch index")?;
                let start = s.cell * n;
                gradient(&x[start..start + n], s.label, &mut current);
                let old = &mut self.memory[index * n..(index + 1) * n];
                for ((sum, old), &new) in self.aggregate[start..start + n]
                    .iter_mut()
                    .zip(old)
                    .zip(&current)
                {
                    if self.multiplicities[s.cell] == 1 {
                        *sum = new;
                    } else {
                        *sum += new - *old;
                    }
                    *old = new;
                }
            }
        }
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(&self.aggregate)
            .map(|((&a, &z), &g)| 2. * a - z - self.step_size * g)
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

    pub fn snapshot(&mut self) -> Result<(Model, f64, Vec<Norms>)> {
        let (table, alpha, norms) =
            self.projector
                .feasible(&self.shadow, self.radius, self.nuclear_budget)?;
        let model = Model {
            groups: self.groups.clone(),
            table,
        };
        model.validate_shape()?;
        Ok((model, alpha, norms))
    }

    pub fn fit_defect(&self, model: &Model) -> f64 {
        let Table {
            vocab: n,
            probabilities,
            ..
        } = &model.table;
        self.samples
            .iter()
            .flat_map(|s| {
                probabilities[s.cell * n..(s.cell + 1) * n]
                    .iter()
                    .enumerate()
                    .map(move |(c, &p)| (p - f64::from(c == s.label)).abs())
            })
            .fold(0., f64::max)
    }

    pub fn full_residual(&mut self) -> Result<f64> {
        let x = self.a(&self.z);
        let g = if self.mode == Mode::Forward {
            vec![0.; x.len()]
        } else {
            self.full_gradient(&x)
        };
        let argument: Vec<_> = x
            .iter()
            .zip(&self.z)
            .zip(g)
            .map(|((&a, &z), g)| 2. * a - z - self.step_size * g)
            .collect();
        let y = self
            .projector
            .ball(&argument, self.radius, self.nuclear_budget)?;
        Ok(distance(&x, &y))
    }

    pub fn diagnostic_state(&self) -> &[f64] {
        &self.z
    }

    /// Endpoint diagnostics only; minibatch memory is not restored for resume.
    pub fn restore_diagnostic_state(&mut self, z: Vec<f64>) -> Result<()> {
        if z.len() != self.z.len() || z.iter().any(|v| !v.is_finite()) {
            return Err("invalid diagnostic splitting point".into());
        }
        self.z = z;
        self.shadow = self.a(&self.z);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minibatch_aggregate_equals_sum_of_retained_original_gradients() -> Result<()> {
        let model = Model {
            groups: Groups::identity(3),
            table: Table::new(3, 3., 3.)?,
        };
        let samples = vec![
            Sample { cell: 0, label: 1 },
            Sample { cell: 0, label: 1 },
            Sample { cell: 1, label: 2 },
        ];
        let mut solver = Solver::new(&model, samples.clone(), Mode::Mini, 1)?;
        for step in 0..20 {
            solver.step(&[step % 3])?;
            let mut sum = vec![0.; 27];
            for (index, s) in samples.iter().enumerate() {
                for c in 0..3 {
                    sum[s.cell * 3 + c] += solver.memory[index * 3 + c];
                }
            }
            assert!(
                sum.iter()
                    .zip(&solver.aggregate)
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
        }
        let mut first = Solver::new(&model, samples.clone(), Mode::Forward, 1)?;
        let mut second = Solver::new(&model, samples, Mode::Forward, 1)?;
        for _ in 0..20 {
            first.step(&[])?;
            second.step(&[0, 1, 2])?;
        }
        assert_eq!(first.diagnostic_state(), second.diagnostic_state());
        assert!(
            first
                .aggregate
                .iter()
                .chain(&first.memory)
                .all(|&g| g == 0.)
        );
        Ok(())
    }
}
