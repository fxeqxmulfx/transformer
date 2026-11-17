//! Training-only scalar component discovery, not an exact convex oracle.
//! Numerical pattern proposals use full-class CE, without recurrent derivatives.
//! CM96 permits skipping only transition objects absent from every batch prefix.
use crate::{C, Result, model::Atom};
use baseline::random::Random;
use bilinear::{model::cross_entropy, train::Record};

pub fn disk(z: C) -> Result<C> {
    let norm = z.norm();
    if !norm.is_finite() {
        return Err("nonfinite scalar proposal".into());
    }
    Ok(if norm > 1. { z / norm } else { z })
}

pub(crate) struct Search<'a, 'r> {
    pub base: &'a [Vec<f64>],
    pub signals: &'a [Vec<f64>],
    pub records: &'a [Record<'r>],
    pub budget: f64,
    pub loss_queries: usize,
}

impl Search<'_, '_> {
    /// Complete complex linear output fit to the observed CE scalar signals.
    /// ForwardMini supplies finite loss differences, never analytic residuals.
    fn fit(&mut self, atom: &mut Atom) -> Result<f64> {
        let states: Vec<_> = self
            .records
            .iter()
            .map(|r| atom.state(r.prefix))
            .collect::<Result<_>>()?;
        atom.output.fill(C::default());
        for (&state, signal) in states.iter().zip(self.signals) {
            for (out, &g) in atom.output.iter_mut().zip(signal) {
                *out += state * (g / self.records.len() as f64);
            }
        }
        for out in &mut atom.output {
            let norm = out.norm();
            if !norm.is_finite() {
                return Err("nonfinite complete output contraction".into());
            }
            if norm == 0. && *out != C::default() {
                return Err("nonzero output contraction norm underflow".into());
            }
            *out = if norm == 0. {
                C::default()
            } else {
                -out.conj() / norm
            };
        }
        let alpha = (1. / self.budget).min(0.1);
        let mut loss = 0.;
        for ((base, &state), record) in self.base.iter().zip(&states).zip(self.records) {
            let logits: Vec<_> = base
                .iter()
                .zip(&atom.output)
                .map(|(&b, &u)| (1. - alpha) * b + alpha * self.budget * (u * state).re)
                .collect();
            loss += cross_entropy(&logits, record.target)?;
        }
        self.loss_queries += self.records.len();
        Ok(loss / self.records.len() as f64)
    }

    pub(crate) fn improve(
        &mut self,
        mut atom: Atom,
        sweeps: usize,
        skip_absent: bool,
    ) -> Result<(Atom, f64)> {
        let mut observed = vec![false; atom.transition.len()];
        for record in self.records {
            for &o in record.prefix {
                *observed
                    .get_mut(usize::try_from(o)?)
                    .ok_or("invalid original object")? = true;
            }
        }
        let mut best = self.fit(&mut atom)?;
        for sweep in 0..sweeps {
            let radius = 2f64.powi(1 - i32::try_from(sweep)?);
            // Coordinate zero is the initial state; every following coordinate
            // is the original object's transition, with no semantic selection.
            for coordinate in 0..=atom.transition.len() {
                if coordinate > 0 && skip_absent && !observed[coordinate - 1] {
                    continue;
                }
                let centre = if coordinate == 0 {
                    atom.initial
                } else {
                    atom.transition[coordinate - 1]
                };
                let mut chosen = atom.clone();
                let mut chosen_loss = best;
                for delta in [
                    C::new(radius, 0.),
                    C::new(-radius, 0.),
                    C::new(0., radius),
                    C::new(0., -radius),
                ] {
                    let mut trial = atom.clone();
                    let value = disk(centre + delta)?;
                    if coordinate == 0 {
                        trial.initial = value;
                    } else {
                        trial.transition[coordinate - 1] = value;
                    }
                    let loss = self.fit(&mut trial)?;
                    if loss < chosen_loss {
                        chosen = trial;
                        chosen_loss = loss;
                    }
                }
                atom = chosen;
                best = chosen_loss;
            }
        }
        atom.validate(atom.transition.len())?;
        Ok((atom, best))
    }

    pub fn propose(
        &mut self,
        objects: usize,
        sweeps: usize,
        rng: &mut Random,
        previous: Option<&Atom>,
    ) -> Result<(Atom, f64)> {
        if sweeps == 0 {
            return Err("empty component search".into());
        }
        let mut starts = vec![Atom::identity_start(objects)];
        let mut random = Atom::identity_start(objects);
        for z in std::iter::once(&mut random.initial).chain(&mut random.transition) {
            let v = C::new(rng.gauss(0., 1.), rng.gauss(0., 1.));
            *z = if v.norm() == 0. {
                C::new(1., 0.)
            } else {
                v / v.norm()
            };
        }
        starts.push(random);
        if let Some(atom) = previous {
            starts.push(atom.clone());
        }
        let mut best: Option<(Atom, f64)> = None;
        for start in starts {
            let trial = self.improve(start, sweeps, true)?;
            if best.as_ref().is_none_or(|b| trial.1 < b.1) {
                best = Some(trial);
            }
        }
        best.ok_or_else(|| "missing proposal".into())
    }
}
