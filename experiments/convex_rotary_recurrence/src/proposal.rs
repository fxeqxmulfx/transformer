//! Approximate complete numerical component discovery, not a global convex oracle.
//! Ported from scalar recurrence1786526f; CM97/98 matrices and frequencies added.
//! Every observed object matrix entry is searched; absent matrices alone can skip.
use crate::{
    C, Result,
    atom::{Atom, Family, State, row},
    coordinates::{self, Coordinate},
};
use baseline::random::Random;
use bilinear::{model::cross_entropy, train::Record};
mod refinement;

/// A feasible complete complex readout direction for a row-l1 budget.
/// All coefficients contribute to the comparison; zeros are values, not removed parameters.
pub(crate) fn readout_direction(a: State) -> Result<State> {
    let n = [a[0].norm(), a[1].norm()];
    for j in 0..2 {
        if !n[j].is_finite() || (n[j] == 0. && a[j] != C::default()) {
            return Err("nonfinite or underflowed complete readout contraction".into());
        }
    }
    let j = usize::from(n[1] > n[0]);
    let mut out = [C::default(); 2];
    if n[j] != 0. {
        out[j] = -a[j].conj() / n[j];
    }
    Ok(out)
}
pub(crate) struct Search<'a, 'r> {
    pub family: Family,
    pub frequency_start: Option<[f64; 2]>,
    pub base: &'a [Vec<f64>],
    pub signals: &'a [Vec<f64>],
    pub records: &'a [Record<'r>],
    pub budget: f64,
    pub loss_queries: usize,
    pub coordinate_trials: [usize; 3],
}
impl Search<'_, '_> {
    fn fit(&mut self, atom: &mut Atom) -> Result<f64> {
        let length = self
            .records
            .iter()
            .map(|r| r.prefix.len())
            .max()
            .ok_or("empty batch")?;
        let phases = atom.phases(length)?;
        let states: Vec<_> = self
            .records
            .iter()
            .map(|r| atom.state_with(r.prefix, phases.as_deref()))
            .collect::<Result<_>>()?;
        atom.output.fill([C::default(); 2]);
        for (&state, signal) in states.iter().zip(self.signals) {
            for (out, &g) in atom.output.iter_mut().zip(signal) {
                for j in 0..2 {
                    out[j] += state[j] * (g / self.records.len() as f64);
                }
            }
        }
        for out in &mut atom.output {
            *out = readout_direction(*out)?;
        }
        let alpha = (1. / self.budget).min(0.1);
        let mut loss = 0.;
        for ((base, &state), record) in self.base.iter().zip(&states).zip(self.records) {
            let logits: Vec<_> = base
                .iter()
                .zip(&atom.output)
                .map(|(&b, &u)| (1. - alpha) * b + alpha * self.budget * row(u, state).re)
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
        let family = self.family;
        let mut observed = vec![false; atom.transition.len()];
        for record in self.records {
            for &o in record.prefix {
                *observed
                    .get_mut(usize::try_from(o)?)
                    .ok_or("invalid original object")? = true;
            }
        }
        let coordinates = coordinates::coordinates(&observed, family, skip_absent);
        let mut best = self.fit(&mut atom)?;
        for sweep in 0..sweeps {
            let radius = 2f64.powi(1 - i32::try_from(sweep)?);
            for &coordinate in &coordinates {
                let mut chosen = atom.clone();
                let mut chosen_loss = best;
                let deltas = [
                    C::new(radius, 0.),
                    C::new(-radius, 0.),
                    C::new(0., radius),
                    C::new(0., -radius),
                ];
                let count = if matches!(coordinate, Coordinate::Frequency(_)) {
                    2
                } else {
                    4
                };
                for &delta in &deltas[..count] {
                    let mut trial = coordinate.apply(&atom, delta)?;
                    let loss = self.fit(&mut trial)?;
                    self.coordinate_trials[coordinate.kind()] += 1;
                    if loss < chosen_loss {
                        chosen = trial;
                        chosen_loss = loss;
                    }
                }
                atom = chosen;
                best = chosen_loss;
            }
        }
        atom.validate(atom.transition.len(), family)?;
        Ok((atom, best))
    }
    pub fn propose(
        &mut self,
        objects: usize,
        family: Family,
        sweeps: usize,
        frequency_refinement_steps: usize,
        rng: &mut Random,
        previous: Option<&Atom>,
    ) -> Result<(Atom, f64)> {
        if sweeps == 0 {
            return Err("empty component search".into());
        }
        let mut starts = coordinates::starts(objects, family, rng, self.frequency_start).to_vec();
        if let Some(a) = previous {
            starts.push(a.clone());
        }
        let mut best: Option<(Atom, f64)> = None;
        for start in starts {
            let mut trial = self.improve(start, sweeps, true)?;
            if frequency_refinement_steps > 0 {
                trial = self.refine_frequencies(trial.0, frequency_refinement_steps)?;
            }
            if best.as_ref().is_none_or(|b| trial.1 < b.1) {
                best = Some(trial);
            }
        }
        best.ok_or_else(|| "missing proposal".into())
    }
}
