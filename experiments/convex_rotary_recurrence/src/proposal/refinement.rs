//! Training-only resolution repair after the unchanged complete coarse search.
//! CM98 allows arbitrary real frequencies; CM99 covers the accepted convex step.
//! Every trial refits all original-class readouts and compares actual complete CE.
use super::Search;
use crate::{
    C, Result,
    atom::{Atom, Family},
    coordinates::Coordinate,
};

impl Search<'_, '_> {
    pub(crate) fn refine_frequencies(
        &mut self,
        mut atom: Atom,
        steps: usize,
    ) -> Result<(Atom, f64)> {
        if self.family != Family::Rotary || steps == 0 || steps > 16 {
            return Err(
                "frequency refinement requires learned rotary and1..16 finer levels".into(),
            );
        }
        let mut best = self.fit(&mut atom)?;
        for level in 0..steps {
            let radius = 2f64.powi(-5 - i32::try_from(level)?);
            for j in 0..2 {
                let mut chosen = atom.clone();
                let mut chosen_loss = best;
                for delta in [radius, -radius] {
                    let mut trial = Coordinate::Frequency(j).apply(&atom, C::new(delta, 0.))?;
                    let loss = self.fit(&mut trial)?;
                    self.coordinate_trials[2] += 1;
                    if loss < chosen_loss {
                        chosen = trial;
                        chosen_loss = loss;
                    }
                }
                atom = chosen;
                best = chosen_loss;
            }
        }
        atom.validate(atom.transition.len(), self.family)?;
        Ok((atom, best))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Model, train};
    use bilinear::{
        model::cross_entropy,
        train::{Mode, Record},
    };

    #[test]
    fn finer_frequency_search_finds_a_feasible_improvement_missed_by_every_old_radius() {
        let records = [
            Record {
                prefix: &[0, 1],
                target: 2,
            },
            Record {
                prefix: &[0, 0, 1],
                target: 2,
            },
        ];
        let base = vec![vec![0.; 3]; 2];
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let signals: Vec<_> = records
                .iter()
                .map(|r| train::scalars(&base[0], r.target, mode, 1e-4).unwrap())
                .collect();
            let mut search = Search {
                family: Family::Rotary,
                frequency_start: None,
                base: &base,
                signals: &signals,
                records: &records,
                budget: 16.,
                loss_queries: 0,
                coordinate_trials: [0; 3],
            };
            let mut atom = Atom::identity_start(3, Family::Rotary);
            atom.initial[1] = C::default();
            atom.frequency = Some([0.02, 0.]);
            atom.transition[1] = [
                [C::default(), C::new(1., 0.)],
                [C::new(1., 0.), C::default()],
            ];
            // The third object is absent, but all of its parameters remain stored.
            atom.transition[2][0][1] = C::new(1e-100, -1e-100);
            let (reference, baseline) = search.improve(atom, 0, false).unwrap();
            for j in 0..2 {
                for level in 0..6 {
                    let radius = 2f64.powi(1 - level);
                    for delta in [radius, -radius] {
                        let trial = Coordinate::Frequency(j)
                            .apply(&reference, C::new(delta, 0.))
                            .unwrap();
                        let (_, loss) = search.improve(trial, 0, false).unwrap();
                        assert!(
                            loss > baseline,
                            "mode={mode:?} delta={delta} loss={loss} baseline={baseline}"
                        );
                    }
                }
            }
            let queries_before = search.loss_queries;
            let (refined, loss) = search.refine_frequencies(reference.clone(), 16).unwrap();
            assert!(loss < baseline - 1e-6);
            assert_eq!(search.coordinate_trials, [0, 0, 64]);
            assert_eq!(search.loss_queries - queries_before, 65 * records.len());
            assert_eq!(refined.initial, reference.initial);
            assert_eq!(refined.transition, reference.transition);
            assert_eq!(refined.output.len(), 3);
            assert_ne!(refined.frequency, reference.frequency);
            let mut model = Model::zero(3, 16., Family::Rotary).unwrap();
            model.mix(refined, 1. / 16.).unwrap();
            let actual: f64 = records
                .iter()
                .map(|r| cross_entropy(&model.logits(r.prefix).unwrap(), r.target).unwrap())
                .sum::<f64>()
                / records.len() as f64;
            assert!((actual - loss).abs() < 2e-15);
            model.validate().unwrap();
        }
    }
}
