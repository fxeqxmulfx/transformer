//! Joint CM57 heads for retained fixed programs. The paired categorical control
//! freezes a training dictionary before CE; it deliberately cannot extrapolate
//! missing quantity-to-label entries. Both arms retain the supplied copy prior.
use crate::{
    discovery::{Compatible, Discovery},
    observations::Observation,
    program::{Atom, Program},
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prior {
    Coordinates,
    Categorical,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub prior: Prior,
    pub programs: Vec<Compatible>,
    pub head: joint::head::Head,
    pub feature_bases: Vec<Vec<Parameters>>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    pub model: Model,
    pub training: joint::training::Outcome,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub logits: Vec<f64>,
    pub defined: Vec<bool>,
    pub span_residuals: Vec<f64>,
    pub active_observable_coverage: bool,
    pub target_realizability_certified: bool,
}

fn feature(
    p: &Compatible,
    atom: Option<Atom>,
    coordinates: &[f64],
    prior: Prior,
) -> Result<(Features, bool)> {
    let Some(atom) = atom else {
        return Ok(([0.; 5], false));
    };
    if prior == Prior::Coordinates {
        return Ok((atom.features(coordinates)?, true));
    }
    let label = match atom {
        Atom::Copy(c) => Some(c),
        _ => p
            .categorical
            .binary_search_by_key(&atom, |(a, _)| *a)
            .ok()
            .map(|i| p.categorical[i].1),
    };
    // This explicit default is an unsupported prediction, never an identified
    // completion of an unseen categorical entry.
    let coordinate = *coordinates
        .get(label.unwrap_or(0))
        .ok_or("categorical entry outside supplied class coordinates")?;
    Ok(([coordinate, 1., 0., 0., 0.], label.is_some()))
}

pub fn records(
    report: &Discovery,
    rows: &[Observation],
    prior: Prior,
) -> Result<Vec<joint::data::Record>> {
    if report.compatible.is_empty() {
        return Err("no observed-compatible program remains".into());
    }
    rows.iter()
        .map(|r| {
            let mut features = vec![];
            for p in &report.compatible {
                let (f, known) = feature(p, p.program.run(&r.tokens), &report.coordinates, prior)?;
                if !known {
                    return Err("retained decoder refuses an observed record".into());
                }
                features.push(f);
            }
            Ok(joint::data::Record {
                id: r.id,
                features,
                label: r.label,
            })
        })
        .collect()
}

pub fn train(
    report: &Discovery,
    rows: &[Observation],
    prior: Prior,
    policy: joint::training::Policy,
    radius: f64,
    minimum_mass: f64,
) -> Result<Trained> {
    let records = records(report, rows, prior)?;
    let budgets = report
        .compatible
        .iter()
        .map(|p| {
            if prior == Prior::Coordinates {
                p.budget
            } else {
                16.
            }
        })
        .collect();
    let training = joint::training::train(
        &records,
        report.coordinates.clone(),
        budgets,
        radius,
        minimum_mass,
        policy,
    )?;
    let model = Model {
        prior,
        programs: report.compatible.clone(),
        head: training.head.clone(),
        feature_bases: training.domain.feature_basis.clone(),
    };
    Ok(Trained { model, training })
}

impl Model {
    fn validate_query(&self, tokens: &[usize]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > 4096
            || tokens.iter().any(|&t| t >= self.head.coordinates.len())
            || self.programs.is_empty()
            || self.programs.len() != self.head.theta.len()
            || self.feature_bases.len() != self.programs.len()
        {
            return Err("invalid query or saved model dimensions".into());
        }
        Ok(())
    }

    pub fn query(&self, tokens: &[usize]) -> Result<Query> {
        self.validate_query(tokens)?;
        let mut inputs = vec![];
        let mut defined = vec![];
        let mut span_residuals = vec![];
        if self.programs.len() != self.head.theta.len()
            || self.feature_bases.len() != self.programs.len()
        {
            return Err("saved model program/basis dimensions differ".into());
        }
        for (p, basis) in self.programs.iter().zip(&self.feature_bases) {
            let (h, known) = feature(p, p.program.run(tokens), &self.head.coordinates, self.prior)?;
            let mut r = [h[0], h[1], h[2], h[3], h[4], 0.];
            joint::projection::linear(&mut r, basis);
            span_residuals.push(
                joint::projection::dot(&r, &r).sqrt()
                    / h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.),
            );
            defined.push(known);
            inputs.push(h);
        }
        let active_observable_coverage = self
            .head
            .theta
            .iter()
            .enumerate()
            .all(|(p, t)| t[5] <= 1e-8 || (defined[p] && span_residuals[p] <= 1e-10));
        Ok(Query {
            logits: self.head.logits(&inputs)?,
            defined,
            span_residuals,
            active_observable_coverage,
            target_realizability_certified: false,
        })
    }

    pub fn independent_logits(&self, tokens: &[usize]) -> Result<Vec<f64>> {
        self.validate_query(tokens)?;
        let inputs = self
            .programs
            .iter()
            .map(|p| {
                feature(
                    p,
                    p.program.interpret(tokens),
                    &self.head.coordinates,
                    self.prior,
                )
                .map(|v| v.0)
            })
            .collect::<Result<Vec<_>>>()?;
        self.head.independent_logits(&inputs)
    }
}

impl Trained {
    pub fn replay(&self, report: &Discovery, rows: &[Observation]) -> Result<()> {
        let rebuilt = train(
            report,
            rows,
            self.model.prior,
            self.training.policy.clone(),
            self.training.domain.radius,
            self.training.domain.minimum_mass,
        )?;
        if *self != rebuilt {
            return Err("complete program/head training replay differs".into());
        }
        Ok(())
    }
}

pub fn bank_programs(report: &Discovery) -> Vec<Program> {
    report
        .compatible
        .iter()
        .map(|p| p.program.clone())
        .collect()
}
