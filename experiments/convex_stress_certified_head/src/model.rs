//! Input-only adapter of synthetic_programs/model.rs at1371ae4f. Preserve its
//! fixed features, categorical default and span diagnostics; only complete-head
//! numerical evaluation changes. No dataset/task/target identifier is accepted.
use crate::*;
use previous::{
    discovery::Compatible,
    model::{Model, Prior, Query},
    program::Atom,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditedQuery {
    pub query: Query,
    pub scores: Scores,
}

pub struct ModelEvaluator<'a> {
    model: &'a Model,
    head: Evaluator<'a>,
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
    let coordinate = *coordinates
        .get(label.unwrap_or(0))
        .ok_or("categorical entry outside supplied class coordinates")?;
    Ok(([coordinate, 1., 0., 0., 0.], label.is_some()))
}

impl<'a> ModelEvaluator<'a> {
    pub fn new(model: &'a Model) -> Result<Self> {
        if model.programs.is_empty()
            || model.programs.len() != model.head.theta.len()
            || model.feature_bases.len() != model.programs.len()
            || model
                .feature_bases
                .iter()
                .any(|basis| basis.len() > 6 || basis.iter().flatten().any(|x| !x.is_finite()))
        {
            return Err("invalid complete model program or basis dimensions".into());
        }
        Ok(Self {
            model,
            head: Evaluator::from_head(&model.head)?,
        })
    }

    pub fn inspect(&self, prefix: &[i64]) -> Result<AuditedQuery> {
        let tokens: Vec<_> = prefix
            .iter()
            .map(|&t| usize::try_from(t))
            .collect::<std::result::Result<_, _>>()?;
        if tokens.is_empty()
            || tokens.len() > 4096
            || tokens
                .iter()
                .any(|&t| t >= self.model.head.coordinates.len())
        {
            return Err("invalid input-only model query".into());
        }
        let mut inputs = vec![];
        let mut independent_inputs = vec![];
        let mut defined = vec![];
        let mut span_residuals = vec![];
        for (p, basis) in self.model.programs.iter().zip(&self.model.feature_bases) {
            let actual = p.program.run(&tokens);
            let independent = p.program.interpret(&tokens);
            if actual != independent {
                return Err("independent program interpreter differs".into());
            }
            let (h, known) = feature(p, actual, &self.model.head.coordinates, self.model.prior)?;
            let (other, other_known) = feature(
                p,
                independent,
                &self.model.head.coordinates,
                self.model.prior,
            )?;
            if h != other || known != other_known {
                return Err("independent fixed program features differ".into());
            }
            let mut r = [h[0], h[1], h[2], h[3], h[4], 0.];
            joint::projection::linear(&mut r, basis);
            span_residuals.push(
                joint::projection::dot(&r, &r).sqrt()
                    / h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.),
            );
            defined.push(known);
            inputs.push(h);
            independent_inputs.push(other);
        }
        if inputs != independent_inputs {
            return Err("independent complete feature vectors differ".into());
        }
        let scores = self.head.evaluate(&inputs)?;
        let active_observable_coverage = self
            .model
            .head
            .theta
            .iter()
            .enumerate()
            .all(|(p, t)| t[5] <= 1e-8 || (defined[p] && span_residuals[p] <= 1e-10));
        let query = Query {
            logits: scores.logits.clone(),
            defined,
            span_residuals,
            active_observable_coverage,
            target_realizability_certified: false,
        };
        Ok(AuditedQuery { query, scores })
    }
}
