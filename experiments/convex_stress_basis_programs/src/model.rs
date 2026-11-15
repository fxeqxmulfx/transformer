//! Fixed input programs plus the actual complete joint head. Categorical
//! controls use observed dictionaries; coordinates use CM57 affine quantities.
//! Observable span coverage is separate from unknown-target realizability.
use crate::{
    discovery::{Candidate, Discovery},
    *,
};
use basis::data::Example;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use weighted::{
    corpus::{Corpus, Limits},
    domain::{Anchors, HeadSpec},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prior {
    Coordinates,
    Categorical,
}

pub fn feature(
    candidate: &Candidate,
    atom: Option<Atom>,
    coords: &[f64],
    prior: Prior,
) -> Result<(Features, bool)> {
    let Some(atom) = atom else {
        return Ok(([0.; 5], false));
    };
    if prior == Prior::Coordinates {
        return Ok((atom.features(coords)?, true));
    }
    let value = candidate.categorical(atom, coords);
    Ok((
        [value.unwrap_or(coords[0]), 1., 0., 0., 0.],
        value.is_some(),
    ))
}

pub fn corpus(report: &Discovery, rows: &[Example], prior: Prior) -> Result<Corpus> {
    if input::inspect(report.input.vocabulary, rows)? != report.input {
        return Err("original training pool binding changed".into());
    }
    let c = report
        .selected
        .as_ref()
        .ok_or("no retained input program")?;
    let limits = Limits {
        records: report.input.records,
        groups: report.input.records.min(4096),
        programs: 1,
        vocabulary: report.input.vocabulary,
    };
    Corpus::build(
        limits,
        input::observations(rows).map(|r| {
            let atom = c.program.run(r.tokens);
            if atom != c.program.interpret(r.tokens) {
                return Err("independent original feature construction differs".into());
            }
            let (h, known) = feature(c, atom, &report.coordinates, prior)?;
            if !known {
                return Err("retained feature refuses original training input".into());
            }
            Ok(joint::data::Record {
                id: r.id,
                features: vec![h],
                label: r.label,
            })
        }),
    )
}

pub fn specification(
    report: &Discovery,
    rows: &[Example],
    corpus: &Corpus,
    prior: Prior,
) -> Result<HeadSpec> {
    let candidate = report
        .selected
        .as_ref()
        .ok_or("no retained input program")?;
    let anchors = if candidate.exact {
        Anchors::All
    } else {
        let mut groups = BTreeSet::new();
        for (r, occurrence) in input::observations(rows).zip(corpus.occurrences()) {
            let atom = candidate
                .program
                .run(r.tokens)
                .ok_or("selected role became undefined")?;
            if candidate.structural_anchor(atom) {
                groups.insert(occurrence.group);
            }
        }
        Anchors::Groups(groups.into_iter().collect())
    };
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    Ok(HeadSpec {
        coordinates: report.coordinates.clone(),
        budgets: vec![if prior == Prior::Coordinates {
            candidate.budget
        } else {
            16.
        }],
        radius: p["fit"]["radius"].as_f64().unwrap(),
        minimum_mass: p["fit"]["minimum_mass"].as_f64().unwrap(),
        anchors,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub prior: Prior,
    pub programs: Vec<Candidate>,
    pub head: joint::head::Head,
    pub feature_bases: Vec<Vec<Parameters>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub logits: Vec<f64>,
    pub defined: Vec<bool>,
    pub span_residuals: Vec<f64>,
    pub active_observable_coverage: bool,
    pub target_realizability_certified: bool,
}
impl Model {
    pub fn validate(&self) -> Result<()> {
        self.head.validate()?;
        if self.programs.len() != 1
            || self.head.theta.len() != 1
            || self.feature_bases.len() != 1
            || !self.programs[0].program.valid(self.head.coordinates.len())
            || self.feature_bases[0].len() > 6
            || self
                .feature_bases
                .iter()
                .flatten()
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err("saved fixed program/complete head dimensions differ".into());
        }
        Ok(())
    }
    fn inputs(&self, tokens: &[i64], independent: bool) -> Result<(Vec<Features>, Vec<bool>)> {
        if tokens.is_empty()
            || tokens.len() > 4096
            || tokens
                .iter()
                .any(|&t| t < 0 || t as usize >= self.head.coordinates.len())
        {
            return Err("query outside supplied input vocabulary".into());
        }
        let mut features = vec![];
        let mut defined = vec![];
        for p in &self.programs {
            let atom = if independent {
                p.program.interpret(tokens)
            } else {
                p.program.run(tokens)
            };
            let (h, known) = feature(p, atom, &self.head.coordinates, self.prior)?;
            features.push(h);
            defined.push(known);
        }
        Ok((features, defined))
    }
    pub fn query(&self, tokens: &[i64]) -> Result<Query> {
        let (h, defined) = self.inputs(tokens, false)?;
        self.query_features(&h, defined)
    }
    pub fn checked_features(&self, tokens: &[i64]) -> Result<Vec<Features>> {
        let (primary, defined) = self.inputs(tokens, false)?;
        let (independent, other_defined) = self.inputs(tokens, true)?;
        if primary != independent || defined != other_defined || defined.iter().any(|b| !*b) {
            return Err("independent complete original features differ or are undefined".into());
        }
        Ok(primary)
    }
    /// Original grouped training rows already contain complete fixed features.
    /// They use the same stored tensor and coverage calculation as raw queries.
    pub fn query_features(&self, h: &[Features], defined: Vec<bool>) -> Result<Query> {
        if h.len() != self.programs.len() || defined.len() != h.len() {
            return Err("incomplete query feature bank".into());
        }
        let span_residuals: Vec<_> = h
            .iter()
            .zip(&self.feature_bases)
            .map(|(h, b)| {
                let mut r = [h[0], h[1], h[2], h[3], h[4], 0.];
                joint::projection::linear(&mut r, b);
                joint::projection::dot(&r, &r).sqrt()
                    / h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.)
            })
            .collect();
        let covered = self
            .head
            .theta
            .iter()
            .enumerate()
            .all(|(p, t)| t[5] <= 1e-8 || (defined[p] && span_residuals[p] <= 1e-10));
        Ok(Query {
            logits: self.head.logits(h)?,
            defined,
            span_residuals,
            active_observable_coverage: covered,
            target_realizability_certified: false,
        })
    }
    pub fn independent_logits(&self, tokens: &[i64]) -> Result<Vec<f64>> {
        let (h, _) = self.inputs(tokens, true)?;
        self.head.independent_logits(&h)
    }
}
