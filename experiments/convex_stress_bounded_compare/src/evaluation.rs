//! Original one-answer pairs: raw-token and exact-sequence denominators both
//! include every original record. A numerical refusal never removes a row.
//! Raw readback independently recounts CE/counts and recomputes predictions.
use crate::{
    model::{Answer, Predictor},
    *,
};
use joint::{Work, head::Certificate};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Raw {
    pub index: usize,
    pub input: [u32; 2],
    pub target: usize,
    pub answer: Option<Answer>,
    pub refusal: Option<String>,
}
fn row(index: usize, original: [u32; 3], predictor: &mut Predictor<'_>, work: &mut Work) -> Raw {
    let input = [original[0], original[1]];
    // No target label enters prediction or its numerical fallback.
    let prediction = predictor.predict(input, work);
    let target = original[2] as usize;
    match prediction {
        Ok(answer) => Raw {
            index,
            input,
            target,
            answer: Some(answer),
            refusal: None,
        },
        Err(error) => Raw {
            index,
            input,
            target,
            answer: None,
            refusal: Some(error.to_string()),
        },
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub targets: usize,
    pub raw_correct: usize,
    pub sequences: usize,
    pub exact_correct: usize,
    pub refusals: usize,
    pub raw_accuracy: f64,
    pub sequence_accuracy: f64,
    pub ordinary_cross_entropy: Option<f64>,
    pub raw99: bool,
    pub exact99: bool,
    pub original_success: bool,
    pub exact_queries: usize,
    pub genuine_ties: usize,
    pub all_states_observed: usize,
    pub unanimous_observed: usize,
    pub maximum_independent_residual: f64,
    pub maximum_enclosure_width: f64,
}
impl Metrics {
    pub fn validate(&self, expected: usize) -> Result<()> {
        let pass =
            self.refusals == 0 && 100u128 * self.raw_correct as u128 >= 99u128 * expected as u128;
        if expected == 0
            || self.targets != expected
            || self.sequences != expected
            || self.raw_correct > expected
            || self.exact_correct != self.raw_correct
            || self.refusals > expected - self.raw_correct
            || self.raw_accuracy != self.raw_correct as f64 / expected as f64
            || self.sequence_accuracy != self.raw_accuracy
            || self.raw99 != pass
            || self.exact99 != pass
            || self.original_success != pass
            || self.exact_queries > expected - self.refusals
            || self.genuine_ties > self.exact_queries
            || self.all_states_observed > expected - self.refusals
            || self.unanimous_observed > self.all_states_observed
            || !self.maximum_independent_residual.is_finite()
            || self.maximum_independent_residual < 0.
            || !self.maximum_enclosure_width.is_finite()
            || self.maximum_enclosure_width < 0.
            || match self.ordinary_cross_entropy {
                Some(ce) => self.refusals != 0 || !ce.is_finite() || ce < 0.,
                None => self.refusals == 0,
            }
        {
            return Err("original metric counts, CE or pass flags are inconsistent".into());
        }
        Ok(())
    }
}
#[derive(Default)]
struct Counts {
    rows: usize,
    correct: usize,
    refusals: usize,
    ce: f64,
    exact: usize,
    ties: usize,
    observed: usize,
    unanimous: usize,
    residual: f64,
    width: f64,
}
impl Counts {
    fn add(&mut self, r: &Raw, classes: usize) -> Result<()> {
        if r.index != self.rows
            || r.target >= classes
            || r.input.iter().any(|&v| v as usize >= classes)
        {
            return Err("raw original row order/range differs".into());
        }
        self.rows += 1;
        match (&r.answer, &r.refusal) {
            (Some(a), None) => {
                let p = &a.numerical;
                if p.logits.len() != classes
                    || p.logits.iter().any(|z| !z.is_finite())
                    || p.winner >= classes
                    || p.maximizers.is_empty()
                    || p.maximizers[0] != p.winner
                    || p.maximizers.iter().any(|&c| c >= classes)
                    || p.maximizers.windows(2).any(|x| x[0] >= x[1])
                    || p.target_realizability_certified
                    || !p.maximum_independent_residual.is_finite()
                    || !p.maximum_enclosure_width.is_finite()
                {
                    return Err("invalid complete raw prediction".into());
                }
                let top = p.logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // Independent metric expression, rather than reading a saved loss.
                let loss = p.logits.iter().map(|z| (z - top).exp()).sum::<f64>().ln()
                    - (p.logits[r.target] - top);
                if !loss.is_finite() || loss < 0. {
                    return Err("invalid ordinary raw CE".into());
                }
                self.ce += loss;
                self.correct += usize::from(p.winner == r.target);
                self.exact += usize::from(p.certificate == Certificate::ExactDyadics);
                self.ties += usize::from(p.maximizers.len() > 1);
                self.observed += usize::from(a.all_latent_states_observed);
                self.unanimous += usize::from(a.unanimous_observed_label.is_some());
                self.residual = self.residual.max(p.maximum_independent_residual);
                self.width = self.width.max(p.maximum_enclosure_width);
            }
            (None, Some(reason)) if !reason.is_empty() => self.refusals += 1,
            _ => return Err("raw refusal and answer fields disagree".into()),
        }
        Ok(())
    }
    fn finish(self, expected: usize) -> Result<Metrics> {
        if expected == 0 || self.rows != expected {
            return Err("original metric denominator changed".into());
        }
        let accuracy = self.correct as f64 / expected as f64;
        let pass =
            self.refusals == 0 && 100u128 * self.correct as u128 >= 99u128 * expected as u128;
        Ok(Metrics {
            targets: expected,
            raw_correct: self.correct,
            sequences: expected,
            exact_correct: self.correct,
            refusals: self.refusals,
            raw_accuracy: accuracy,
            sequence_accuracy: accuracy,
            ordinary_cross_entropy: if self.refusals == 0 {
                Some(self.ce / expected as f64)
            } else {
                None
            },
            raw99: pass,
            exact99: pass,
            original_success: pass,
            exact_queries: self.exact,
            genuine_ties: self.ties,
            all_states_observed: self.observed,
            unanimous_observed: self.unanimous,
            maximum_independent_residual: self.residual,
            maximum_enclosure_width: self.width,
        })
    }
}
pub fn score(
    path: &Path,
    records: &[[u32; 3]],
    classes: usize,
    predictor: &mut Predictor<'_>,
    work: &mut Work,
    guard: &resources::Guard,
) -> Result<(disk::Artifact, Metrics)> {
    let limit = records.len() as u64 * (2048 + 48 * classes as u64);
    let mut out = disk::Writer::new(path, limit, guard)?;
    let mut counts = Counts::default();
    for (i, &original) in records.iter().enumerate() {
        let raw = row(i, original, predictor, work);
        counts.add(&raw, classes)?;
        serde_json::to_writer(&mut out, &raw)?;
        out.write_all(b"\n")?;
    }
    Ok((out.finish()?, counts.finish(records.len())?))
}
pub fn replay(
    path: &Path,
    artifact: &disk::Artifact,
    records: &[[u32; 3]],
    classes: usize,
    predictor: &mut Predictor<'_>,
    work: &mut Work,
) -> Result<Metrics> {
    if disk::hash(path)? != *artifact {
        return Err("raw artifact hash differs".into());
    }
    let reader = BufReader::new(std::fs::File::open(path)?);
    let mut count = Counts::default();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.len() > 2048 + 48 * classes {
            return Err("raw row exceeds its forecast".into());
        }
        let saved: Raw = serde_json::from_str(&line)?;
        let original = *records.get(index).ok_or("extra raw prediction")?;
        let expected = row(index, original, predictor, work);
        if saved != expected {
            return Err("fresh complete-head prediction differs from saved raw row".into());
        }
        count.add(&saved, classes)?;
    }
    count.finish(records.len())
}
