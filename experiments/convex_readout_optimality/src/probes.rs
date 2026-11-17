//! Actual feasible full-logit probes of the CM103 linear minimizer.
//! These are diagnostic evaluations; no candidate is installed or checkpointed.
use crate::Result;
use bilinear::{model::cross_entropy, train::Record};
use rotary::model::Model;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn mix(base: &Model, best: &Model, eta: f64) -> Result<Model> {
    base.validate()?;
    best.validate()?;
    if !eta.is_finite()
        || !(0. ..=1.).contains(&eta)
        || base.objects != best.objects
        || base.weights != best.weights
        || readouts::source::content_hash(base)? != readouts::source::content_hash(best)?
    {
        return Err("incompatible conditional probe".into());
    }
    let mut trial = base.clone();
    for (a, b) in trial.atoms.iter_mut().zip(&best.atoms) {
        for (u, v) in a.output.iter_mut().zip(&b.output) {
            for k in 0..2 {
                u[k] = u[k] * (1. - eta) + v[k] * eta;
            }
        }
    }
    trial.validate()?;
    Ok(trial)
}
#[derive(Debug, Serialize)]
pub struct Score {
    pub records: usize,
    pub ce: f64,
    pub correct: usize,
    pub raw_logits_sha256: String,
}
#[derive(Default)]
pub struct ScoreScan {
    n: usize,
    loss: f64,
    correct: usize,
    hash: Sha256,
}
impl ScoreScan {
    pub fn observe(&mut self, model: &Model, records: &[Record<'_>]) -> Result<()> {
        for r in records {
            let logits = model.logits(r.prefix)?;
            self.loss += cross_entropy(&logits, r.target)?;
            let c = (1..logits.len()).fold(0, |b, c| if logits[c] > logits[b] { c } else { b });
            self.correct += usize::from(c == r.target);
            self.n += 1;
            self.hash.update((r.prefix.len() as u64).to_le_bytes());
            for &o in r.prefix {
                self.hash.update(o.to_le_bytes());
            }
            self.hash.update((r.target as u64).to_le_bytes());
            for x in logits {
                self.hash.update(x.to_le_bytes());
            }
        }
        Ok(())
    }
    pub fn finish(self) -> Result<Score> {
        if self.n == 0 || !self.loss.is_finite() {
            return Err("invalid complete score scan".into());
        }
        Ok(Score {
            records: self.n,
            ce: self.loss / self.n as f64,
            correct: self.correct,
            raw_logits_sha256: self
                .hash
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        })
    }
}
