//! Training-CE comparison with supplied earlier decoded vertices. All prior
//! models and all new attempts remain present, including refusals. This is not
//! the all35 comparison with the other model families, whose queues are frozen.
use crate::{
    Model, Result,
    search::{self, Proposal},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prior {
    pub model: Model,
    pub audit: Value,
    pub train_ce: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Choice {
    Prior(usize),
    Density(usize),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub proposal: Proposal,
    pub prior: Vec<Prior>,
    pub selected: Option<Choice>,
}

pub fn compare(proposal: Proposal, previous: Vec<Model>) -> Result<Comparison> {
    if previous.len() > 32 {
        return Err("prior decoded-candidate cap exceeded".into());
    }
    let samples = search::validate(&proposal.input)?;
    search::replay(&proposal.input, &proposal)?;
    let mut out = Comparison {
        proposal,
        prior: vec![],
        selected: None,
    };
    let mut best = f64::INFINITY;
    for model in previous {
        if model.n != out.proposal.input.n || model.vocab != out.proposal.input.vocab {
            return Err("prior candidate scope mismatch".into());
        }
        let audit = model.audit(&samples)?;
        let ce = audit["train_ce"]
            .as_f64()
            .ok_or("prior has no audited CE")?;
        if ce + 1e-12 < best {
            best = ce;
            out.selected = Some(Choice::Prior(out.prior.len()));
        }
        out.prior.push(Prior {
            model,
            audit,
            train_ce: ce,
        });
    }
    for (i, attempt) in out.proposal.attempts.iter().enumerate() {
        if let Some(ce) = attempt.train_ce
            && ce + 1e-12 < best
        {
            best = ce;
            out.selected = Some(Choice::Density(i));
        }
    }
    Ok(out)
}

pub fn replay(original: &search::Input, previous: &[Model], saved: &Comparison) -> Result<()> {
    if &saved.proposal.input != original {
        return Err("comparison original input mismatch".into());
    }
    if compare(saved.proposal.clone(), previous.to_vec())? != *saved {
        return Err("decoded-candidate comparison replay mismatch".into());
    }
    Ok(())
}
