//! Original all-supervised-prefix scoring and generative EOS/limit scoring.
//! Stochastic control labels are scored as sampled, never replaced by an oracle.
use crate::*;
pub use generative::scoring::{Score, rollout, score};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixScore {
    pub targets: usize,
    pub correct: usize,
    pub exact: bool,
    pub transition_targets: usize,
    pub transition_correct: usize,
}
pub fn prefix(e: &data::Example, predictions: &[usize]) -> Result<PrefixScore> {
    if !e.prompt.is_empty()
        || !e.answer.is_empty()
        || e.limit != 0
        || e.tokens.len() != e.targets.len()
        || e.tokens.len() != e.changes.len()
        || predictions.len() != e.tokens.len()
    {
        return Err("prefix scorer requires all original input positions".into());
    }
    let (mut targets, mut correct, mut transition_targets, mut transition_correct) = (0, 0, 0, 0);
    let mut previous = None;
    for (i, &label) in e.targets.iter().enumerate() {
        let changed = label >= 0 && previous != Some(label);
        if e.changes[i] != changed {
            return Err("prefix change mask differs".into());
        }
        if label >= 0 {
            targets += 1;
            let hit = predictions[i] == label as usize;
            correct += usize::from(hit);
            transition_targets += usize::from(changed);
            transition_correct += usize::from(changed && hit);
            previous = Some(label);
        } else if label != -100 {
            return Err("invalid ignore label".into());
        }
    }
    if targets == 0 {
        return Err("no supervised prefixes".into());
    }
    Ok(PrefixScore {
        targets,
        correct,
        exact: targets == correct,
        transition_targets,
        transition_correct,
    })
}
