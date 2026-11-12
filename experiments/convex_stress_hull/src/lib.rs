//! CM8: fixed causal and complex spectral families in one prediction hull.
//! Learners are imported unchanged from counts 981a7eca and Schatten deef98cb.
//! Selection uses actual teacher-forced CE, never generated or held-out accuracy.
pub use convex_stress_counts as counts;
pub use convex_stress_schatten as spectral;
use metrics::Predictor;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
pub use spectral::{Result, data, hash, metrics, random};
use std::collections::BTreeSet;
pub mod meta;
pub mod training;

/// Eligibility uses the visible training input and supervision layout only.
/// Repeated pairs are rejected even when their labels agree: the imported
/// spectral solver requires unique cells. Labels are not deduplicated or replaced.
pub fn pair_eligibility(vocab: usize, examples: &[data::Example]) -> Option<String> {
    if !(2..=256).contains(&vocab) || examples.is_empty() {
        return Some("unsupported_vocabulary_or_empty_train".into());
    }
    let mut seen = BTreeSet::new();
    for e in examples {
        if e.tokens.len() != 2
            || e.targets.len() != 2
            || !e.prompt.is_empty()
            || e.targets[0] >= 0
            || e.targets[1] < 0
        {
            return Some("requires_two_tokens_and_one_final_target".into());
        }
        if e.tokens.iter().any(|&t| t < 0 || t as usize >= vocab) || e.targets[1] as usize >= vocab
        {
            return Some("invalid_token_or_label".into());
        }
        if !seen.insert((e.tokens[0], e.tokens[1])) {
            return Some("repeated_training_pair".into());
        }
    }
    None
}

/// One fixed scale per dataset, determined before training. All candidates
/// share it: unit scale on supported pair layouts, ten otherwise.
pub fn scale(task: &data::Task) -> f64 {
    if pair_eligibility(task.vocab, &task.splits["train"]).is_none() {
        1.
    } else {
        10.
    }
}
pub fn features(task: &data::Task) -> counts::Aligned {
    counts::Aligned {
        base: counts::features::Adapter {
            vocab: task.vocab,
            // Only input/rollout length metadata sets the existing context capacity.
            context: task
                .splits
                .values()
                .flatten()
                .map(|e| {
                    e.tokens
                        .len()
                        .max(e.prompt.len() + e.limit.saturating_sub(1))
                })
                .max()
                .unwrap(),
        },
    }
}
pub fn encoded(
    a: counts::Aligned,
    examples: &[data::Example],
) -> (Vec<counts::input::Observation>, Vec<Vec<usize>>) {
    let mut records = vec![];
    let mut indices = vec![];
    for e in examples {
        let mut ids = vec![];
        for (position, &label) in e.targets.iter().enumerate() {
            if label < 0 {
                continue;
            }
            ids.push(records.len());
            records.push(a.observation(&e.tokens[..=position], label as usize));
        }
        indices.push(ids);
    }
    (records, indices)
}

pub fn budgets(candidate: &str, n: usize) -> Result<(f64, f64)> {
    let n = n as f64;
    match candidate {
        "rank_one" => Ok((n, n)),
        "full_rank" => Ok((n.sqrt(), n * n.sqrt())),
        _ => Err("not a spectral candidate".into()),
    }
}
pub fn ideal_ce(vocab: usize, scale: f64) -> f64 {
    // Stable form of log(exp(scale)+V-1)-scale for the fixed positive scales.
    ((vocab as f64 - 1.) * (-scale).exp()).ln_1p()
}
pub fn teacher_ce(model: &impl Predictor, examples: &[data::Example]) -> Result<f64> {
    let (mut sum, mut count) = (0., 0);
    for e in examples {
        for (position, &target) in e.targets.iter().enumerate() {
            if target < 0 {
                continue;
            }
            let logits = model.logits(&e.tokens[..=position]);
            sum += spectral::loss(&logits, target as usize);
            count += 1;
        }
    }
    if count == 0 || !sum.is_finite() {
        return Err("invalid observed CE".into());
    }
    Ok(sum / count as f64)
}
pub fn all_metrics(model: &impl Predictor, task: &data::Task) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for (name, examples) in &task.splits {
        out.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(out))
}
pub fn passes(v: &Value) -> bool {
    v["exact"]
        .as_u64()
        .zip(v["examples"].as_u64())
        .is_some_and(|(hits, n)| n > 0 && hits * 100 >= n * 99)
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "model")]
pub enum Expert {
    Counts(counts::AdaptiveModel),
    Moments(spectral::Model),
}
impl Predictor for Expert {
    fn vocab(&self) -> usize {
        match self {
            Self::Counts(m) => m.vocab(),
            Self::Moments(m) => m.vocab(),
        }
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        match self {
            Self::Counts(m) => m.logits(prefix),
            Self::Moments(m) => m.logits(prefix),
        }
    }
}
impl Expert {
    pub fn validate(&self, task: &data::Task, candidate: &str, workers: usize) -> Result<Value> {
        if self.vocab() != task.vocab {
            return Err("model vocabulary mismatch".into());
        }
        match self {
            Self::Counts(m) => {
                let a = features(task);
                if candidate != "counts"
                    || m.scale != scale(task)
                    || m.features.base.context != a.base.context
                    || m.features.base.vocab != a.base.vocab
                {
                    return Err("count model configuration mismatch".into());
                }
                Ok(json!({"feasibility_error":m.feasibility_error()?}))
            }
            Self::Moments(m) => {
                if pair_eligibility(task.vocab, &task.splits["train"]).is_some()
                    || (m.radius, m.nuclear_budget) != budgets(candidate, task.vocab)?
                {
                    return Err("spectral model configuration mismatch".into());
                }
                let norms =
                    m.validate(&mut spectral::projection::Projector::new(m.vocab, workers)?)?;
                Ok(json!({"moment_norms":norms}))
            }
        }
    }
}

#[cfg(test)]
mod tests;
