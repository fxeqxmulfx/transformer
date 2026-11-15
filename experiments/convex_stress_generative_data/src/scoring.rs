//! Scalar port of frozen synthetic/generation.py rollout and score semantics.
//! Predictors see only the growing prefix. Answers enter the separate scorer.
use crate::Result;
use serde::{Deserialize, Serialize};

pub fn rollout(
    prompt: &[i64],
    limit: usize,
    vocabulary: usize,
    mut predict: impl FnMut(&[i64]) -> Result<usize>,
) -> Result<Vec<usize>> {
    if prompt.is_empty()
        || limit == 0
        || limit > 4097
        || vocabulary <= 17
        || prompt.len() + limit > 8195
        || prompt.iter().any(|&t| t < 0 || t as usize >= vocabulary)
    {
        return Err("invalid generation input or bound".into());
    }
    let mut prefix = prompt.to_vec();
    let mut generated = Vec::with_capacity(limit);
    for _ in 0..limit {
        let token = predict(&prefix)?;
        if token >= vocabulary {
            return Err("generated token outside vocabulary".into());
        }
        generated.push(token);
        if token == 17 {
            break;
        }
        prefix.push(token as i64);
    }
    Ok(generated)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub targets: usize,
    pub correct: usize,
    pub exact: bool,
    pub ended: bool,
    pub generated_tokens: usize,
    pub extra_tokens: usize,
    pub transition_targets: usize,
    pub transition_correct: usize,
}

pub fn score(answer: &[i64], generated: &[usize]) -> Result<Score> {
    if answer.is_empty()
        || answer.last() != Some(&17)
        || answer[..answer.len() - 1].contains(&17)
        || answer.iter().any(|&t| t < 0)
        || generated.is_empty()
        || generated[..generated.len() - 1].contains(&17)
    {
        return Err("invalid reference or generated EOS trajectory".into());
    }
    let mut correct = 0;
    let mut transition_targets = 0;
    let mut transition_correct = 0;
    for (i, &target) in answer.iter().enumerate() {
        let hit = generated.get(i).copied() == Some(target as usize);
        correct += usize::from(hit);
        if i == 0 || target != answer[i - 1] {
            transition_targets += 1;
            transition_correct += usize::from(hit);
        }
    }
    Ok(Score {
        targets: answer.len(),
        correct,
        exact: generated.len() == answer.len() && correct == answer.len(),
        ended: generated.last() == Some(&17),
        generated_tokens: generated.len(),
        extra_tokens: generated.len().saturating_sub(answer.len()),
        transition_targets,
        transition_correct,
    })
}
