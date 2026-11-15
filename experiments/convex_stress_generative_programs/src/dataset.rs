//! Dataset names belong only to this adapter. The model sees raw observations.
use crate::*;
pub use sequences::dataset::Dataset;
pub fn cases() -> Result<Vec<String>> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["cases"].clone(),
    )?)
}
pub fn build(case: &str) -> Result<Dataset> {
    if generative::CASES.contains(&case) {
        generative::dataset::build(case)
    } else {
        parent::dataset::build(case)
    }
}
pub fn metadata(data: &Dataset) -> serde_json::Value {
    if generative::CASES.contains(&data.name.as_str()) {
        generative::dataset::metadata(data)
    } else {
        data.metadata()
    }
}

/// Preserve the old prefix adapter exactly, and expose only each original
/// supervised causal prefix for generated answers. No names or test rows.
pub fn input(vocabulary: usize, rows: &[data::Example]) -> Result<Input> {
    if rows.iter().all(|e| e.prompt.is_empty()) {
        return Ok(sequences::adapter::request(vocabulary, rows)?
            .discovery
            .input);
    }
    if rows.is_empty() || rows.len() > 20_000 {
        return Err("empty or oversized generative input".into());
    }
    let context = rows.iter().map(|e| e.tokens.len()).max().unwrap();
    let mut observations = vec![];
    let mut prefix_tokens = 0usize;
    for e in rows {
        if e.prompt.is_empty()
            || e.answer.is_empty()
            || e.field.is_some()
            || e.tokens.len() != e.targets.len()
            || e.tokens.len() != e.changes.len()
            || e.prompt.last() != Some(&18)
            || e.answer.last() != Some(&17)
            || e.answer[..e.answer.len() - 1].contains(&17)
            || e.limit < e.answer.len()
            || e.tokens
                != e.prompt
                    .iter()
                    .chain(e.answer[..e.answer.len() - 1].iter())
                    .copied()
                    .collect::<Vec<_>>()
            || e.targets
                != std::iter::repeat_n(-100, e.prompt.len() - 1)
                    .chain(e.answer.iter().copied())
                    .collect::<Vec<_>>()
        {
            return Err("invalid shifted causal answer record".into());
        }
        for (position, &label) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
            prefix_tokens += position + 1;
            if observations.len() == 200_000 || prefix_tokens > 20_000_000 {
                return Err("training prefix storage cap".into());
            }
            observations.push(Observation {
                tokens: e.tokens[..=position]
                    .iter()
                    .map(|&t| usize::try_from(t))
                    .collect::<std::result::Result<_, _>>()?,
                label: usize::try_from(label)?,
            });
        }
    }
    let out = Input {
        vocabulary,
        context,
        observations,
    };
    sequences::complex::Request {
        input: out.clone(),
        policy: Default::default(),
    }
    .validate()?;
    Ok(out)
}
