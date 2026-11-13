//! Independent interpretation of every retained contradiction and every raw
//! label fitted by a selected decoder. Deterministic replay additionally checks
//! exact candidate order, early exits, refusal scopes and all counters.
use super::*;
pub fn check(request: &Request, report: &Report) -> Result<()> {
    let x = &request.input;
    if report.request != *request
        || report.buffer_preflight_bytes != request.validate()?
        || report.unseen_identification
    {
        return Err("causal discovery input, buffer bound or claim changed".into());
    }
    let library = features::library(x.vocabulary, x.context);
    let mut queries = std::collections::BTreeSet::new();
    let (mut visits, mut attempts) = (0, 0);
    for group in &report.groups {
        if group.query >= x.vocabulary
            || !queries.insert(group.query)
            || group.observations
                != x.observations
                    .iter()
                    .filter(|s| s.tokens.last() == Some(&group.query))
                    .count()
        {
            return Err("causal query coverage changed".into());
        }
        for attempt in &group.attempts {
            if !attempt.candidate.valid(library.len()) {
                return Err("invalid candidate index".into());
            }
            visits += attempt.visits;
            attempts += 1;
            if let Decision::Conflict { first, second } = attempt.decision {
                let a = x
                    .observations
                    .get(first)
                    .ok_or("bad first conflict index")?;
                let b = x
                    .observations
                    .get(second)
                    .ok_or("bad second conflict index")?;
                if first >= second
                    || a.tokens.last() != Some(&group.query)
                    || b.tokens.last() != Some(&group.query)
                    || a.label == b.label
                    || attempt
                        .candidate
                        .original(&library, x.vocabulary, &a.tokens)
                        != attempt
                            .candidate
                            .original(&library, x.vocabulary, &b.tokens)
                {
                    return Err("reported original-label contradiction is false".into());
                }
            }
        }
        if let Some(decoder) = &group.decoder {
            if !decoder.candidate.valid(library.len())
                || decoder.values.is_empty()
                || decoder.values.len() != decoder.labels.len()
                || decoder.values.windows(2).any(|w| w[0] >= w[1])
                || decoder.labels.iter().any(|&c| c >= x.vocabulary)
                || group.attempts.last().is_none_or(|a| {
                    a.candidate != decoder.candidate
                        || a.decision != Decision::Admitted
                        || a.visits != group.observations
                })
            {
                return Err("invalid retained categorical decoder".into());
            }
            let mut used = vec![false; decoder.values.len()];
            for s in x
                .observations
                .iter()
                .filter(|s| s.tokens.last() == Some(&group.query))
            {
                let value = decoder
                    .candidate
                    .original(&library, x.vocabulary, &s.tokens);
                let state = decoder.state(value).ok_or("original training value lost")?;
                used[state] = true;
                if decoder.labels[state] != s.label {
                    return Err("selected decoder fails an original raw label".into());
                }
            }
            if used.contains(&false) {
                return Err("unobserved decoder value was invented".into());
            }
        }
    }
    if visits != report.work.visits
        || attempts != report.work.attempts
        || visits > request.policy.visits
        || attempts > request.policy.attempts
        || report.work.validated_tokens
            != x.observations.iter().map(|s| s.tokens.len()).sum::<usize>()
    {
        return Err("retained search work differs".into());
    }
    if report.reason.is_none()
        && (report.groups.iter().any(|g| g.decoder.is_none())
            || x.observations
                .iter()
                .any(|s| !queries.contains(s.tokens.last().unwrap())))
    {
        return Err("completed discovery lost an observed query".into());
    }
    Ok(())
}
