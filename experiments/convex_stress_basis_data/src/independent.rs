//! Separate data oracle: recount each prefix and scan writes backward. This
//! never calls the sampling oracle or enters a model's features or fitting.
use crate::*;
use serde_json::Value;
use std::collections::BTreeSet;

pub fn check(s: &Value, e: &data::Example) -> Result<()> {
    if e.tokens.is_empty()
        || e.tokens[0] != 1
        || e.tokens.len() != e.targets.len()
        || e.tokens.len() != e.changes.len()
        || e.field.is_some()
        || e.row_index.is_some()
        || e.task != s["task"]
    {
        return Err("invalid source record structure".into());
    }
    let mut expected = vec![-100; e.tokens.len()];
    let size = match e.task.as_str() {
        "blocks" => {
            if !e.prompt.is_empty()
                || !e.answer.is_empty()
                || e.limit != 0
                || e.tokens[1..].iter().any(|t| ![9, 10, 11].contains(t))
            {
                return Err("invalid block input".into());
            }
            if s["neutral_fraction"] == 0. && e.tokens.contains(&11) {
                return Err("neutral token in neutral-free language".into());
            }
            let blocks = generator::number(s, "blocks")?;
            for (i, label) in expected.iter_mut().enumerate().skip(1) {
                let active: Vec<_> = e.tokens[1..=i]
                    .iter()
                    .filter(|&&t| t != 11)
                    .copied()
                    .collect();
                let runs = usize::from(!active.is_empty())
                    + active.windows(2).filter(|w| w[0] != w[1]).count();
                *label = if active.first() == Some(&9) && runs == blocks {
                    16
                } else {
                    15
                };
            }
            e.tokens.len()
        }
        "mqar" => {
            let pairs = generator::number(s, "pairs")?;
            let symbols = generator::number(s, "symbols")? as i64;
            let queries = generator::number(s, "queries")?;
            let gap = generator::number(s, "query_gap")?;
            let overwrites = s["overwrites"].as_u64().unwrap_or(0) as usize;
            let end = 1 + 2 * pairs;
            if e.tokens.len() < end || !e.prompt.is_empty() || !e.answer.is_empty() || e.limit != 0
            {
                return Err("invalid recall input".into());
            }
            let writes: Vec<_> = e.tokens[1..end].as_chunks::<2>().0.iter().collect();
            let mut distinct = BTreeSet::new();
            for (i, kv) in writes.iter().enumerate() {
                if !(36..36 + symbols).contains(&kv[0])
                    || !(36 + symbols..36 + 2 * symbols).contains(&kv[1])
                {
                    return Err("write outside raw vocabulary".into());
                }
                if let Some(old) = writes[..i].iter().rev().find(|old| old[0] == kv[0])
                    && (overwrites == 0 || old[1] == kv[1])
                {
                    return Err("rewrite does not change its value".into());
                }
                distinct.insert(kv[0]);
            }
            if distinct.len() != pairs - overwrites {
                return Err("wrong distinct-key count".into());
            }
            let mut seen = BTreeSet::new();
            for (i, label) in expected.iter_mut().enumerate().skip(end) {
                let key = e.tokens[i];
                if (36..36 + symbols).contains(&key) {
                    if i < end + gap || !seen.insert(key) {
                        return Err("invalid or repeated query".into());
                    }
                    *label = writes
                        .iter()
                        .rev()
                        .find(|kv| kv[0] == key)
                        .ok_or("query lacks prior binding")?[1];
                } else if !(36 + symbols..36 + 2 * symbols).contains(&key) {
                    return Err("invalid filler".into());
                }
            }
            if seen.len() != queries {
                return Err("wrong query count".into());
            }
            e.tokens.len()
        }
        "parity" => {
            if e.prompt.len() < 3
                || e.prompt[0] != 1
                || e.prompt.last() != Some(&18)
                || e.prompt[1..e.prompt.len() - 1]
                    .iter()
                    .any(|t| ![21, 22].contains(t))
                || e.limit != 2
            {
                return Err("invalid parity prompt or limit".into());
            }
            let one_count = e.prompt[1..e.prompt.len() - 1]
                .iter()
                .filter(|&&t| t == 22)
                .count();
            let answer = [if one_count % 2 == 0 { 24 } else { 25 }, 17];
            let full: Vec<_> = e.prompt.iter().chain(&answer[..1]).copied().collect();
            if e.answer != answer || e.tokens != full {
                return Err("parity answer or teacher forcing differs".into());
            }
            expected[e.prompt.len() - 1..].copy_from_slice(&answer);
            e.prompt.len() - 2
        }
        _ => return Err("unknown independent oracle".into()),
    };
    let maximum = generator::number(s, "length")?;
    let minimum = if s["min_length"].is_null() {
        maximum
    } else {
        generator::number(s, "min_length")?
    };
    if !(minimum..=maximum).contains(&size) || expected != e.targets {
        return Err("independent labels or problem length differ".into());
    }
    let mut prior = None;
    for (&label, &change) in e.targets.iter().zip(&e.changes) {
        if change != (label >= 0 && prior != Some(label)) {
            return Err("independent state-change mask differs".into());
        }
        if label >= 0 {
            prior = Some(label);
        }
    }
    Ok(())
}
