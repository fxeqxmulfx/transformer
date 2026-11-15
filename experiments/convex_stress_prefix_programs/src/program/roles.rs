//! Deterministic role discovery from inputs alone, before decoder fitting.
use super::*;
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roles {
    pub vocabulary: usize,
    pub minimal_prefixes: usize,
    pub delimiters: Vec<usize>,
    pub counters: Vec<usize>,
    pub frequencies: Vec<usize>,
    pub scalar_work_forecast: u64,
}
pub fn infer(input: &Input) -> Result<Roles> {
    if !(2..=240).contains(&input.vocabulary)
        || input.observations.is_empty()
        || input.observations.len() > 200_000
    {
        return Err("count role dimensions refused".into());
    }
    let total: usize = input.observations.iter().map(|r| r.tokens.len()).sum();
    if total > 20_000_000
        || input.observations.iter().any(|r| {
            r.tokens.is_empty()
                || r.tokens.len() > 4096
                || r.tokens.iter().any(|&t| t >= input.vocabulary)
        })
    {
        return Err("count role prefix bounds refused".into());
    }
    let sorted: BTreeSet<_> = input
        .observations
        .iter()
        .map(|r| r.tokens.clone())
        .collect();
    let mut minimal: Vec<Vec<usize>> = vec![];
    for row in sorted {
        if minimal
            .last()
            .is_none_or(|previous| !row.starts_with(previous))
        {
            minimal.push(row);
        }
    }
    let mut frequencies = vec![0; input.vocabulary];
    for row in &minimal {
        for &t in row {
            frequencies[t] += 1;
        }
    }
    let delimiters = minimal
        .iter()
        .map(|r| *r.last().unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let counters = frequencies
        .iter()
        .enumerate()
        .filter_map(|(t, &n)| (n >= minimal.len()).then_some(t))
        .collect();
    let out = Roles {
        vocabulary: input.vocabulary,
        minimal_prefixes: minimal.len(),
        delimiters,
        counters,
        frequencies,
        scalar_work_forecast: 16 * total as u64 * (1 + input.observations.len().ilog2() as u64),
    };
    catalogue(&out)?;
    Ok(out)
}
pub fn catalogue(roles: &Roles) -> Result<Vec<Program>> {
    if !(2..=240).contains(&roles.vocabulary)
        || roles.minimal_prefixes == 0
        || roles.frequencies.len() != roles.vocabulary
        || roles.delimiters.is_empty()
        || roles.delimiters.len() > 16
        || roles.counters.is_empty()
        || roles.counters.len() > 8
        || [&roles.delimiters, &roles.counters]
            .iter()
            .any(|v| v.iter().any(|&t| t >= roles.vocabulary) || v.windows(2).any(|w| w[0] >= w[1]))
    {
        return Err("count role catalogue refused".into());
    }
    let count = roles.delimiters.len() * roles.counters.len() * 7 * 4 * 3 * 2 * 7;
    if count > 16384 {
        return Err("count program candidate cap16384".into());
    }
    let mut out = Vec::with_capacity(count);
    for &delimiter in &roles.delimiters {
        for &counter in &roles.counters {
            for emit in [Emit::Final, Emit::All, Emit::Selected] {
                for modulus in 2..=8 {
                    for skip in 0..=3 {
                        for (stride, offset) in [(1, 0), (2, 0), (2, 1)] {
                            for latest in [false, true] {
                                let copies = if emit == Emit::Final {
                                    vec![None]
                                } else {
                                    vec![None, Some(Side::Before), Some(Side::After)]
                                };
                                for copy in copies {
                                    out.push(Program {
                                        delimiter,
                                        counter,
                                        modulus,
                                        skip,
                                        stride,
                                        offset,
                                        latest,
                                        emit,
                                        copy,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}
