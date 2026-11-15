//! Input-only delimiter discovery and a fixed Cartesian grammar. Candidate
//! order is deterministic and independent of heldout records and task names.
use super::*;
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roles {
    pub vocabulary: usize,
    pub delimiters: Vec<usize>,
    pub minimal_prefixes: usize,
    pub scalar_work_forecast: u64,
}
pub fn infer(input: &Input) -> Result<Roles> {
    if !(2..=1024).contains(&input.vocabulary)
        || input.observations.is_empty()
        || input.observations.len() > 200_000
        || input.observations.iter().any(|r| {
            r.tokens.is_empty()
                || r.tokens.len() > 4096
                || r.tokens.iter().any(|&t| t >= input.vocabulary)
        })
    {
        return Err("aggregate role input bounds refused".into());
    }
    let total: usize = input.observations.iter().map(|r| r.tokens.len()).sum();
    if total > 20_000_000 {
        return Err("aggregate role input buffer refused".into());
    }
    let sorted: BTreeSet<_> = input
        .observations
        .iter()
        .map(|r| r.tokens.clone())
        .collect();
    let mut minimal: Vec<Vec<usize>> = vec![];
    for row in sorted {
        if minimal.last().is_none_or(|last| !row.starts_with(last)) {
            minimal.push(row);
        }
    }
    let delimiters: Vec<_> = minimal
        .iter()
        .map(|r| *r.last().unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if delimiters.len() > 16 {
        return Err("too many training delimiter candidates".into());
    }
    Ok(Roles {
        vocabulary: input.vocabulary,
        delimiters,
        minimal_prefixes: minimal.len(),
        scalar_work_forecast: 32 * total as u64 * (1 + input.observations.len().ilog2() as u64),
    })
}
pub fn catalogue(roles: &Roles) -> Result<Vec<Program>> {
    if !(2..=1024).contains(&roles.vocabulary)
        || roles.delimiters.is_empty()
        || roles.delimiters.len() > 16
        || roles.delimiters.windows(2).any(|w| w[0] >= w[1])
        || roles.delimiters.iter().any(|&d| d >= roles.vocabulary)
    {
        return Err("invalid fixed aggregate catalogue dimensions".into());
    }
    let mut out = vec![];
    for &delimiter in &roles.delimiters {
        for latest in [false, true] {
            for skip in 0..=1 {
                for basis in [
                    Basis::Positions,
                    Basis::Distinct,
                    Basis::Maximum,
                    Basis::Minimum,
                ] {
                    for order in [
                        Order::Source,
                        Order::Ascending,
                        Order::Descending,
                        Order::Reverse,
                    ] {
                        for statistic in [
                            Statistic::Frequency,
                            Statistic::EqualFrequencyClasses,
                            Statistic::PrefixFrequency,
                            Statistic::Position,
                        ] {
                            for emit in [
                                Emit::Quantity,
                                Emit::Copy,
                                Emit::QuantityCopy,
                                Emit::CopyQuantity,
                            ] {
                                for prefix in [false, true] {
                                    for padding in [false, true] {
                                        for tail in [Tail::None, Tail::Maximum, Tail::Minimum] {
                                            let p = Program {
                                                vocabulary: roles.vocabulary,
                                                delimiter,
                                                latest,
                                                skip,
                                                basis,
                                                order,
                                                statistic,
                                                emit,
                                                prefix,
                                                padding,
                                                tail,
                                            };
                                            if p.validate().is_ok() {
                                                out.push(p);
                                            }
                                        }
                                    }
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
