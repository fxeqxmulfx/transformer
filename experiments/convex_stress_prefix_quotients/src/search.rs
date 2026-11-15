//! Bounded quotient expansion and root-constraint search on training inputs.
use crate::{bank::Bank, pool::Pool, *};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub depth: usize,
    pub max_classes: usize,
    pub retained: usize,
    pub latent_classes: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            depth: 3,
            max_classes: 32768,
            retained: 32,
            latent_classes: 262144,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    pub depth: usize,
    pub child_signatures: usize,
    pub expansion_candidates: u64,
    pub root: root::Stats,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expansion {
    pub depth: usize,
    pub visited: u64,
    pub classes: usize,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Search {
    pub policy: Policy,
    pub observed_records: usize,
    pub closure_upper_bound: usize,
    pub closure_nodes: usize,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
    pub stages: Vec<Stage>,
    pub expansions: Vec<Expansion>,
    pub deep: Option<deep::Search>,
    pub programs: Vec<Program>,
    pub refusal: Option<String>,
}
fn admit(out: &mut Search, work: u128, bytes: u128) -> Result<()> {
    let total = u128::from(out.scalar_forecast) + work;
    if total > 40_000_000_000_000 || bytes > prior::joint::MAX_BYTES {
        return Err("quotient search forecast refused".into());
    }
    out.scalar_forecast = total as u64;
    out.buffer_forecast = out.buffer_forecast.max(bytes as u64);
    Ok(())
}
/// Input-only storage preflight for the complete search; no grammar fitting.
pub fn preflight(rows: &[Observation], vocabulary: usize, policy: &Policy) -> Result<(u64, u64)> {
    prior::observations::validate(rows, vocabulary)?;
    let upper: usize = rows.iter().map(|r| r.tokens.len()).sum();
    let initial = 1024u128.pow(3) + upper as u128 * 512;
    if initial > prior::joint::MAX_BYTES {
        return Err("prefix closure storage refused".into());
    }
    if rows
        .iter()
        .any(|r| prior::prefix::payload(&r.tokens).is_none())
    {
        return Ok((upper as u64 * 256, initial as u64));
    }
    let pool = Pool::new(rows)?;
    let n = pool.nodes.len() as u128;
    let mut bytes = initial + 2 * policy.max_classes as u128 * (n * 8 + 65536);
    if policy.depth == 3 {
        bytes += 2 * policy.latent_classes as u128 * (n * 8 + 65536)
            + pairs::buffer_bound(&pool, policy.latent_classes);
    }
    if bytes > prior::joint::MAX_BYTES {
        return Err("complete search buffer forecast refused".into());
    }
    Ok((40_000_000_000_000, bytes as u64))
}
pub fn run(rows: &[Observation], vocabulary: usize, policy: Policy) -> Result<Search> {
    prior::observations::validate(rows, vocabulary)?;
    if !(1..=3).contains(&policy.depth)
        || !(8..=32768).contains(&policy.max_classes)
        || !(1..=64).contains(&policy.retained)
        || !(8..=262144).contains(&policy.latent_classes)
    {
        return Err("quotient search policy refused".into());
    }
    let upper: usize = rows.iter().map(|r| r.tokens.len()).sum();
    let mut out = Search {
        policy: policy.clone(),
        observed_records: rows.len(),
        closure_upper_bound: upper,
        closure_nodes: 0,
        scalar_forecast: 0,
        buffer_forecast: 0,
        stages: vec![],
        expansions: vec![],
        deep: None,
        programs: vec![],
        refusal: None,
    };
    admit(
        &mut out,
        upper as u128 * 256,
        upper as u128 * 512 + 1024u128.pow(3),
    )?;
    if rows
        .iter()
        .any(|r| prior::prefix::payload(&r.tokens).is_none())
    {
        out.refusal = Some("input prefixes do not use supplied three-symbol alphabet".into());
        return Ok(out);
    }
    if rows.iter().map(|r| r.label).collect::<BTreeSet<_>>().len() != 2 {
        out.programs = grammar::base().into_iter().map(Program::Prefix).collect();
        out.refusal = Some("two observed raw classes required; literal bank retained".into());
        return Ok(out);
    }
    let pool = Pool::new(rows)?;
    out.closure_nodes = pool.nodes.len();
    let n = pool.nodes.len() as u128;
    let bytes =
        1024u128.pow(3) + upper as u128 * 512 + policy.max_classes as u128 * (n * 8 + 65536) * 2;
    let independent = policy.retained as u128
        * 128
        * 4096
        * rows.iter().map(|r| r.tokens.len() as u128).sum::<u128>();
    admit(&mut out, independent + n * 4096 * 128, bytes)?;
    let mut bank = Bank::initial(&pool, policy.max_classes)?;
    for depth in 1..=policy.depth.min(2) {
        let k = bank.rows.len() as u128;
        // Four extra counts, seven gates, two label orientations, two negations;
        // each constraint scans at most every original observation. Early rejects
        // only decrease actual row comparisons, recorded separately in root stats.
        admit(
            &mut out,
            k * k * 112 * (rows.len() as u128 * 32 + 1024),
            bytes,
        )?;
        let (perfect, stats) = match root::run(&pool, rows, &bank, policy.retained) {
            Ok(value) => value,
            Err(e) => {
                out.refusal = Some(e.to_string());
                break;
            }
        };
        out.stages.push(Stage {
            depth,
            child_signatures: bank.rows.len(),
            expansion_candidates: bank.visited,
            root: stats,
        });
        if !perfect.is_empty() {
            out.programs = perfect.into_iter().map(Program::Prefix).collect();
            return Ok(out);
        }
        if depth == 1 && policy.depth > 1 {
            admit(
                &mut out,
                k * k * 20 * (n * 64 + 14 * (n.div_ceil(64) * 64 + 2048)),
                bytes,
            )?;
            match Bank::expanded(&pool, &bank, policy.max_classes) {
                Ok(next) => {
                    out.expansions.push(Expansion {
                        depth,
                        visited: next.bank.visited,
                        classes: next.bank.rows.len(),
                        refusal: next.refusal.clone(),
                    });
                    if next.refusal.is_some() {
                        out.refusal = next.refusal;
                        break;
                    }
                    bank = next.bank;
                }
                Err(e) => {
                    out.refusal = Some(e.to_string());
                    break;
                }
            }
        }
    }
    if policy.depth == 3 && out.programs.is_empty() && out.refusal.is_none() {
        let deep_bytes = bytes
            + 2 * policy.latent_classes as u128 * (n * 8 + 65536)
            + pairs::buffer_bound(&pool, policy.latent_classes);
        if out.scalar_forecast > 40_000_000_000_000 || deep_bytes > prior::joint::MAX_BYTES {
            out.refusal = Some("transition search memory/work forecast refused".into());
        } else {
            out.buffer_forecast = out.buffer_forecast.max(deep_bytes as u64);
            match deep::run(
                &pool,
                rows,
                &bank,
                policy.latent_classes,
                policy.retained,
                out.scalar_forecast,
            ) {
                Ok(d) => {
                    out.programs = d.programs.clone();
                    out.refusal = d.refused.clone();
                    out.deep = Some(d);
                }
                Err(e) => out.refusal = Some(e.to_string()),
            }
            out.scalar_forecast = 40_000_000_000_000;
        }
    }
    if out.programs.is_empty() && out.refusal.is_none() {
        out.refusal = Some("no compatible root within declared count depth".into());
    }
    Ok(out)
}
