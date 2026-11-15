//! Preserve the old depth-one/two search; replace only its depth-three join.
use crate::{bank::Bank, pool::Pool, *};
use serde::{Deserialize, Serialize};
pub const WORK_LIMIT: u64 = 40_000_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub quotient: previous::search::Policy,
    pub cache_entries: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            quotient: previous::search::Policy::default(),
            cache_entries: 65536,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Search {
    pub policy: Policy,
    pub observed_records: usize,
    pub closure_nodes: usize,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
    pub shallow: previous::search::Search,
    pub deep: Option<deep::Search>,
    pub programs: Vec<Program>,
    pub refusal: Option<String>,
}
pub fn preflight(rows: &[Observation], vocabulary: usize, policy: &Policy) -> Result<(u64, u64)> {
    let q = &policy.quotient;
    if policy.cache_entries > 65536
        || !(1..=3).contains(&q.depth)
        || !(8..=32768).contains(&q.max_classes)
        || !(8..=262144).contains(&q.latent_classes)
        || !(1..=64).contains(&q.retained)
    {
        return Err("stream search policy refused".into());
    }
    let (work, bytes) = previous::search::preflight(rows, vocabulary, q)?;
    let upper = rows.iter().map(|r| r.tokens.len()).sum();
    let bytes = u128::from(bytes) + cache::buffer_bound(upper, policy.cache_entries);
    if bytes > prior::joint::MAX_BYTES {
        return Err("stream search and cache buffer forecast refused".into());
    }
    Ok((work, u64::try_from(bytes)?))
}
pub fn run(rows: &[Observation], vocabulary: usize, policy: Policy) -> Result<Search> {
    let (scalar_forecast, buffer_forecast) = preflight(rows, vocabulary, &policy)?;
    let mut shallow_policy = policy.quotient.clone();
    shallow_policy.depth = shallow_policy.depth.min(2);
    let shallow = previous::search::run(rows, vocabulary, shallow_policy)?;
    let mut out = Search {
        policy: policy.clone(),
        observed_records: rows.len(),
        closure_nodes: shallow.closure_nodes,
        scalar_forecast,
        buffer_forecast,
        programs: shallow.programs.clone(),
        refusal: shallow.refusal.clone(),
        shallow,
        deep: None,
    };
    if policy.quotient.depth != 3
        || !out.programs.is_empty()
        || out.refusal.as_deref() != Some("no compatible root within declared count depth")
    {
        return Ok(out);
    }
    let pool = Pool::new(rows)?;
    // Reconstruct the small child bank rather than changing the retained344 API.
    // Charge that extra construction before executing it, in addition to shallow.
    let mut meter = latent::Meter {
        work: out.shallow.scalar_forecast,
        limit: WORK_LIMIT,
    };
    let n = pool.nodes.len() as u64;
    if let Err(e) =
        meter.charge(n * 4096 * 128 + 64 * 20 * (n * 64 + 14 * (n.div_ceil(64) * 64 + 2048)))
    {
        out.refusal = Some(e.to_string());
        return Ok(out);
    }
    let initial = Bank::initial(&pool, policy.quotient.max_classes)?;
    let children = Bank::expanded(&pool, &initial, policy.quotient.max_classes)?;
    if children.refusal.is_some() {
        out.refusal = children.refusal;
        return Ok(out);
    }
    let d = deep::run(&pool, rows, &children.bank, &policy, meter.work)?;
    out.programs = d.programs.clone();
    out.refusal = d.refused.clone();
    out.deep = Some(d);
    Ok(out)
}
