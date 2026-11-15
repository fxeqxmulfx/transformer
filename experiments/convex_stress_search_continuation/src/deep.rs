//! Continue task347's exact join at1371ae4f after a completed fit. Keep the
//! smallest actual ASTs encountered, preserving fits on an explicit work stop.
//! Capped anchors and interrupted query streams remain incomplete.
use crate::{
    bank::Bank,
    cache::Cache,
    index::{Index, Side},
    latent::Meter,
    pool::Pool,
    *,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub config: transitions::Config,
    pub left: latent::Stats,
    pub right: Option<latent::Stats>,
    pub anchor_side: Option<Side>,
    pub anchor_complete: bool,
    pub anchor_classes: usize,
    pub retained_query_complete: bool,
    pub stream: stream::Stats,
    pub index: index::Stats,
    pub cache_hits: u64,
    pub cheaper_retests: u64,
    pub cache_evictions: u64,
    pub cache_peak_entries: usize,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Search {
    pub possible_branches: usize,
    pub attempts: Vec<Attempt>,
    pub programs: Vec<Program>,
    pub charged_scalar_bound: u64,
    pub refused: Option<String>,
    pub stopped_reason: Option<String>,
    pub retention: retained::Stats,
    pub final_audit_reserved_work: u64,
}
fn audit(programs: &[Program], rows: &[Observation]) -> Result<()> {
    for p in programs {
        let mut labels = BTreeMap::new();
        for r in rows {
            let atom = p.run(&r.tokens).ok_or("streamed candidate undefined")?;
            if Some(atom) != p.interpret(&r.tokens) {
                return Err("independent streamed interpreter mismatch".into());
            }
            if let Some(old) = labels.insert(atom, r.label)
                && old != r.label
            {
                return Err("streamed candidate fails original labels".into());
            }
        }
    }
    Ok(())
}
pub fn run(
    pool: &Pool,
    rows: &[Observation],
    children: &Bank,
    policy: &search::Policy,
    used: u64,
) -> Result<Search> {
    run_with_limit(pool, rows, children, policy, used, search::WORK_LIMIT)
}

pub(crate) fn run_with_limit(
    pool: &Pool,
    rows: &[Observation],
    children: &Bank,
    policy: &search::Policy,
    used: u64,
    work_limit: u64,
) -> Result<Search> {
    let branches = transitions::branches(pool, rows)?;
    let audit_work = audit_bound(rows, policy.quotient.retained)?;
    let mut out = Search {
        possible_branches: branches.len(),
        attempts: vec![],
        programs: vec![],
        charged_scalar_bound: used,
        refused: None,
        stopped_reason: None,
        retention: retained::Stats::default(),
        final_audit_reserved_work: audit_work,
    };
    let mut meter = Meter {
        work: used,
        limit: work_limit
            .checked_sub(audit_work)
            .ok_or("final fit audit work refused")?,
    };
    if used > meter.limit {
        return Err("prior work leaves no reserved fit audit".into());
    }
    let mut retained = retained::Retained::new(policy.quotient.retained)?;
    let high = rows.iter().map(|r| r.label).max().ok_or("missing labels")?;
    let observed = pool
        .row_nodes
        .iter()
        .zip(rows)
        .map(|(&i, r)| (i, r.label == high))
        .collect();
    let extras: Vec<_> = (0..3)
        .map(|s| Ok(pool.counts(&pool.signature(&grammar::literal(s))?)))
        .collect::<Result<_>>()?;
    for branch in branches.into_iter().take(64) {
        let (left, left_stats) = latent::synthesize(
            pool,
            children,
            &branch.left,
            policy.quotient.latent_classes,
            &mut meter,
        )?;
        let mut attempt = Attempt {
            config: branch.config.clone(),
            left: left_stats,
            right: None,
            anchor_side: None,
            anchor_complete: false,
            anchor_classes: 0,
            retained_query_complete: false,
            stream: stream::Stats::default(),
            index: index::Stats::default(),
            cache_hits: 0,
            cheaper_retests: 0,
            cache_evictions: 0,
            cache_peak_entries: 0,
            refusal: None,
        };
        let completed_before = retained.stats.successful_queries;
        let result = (|| -> Result<()> {
            if attempt.left.work_exhausted {
                return Err("left anchor work cap exhausted".into());
            }
            if left.rows.is_empty() {
                attempt.stream.complete = true;
                return Ok(());
            }
            let (right, right_stats) = latent::synthesize(
                pool,
                children,
                &branch.right,
                policy.quotient.latent_classes,
                &mut meter,
            )?;
            attempt.right = Some(right_stats.clone());
            if right_stats.work_exhausted {
                return Err("right anchor work cap exhausted".into());
            }
            if right.rows.is_empty() {
                attempt.stream.complete = true;
                return Ok(());
            }
            let side = if !attempt.left.capped
                && (right_stats.capped || left.rows.len() <= right.rows.len())
            {
                Side::Left
            } else if !right_stats.capped {
                Side::Right
            } else {
                Side::Left
            };
            let (anchor, query_bank, wanted, anchor_complete, query_complete) =
                if side == Side::Left {
                    (
                        left,
                        right,
                        &branch.right,
                        !attempt.left.capped,
                        !right_stats.capped,
                    )
                } else {
                    (
                        right,
                        left,
                        &branch.left,
                        !right_stats.capped,
                        !attempt.left.capped,
                    )
                };
            attempt.anchor_side = Some(side);
            attempt.anchor_complete = anchor_complete;
            attempt.anchor_classes = anchor.rows.len();
            attempt.retained_query_complete = query_complete;
            // Drop a truncated query bank before allocating the persistent index.
            let query_bank = if query_complete {
                Some(query_bank)
            } else {
                drop(query_bank);
                None
            };
            let bounds = count_bounds::partners(
                pool,
                &observed,
                &anchor,
                &branch.config,
                &extras,
                side == Side::Left,
                &mut meter,
            )?;
            let mut index = Index::new(
                pool,
                &observed,
                &anchor,
                side,
                &branch.config,
                &extras,
                &mut meter,
            )?;
            let mut cache = Cache::new(policy.cache_entries);
            let mut visit = |q: stream::Query<'_>, meter: &mut Meter| -> Result<bool> {
                meter.charge(q.bits.len() as u64 * 128 + 8192)?;
                if cache.skip(q.bits, q.nodes) {
                    return Ok(false);
                }
                let ids = index.query(q.counts, q.nodes, meter)?;
                if ids.is_empty() {
                    cache.record_refusal(q.bits, q.nodes);
                    return Ok(false);
                }
                let formula = (q.make)();
                if formula.nodes() != q.nodes {
                    return Err("streamed AST size differs".into());
                }
                // Charge all temporary assembled roots, bounded sorting and
                // retention before allocation. Positive matches never enter
                // the refusal cache: unseen semantics can differ (CM63).
                meter.charge((ids.len() as u64 + 32) * 127 * 256)?;
                let complete = index.assemble(&formula, &ids, policy.quotient.retained);
                retained.offer(complete, meter.work)?;
                Ok(false)
            };
            let streamed = if let Some(bank) = query_bank {
                (|| -> Result<()> {
                    for c in &bank.rows {
                        attempt.stream.visited_queries += 1;
                        if visit(
                            stream::Query {
                                bits: &c.bits,
                                counts: &c.counts,
                                nodes: c.nodes,
                                make: &|| c.formula.clone(),
                            },
                            &mut meter,
                        )? {
                            attempt.stream.found = true;
                            return Ok(());
                        }
                    }
                    attempt.stream.complete = true;
                    Ok(())
                })()
            } else {
                stream::enumerate(
                    pool,
                    children,
                    wanted,
                    &bounds,
                    &mut meter,
                    &mut attempt.stream,
                    &mut visit,
                )
            };
            attempt.index = index.stats;
            attempt.cache_hits = cache.hits;
            attempt.cheaper_retests = cache.cheaper_retests;
            attempt.cache_evictions = cache.evictions;
            attempt.cache_peak_entries = cache.peak_entries;
            attempt.stream.found = retained.stats.successful_queries > completed_before;
            streamed
        })();
        match result {
            Ok(()) => (),
            Err(e) => {
                let reason = e.to_string();
                if !is_budget_stop(&reason) {
                    return Err(e);
                }
                attempt.refusal = Some(reason.clone());
                out.stopped_reason = Some(reason);
                retained.stats.interrupted_after_fit = !retained.formulas().is_empty();
            }
        }
        eprintln!(
            "stream branch {}: {} anchor classes, {} queries, {} programs, work {}, refusal {:?}",
            out.attempts.len() + 1,
            attempt.anchor_classes,
            attempt.index.queries,
            retained.formulas().len(),
            meter.work,
            attempt.refusal
        );
        out.attempts.push(attempt);
        if out.stopped_reason.is_some() {
            break;
        }
    }
    if out.stopped_reason.is_none() && out.attempts.len() < out.possible_branches {
        out.stopped_reason = Some("64-branch search cap reached".into());
        retained.stats.interrupted_after_fit = !retained.formulas().is_empty();
    }
    out.retention = retained.stats.clone();
    out.programs = retained
        .into_formulas()
        .into_iter()
        .map(Program::Prefix)
        .collect();
    meter.limit = work_limit;
    meter.charge(audit_work)?;
    out.charged_scalar_bound = meter.work;
    if out.programs.is_empty() && out.refused.is_none() {
        out.refused = Some(out.stopped_reason.clone().unwrap_or_else(|| {
            "no fit in streamed joins relative to retained anchors within64 branches".into()
        }));
    }
    audit(&out.programs, rows)?;
    Ok(out)
}

pub(crate) fn is_budget_stop(reason: &str) -> bool {
    matches!(
        reason,
        "transition search work cap exhausted"
            | "left anchor work cap exhausted"
            | "right anchor work cap exhausted"
    )
}

pub(crate) fn audit_bound(rows: &[Observation], retained: usize) -> Result<u64> {
    // Two independent interpreters, AST traversals, feature/label maps and
    // allocator slack; every syntax has at most127 nodes and prefixes4096.
    let words: u128 = rows.iter().map(|r| r.tokens.len() as u128 + 1).sum();
    Ok(u64::try_from(
        1048576 + words * retained as u128 * 127 * 256,
    )?)
}
