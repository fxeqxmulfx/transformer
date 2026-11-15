//! Prefer a complete anchor, then stream the opposite grammar. A capped anchor
//! remains explicitly conditional even when its opposite stream completes.
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
    let branches = transitions::branches(pool, rows)?;
    let mut out = Search {
        possible_branches: branches.len(),
        attempts: vec![],
        programs: vec![],
        charged_scalar_bound: used,
        refused: None,
    };
    let mut meter = Meter {
        work: used,
        limit: search::WORK_LIMIT,
    };
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
        let result = (|| -> Result<Vec<Formula>> {
            if attempt.left.work_exhausted {
                return Err("left anchor work cap exhausted".into());
            }
            if left.rows.is_empty() {
                attempt.stream.complete = true;
                return Ok(vec![]);
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
                return Ok(vec![]);
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
            let mut perfect = vec![];
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
                perfect = index.assemble(&formula, &ids, policy.quotient.retained);
                Ok(true)
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
            streamed?;
            Ok(perfect)
        })();
        match result {
            Ok(formulas) => out.programs = formulas.into_iter().map(Program::Prefix).collect(),
            Err(e) => {
                attempt.refusal = Some(e.to_string());
                out.refused = Some(e.to_string());
            }
        }
        eprintln!(
            "stream branch {}: {} anchor classes, {} queries, {} programs, work {}, refusal {:?}",
            out.attempts.len() + 1,
            attempt.anchor_classes,
            attempt.index.queries,
            out.programs.len(),
            meter.work,
            attempt.refusal
        );
        out.attempts.push(attempt);
        if out.refused.is_some() || !out.programs.is_empty() {
            break;
        }
    }
    out.charged_scalar_bound = meter.work;
    if out.programs.is_empty() && out.refused.is_none() {
        out.refused =
            Some("no fit in streamed joins relative to retained anchors within64 branches".into());
    }
    audit(&out.programs, rows)?;
    Ok(out)
}
