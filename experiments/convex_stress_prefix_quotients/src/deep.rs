//! CM59 branch-conditioned depth-three discovery. Stop at the first fitting
//! branch in declared order; truncated child banks remain explicit in the report.
use crate::{
    bank::Bank,
    latent::{Meter, Stats},
    pool::Pool,
    transitions::{self, Config},
    *,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub config: Config,
    pub left_constraints: usize,
    pub right_constraints: usize,
    pub left: Stats,
    pub right: Option<Stats>,
    pub root_pairs: u64,
    pub root_row_comparisons: u64,
    pub fitting_syntaxes: u64,
    pub work_exhausted: bool,
    pub index: Option<pairs::Stats>,
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
            let atom = p.run(&r.tokens).ok_or("deep candidate undefined")?;
            if Some(atom) != p.interpret(&r.tokens) {
                return Err("independent deep interpreter mismatch".into());
            }
            if let Some(old) = labels.insert(atom, r.label)
                && old != r.label
            {
                return Err("deep candidate fails original labels".into());
            }
        }
    }
    Ok(())
}
pub fn run(
    pool: &Pool,
    rows: &[Observation],
    children: &Bank,
    cap: usize,
    retained: usize,
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
        limit: 40_000_000_000_000,
    };
    let high = rows.iter().map(|r| r.label).max().ok_or("missing labels")?;
    let mut observed = BTreeMap::new();
    for (&i, r) in pool.row_nodes.iter().zip(rows) {
        observed.insert(i, r.label == high);
    }
    let symbol_counts: Vec<_> = (0..3)
        .map(|s| Ok(pool.counts(&pool.signature(&grammar::literal(s))?)))
        .collect::<Result<_>>()?;
    for branch in branches.into_iter().take(64) {
        let (left, left_stats) = latent::synthesize(pool, children, &branch.left, cap, &mut meter)?;
        let mut attempt = Attempt {
            config: branch.config.clone(),
            left_constraints: branch.left.len(),
            right_constraints: branch.right.len(),
            left: left_stats,
            right: None,
            root_pairs: 0,
            root_row_comparisons: 0,
            fitting_syntaxes: 0,
            work_exhausted: false,
            index: None,
        };
        if attempt.left.work_exhausted {
            out.attempts.push(attempt);
            out.refused = Some("transition search work cap exhausted".into());
            break;
        }
        if left.rows.is_empty() {
            out.attempts.push(attempt);
            continue;
        }
        let (right, right_stats) =
            latent::synthesize(pool, children, &branch.right, cap, &mut meter)?;
        let exhausted = right_stats.work_exhausted;
        attempt.right = Some(right_stats);
        if exhausted {
            out.attempts.push(attempt);
            out.refused = Some("transition search work cap exhausted".into());
            break;
        }
        if right.rows.is_empty() {
            out.attempts.push(attempt);
            continue;
        }
        let (perfect, index) = pairs::run(
            pool,
            &observed,
            (&left, &right),
            &branch.config,
            &symbol_counts,
            retained,
            &mut meter,
        )?;
        attempt.root_pairs = index.matching_pairs;
        attempt.fitting_syntaxes = index.matching_pairs;
        attempt.work_exhausted = index.work_exhausted;
        attempt.index = Some(index);
        let exhausted = attempt.work_exhausted;
        out.attempts.push(attempt);
        if exhausted {
            out.refused = Some("transition root-pair work cap exhausted".into());
            break;
        }
        if !perfect.is_empty() {
            out.programs = perfect.into_iter().map(Program::Prefix).collect();
            audit(&out.programs, rows)?;
            break;
        }
    }
    out.charged_scalar_bound = meter.work;
    if out.programs.is_empty() && out.refused.is_none() {
        out.refused =
            Some("no compatible root among retained latent programs within64 attempts".into());
    }
    Ok(out)
}
