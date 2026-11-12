//! Bounded proposals for the exact CM18 checker. This is native propagation,
//! not the classical Lean `recover`: no exhaustive global consistency claim.
//! Anchor choice and selection use only observed labels and certified coverage.
use crate::{Certificate, Result, Training, Verified, check};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    pub rectangles: u64,
    pub candidates: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            rectangles: 4096,
            candidates: 16384,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCosts {
    pub row_pairs: u64,
    pub rectangles: u64,
    pub candidates: u64,
    pub phase_tests: u64,
    pub checked_certificates: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStop {
    CompleteCoverage,
    SearchLimit,
    SearchExhausted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum AttemptOutcome {
    PropagationRefused,
    CheckerRefused,
    Accepted { covered_cells: usize },
}

/// Every tried phase proposal, including refusals. The deterministic search can
/// reconstruct its phase vectors from this training-only anchor and difference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Attempt {
    pub anchor: [usize; 4],
    pub delta: usize,
    pub outcome: AttemptOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Discovery {
    pub verified: Option<Verified>,
    pub costs: SearchCosts,
    /// Exhaustion or a limit does not assert that no latent completion exists.
    pub stop: SearchStop,
    pub attempts: Vec<Attempt>,
}

/// At most two candidates from one label, checked against every reached neighbor.
/// A solution must be unique to extend the reached region. CM18 `step_unique`
/// proves uniqueness when two neighbors have distinct phases.
fn infer(
    training: &Training,
    neighbors: &[(usize, usize)],
    costs: &mut SearchCosts,
) -> Option<usize> {
    let q = training.shape().latent_states();
    let &(phase, label) = neighbors.first()?;
    let mut solutions = training
        .states(label)
        .into_iter()
        .map(|z| (z + q - phase) % q)
        .filter(|&candidate| {
            neighbors.iter().all(|&(other, observed)| {
                costs.phase_tests += 1;
                training.decoder().class_of_state((candidate + other) % q) == observed
            })
        });
    let first = solutions.next()?;
    solutions.next().is_none().then_some(first)
}

fn neighbors(
    phases: &[Option<usize>],
    mut label: impl FnMut(usize) -> Option<usize>,
) -> Vec<(usize, usize)> {
    phases
        .iter()
        .enumerate()
        .filter_map(|(i, &phase)| Some((phase?, label(i)?)))
        .collect()
}

fn distinct(values: &[(usize, usize)]) -> bool {
    values
        .first()
        .is_some_and(|&(phase, _)| values.iter().any(|&(p, _)| p != phase))
}

fn propose(
    training: &Training,
    anchor: [usize; 4],
    delta: usize,
    costs: &mut SearchCosts,
) -> Option<Certificate> {
    let [top, bot, left, right] = anchor;
    let n = training.shape().side;
    let mut row = vec![None; n];
    let mut col = vec![None; n];
    row[top] = Some(0);
    row[bot] = Some(delta);
    for (j, phase) in col.iter_mut().enumerate() {
        if let (Some(a), Some(b)) = (training.at(top, j), training.at(bot, j)) {
            *phase = Some(infer(training, &[(0, a), (delta, b)], costs)?);
        }
    }
    loop {
        let mut changed = false;
        for (i, phase) in row.iter_mut().enumerate() {
            if phase.is_none() {
                let adjacent = neighbors(&col, |j| training.at(i, j));
                if distinct(&adjacent) {
                    *phase = Some(infer(training, &adjacent, costs)?);
                    changed = true;
                }
            }
        }
        for (j, phase) in col.iter_mut().enumerate() {
            if phase.is_none() {
                let adjacent = neighbors(&row, |i| training.at(i, j));
                if distinct(&adjacent) {
                    *phase = Some(infer(training, &adjacent, costs)?);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Some(Certificate {
        row: row.into_iter().map(|z| z.unwrap_or(0)).collect(),
        col: col.into_iter().map(|z| z.unwrap_or(0)).collect(),
        top,
        bot,
        left,
        right,
    })
}

pub fn discover(training: &Training, limits: Limits) -> Result<Discovery> {
    if limits.rectangles > 1_000_000 || limits.candidates > 4_000_000 {
        return Err("proposal limits exceed the declared work cap".into());
    }
    let (n, q) = (training.shape().side, training.shape().latent_states());
    let mut out = Discovery {
        verified: None,
        costs: SearchCosts::default(),
        stop: SearchStop::SearchExhausted,
        attempts: vec![],
    };
    for top in 0..n {
        for bot in top + 1..n {
            out.costs.row_pairs += 1;
            let shared: Vec<_> = (0..n)
                .filter(|&j| training.at(top, j).is_some() && training.at(bot, j).is_some())
                .collect();
            if !shared
                .iter()
                .any(|&j| training.at(top, j) != training.at(bot, j))
            {
                continue;
            }
            for (index, &left) in shared.iter().enumerate() {
                for &right in &shared[index + 1..] {
                    if out.costs.rectangles == limits.rectangles {
                        out.stop = SearchStop::SearchLimit;
                        return Ok(out);
                    }
                    out.costs.rectangles += 1;
                    let pair = (
                        training.at(top, left).unwrap(),
                        training.at(bot, left).unwrap(),
                    );
                    let other = (
                        training.at(top, right).unwrap(),
                        training.at(bot, right).unwrap(),
                    );
                    if pair == other || pair == (other.1, other.0) {
                        continue;
                    }
                    let mut deltas = Vec::new();
                    for a in training.states(pair.0) {
                        for b in training.states(pair.1) {
                            let delta = (b + q - a) % q;
                            if delta != 0 && !deltas.contains(&delta) {
                                deltas.push(delta);
                            }
                        }
                    }
                    for delta in deltas {
                        if out.costs.candidates == limits.candidates {
                            out.stop = SearchStop::SearchLimit;
                            return Ok(out);
                        }
                        out.costs.candidates += 1;
                        let index = out.attempts.len();
                        out.attempts.push(Attempt {
                            anchor: [top, bot, left, right],
                            delta,
                            outcome: AttemptOutcome::PropagationRefused,
                        });
                        let Some(c) =
                            propose(training, [top, bot, left, right], delta, &mut out.costs)
                        else {
                            continue;
                        };
                        out.costs.checked_certificates += 1;
                        out.attempts[index].outcome = AttemptOutcome::CheckerRefused;
                        let Ok(verified) = check(training, &c) else {
                            continue;
                        };
                        let coverage = verified.covered_cells();
                        out.attempts[index].outcome = AttemptOutcome::Accepted {
                            covered_cells: coverage,
                        };
                        if out
                            .verified
                            .as_ref()
                            .is_none_or(|old| coverage > old.covered_cells())
                        {
                            out.verified = Some(verified);
                        }
                        if coverage == n * n {
                            out.stop = SearchStop::CompleteCoverage;
                            return Ok(out);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}
