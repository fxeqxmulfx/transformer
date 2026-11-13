//! CFt4 scan over original labels. The first surviving candidate is a declared
//! prior; a consistent candidate is not a proof of arbitrary unseen answers.
use crate::{
    features::{self, Candidate},
    *,
};
use std::collections::BTreeMap;
mod check;
pub use check::check;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Conflict { first: usize, second: usize },
    Admitted,
    VisitCap,
    StateCap,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub candidate: Candidate,
    pub visits: usize,
    pub distinct_values: usize,
    pub decision: Decision,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decoder {
    pub candidate: Candidate,
    pub values: Vec<[u32; 2]>,
    pub labels: Vec<usize>,
}
impl Decoder {
    pub fn state(&self, value: [u32; 2]) -> Option<usize> {
        self.values.binary_search(&value).ok()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub query: usize,
    pub observations: usize,
    pub attempts: Vec<Attempt>,
    pub decoder: Option<Decoder>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub validated_tokens: usize,
    pub feature_cells: usize,
    pub visits: usize,
    pub attempts: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub groups: Vec<Group>,
    pub work: Work,
    pub reason: Option<String>,
    pub buffer_preflight_bytes: u64,
    pub unseen_identification: bool,
}
pub fn run(request: Request) -> Result<Report> {
    let bytes = request.validate()?;
    let x = &request.input;
    let mut out = Report {
        request: request.clone(),
        groups: vec![],
        work: Work {
            validated_tokens: x.observations.iter().map(|s| s.tokens.len()).sum(),
            ..Work::default()
        },
        reason: None,
        buffer_preflight_bytes: bytes,
        unseen_identification: false,
    };
    let keys = features::library(x.vocabulary, x.context).len();
    let cells = keys * x.observations.len();
    if cells > request.policy.feature_cells {
        out.reason = Some("feature cache cap".into());
        return Ok(out);
    }
    let cache: Vec<_> = x
        .observations
        .iter()
        .map(|s| features::encode(x.vocabulary, x.context, &s.tokens))
        .collect();
    out.work.feature_cells = cells;
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for (i, s) in x.observations.iter().enumerate() {
        grouped
            .entry(*s.tokens.last().unwrap())
            .or_default()
            .push(i);
    }
    for (query, indices) in grouped {
        let mut group = Group {
            query,
            observations: indices.len(),
            attempts: vec![],
            decoder: None,
        };
        for candidate in features::catalogue(x.vocabulary, x.context) {
            if out.work.attempts == request.policy.attempts {
                out.reason = Some("candidate trace cap".into());
                break;
            }
            out.work.attempts += 1;
            let mut seen = BTreeMap::<[u32; 2], (usize, usize)>::new();
            let mut attempt = Attempt {
                candidate,
                visits: 0,
                distinct_values: 0,
                decision: Decision::Admitted,
            };
            for &index in &indices {
                if out.work.visits == request.policy.visits {
                    attempt.decision = Decision::VisitCap;
                    break;
                }
                out.work.visits += 1;
                attempt.visits += 1;
                let value = candidate.cached(&cache[index]);
                let label = x.observations[index].label;
                if let Some(&(old, first)) = seen.get(&value) {
                    if old != label {
                        attempt.decision = Decision::Conflict {
                            first,
                            second: index,
                        };
                        break;
                    }
                } else {
                    if seen.len() == request.policy.states_per_query {
                        attempt.decision = Decision::StateCap;
                        break;
                    }
                    seen.insert(value, (label, index));
                }
            }
            attempt.distinct_values = seen.len();
            match attempt.decision {
                Decision::Admitted => {
                    group.decoder = Some(Decoder {
                        candidate,
                        values: seen.keys().copied().collect(),
                        labels: seen.values().map(|p| p.0).collect(),
                    })
                }
                Decision::VisitCap => out.reason = Some("candidate observation-visit cap".into()),
                Decision::StateCap | Decision::Conflict { .. } => {}
            }
            group.attempts.push(attempt);
            if group.decoder.is_some() || out.reason.is_some() {
                break;
            }
        }
        let missing = group.decoder.is_none();
        out.groups.push(group);
        if missing {
            out.reason
                .get_or_insert("no consistent candidate in bounded catalogue".into());
            break;
        }
    }
    Ok(out)
}
pub fn replay(request: &Request, report: &Report) -> Result<()> {
    check(request, report)?;
    if run(request.clone())? != *report {
        return Err("original causal search trace/state/work differs".into());
    }
    Ok(())
}
