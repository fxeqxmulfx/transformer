//! CM43 initializer_logits and the CM42 diagnostic atc62022c5. The original
//! context is retained; only confidence is selected. No trained tensor read.
use crate::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub scale: f64,
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub records: usize,
    pub excluded_state_records: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub selected_scale: f64,
    pub selected_neighbors: usize,
    pub candidates: Vec<Candidate>,
}

pub(crate) fn select(model: &context::Model) -> Result<Selection> {
    let source = model.source();
    let k = model.selection().selected_neighbors;
    let mut candidates = BANK.map(|scale| Candidate {
        scale,
        canonical_ce: 0.,
        correct_records: 0,
        records: model.records().len(),
        excluded_state_records: 0,
    });
    for record in model.records() {
        let (block, _) = source.location(record.cell);
        let b = &source.blocks[block];
        let mut distances = Vec::new();
        if k > 0 && b.observed_labels.iter().collect::<BTreeSet<_>>().len() > 1 {
            for (state, &other) in b.observed.iter().enumerate() {
                if other == record.cell {
                    continue;
                }
                let dx = (record.cell % source.n).abs_diff(other % source.n);
                let dy = (record.cell / source.n).abs_diff(other / source.n);
                distances.push((dx * dx + dy * dy, other, state));
            }
            distances.sort_unstable();
            distances.truncate(k);
        }
        let mut base = vec![0.; source.vocabulary];
        if distances.is_empty() {
            base[b.default_label] = context::MARGIN;
        } else {
            for &(_, _, state) in &distances {
                base[b.observed_labels[state]] += context::MARGIN / distances.len() as f64;
            }
        }
        for candidate in &mut candidates {
            // Unmodified rows retain scale one in the diagnostic as in prediction.
            let scale = if distances.is_empty() {
                1.
            } else {
                candidate.scale
            };
            let z: Vec<_> = base.iter().map(|v| scale * v).collect();
            candidate.canonical_ce += kernel::training::ce(&z, record.label);
            let winner = z
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .ok_or("empty canonical vocabulary")?
                .0;
            candidate.correct_records += usize::from(winner == record.label);
            candidate.excluded_state_records += usize::from(!distances.is_empty());
        }
    }
    for candidate in &mut candidates {
        candidate.canonical_ce /= candidate.records as f64;
    }
    let original = model
        .selection()
        .candidates
        .iter()
        .find(|c| c.neighbors == k)
        .ok_or("missing original context diagnostic")?;
    if (candidates[0].canonical_ce - original.canonical_ce).abs() > 1e-12
        || candidates[0].correct_records != original.correct_records
        || candidates[0].excluded_state_records != original.excluded_state_records
    {
        return Err("scale-one diagnostic differs from reviewed context policy".into());
    }
    let mut selected = 0;
    for i in 1..candidates.len() {
        if candidates[i].canonical_ce + 1e-10 < candidates[selected].canonical_ce {
            selected = i;
        }
    }
    Ok(Selection {
        selected_scale: candidates[selected].scale,
        selected_neighbors: k,
        candidates: candidates.to_vec(),
    })
}
