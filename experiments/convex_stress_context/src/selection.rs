//! CM42 initializer_logits: selection is fixed from original source labels,
//! not trained coefficients. The canonical logits equal margin times raw-label
//! neighbor weight. This is a training diagnostic, not a held-out certificate.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub neighbors: usize,
    pub canonical_ce: f64,
    pub correct_records: usize,
    pub records: usize,
    pub excluded_state_records: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub selected_neighbors: usize,
    pub candidates: Vec<Candidate>,
}

pub(crate) fn select(source: &kernel::Model, records: &[records::Record]) -> Selection {
    let mut candidates = Vec::new();
    for k in BANK {
        let mut ce = 0.;
        let mut correct = 0;
        let mut excluded = 0;
        for record in records {
            let (block, _) = source.location(record.cell);
            let b = &source.blocks[block];
            let states = if k > 0 && ambiguous(b) {
                nearest(source, block, record.cell, k)
            } else {
                vec![]
            };
            let mut logits = vec![0.; source.vocabulary];
            if states.is_empty() {
                logits[b.default_label] = MARGIN;
            } else {
                excluded += 1;
                for &s in &states {
                    logits[b.observed_labels[s]] += MARGIN / states.len() as f64;
                }
            }
            ce += kernel::training::ce(&logits, record.label);
            let winner = logits
                .iter()
                .enumerate()
                .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                .unwrap()
                .0;
            correct += usize::from(winner == record.label);
        }
        candidates.push(Candidate {
            neighbors: k,
            canonical_ce: ce / records.len() as f64,
            correct_records: correct,
            records: records.len(),
            excluded_state_records: excluded,
        });
    }
    let mut selected = 0;
    for i in 1..candidates.len() {
        if candidates[i].canonical_ce + 1e-10 < candidates[selected].canonical_ce {
            selected = i;
        }
    }
    Selection {
        selected_neighbors: candidates[selected].neighbors,
        candidates,
    }
}
