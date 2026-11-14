//! CM44 row_ce and batch_ce. Repeated inputs retain distinct record IDs.
//! This canonical identity selects a margin; actual products are audited later.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub margin: f64,
    pub mean_ce: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub selected_margin: f64,
    pub original_records: usize,
    pub raw_errors: usize,
    pub candidates: Vec<Candidate>,
}

pub(crate) fn margin_ce(vocab: usize, margin: f64) -> f64 {
    ((vocab - 1) as f64 * (-margin).exp()).ln_1p()
}

pub(crate) fn select(source: &tagged::Model, records: &[records::Record]) -> Selection {
    let errors = records
        .iter()
        .filter(|r| source.output_labels[r.cell] != r.label)
        .count();
    let rate = errors as f64 / records.len() as f64;
    let candidates: Vec<_> = BANK
        .iter()
        .map(|&margin| Candidate {
            margin,
            mean_ce: margin_ce(source.vocab, margin) + margin * rate,
        })
        .collect();
    let mut best = &candidates[0];
    for candidate in &candidates[1..] {
        if candidate.mean_ce + 1e-10 < best.mean_ce {
            best = candidate;
        }
    }
    Selection {
        selected_margin: best.margin,
        original_records: records.len(),
        raw_errors: errors,
        candidates,
    }
}
