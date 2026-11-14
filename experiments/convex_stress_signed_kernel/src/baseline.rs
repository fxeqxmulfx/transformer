//! Source-excluded CM46 baseline, ported from gated_context/bank.rs and
//! diagnostic.rs atc60816fa. Exact original diagnostic totals are checked.
use crate::*;

pub(crate) fn gated(parent: &parent::Model) -> &gated::Model {
    parent.parent().parent().parent()
}

pub(crate) fn logits(parent: &parent::Model, cell: usize, labels: &[usize]) -> Vec<f64> {
    let g = gated(parent);
    let source = parent.source();
    let (group, _) = source.location(cell);
    let b = &source.blocks[group];
    let ambiguous = b
        .observed_labels
        .first()
        .is_some_and(|y| b.observed_labels.iter().any(|v| v != y));
    let sorted = [[1, 1], [4, 1], [1, 4]].map(|axis| {
        let mut items: Vec<_> = b
            .observed
            .iter()
            .enumerate()
            .filter(|&(_, &other)| other != cell)
            .map(|(state, &other)| {
                let dx = (cell % source.n).abs_diff(other % source.n);
                let dy = (cell / source.n).abs_diff(other / source.n);
                (axis[0] * dx * dx + axis[1] * dy * dy, other, state)
            })
            .collect();
        items.sort_unstable();
        items
    });
    let mut bank = Vec::with_capacity(g.bank().len());
    for context in g.bank() {
        let metric = [[1, 1], [4, 1], [1, 4]]
            .iter()
            .position(|axis| *axis == context.axis_weights)
            .unwrap();
        let items = &sorted[metric][..context.neighbors.min(sorted[metric].len())];
        let mut z = vec![0.; labels.len()];
        if !ambiguous || items.is_empty() {
            z[labels.binary_search(&b.default_label).unwrap()] = gated::scale::context::MARGIN;
            bank.push(z);
            continue;
        }
        let raw = |d: usize| 1. / (d as f64).powi(context.inverse_power as i32);
        let total: f64 = items.iter().map(|&(d, _, _)| raw(d)).sum();
        for &(d, _, state) in items {
            let mass = context.scale * raw(d) / total;
            z[labels.binary_search(&b.observed_labels[state]).unwrap()] +=
                gated::scale::context::MARGIN * mass;
        }
        bank.push(z);
    }
    let mut anchor = vec![0.; labels.len()];
    for (&alpha, z) in g.anchor().iter().zip(&bank) {
        for (value, &raw) in anchor.iter_mut().zip(z) {
            *value += alpha * raw;
        }
    }
    let mut result: Vec<_> = anchor.iter().map(|v| (1. - g.radius()) * v).collect();
    for (&alpha, z) in g.selection().weights[group].iter().zip(&bank) {
        for (value, &raw) in result.iter_mut().zip(z) {
            *value += g.radius() * alpha * raw;
        }
    }
    result
}
