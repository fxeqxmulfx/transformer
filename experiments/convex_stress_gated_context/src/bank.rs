//! Distance/weight implementation ported unchanged from CM45 bank.rs at14b8ad6e.
//! The bank itself comes from the externally bound parent, with no new contexts.
use crate::*;

pub(crate) const METRICS: [[usize; 2]; 3] = [[1, 1], [4, 1], [1, 4]];
type Distance = (usize, usize, usize);

/// Sort once per metric, including the exact reviewed tie order.
pub(crate) fn distances(source: &kernel::Model, cell: usize) -> [Vec<Distance>; 3] {
    let (block, _) = source.location(cell);
    METRICS.map(|axis| {
        let mut items: Vec<_> = source.blocks[block]
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
    })
}

pub(crate) fn weights(context: &Context, sorted: &[Vec<Distance>; 3]) -> Vec<Weight> {
    let metric = METRICS
        .iter()
        .position(|m| *m == context.axis_weights)
        .unwrap();
    let items = &sorted[metric][..context.neighbors.min(sorted[metric].len())];
    let raw = |d: usize| 1. / (d as f64).powi(context.inverse_power as i32);
    let total: f64 = items.iter().map(|&(d, _, _)| raw(d)).sum();
    items
        .iter()
        .map(|&(d, _, state)| Weight {
            state,
            mass: context.scale * raw(d) / total,
        })
        .collect()
}

pub(crate) fn ambiguous(b: &kernel::model::Block) -> bool {
    b.observed_labels
        .first()
        .is_some_and(|y| b.observed_labels.iter().any(|v| v != y))
}
