//! Fixed ordinal context bank. Input coordinates set distances; raw labels
//! only identify ambiguous source blocks and the training diagnostic targets.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub neighbors: usize,
    pub inverse_power: u32,
    pub axis_weights: [usize; 2],
    pub scale: f64,
}

pub(crate) const METRICS: [[usize; 2]; 3] = [[1, 1], [4, 1], [1, 4]];
type Distance = (usize, usize, usize);

pub(crate) fn build(parent: &parent::Model) -> Vec<Context> {
    let mut bank = vec![Context {
        neighbors: parent.selection().selected_neighbors,
        inverse_power: 0,
        axis_weights: METRICS[0],
        scale: parent.selection().selected_scale,
    }];
    for neighbors in [1, 3, 5, 9, 17] {
        for inverse_power in [0, 1, 2] {
            for axis_weights in METRICS {
                for scale in [0.25, 0.5] {
                    bank.push(Context {
                        neighbors,
                        inverse_power,
                        axis_weights,
                        scale,
                    });
                }
            }
        }
    }
    bank
}

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
