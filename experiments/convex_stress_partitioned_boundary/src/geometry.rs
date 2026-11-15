//! Input-only extrema. Raw class labels, labels' numerical names, task names,
//! generator rules and original evaluation queries never enter this map.
use crate::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frame {
    Grid,
    NonmodalBox,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bank {
    Axes,
    Diagonals,
}
pub const FRAMES: [Frame; 2] = [Frame::Grid, Frame::NonmodalBox];
pub const BANKS: [Bank; 2] = [Bank::Axes, Bank::Diagonals];
impl Bank {
    pub fn directions(self) -> &'static [[f64; 2]] {
        match self {
            Self::Axes => &[[1., 0.], [0., 1.]],
            Self::Diagonals => &[[1., 0.], [0., 1.], [1., 1.], [1., -1.]],
        }
    }
    pub fn dimension(self) -> usize {
        1 + 2 * self.directions().len()
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    pub frame: Frame,
    pub bank: Bank,
    pub modal: usize,
    pub bounds: [usize; 4],
    pub counts: Vec<usize>,
    pub descriptors: Vec<Vec<f64>>,
}
impl Geometry {
    pub fn new(input: &Input, frame: Frame, bank: Bank) -> Result<Self> {
        input.validate()?;
        let mut frequency = vec![0usize; input.blocks];
        for &b in &input.route {
            frequency[b] += 1;
        }
        let modal = (0..input.blocks)
            .max_by_key(|&b| (frequency[b], std::cmp::Reverse(b)))
            .unwrap();
        let mut bounds = [0, input.n - 1, 0, input.n - 1];
        if frame == Frame::NonmodalBox {
            let cells: Vec<_> = input
                .route
                .iter()
                .enumerate()
                .filter(|&(_, &b)| b != modal)
                .map(|(i, _)| i)
                .collect();
            if cells.is_empty() {
                return Err("nonmodal route rectangle is empty".into());
            }
            bounds = [
                cells.iter().map(|i| i / input.n).min().unwrap(),
                cells.iter().map(|i| i / input.n).max().unwrap(),
                cells.iter().map(|i| i % input.n).min().unwrap(),
                cells.iter().map(|i| i % input.n).max().unwrap(),
            ];
        }
        let mut descriptors = vec![vec![0.; bank.dimension()]; input.blocks];
        let mut counts = vec![0usize; input.blocks];
        for (cell, &b) in input.route.iter().enumerate() {
            let (u, v) = (cell / input.n, cell % input.n);
            if u < bounds[0] || u > bounds[1] || v < bounds[2] || v > bounds[3] {
                continue;
            }
            let point = convex_stress_shared_head::features::point(input.n, cell);
            let row = &mut descriptors[b];
            row[0] = 1.;
            for (j, dir) in bank.directions().iter().enumerate() {
                let x = (dir[0] * point[0] + dir[1] * point[1]) / (dir[0].abs() + dir[1].abs());
                if counts[b] == 0 {
                    row[1 + 2 * j] = x;
                    row[2 + 2 * j] = x;
                } else {
                    row[1 + 2 * j] = row[1 + 2 * j].min(x);
                    row[2 + 2 * j] = row[2 + 2 * j].max(x);
                }
            }
            counts[b] += 1;
        }
        Ok(Self {
            frame,
            bank,
            modal,
            bounds,
            counts,
            descriptors,
        })
    }
    pub fn validate(&self, input: &Input) -> Result<()> {
        if self != &Self::new(input, self.frame, self.bank)? {
            return Err("stored geometry differs from complete input route".into());
        }
        Ok(())
    }
}
