//! Independent scalar FP64 oracle for the equations, never used for training.
use crate::domain::model::{Arm, E, W};
fn norm(x: &[f64]) -> Vec<f64> {
    let scale =
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f32::EPSILON as f64).sqrt();
    x.iter().map(|v| v / scale).collect()
}
fn linear(x: &[f64], weight: &[f64]) -> Vec<f64> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect()
}
pub fn product(z: [f64; 4], arm: Arm) -> [f64; 2] {
    let [a, b, u, v] = z;
    let gate = if arm == Arm::Plain {
        1.0
    } else {
        1.0 / (1.0 + (-(a * a + b * b + f32::EPSILON as f64).sqrt()).exp())
    };
    if arm == Arm::Diagonal {
        [gate * a * u, gate * b * v]
    } else {
        [
            gate * (a * u - b * v) / 2.0f64.sqrt(),
            gate * (a * v + b * u) / 2.0f64.sqrt(),
        ]
    }
}
pub fn logits(arm: Arm, p: &[f64], rows: &[u32]) -> Vec<f64> {
    let h = arm.hidden();
    let mut logits = Vec::new();
    for row in rows.as_chunks::<3>().0.iter() {
        let (a, b) = (row[0] as usize, row[1] as usize);
        let input: Vec<f64> = (0..W)
            .map(|c| {
                p[a * W + c]
                    + if c % 2 == 0 {
                        -p[b * W + c + 1]
                    } else {
                        p[b * W + c - 1]
                    }
            })
            .collect();
        let x = norm(&input);
        let gv = linear(&norm(&x), &p[E..E + 2 * h * W]);
        let mut hidden = vec![0.0; h];
        if arm == Arm::Swiglu {
            for c in 0..h {
                hidden[c] = gv[c] / (1.0 + (-gv[c]).exp()) * gv[c + h];
            }
        } else {
            for c in 0..h / 2 {
                let z = product([gv[c], gv[c + h / 2], gv[c + h], gv[c + 3 * h / 2]], arm);
                hidden[c] = z[0];
                hidden[c + h / 2] = z[1];
            }
        }
        let branch = norm(&linear(&hidden, &p[E + 2 * h * W..]));
        let out = norm(&x.iter().zip(branch).map(|(a, b)| a + b).collect::<Vec<_>>());
        logits.extend(linear(&out, &p[..E]));
    }
    logits
}
pub fn loss(arm: Arm, p: &[f64], rows: &[u32]) -> f64 {
    logits(arm, p, rows)
        .as_chunks::<97>()
        .0
        .iter()
        .zip(rows.as_chunks::<3>().0.iter())
        .map(|(row, input)| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max - row[input[2] as usize]
        })
        .sum::<f64>()
        / (rows.len() / 3) as f64
}
