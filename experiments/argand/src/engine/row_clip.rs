//! Projection of each complex output row of the seven learned decoder
//! matrices onto a Euclidean ball. Tied embeddings and scalar gates are separate.
use super::{Engine, Result};
use kernite_cuda::{Buffer, Grid};

pub(super) struct RowClip {
    rows: Buffer<u32>,
    count: u32,
    radius: f32,
}

fn descriptors(engine: &Engine) -> Result<Vec<[u32; 3]>> {
    let mut rows = Vec::new();
    for layer in &engine.parameters.layers {
        for projection in layer.projections() {
            let (offset, inputs, outputs) = projection.layout();
            for j in 0..outputs {
                rows.push([
                    u32::try_from(offset + 2 * j)?,
                    u32::try_from(2 * outputs)?,
                    u32::try_from(inputs)?,
                ]);
            }
        }
    }
    Ok(rows)
}

impl Engine {
    pub(crate) fn clip_projection_rows(&self, radius: f32) -> Result<()> {
        RowClip::new(self, radius)?.apply(self)
    }

    /// Empirical quantiles of complex Euclidean output-row norms, in FP64.
    pub(crate) fn projection_row_norms(&self) -> Result<serde_json::Value> {
        let values = self.weights()?;
        let mut norms = Vec::new();
        for [start, stride, count] in descriptors(self)? {
            let squared = (0..count)
                .map(|i| {
                    let k = start as usize + i as usize * stride as usize;
                    let re = f64::from(values[k]);
                    let im = f64::from(values[k + 1]);
                    re * re + im * im
                })
                .sum::<f64>();
            norms.push(squared.sqrt());
        }
        if norms.is_empty() {
            return Ok(serde_json::json!({"count":0}));
        }
        norms.sort_by(f64::total_cmp);
        let n = norms.len();
        Ok(
            serde_json::json!({"count":n,"median":norms[n/2],"p90":norms[9*n/10],"maximum":norms[n-1]}),
        )
    }
}

impl RowClip {
    pub fn new(engine: &Engine, radius: f32) -> Result<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err("row radius must be positive and finite".into());
        }
        let rows = descriptors(engine)?;
        Ok(Self {
            count: u32::try_from(rows.len())?,
            rows: engine.cuda.upload(rows.as_flattened())?,
            radius,
        })
    }

    pub fn apply(&self, engine: &Engine) -> Result<()> {
        // SAFETY: one warp owns an output row, including both independent real
        // coordinates of every coefficient. The row descriptors are disjoint.
        if self.count > 0 {
            unsafe {
                engine.kernel.clip_complex_rows(
                    Grid::new(self.count, 32),
                    engine.parameters.value.ptr(),
                    self.rows.ptr(),
                    self.radius,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::Shape;

    #[test]
    #[ignore = "requires CUDA"]
    fn output_rows_share_a_complex_norm_and_leave_other_parameters_untouched() -> Result<()> {
        let mut engine = Engine::new(
            Shape {
                width: 16,
                heads: 2,
                layers: 1,
                vocab: 4,
                max_batch: 2,
                max_length: 2,
            },
            1,
            17,
        )?;
        let rows = RowClip::new(&engine, 2.0)?;
        let mut input = vec![0.0_f32; engine.parameters.len()];
        input[0] = 9.0;
        let layer = &engine.parameters.layers[0];
        input[layer.score] = 3.0;
        let mut expected = input.clone();
        for proj in layer.projections() {
            let (base, _, outputs) = proj.layout();
            input[base] = 3.0;
            input[base + 2 * outputs + 1] = 4.0;
            input[base + 2] = 1.5;
            expected[base] = 1.2;
            expected[base + 2 * outputs + 1] = 1.6;
            expected[base + 2] = 1.5;
        }
        engine.restore(&input)?;
        assert!(RowClip::new(&engine, f32::NAN).is_err());
        assert_eq!(engine.projection_row_norms()?["maximum"], 5.0);
        rows.apply(&engine)?;
        let after = engine.projection_row_norms()?;
        assert!((after["maximum"].as_f64().unwrap() - 2.0).abs() < 3e-7);
        for (out, want) in engine.weights()?.into_iter().zip(&expected) {
            assert!((out - want).abs() < 3e-7);
        }
        Ok(())
    }
}
